#ifndef CLUSTERER_CLASS_HEADER
#define CLUSTERER_CLASS_HEADER

#include <fstream>
#include <array>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <condition_variable> 

#include <tbb/flow_graph.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_access.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/renderer/barrier_operations.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"

#define MAX_LIGHTS 1024u
#define MAX_WORDS uint32_t(std::ceil(MAX_LIGHTS / 32u))
#define Z_BIN_COUNT 8096u
#define CLUSTERED_BUFFERS_SIZE MAX_LIGHTS * 48 + Z_BIN_COUNT * sizeof(uint16_t) * 2 + MAX_LIGHTS * sizeof(uint16_t) + MAX_LIGHTS * sizeof(uint16_t) + 512/*possible alignment correction*/
#define TILE_DATA_SIZE  MAX_WORDS * 4
#define TILE_PIXEL_WIDTH 8
#define TILE_PIXEL_HEIGHT 8
#define TILE_TEST_SAMPLE_COUNT VK_SAMPLE_COUNT_8_BIT
#define POINT_LIGHT_BV_VERTEX_COUNT 96
#define POINT_LIGHT_BV_SIZE sizeof(float) * 3 * POINT_LIGHT_BV_VERTEX_COUNT
#define SPOT_LIGHT_BV_VERTEX_COUNT 30

namespace LightTypes
{
	class LightBase;
	class PointLight;
	class SpotLight;
}

class Clusterer
{
public:
	struct LightFormat
	{
		alignas(16) glm::vec3 position{};
		alignas(4)  float length{};
		alignas(16) glm::vec3 spectrum{};
		alignas(4)  float cutoffCos{};
		alignas(16) glm::vec3 lightDir{};
		alignas(4)  float falloffCos{};

		enum Types : uint8_t
		{
			TYPE_POINT = 0,
			TYPE_SPOT = 1,
			ALL_TYPES
		};
	};
	static_assert(sizeof(LightFormat) == 48);

private:
	VkDevice m_device;
	BufferBaseHostAccessible m_motherBufferShared;
	BufferMapped m_sortedLightData;
	BufferMapped m_binsMinMax;
	BufferBaseHostAccessible m_sortedTypeData;
	BufferBaseHostInaccessible m_tileData;
	std::vector<LightFormat> m_lightData{};
	std::vector<LightFormat::Types> m_typeData{};
	std::vector<glm::vec4> m_boundingSpheres{};
	BufferBaseHostInaccessible m_constData;
	BufferBaseHostInaccessible m_lightBoundingVolumeVertexData;

	uint32_t m_widthInTiles{};
	uint32_t m_heightInTiles{};

	struct CulledLightData
	{
		uint32_t index;
		float front;
		float back;
	};
	CulledLightData* m_nonculledLightsData{ nullptr };
	uint32_t m_nonculledLightsCount{};
	float m_currentFurthestLight{};

	Pipeline m_pointLightTileTestPipeline{};
	uint32_t m_nonculledPointLightCount{ 0 };
	BufferMapped m_instancePointLightIndexData{};
	Pipeline m_spotLightTileTestPipeline{};
	uint32_t m_nonculledSpotLightCount{ 0 };
	BufferMapped m_instanceSpotLightIndexData{};

	glm::mat4 m_currentViewMat{};
	std::array<glm::vec4, 5> m_frustumPlanes{};

	typedef oneapi::tbb::flow::continue_node<oneapi::tbb::flow::continue_msg> node_t;
	typedef const oneapi::tbb::flow::continue_msg& msg_t;
	oneapi::tbb::flow::graph m_flowGraph{};
	node_t m_cullNode{ m_flowGraph, [this](msg_t) { cullLights(); } };
	node_t m_sortNode{ m_flowGraph, [this](msg_t) { sortLights(); } };
	node_t m_fillBuffersNode{ m_flowGraph, [this](msg_t) { fillLightBuffers(); } };
	node_t m_fillBinsNode{ m_flowGraph, [this](msg_t) { fillZBins(); } };

	bool m_countDataReady{ false };

	std::mutex m_mutex{};
	std::condition_variable m_cv{};

public:
	Clusterer(VkDevice device, FrameCommandBufferSet& cmdBufferSet, VkQueue queue, uint32_t windowWidth, uint32_t windowHeight, const BufferMapped& viewprojDataUB);
	~Clusterer();

	void submitPointLight(const glm::vec3& position, const glm::vec3& color, float power, float radius);
	void submitSpotLight(const glm::vec3& position, const glm::vec3& color, float power, float length, glm::vec3 lightDir, float cutoffStartAngle, float cutoffEndAngle);
	void submitFrustum(double near, double far, double aspect, double FOV);
	void submitViewMatrix(const glm::mat4& viewMat);

	void startClusteringProcess()
	{
		m_cullNode.try_put(oneapi::tbb::flow::continue_msg());
	}
	void waitClusteringProcess()
	{
		m_flowGraph.wait_for_all();
	}

	void cmdPassConductTileTest(VkCommandBuffer cb, DescriptorManager& descriptorManager);
	void cmdDrawBVs(VkCommandBuffer cb, DescriptorManager& descriptorManager, Pipeline& pointLPipeline, Pipeline& spotLPipeline, VkRenderingInfo& renderInfo);

	const BufferMapped& getSortedLightsUB() const
	{
		return m_sortedLightData;
	}
	const BufferBaseHostAccessible& getSortedTypeDataUB() const
	{
		return m_sortedTypeData;
	}
	const BufferBaseHostInaccessible& getTileDataSSBO() const
	{
		return m_tileData;
	}
	const BufferMapped& getPointIndicesUB() const
	{
		return m_instancePointLightIndexData;
	}
	const BufferMapped& getSpotIndicesUB() const
	{
		return m_instanceSpotLightIndexData;
	}
	const BufferMapped& getZBinUB() const
	{
		return m_binsMinMax;
	}
	uint32_t getLightNumber() const
	{
		return m_lightData.size();
	}
	float getCurrentBinWidth() const
	{
		return m_currentFurthestLight / Z_BIN_COUNT; //Min value to avoid FP artifacts
	}
	uint32_t getWidthInTiles() const
	{
		return m_widthInTiles;
	}
	uint32_t getHeightInTiles() const
	{
		return m_heightInTiles;
	}

private:
	void cullLights();
	void sortLights();
	void fillLightBuffers();
	void fillZBins();

	bool testSphereAgainstFrustum(const glm::vec4& sphereData);
	void computeFrontAndBack(const LightFormat& light, LightFormat::Types type, float& front, float& back);

	void createTileTestObjects(const BufferMapped& viewprojDataUB);
	void uploadBuffersData(FrameCommandBufferSet& cmdBufferSet, VkQueue queue);
	void getNewLight(Clusterer::LightFormat* lightData, glm::vec4* boundingSphere, LightFormat::Types type);


	Clusterer() = delete;
	Clusterer(Clusterer&) = delete;
	void operator=(Clusterer&) = delete;

	friend class LightTypes::LightBase;
	friend class LightTypes::PointLight;
	friend class LightTypes::SpotLight;
};

#endif