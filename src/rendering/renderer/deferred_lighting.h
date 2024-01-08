#ifndef UV_BUFFER_CLASS_HEADER
#define UV_BUFFER_CLASS_HEADER

#include <glm/glm.hpp>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/renderer/depth_buffer.h"
#include "src/rendering/renderer/culling.h"
#include "src/rendering/renderer/sync_operations.h"
#include "src/rendering/renderer/HBAO.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/tools/arraysize.h"
#include "src/tools/comp_s.h"

class DeferredLighting
{
private:
	Image m_UV;
	Image m_tangentFrame;
	Image m_drawID;

	Image m_outputFramebuffer;

	const DepthBuffer& m_depthBuffer;

	Pipeline m_uvBufferPipeline{};
	Pipeline m_lightingComputePipeline{};

	std::array<VkImageMemoryBarrier2, 6> m_imageBarriers{};
	VkDependencyInfo m_dependencyInfo{};

	ResourceSet m_resSet{};
	
	struct
	{
		glm::vec3 camPos;
		float binWidth;
		glm::vec2 invResolution;
		uint32_t windowTileWidth;
		float nearPlane;
		glm::vec3 giSceneCenter;
		float farPlane;
	} m_pcData;

public:
	DeferredLighting(VkDevice device, uint32_t width, uint32_t height,
		const DepthBuffer& depthBuffer,
		const HBAO& hbao,
		const ResourceSet& viewprojRS,
		const ResourceSet& transformMatricesRS,
		const ResourceSet& materialsTexturesRS,
		const ResourceSet& shadowMapsRS,
		const ResourceSet& indirectDiffiseLightingRS,
		const ResourceSet& indirectSpecularLightingRS,
		const ResourceSet& indirectLightingMetadataRS,
		const ResourceSet& drawDataRS,
		const ResourceSet& pbrRS,
		const ResourceSet& directLightingRS,
		VkSampler generalSampler);
	~DeferredLighting() = default;

	void updateCameraPosition(const glm::vec3& camPos)
	{
		m_pcData.camPos = camPos;
	}
	void updateProjectionData(double near, double far)
	{
		m_pcData.nearPlane = near;
		m_pcData.farPlane = far;
	}
	void updateLightBinWidth(float binWidth)
	{
		m_pcData.binWidth = binWidth;
	}
	void updateTileWidth(uint32_t tileWidth)
	{
		m_pcData.windowTileWidth = tileWidth;
	}
	void updateGISceneCenter(const glm::vec3& giSceneCenter)
	{
		m_pcData.giSceneCenter = giSceneCenter;
	}

	const Image& getFramebuffer() const
	{
		return m_outputFramebuffer;
	}
	VkImage getFramebufferImageHandle() const
	{
		return m_outputFramebuffer.getImageHandle();
	}
	VkImageView getFramebufferImageView() const
	{
		return m_outputFramebuffer.getImageView();
	}
	VkImageSubresourceRange getFramebufferSubresourceRange()
	{
		return m_outputFramebuffer.getSubresourceRange();
	}

	void cmdPassDrawToUVBuffer(VkCommandBuffer cb, const Culling& culling, const Buffer& vertexData, const Buffer& indexData);

	const VkDependencyInfo& getDependency()
	{
		return m_dependencyInfo;
	}

	void cmdDispatchLightingCompute(VkCommandBuffer cb, uint32_t indirectCurrentSet);
};

#endif