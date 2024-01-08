#ifndef WORLD_TRANSFORM_CLASS_HEADER
#define WORLD_TRANSFORM_CLASS_HEADER

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/tools/projection.h"

class CoordinateTransformation
{
private:
	struct CoordinateTransformationData
	{
		glm::mat4 ndcFromWorld;
		glm::mat4 viewFromWorld;
		glm::mat4 ndcFromView;
		glm::mat4 worldFromNdc;
		glm::mat4 worldFromView;
		glm::mat4 viewFromNdc;
		glm::mat4 ndcFromWorldPrev;
	};
	BufferMapped m_data{};
	ResourceSet m_resSet{};

	uint32_t m_jitterIndex{ 0 };

	glm::dvec2 m_HaltonSequenceJitter[16]
	{
		glm::dvec2{0.500000, 0.333333} - 0.5,
		glm::dvec2{0.250000, 0.666667} - 0.5,
		glm::dvec2{0.750000, 0.111111} - 0.5,
		glm::dvec2{0.125000, 0.444444} - 0.5,
		glm::dvec2{0.625000, 0.777778} - 0.5,
		glm::dvec2{0.375000, 0.222222} - 0.5,
		glm::dvec2{0.875000, 0.555556} - 0.5,
		glm::dvec2{0.062500, 0.888889} - 0.5,
		glm::dvec2{0.562500, 0.037037} - 0.5,
		glm::dvec2{0.312500, 0.370370} - 0.5,
		glm::dvec2{0.812500, 0.703704} - 0.5,
		glm::dvec2{0.187500, 0.148148} - 0.5,
		glm::dvec2{0.687500, 0.481481} - 0.5,
		glm::dvec2{0.437500, 0.814815} - 0.5,
		glm::dvec2{0.937500, 0.259259} - 0.5,
		glm::dvec2{0.031250, 0.592593} - 0.5
	};

public:
	CoordinateTransformation(VkDevice device, BufferBaseHostAccessible& baseHostCachedBuffer) : m_data{ baseHostCachedBuffer, sizeof(CoordinateTransformationData) }
	{
		VkDescriptorSetLayoutBinding worldTransformBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, 
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorAddressInfoEXT worldTransformAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_data.getDeviceAddress(), .range = m_data.getSize() };
		m_resSet.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
			std::array{ worldTransformBinding },
			std::array<VkDescriptorBindingFlags, 0>{},
			std::vector<std::vector<VkDescriptorDataEXT>>{ std::vector<VkDescriptorDataEXT>{ {.pUniformBuffer = &worldTransformAddressInfo} } },
			false);
	}

	VkBuffer getBufferHandle() const { return m_data.getBufferHandle(); }
	VkDeviceSize getBufferOffset() const { return m_data.getOffset(); }
	const ResourceSet& getResourceSet() const { return m_resSet; };
	const glm::vec2& getCurrentJitter()
	{
		return m_HaltonSequenceJitter[m_jitterIndex];
	}

	const glm::mat4& getViewMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->viewFromWorld; }
	const glm::mat4& getInverseViewMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->worldFromView; }
	const glm::mat4& getProjectionMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->ndcFromView; }
	const glm::mat4& getInverseProjectionMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->viewFromNdc; }
	const glm::mat4& getViewProjectionMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->ndcFromWorld; }
	const glm::mat4& getInverseViewProjectionMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->worldFromNdc; }

	void updateViewMatrix(const glm::vec3& eye, const glm::vec3& gazePoint, const glm::vec3& up)
	{
		CoordinateTransformationData* data{ reinterpret_cast<CoordinateTransformationData*>(m_data.getData()) };

		data->viewFromWorld = glm::lookAt(eye, gazePoint, up);
		data->worldFromView = glm::inverse(data->viewFromWorld);

		data->ndcFromWorldPrev = data->ndcFromWorld;
		data->ndcFromWorld = data->ndcFromView * data->viewFromWorld;
		data->worldFromNdc = glm::inverse(data->ndcFromWorld);
	}
	void updateProjectionMatrix(float FOV, float aspect, float zNear, float zFar)
	{
		CoordinateTransformationData* data{ reinterpret_cast<CoordinateTransformationData*>(m_data.getData()) };

		data->ndcFromView = getProjectionRZ(FOV, aspect, zNear, zFar);
		data->viewFromNdc = glm::inverse(data->ndcFromView);

		data->ndcFromWorldPrev = data->ndcFromWorld;
		data->ndcFromWorld = data->ndcFromView * data->viewFromWorld;
		data->worldFromNdc = glm::inverse(data->ndcFromWorld);
	}
	void updateProjectionMatrixJitter()
	{
		CoordinateTransformationData* data{ reinterpret_cast<CoordinateTransformationData*>(m_data.getData()) };

		m_jitterIndex = (m_jitterIndex + 1) % ARRAYSIZE(m_HaltonSequenceJitter);
		data->ndcFromView[2][0] = static_cast<float>(m_HaltonSequenceJitter[m_jitterIndex].x);
		data->ndcFromView[2][1] = static_cast<float>(m_HaltonSequenceJitter[m_jitterIndex].y);
		data->viewFromNdc = glm::inverse(data->ndcFromView);

		data->ndcFromWorld = data->ndcFromView * data->viewFromWorld;
		data->worldFromNdc = glm::inverse(data->ndcFromWorld);
	}
	void updateScreenDimensions(uint32_t width, uint32_t height)
	{
		for (auto& jit : m_HaltonSequenceJitter)
			jit /= glm::vec2(float(width), float(height));
	}
};

#endif