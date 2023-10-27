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
	};
	BufferMapped m_data{};
	ResourceSet m_resSet{};

public:
	CoordinateTransformation(VkDevice device, BufferBaseHostAccessible& baseHostCachedBuffer) : m_data{ baseHostCachedBuffer, sizeof(CoordinateTransformationData) }
	{
		VkDescriptorSetLayoutBinding worldTransformBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, 
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
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

	const glm::mat4& getViewMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->viewFromWorld; }
	const glm::mat4& getProjectionMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->ndcFromView; }
	const glm::mat4& getViewProjectionMatrix() const { return reinterpret_cast<CoordinateTransformationData*>(m_data.getData())->ndcFromWorld; }

	void updateViewMatrix(const glm::vec3& eye, const glm::vec3& gazePoint, const glm::vec3& up)
	{
		CoordinateTransformationData* data{ reinterpret_cast<CoordinateTransformationData*>(m_data.getData()) };

		data->viewFromWorld = glm::lookAt(eye, gazePoint, up);
		data->worldFromView = glm::inverse(data->viewFromWorld);

		data->ndcFromWorld = data->ndcFromView * data->viewFromWorld;
		data->worldFromNdc = glm::inverse(data->ndcFromWorld);
	}
	void updateProjectionMatrix(float FOV, float aspect, float zNear, float zFar)
	{
		CoordinateTransformationData* data{ reinterpret_cast<CoordinateTransformationData*>(m_data.getData()) };

		data->ndcFromView = getProjectionRZ(FOV, aspect, zNear, zFar);
		data->viewFromNdc = glm::inverse(data->ndcFromView);

		data->ndcFromWorld = data->ndcFromView * data->viewFromWorld;
		data->worldFromNdc = glm::inverse(data->ndcFromWorld);
	}
};

#endif