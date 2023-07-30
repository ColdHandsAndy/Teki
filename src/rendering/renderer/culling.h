#ifndef CULLING_CLASS_HEADER
#define CULLING_CLASS_HEADER

#include <cstdint>

#include <glm/glm.hpp>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/data_management/memory_manager.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/data_abstraction/BB.h"
#include "src/rendering/renderer/barrier_operations.h"
#include "src/rendering/renderer/depth_buffer.h"
#include "src/tools/comp_s.h"

struct IndirectData
{
	VkDrawIndexedIndirectCommand cmd{};
	//Bounding sphere data
	float bsPos[3]{};
	float bsRad{};
};

struct FrustumInfo
{
	glm::vec4 planes[6]{};
	glm::vec3 points[8]{};
};

class Culling
{
private:
	Pipeline m_occlusionPass{};
	
	BufferMapped m_indicesCmds{};
	Buffer m_drawCount{};
	Buffer m_targetDrawCommands{};
	Buffer m_targetDrawDataIndices{};

	uint32_t m_frustumNonculledCount{};
	uint32_t m_hiZmipmax{};
	float m_zNear{};

public:
	Culling(VkDevice device,
		uint32_t drawCommandsMax,
		float zNearProjPlane,
		BufferBaseHostAccessible& baseHost, BufferBaseHostAccessible& baseShared, BufferBaseHostInaccessible& baseDevice,
		const BufferMapped& viewprojUB,
		const BufferMapped& indirectDataBuffer,
		const DepthBuffer& depthBuffer)
	{
		m_hiZmipmax = depthBuffer.getMipLevelCountHiZ() - 1;
		m_zNear = zNearProjPlane;

		m_indicesCmds.initialize(baseShared, sizeof(uint32_t) * drawCommandsMax);
		m_drawCount.initialize(baseDevice, sizeof(uint32_t));
		m_targetDrawCommands.initialize(baseDevice, sizeof(VkDrawIndexedIndirectCommand) * drawCommandsMax);
		m_targetDrawDataIndices.initialize(baseDevice, sizeof(uint32_t) * drawCommandsMax);

		std::vector<ResourceSet> resourceSets{};

		VkDescriptorSetLayoutBinding indicesBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorAddressInfoEXT indicesAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_indicesCmds.getDeviceAddress(), .range = m_indicesCmds.getSize() };

		VkDescriptorSetLayoutBinding cmdAndSpheresBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorAddressInfoEXT cmdAndSpheresAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = indirectDataBuffer.getDeviceAddress(), .range = indirectDataBuffer.getSize() };

		VkDescriptorSetLayoutBinding targetCmdsBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorAddressInfoEXT targetCmdsAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_targetDrawCommands.getDeviceAddress(), .range = m_targetDrawCommands.getSize() };

		VkDescriptorSetLayoutBinding drawCountBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorAddressInfoEXT drawCountAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_drawCount.getDeviceAddress(), .range = m_drawCount.getSize() };

		VkDescriptorSetLayoutBinding drawDataIndicesBinding{ .binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorAddressInfoEXT drawDataIndicesAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_targetDrawDataIndices.getDeviceAddress(), .range = m_targetDrawDataIndices.getSize() };

		VkDescriptorSetLayoutBinding viewprojBinding{ .binding = 6, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorAddressInfoEXT viewprojAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojUB.getDeviceAddress(), .range = viewprojUB.getSize() };

		VkDescriptorSetLayoutBinding hiZBinding{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorImageInfo hiZImageInfo{ .sampler = depthBuffer.getReductionSampler(), .imageView = depthBuffer.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

		resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
			{{indicesBinding, cmdAndSpheresBinding, targetCmdsBinding, drawCountBinding, hiZBinding, drawDataIndicesBinding, viewprojBinding}},  {},
			{{{.pStorageBuffer = &indicesAddressinfo}},
				{{.pStorageBuffer = &cmdAndSpheresAddressinfo}},
					{{.pStorageBuffer = &targetCmdsAddressinfo}},
						{{.pStorageBuffer = &drawCountAddressinfo}},
							{{.pCombinedImageSampler = &hiZImageInfo}},
								{{.pStorageBuffer = &drawDataIndicesAddressinfo}},
									{{.pUniformBuffer = &viewprojAddressinfo}}}, true });

		m_occlusionPass.initializaCompute(device,
			"shaders/cmpld/occlusion_culling_comp.spv",
			resourceSets,
			{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(uint32_t) * 2 + sizeof(float)}} });
	}

	void cullAgainstFrustum(const OBBs& boundingBoxes, const FrustumInfo& frustumInfo, const glm::mat4& viewMat)
	{
		glm::mat4 viewInv{ glm::inverse(viewMat) };

		__m128 planes[6]{};
		auto transformPlanes{ [&planes](const FrustumInfo& frustum, const glm::mat4& viewMatr)
			{
				for (int i{ 0 }; i < 6; ++i)
				{
					glm::vec3 newNormal{ glm::mat3{viewMatr} *glm::vec3{frustum.planes[i]} };
					float dot{ glm::dot(glm::vec3{viewMatr[3][0], viewMatr[3][1], viewMatr[3][2]}, newNormal) };
					float newDist{ -(dot - frustum.planes[i].w) };
					planes[i] = _mm_set_ps(newNormal.x, newNormal.y, newNormal.z, newDist);
				}
			} };
		transformPlanes(frustumInfo, viewInv);

		m_frustumNonculledCount = 0;
		uint32_t* indices{ reinterpret_cast<uint32_t*>(m_indicesCmds.getData()) };

		for (int i{ 0 }; i < boundingBoxes.getBBCount(); ++i)
		{
			float* xs{};
			float* ys{};
			float* zs{};
			boundingBoxes.getOBB(i, &xs, &ys, &zs);
			__m128 points[8]{};
			for (int j{ 0 }; j < 8; ++j)
			{
				points[j] = _mm_set_ps(xs[j], ys[j], zs[j], 1.0);
			}
			for (int j{ 0 }; j < 6; ++j)
			{
				int out = 0;
				out += ((_mm_dp_ps(planes[j], points[0], 0xF1).m128_f32[0] > 0.0f) ? 1 : 0);
				out += ((_mm_dp_ps(planes[j], points[1], 0xF1).m128_f32[0] > 0.0f) ? 1 : 0);
				out += ((_mm_dp_ps(planes[j], points[2], 0xF1).m128_f32[0] > 0.0f) ? 1 : 0);
				out += ((_mm_dp_ps(planes[j], points[3], 0xF1).m128_f32[0] > 0.0f) ? 1 : 0);
				out += ((_mm_dp_ps(planes[j], points[4], 0xF1).m128_f32[0] > 0.0f) ? 1 : 0);
				out += ((_mm_dp_ps(planes[j], points[5], 0xF1).m128_f32[0] > 0.0f) ? 1 : 0);
				out += ((_mm_dp_ps(planes[j], points[6], 0xF1).m128_f32[0] > 0.0f) ? 1 : 0);
				out += ((_mm_dp_ps(planes[j], points[7], 0xF1).m128_f32[0] > 0.0f) ? 1 : 0);

				if (out == 8)
					goto next;
			}

			++m_frustumNonculledCount;
			*(indices++) = i;
			continue;

		next:;
		}
	}

	void cmdCullOccluded(VkCommandBuffer cb, DescriptorManager& descriptorManager)
	{
		uint32_t zero{ 0 };
		vkCmdUpdateBuffer(cb, m_drawCount.getBufferHandle(), m_drawCount.getOffset(), sizeof(zero), &zero);

		BarrierOperations::cmdExecuteBarrier(cb, {{BarrierOperations::constructMemoryBarrier(
				VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)}
		});

		m_occlusionPass.cmdBind(cb);

		descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_occlusionPass.getResourceSets(), m_occlusionPass.getResourceSetsInUse(), m_occlusionPass.getPipelineLayoutHandle());
		struct {uint32_t commandCount; uint32_t mipMax; float zNear;} pcData;
		pcData.commandCount = m_frustumNonculledCount;
		pcData.mipMax = m_hiZmipmax;
		pcData.zNear = m_zNear;
		vkCmdPushConstants(cb, m_occlusionPass.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t) * 2 + sizeof(float), &pcData);
		constexpr uint32_t groupsizeX{ 64 };
		vkCmdDispatch(cb, DISPATCH_SIZE(m_frustumNonculledCount, groupsizeX), 1, 1);
	}

	VkBuffer getDrawCommandBufferHandle() const
	{
		return m_targetDrawCommands.getBufferHandle();
	}
	VkDeviceSize getDrawCommandBufferOffset() const
	{
		return m_targetDrawCommands.getOffset();
	}
	VkDeviceSize getDrawCommandBufferStride() const
	{
		return sizeof(VkDrawIndexedIndirectCommand);
	}

	VkBuffer getDrawCountBufferHandle() const
	{
		return m_drawCount.getBufferHandle();
	}
	VkDeviceSize getDrawCountBufferOffset() const
	{
		return m_drawCount.getOffset();
	}

	const Buffer& getDrawDataIndexBuffer()
	{
		return m_targetDrawDataIndices;
	}

};

#endif