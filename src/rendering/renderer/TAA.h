#ifndef TAA_CLASS_HEADER
#define TAA_CLASS_HEADER

#include <cstdint>

#include <vulkan/vulkan.h>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/sync_operations.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/tools/comp_s.h"

class TAA
{
private:
	VkDevice m_device{};

	uint32_t m_windowWidth{};
	uint32_t m_windowHeight{};

	bool resCopyIndex{ 0 };

	Image m_historyFramebuffers[2];

	ResourceSet m_resSet0{};
	ResourceSet m_resSet1{};

	Pipeline m_TAApass{};

	const double m_minSmoothingConstValue{ 0.1 };
	const double m_maxSmoothingConstValue{ 0.4 };
	struct 
	{
		glm::uvec2 resolution;
		glm::vec2 invResolution;
		glm::vec2 jitterValue;
		glm::vec2 jitterValuePrev;
		float smoothingFactor;
	} m_pcData;

	VkSampler m_sampler{};

public:
	TAA(VkDevice device, const DepthBuffer& depthBuffer, const Image& framebuffer, const ResourceSet& viewportRS, CommandBufferSet& cmdBufferSet, VkQueue queue) :
		m_device{ device }, m_windowWidth{ framebuffer.getWidth() }, m_windowHeight{ framebuffer.getHeight() },
		m_historyFramebuffers{ Image{ device, framebuffer.getFormat(), framebuffer.getWidth(), framebuffer.getHeight(), VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT},
							   Image{ device, framebuffer.getFormat(), framebuffer.getWidth(), framebuffer.getHeight(), VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT}}
	{
		VkSamplerCreateInfo samplerCI{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = 128.0f,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE };
		vkCreateSampler(device, &samplerCI, nullptr, &m_sampler);

		VkDescriptorSetLayoutBinding inputImageBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorImageInfo inputImageInfo{ .sampler = m_sampler, .imageView = framebuffer.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		VkDescriptorSetLayoutBinding depthBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorImageInfo depthImageInfo{ .sampler = m_sampler, .imageView = depthBuffer.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL};
		VkDescriptorSetLayoutBinding outputBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };

		m_resSet0.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlags{},
			std::array{ inputImageBinding, depthBinding, outputBinding }, std::array<VkDescriptorBindingFlags, 0>{},
			std::vector<std::vector<VkDescriptorDataEXT>>{
				std::vector<VkDescriptorDataEXT>{{.pCombinedImageSampler = &inputImageInfo}},
				std::vector<VkDescriptorDataEXT>{{.pCombinedImageSampler = &depthImageInfo}},
				std::vector<VkDescriptorDataEXT>{}},
			true);

		VkDescriptorSetLayoutBinding oldHistoryBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorSetLayoutBinding newHistoryBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		VkDescriptorImageInfo oldHistoryImageInfo0{ .sampler = m_sampler, .imageView = m_historyFramebuffers[0].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		VkDescriptorImageInfo newHistoryImageInfo0{ .imageView = m_historyFramebuffers[1].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		VkDescriptorImageInfo oldHistoryImageInfo1{ .sampler = m_sampler, .imageView = m_historyFramebuffers[1].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		VkDescriptorImageInfo newHistoryImageInfo1{ .imageView = m_historyFramebuffers[0].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

		m_resSet1.initializeSet(device, 2, VkDescriptorSetLayoutCreateFlags{},
			std::array{ oldHistoryBinding, newHistoryBinding }, std::array<VkDescriptorBindingFlags, 0>{},
			std::vector<std::vector<VkDescriptorDataEXT>>{
				std::vector<VkDescriptorDataEXT>{{.pCombinedImageSampler = &oldHistoryImageInfo0}, {.pCombinedImageSampler = &oldHistoryImageInfo1}},
				std::vector<VkDescriptorDataEXT>{{.pStorageImage = &newHistoryImageInfo0}, {.pStorageImage = &newHistoryImageInfo1}} },
			true);

		std::reference_wrapper<const ResourceSet> resSets[3]{ viewportRS, m_resSet0, m_resSet1 };

		m_TAApass.initializaCompute(device, "shaders/cmpld/taa_comp.spv", resSets, 
			{{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(m_pcData)}}});

		m_pcData.resolution = {m_windowWidth, m_windowHeight};
		m_pcData.invResolution = { 1.0 / (m_windowWidth), 1.0 / (m_windowHeight) };
		m_pcData.smoothingFactor = 0.1;

		VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };
			SyncOperations::cmdExecuteBarrier(cb, 
			{{SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_NONE,
				0, 0,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				m_historyFramebuffers[0].getImageHandle(), m_historyFramebuffers[0].getSubresourceRange()),
			SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_NONE,
				0, 0,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				m_historyFramebuffers[1].getImageHandle(), m_historyFramebuffers[1].getSubresourceRange())}});
		cmdBufferSet.endRecording(cb);
		VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(queue);
		cmdBufferSet.resetAll();
	}
	~TAA()
	{
		vkDestroySampler(m_device, m_sampler, nullptr);
	}

	void updateJitterValue(const glm::vec2& jitter)
	{
		m_pcData.jitterValuePrev = m_pcData.jitterValue;
		m_pcData.jitterValue = jitter;
	}

	void adjustSmoothingFactor(double deltaTime, double camSpeed, bool camPosChanged)
	{
		constexpr double convergenceTime{ 0.07 };
		m_pcData.smoothingFactor = static_cast<float>(glm::clamp(1.0 - glm::exp(-deltaTime / convergenceTime) + (camPosChanged ? glm::sqrt(camSpeed * 0.1) * 0.15 : 0.0), 0.05, 0.4));
	}

	void cmdDispatchTAA(VkCommandBuffer cb, VkImageView outputAttachment)
	{
		constexpr uint32_t outputImageSetIndex{ 1 };
		constexpr uint32_t outputImageBindingIndex{ 2 };
		VkDescriptorImageInfo outputImageInfo{ .imageView = outputAttachment, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
		VkDescriptorDataEXT descData{ .pStorageImage = &outputImageInfo };
		m_TAApass.rewriteDescriptor(outputImageSetIndex, outputImageBindingIndex, 0, 0, descData);
		constexpr uint32_t historyBufferResSetIndex{ 2 };
		m_TAApass.setResourceInUse(historyBufferResSetIndex, resCopyIndex);
		m_TAApass.cmdBindResourceSets(cb);
		m_TAApass.cmdBind(cb);
		vkCmdPushConstants(cb, m_TAApass.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_pcData), &m_pcData);
		constexpr uint32_t groupSize{ 8 };
		vkCmdDispatch(cb, DISPATCH_SIZE(m_windowWidth, groupSize), DISPATCH_SIZE(m_windowHeight, groupSize), 1);

		SyncOperations::cmdExecuteBarrier(cb, 
			{{SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
				m_historyFramebuffers[resCopyIndex].getImageHandle(), m_historyFramebuffers[resCopyIndex].getSubresourceRange()),
			SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				m_historyFramebuffers[!resCopyIndex].getImageHandle(), m_historyFramebuffers[!resCopyIndex].getSubresourceRange())}});

		resCopyIndex = !resCopyIndex;
	}
};

#endif