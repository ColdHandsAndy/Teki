#ifndef DEPTH_BUFFER_CLASS
#define DEPTH_BUFFER_CLASS

#include <cstdint>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/barrier_operations.h"
#include "src/rendering/data_management/image_classes.h"

class DepthBuffer
{
private:
	VkDevice m_device{};

	Image m_depthImage;
	Image m_hierarchicalZ;

	Pipeline m_calcHiZ{};
	int m_hiZopCount{};

	VkSampler m_samplerHiZ{};

	VkImageView* m_imageViewsHiZ{};

public:
	DepthBuffer(VkDevice device, uint32_t width, uint32_t heigth)
		: m_device{ device },
		m_depthImage{ device, VK_FORMAT_D32_SFLOAT, width, heigth, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_DEPTH_BIT },
		m_hierarchicalZ{ device, VK_FORMAT_R32_SFLOAT, width / 2, heigth / 2, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, true }
	{
		VkSamplerCreateInfo samplerCI{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.minLod = 0.0f,
			.maxLod = 1.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE };
		VkSamplerReductionModeCreateInfo samplerReductionCI{ VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT };
		samplerReductionCI.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;
		samplerCI.pNext = &samplerReductionCI;
		vkCreateSampler(device, &samplerCI, nullptr, &m_samplerHiZ);

		const uint32_t operationNum = m_hierarchicalZ.getMipLevelCount() - 1;
		const uint32_t activeMips = m_hierarchicalZ.getMipLevelCount() - 1;
		m_hiZopCount = operationNum;

		m_imageViewsHiZ = { new VkImageView[activeMips] };
		for (int i{ 0 }; i < activeMips; ++i)
		{
			VkImageViewCreateInfo imageViewCI{};
			imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			imageViewCI.image = m_hierarchicalZ.getImageHandle();
			imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
			imageViewCI.format = m_hierarchicalZ.getFormat();
			imageViewCI.components = { .r = VK_COMPONENT_SWIZZLE_R, .g = VK_COMPONENT_SWIZZLE_G, .b = VK_COMPONENT_SWIZZLE_B, .a = VK_COMPONENT_SWIZZLE_A };
			imageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageViewCI.subresourceRange.baseMipLevel = i;
			imageViewCI.subresourceRange.levelCount = 1;
			imageViewCI.subresourceRange.baseArrayLayer = 0;
			imageViewCI.subresourceRange.layerCount = 1;
			vkCreateImageView(device, &imageViewCI, nullptr, &m_imageViewsHiZ[i]);
		}

		std::vector<VkDescriptorSetLayoutBinding> bindings(2);
		std::vector<VkDescriptorImageInfo> imageInfos(operationNum * 2);

		bindings[0] = { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
		bindings[1] = { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };

		imageInfos[0] = { .sampler = m_samplerHiZ, .imageView = m_depthImage.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		imageInfos[1] = { .imageView = m_imageViewsHiZ[0], .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		for (int i{ 1 }; i < activeMips; ++i)
		{
			imageInfos[2 * i] = { .sampler = m_samplerHiZ, .imageView = m_imageViewsHiZ[i - 1], .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
			imageInfos[2 * i + 1] = { .imageView = m_imageViewsHiZ[i + 0], .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		}

		std::vector<std::vector<VkDescriptorDataEXT>> descData(2);

		for (int i{ 0 }; i < m_hiZopCount; ++i)
		{
			descData[0].push_back(VkDescriptorDataEXT{ .pCombinedImageSampler = &imageInfos[2 * i + 0] });
			descData[1].push_back(VkDescriptorDataEXT{ .pStorageImage = &imageInfos[2 * i + 1] });
		}

		std::vector<ResourceSet> resourceSets{};
		resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, operationNum,
			bindings,  {},
			descData });
		m_calcHiZ.initializaCompute(device, 
			"shaders/cmpld/calc_hi_z_comp.spv",
			resourceSets, 
			{{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(uint32_t) * 2 + sizeof(float) * 2}}});
	}
	~DepthBuffer()
	{
		vkDestroySampler(m_device, m_samplerHiZ, nullptr);
		for (int i{ 0 }; i < m_hiZopCount; ++i)
		{
			vkDestroyImageView(m_device, m_imageViewsHiZ[i], nullptr);
		}
		delete[] m_imageViewsHiZ;
	}

	void cmdCalcHiZ(VkCommandBuffer cb, DescriptorManager& descriptorManager)
	{
#define DISPATCH_SIZE(groupSize, elementsCount) ((elementsCount + groupSize - 1) / groupSize)

		BarrierOperations::cmdExecuteBarrier(cb, { 
			{BarrierOperations::constructImageBarrier(
					VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					m_depthImage.getImageHandle(),
					{.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 }),
			BarrierOperations::constructImageBarrier(
					VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0, 0,
					VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
					m_hierarchicalZ.getImageHandle(),
					{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = static_cast<uint32_t>(m_hiZopCount), .baseArrayLayer = 0, .layerCount = 1 })} });

		uint32_t width{ m_hierarchicalZ.getWidth() };
		uint32_t height{ m_hierarchicalZ.getHeight() };

		struct { uint32_t width; uint32_t height; float invWidth; float invHeight; } pushC;
		m_calcHiZ.cmdBind(cb);

		m_calcHiZ.setResourceInUse(0, 0);
		descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_calcHiZ.getResourceSets(), m_calcHiZ.getResourceSetsInUse(), m_calcHiZ.getPipelineLayoutHandle());
		constexpr uint32_t groupsizeX{ 16 };
		constexpr uint32_t groupsizeY{ 16 };
		pushC.width = width;
		pushC.height = height;
		pushC.invWidth = 1.0f / width;
		pushC.invHeight = 1.0f / height;
		vkCmdPushConstants(cb, m_calcHiZ.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushC), &pushC);
		vkCmdDispatch(cb, DISPATCH_SIZE(groupsizeX, width), DISPATCH_SIZE(groupsizeY, height), 1);

		VkImageMemoryBarrier2 barrier{ 
			BarrierOperations::constructImageBarrier(
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					m_hierarchicalZ.getImageHandle(),
					{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 }) };

		BarrierOperations::cmdExecuteBarrier(cb, { {barrier} });
		
		for (int i{ 1 }; i < m_hiZopCount; ++i)
		{
			width = width / 2;
			height = height / 2;
			m_calcHiZ.setResourceInUse(0, i);
			descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_calcHiZ.getResourceSets(), m_calcHiZ.getResourceSetsInUse(), m_calcHiZ.getPipelineLayoutHandle());
			pushC.width = width;
			pushC.height = height;
			pushC.invWidth = 1.0f / width;
			pushC.invHeight = 1.0f / height;
			vkCmdPushConstants(cb, m_calcHiZ.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushC), &pushC);
			vkCmdDispatch(cb, DISPATCH_SIZE(groupsizeX, width), DISPATCH_SIZE(groupsizeY, height), 1);
		
			barrier.subresourceRange.baseMipLevel = i;
		
			BarrierOperations::cmdExecuteBarrier(cb, { {barrier} });
		}

		BarrierOperations::cmdExecuteBarrier(cb, {
			{BarrierOperations::constructImageBarrier(
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
					0, 0,
					VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
					m_depthImage.getImageHandle(),
					{.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 }) } });

#undef DISPATCH_SIZE
	}

	uint32_t getMipLevelCountHiZ()
	{
		return m_hiZopCount;
	}

	VkImage getImageHandle()
	{
		return m_depthImage.getImageHandle();
	}
	VkImageView getImageView()
	{
		return m_depthImage.getImageView();
	}
	VkImage getImageHandleHiZ()
	{
		return m_hierarchicalZ.getImageHandle();
	}
	VkImageView getImageViewHiZ()
	{
		return m_hierarchicalZ.getImageView();
	}


	void cmdVisualizeHiZ(VkCommandBuffer cb, DescriptorManager& descriptorManager, VkImage outImage, VkImageLayout outImageLayout, uint32_t mipLevel)
	{
		BarrierOperations::cmdExecuteBarrier(cb, {
			{BarrierOperations::constructImageBarrier(
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				m_hierarchicalZ.getImageHandle(),
				{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = static_cast<uint32_t>(m_hiZopCount), .baseArrayLayer = 0, .layerCount = 1 }),
			BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0,
				outImageLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				outImage,
				{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 })} });

		VkImageBlit blit{};
		blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.srcSubresource.mipLevel = mipLevel;
		blit.srcSubresource.layerCount = 1;
		blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		blit.dstSubresource.layerCount = 1;
		blit.srcOffsets[0] = { 0, 0, 0 };
		blit.srcOffsets[1] = { int((m_hierarchicalZ.getWidth() + 0.5) / std::pow(2.0, mipLevel)), int((m_hierarchicalZ.getHeight() + 0.5) / std::pow(2.0, mipLevel)), 1 };
		blit.dstOffsets[0] = { 0, 0, 0 };
		blit.dstOffsets[1] = { int(m_depthImage.getWidth()), int(m_depthImage.getHeight()), 1 };

		vkCmdBlitImage(cb, m_hierarchicalZ.getImageHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, outImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

		BarrierOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
			{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				0, 0,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, outImageLayout,
				outImage,
				{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1 })} });
	}
};

#endif