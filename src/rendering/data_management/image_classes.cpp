#include "src/rendering/data_management/image_classes.h"

#include <cmath>

#include "src/rendering/renderer/barrier_operations.h"

ImageList::ImageList(VkDevice device, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usageFlags, bool allocateMips, uint32_t layercount)
{
	VkImageCreateInfo imageCI{};
	imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCI.imageType = VK_IMAGE_TYPE_2D;
	imageCI.format = format;
	m_format = format;
	m_width = width;
	m_height = height;
	imageCI.extent = {.width = width, .height = height, .depth = 1};
	if (allocateMips)
	{
		m_mipLevelCount = std::floor(std::log2(std::max(width, height))) + 1;
	}
	else
	{
		m_mipLevelCount = 1;
	}
	imageCI.mipLevels = m_mipLevelCount;
	if (layercount == 0)
	{
		m_arrayLayerCount = 4; /*calculate array layers function*/
	}
	else
	{
		m_arrayLayerCount = layercount;
	}
	imageCI.arrayLayers = m_arrayLayerCount;
	imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCI.usage = usageFlags;
	imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	//Allocation performed on device through VMA
	VmaAllocationCreateInfo allocCI{};
	allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	auto allocationIter{ m_memoryManager->addAllocation() };
	ASSERT_DEBUG(vmaCreateImage(m_memoryManager->getAllocator(), &imageCI, &allocCI, &m_imageHandle, &(*allocationIter), nullptr) == VK_SUCCESS, "VMA", "Image creation failed.");
	m_imageAllocIter = allocationIter;

	VkImageViewCreateInfo imageViewCI{};
	imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewCI.image = m_imageHandle;
	imageViewCI.format = m_format;
	imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
	imageViewCI.components = {.r = VK_COMPONENT_SWIZZLE_R, .g = VK_COMPONENT_SWIZZLE_G, .b = VK_COMPONENT_SWIZZLE_B, .a = VK_COMPONENT_SWIZZLE_A};
	imageViewCI.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = m_mipLevelCount, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount};
	ASSERT_DEBUG(vkCreateImageView(device, &imageViewCI, nullptr, &m_imageViewHandle) == VK_SUCCESS, "Vulkan", "Image view creation failed.");

	for (int i{ static_cast<int>(m_arrayLayerCount - 1) }; i >= 0; --i)
	{
		m_freeLayers.push_back(static_cast<uint32_t>(i));
	}
}
ImageList::ImageList(ImageList&& src) noexcept
{
	m_imageHandle = src.m_imageHandle;
	m_imageViewHandle = src.m_imageViewHandle;

	m_format = src.m_format;
	m_width = src.m_width;
	m_height = src.m_height;
	m_mipLevelCount = src.m_mipLevelCount;
	m_arrayLayerCount = src.m_arrayLayerCount;

	m_freeLayers = std::move(src.m_freeLayers);

	m_imageAllocIter = src.m_imageAllocIter;
	
	src.m_invalid = true;
}


uint32_t ImageList::getWidth() const
{
	return m_width;
}

uint32_t ImageList::getHeight() const
{
	return m_height;
}

uint32_t ImageList::getLayerCount() const
{
	return m_arrayLayerCount;
}


VkImageSubresourceRange ImageList::getSubresourceRange() const
{
	return VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = m_mipLevelCount, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount};
}

bool ImageList::getLayer(uint16_t& slotIndex)
{
	if (m_freeLayers.empty())
		return false;
	slotIndex = m_freeLayers.back();
	m_freeLayers.pop_back();
	return true;
}

void ImageList::freeLayer(uint16_t freedSlotIndex)
{
	m_freeLayers.push_back(freedSlotIndex);
}

void ImageList::cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, uint32_t regionCount, VkDeviceSize* bufferOffset, uint32_t* width, uint32_t* height, uint32_t* dstImageLayerIndex, uint32_t* mipLevel)
{
	VkBufferImageCopy* bufferImageCopies{ new VkBufferImageCopy[regionCount] };

	if (mipLevel == nullptr)
	{
		for (uint32_t i{ 0 }; i < regionCount; ++i)
		{
			VkImageSubresourceLayers imageSubresource{};
			imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageSubresource.layerCount = 1;
			imageSubresource.baseArrayLayer = dstImageLayerIndex[i];
			imageSubresource.mipLevel = 0;

			bufferImageCopies[i] = VkBufferImageCopy{
				.bufferOffset = bufferOffset[i],
				.imageSubresource = imageSubresource,
				.imageOffset = { 0, 0, 0 },
				.imageExtent = { width[i], height[i], 1 }
			};
		}
	}
	else
	{
		for (uint32_t i{ 0 }; i < regionCount; ++i)
		{
			bufferImageCopies[i].bufferOffset = bufferOffset[i];
			bufferImageCopies[i].imageOffset = { 0, 0, 0 };
			bufferImageCopies[i].imageExtent = { width[i], height[i], 1 };

			VkImageSubresourceLayers imageSubresource{};
			imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			imageSubresource.layerCount = 1;
			imageSubresource.baseArrayLayer = dstImageLayerIndex[i];
			imageSubresource.mipLevel = mipLevel[i];

			bufferImageCopies[i].imageSubresource = imageSubresource;
		}
	}

	vkCmdCopyBufferToImage(cb, srcBuffer, m_imageHandle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regionCount, bufferImageCopies);
	delete[] bufferImageCopies;
}

void ImageList::cmdCreateMipmaps(VkCommandBuffer cb, VkImageLayout currentListLayout)
{
	if (m_mipLevelCount <= 1 || (m_width == 1 && m_height == 1))
		return;

	std::vector<int> busyLayers{};
	busyLayers.reserve(m_arrayLayerCount);
	for (int i{ 0 }; i < m_arrayLayerCount; ++i)
	{
		busyLayers.push_back(i);
	}
	for (auto freeLayer : m_freeLayers)
	{
		busyLayers.erase(busyLayers.begin() + freeLayer);
	}

	uint32_t blitCount{ static_cast<uint32_t>(busyLayers.size())};

	VkImageBlit* imageBlits{ new VkImageBlit[blitCount] };
	for (int i{ 0 }; i < blitCount; ++i)
	{
		VkImageBlit& blit{ imageBlits[i] };
		blit = VkImageBlit{};
	}

	VkImageMemoryBarrier2 initialBarriers[2] = {
		VkImageMemoryBarrier2{
			BarrierOperations::constructImageBarrier(
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, VK_ACCESS_TRANSFER_READ_BIT,
				currentListLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				m_imageHandle,
				VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount }) },
		VkImageMemoryBarrier2{
			BarrierOperations::constructImageBarrier(
				VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, VK_ACCESS_TRANSFER_WRITE_BIT,
				currentListLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
				m_imageHandle,
				VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 1, .levelCount = m_mipLevelCount - 1, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount }) }
		};

	VkImageMemoryBarrier2 memBarrier{ 
		BarrierOperations::constructImageBarrier(
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			m_imageHandle,
			VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount}) };

	BarrierOperations::cmdExecuteBarrier(cb, std::span<VkImageMemoryBarrier2>{initialBarriers, initialBarriers + 2});

	int mipWidth{ static_cast<int>(m_width) };
	int mipHeight{ static_cast<int>(m_height) };
	for (uint32_t i{ 1 }; i < m_mipLevelCount; ++i)
	{
		for (uint32_t j{ 0 }; j < blitCount; ++j)
		{
			VkImageBlit& blit{ imageBlits[j] };
			blit.srcOffsets[0] = VkOffset3D{ 0, 0, 0 };
			blit.srcOffsets[1] = VkOffset3D{ mipWidth, mipHeight, 1 };
			blit.srcSubresource = VkImageSubresourceLayers{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = i - 1, .baseArrayLayer = static_cast<uint32_t>(busyLayers[j]), .layerCount = 1};
			blit.dstOffsets[0] = VkOffset3D{ 0, 0, 0 };
			blit.dstOffsets[1] = VkOffset3D{ mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1 };
			blit.dstSubresource = VkImageSubresourceLayers{ .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = i, .baseArrayLayer = static_cast<uint32_t>(busyLayers[j]), .layerCount = 1 };
		}
		vkCmdBlitImage(cb, m_imageHandle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_imageHandle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, blitCount, imageBlits, VK_FILTER_LINEAR);
		mipWidth = mipWidth > 1 ? mipWidth / 2 : 1;
		mipHeight = mipHeight > 1 ? mipHeight / 2 : 1;

		memBarrier.subresourceRange.baseMipLevel = i;
		BarrierOperations::cmdExecuteBarrier(cb, { {memBarrier} });
	}

	BarrierOperations::cmdExecuteBarrier(cb, { {BarrierOperations::constructImageBarrier(
												VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
												VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
												VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
												m_imageHandle,
												VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = m_mipLevelCount, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount})} });
	delete[] imageBlits;
}



ImageCubeMap::ImageCubeMap(VkDevice device, VkFormat format, uint32_t sideLength, VkImageUsageFlags usageFlags)
{
	VkImageCreateInfo imageCI{};
	imageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCI.imageType = VK_IMAGE_TYPE_2D;
	imageCI.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
	imageCI.format = format;
	m_format = format;
	imageCI.extent = { .width = sideLength, .height = sideLength, .depth = 1 };
	m_sideLength = sideLength;
	m_mipLevelCount = 1;
	imageCI.mipLevels = m_mipLevelCount;
	constexpr uint32_t cubemapImageLayers{ 6 };
	m_arrayLayerCount = cubemapImageLayers;
	imageCI.arrayLayers = m_arrayLayerCount;
	imageCI.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCI.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCI.usage = usageFlags;
	imageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	//Allocation performed on device through VMA
	VmaAllocationCreateInfo allocCI{};
	allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
	allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	auto allocationIter{ m_memoryManager->addAllocation() };
	ASSERT_DEBUG(vmaCreateImage(m_memoryManager->getAllocator(), &imageCI, &allocCI, &m_imageHandle, &(*allocationIter), nullptr) == VK_SUCCESS, "VMA", "Image creation failed.");
	m_imageAllocIter = allocationIter;

	VkImageViewCreateInfo imageViewCI{};
	imageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	imageViewCI.image = m_imageHandle;
	imageViewCI.format = m_format;
	imageViewCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
	imageViewCI.components = { .r = VK_COMPONENT_SWIZZLE_R, .g = VK_COMPONENT_SWIZZLE_G, .b = VK_COMPONENT_SWIZZLE_B, .a = VK_COMPONENT_SWIZZLE_A };
	imageViewCI.subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = m_mipLevelCount, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount };
	ASSERT_DEBUG(vkCreateImageView(device, &imageViewCI, nullptr, &m_imageViewHandle) == VK_SUCCESS, "Vulkan", "Image view creation failed.");

	VkSamplerCreateInfo samplerCI{};
	samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerCI.magFilter = VK_FILTER_LINEAR;
	samplerCI.minFilter = VK_FILTER_LINEAR;
	samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	samplerCI.anisotropyEnable = VK_TRUE;
	samplerCI.maxAnisotropy = m_memoryManager->m_physDevLimits.maxSamplerAnisotropy;
	samplerCI.borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE;
	samplerCI.unnormalizedCoordinates = VK_FALSE;
	samplerCI.compareEnable = VK_FALSE;
	samplerCI.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCI.minLod = 0.0f;
	samplerCI.maxLod = 1.0f;
	samplerCI.mipLodBias = 0.0f;
	vkCreateSampler(device, &samplerCI, nullptr, &m_sampler);
}
ImageCubeMap::ImageCubeMap(ImageCubeMap&& src) noexcept
{
	m_imageHandle = src.m_imageHandle;
	m_imageViewHandle = src.m_imageViewHandle;
	m_sampler = src.m_sampler;

	m_format = src.m_format;
	m_sideLength = src.m_sideLength;
	m_mipLevelCount = src.m_mipLevelCount;
	m_arrayLayerCount = src.m_arrayLayerCount;

	m_imageAllocIter = src.m_imageAllocIter;

	src.m_invalid = true;
}
ImageCubeMap::~ImageCubeMap()
{
	if (!m_invalid)
	{
		vkDestroySampler(m_memoryManager->m_device, m_sampler, nullptr);
	}
}


void ImageCubeMap::cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, uint32_t regionCount, uint32_t sideLength, VkDeviceSize* bufferOffset, uint32_t* dstImageLayerIndex)
{
	VkBufferImageCopy* bufferImageCopies{ new VkBufferImageCopy[regionCount] };

	for (uint32_t i{ 0 }; i < regionCount; ++i)
	{
		VkImageSubresourceLayers imageSubresource{};
		imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageSubresource.layerCount = 1;
		imageSubresource.baseArrayLayer = dstImageLayerIndex[i];
		imageSubresource.mipLevel = 0;

		bufferImageCopies[i] = VkBufferImageCopy{
			.bufferOffset = bufferOffset[i],
			.imageSubresource = imageSubresource,
			.imageOffset = { 0, 0, 0 },
			.imageExtent = { sideLength, sideLength, 1 }
		};
	}

	vkCmdCopyBufferToImage(cb, srcBuffer, m_imageHandle, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regionCount, bufferImageCopies);
	delete[] bufferImageCopies;
}