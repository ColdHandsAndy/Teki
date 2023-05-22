#include "image_list.h"

#include <cmath>

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
	imageCI.queueFamilyIndexCount = 1;
	imageCI.pQueueFamilyIndices = &m_memoryManager->m_transferQueueFamilyIndex;
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
		m_freeLayers.push(static_cast<uint32_t>(i));
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
ImageList::~ImageList()
{
	if (!m_invalid)
	{
		vkDestroyImageView(m_memoryManager->m_device, m_imageViewHandle, nullptr);
		m_memoryManager->destroyImage(m_imageHandle, m_imageAllocIter);
	}
}

VkImage ImageList::getImageHandle()
{
	return m_imageHandle;
}

VkImageView ImageList::getImageView()
{
	return m_imageViewHandle;
}

VkFormat ImageList::getFormat()
{
	return m_format;
}

uint32_t ImageList::getWidth()
{
	return m_width;
}

uint32_t ImageList::getHeight()
{
	return m_height;
}

uint32_t ImageList::getLayerCount()
{
	return m_arrayLayerCount;
}


VkImageSubresourceRange ImageList::getSubresourceRange()
{
	return VkImageSubresourceRange{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = m_mipLevelCount, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount};
}

bool ImageList::getLayer(uint16_t& slotIndex)
{
	if (m_freeLayers.empty())
		return false;
	slotIndex = m_freeLayers.top();
	m_freeLayers.pop();
	return true;
}

void ImageList::freeLayer(uint16_t freedSlotIndex)
{
	m_freeLayers.push(freedSlotIndex);
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

void ImageList::cmdCreateMipmaps(VkCommandBuffer cb)
{

}

void ImageList::assignGlobalMemoryManager(MemoryManager& memManager)
{
	m_memoryManager = &memManager;
}
