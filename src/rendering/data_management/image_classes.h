#ifndef IMAGE_LIST_CLASS_HEADER
#define IMAGE_LIST_CLASS_HEADER

#include <list>
#include <deque>

#include "vulkan/vulkan.h"
#include "vma/vk_mem_alloc.h"

#include "src/rendering/data_management/memory_manager.h"

class ImageBase
{
protected:
	VkImage m_imageHandle{};
	VkImageView m_imageViewHandle{};

	VkFormat m_format{};
	uint32_t m_mipLevelCount{};
	uint32_t m_arrayLayerCount{};

	std::list<VmaAllocation>::const_iterator m_imageAllocIter{};

	bool m_invalid{ false };

	inline static MemoryManager* m_memoryManager{ nullptr };

public:
	ImageBase() = default;
	~ImageBase()
	{
		if (!m_invalid)
		{
			vkDestroyImageView(m_memoryManager->m_device, m_imageViewHandle, nullptr);
			m_memoryManager->destroyImage(m_imageHandle, m_imageAllocIter);
		}
	}

	VkImage getImageHandle() const
	{
		return m_imageHandle;
	}
	VkImageView getImageView() const
	{
		return m_imageViewHandle;
	}
	VkFormat getFormat() const
	{
		return m_format;
	}

	static void assignGlobalMemoryManager(MemoryManager& memManager)
	{
		m_memoryManager = &memManager;
	}

};



class ImageList : public ImageBase
{
private:
	uint32_t m_width{};
	uint32_t m_height{};
	uint32_t m_mipLevelCount{};
	uint32_t m_arrayLayerCount{};

	std::deque<uint16_t> m_freeLayers{};

public:
	ImageList() = delete;
	ImageList(VkDevice device, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usageFlags, bool allocateMips = false, uint32_t layercount = 0); //if layerCount is zero default layerCount is used
	ImageList(ImageList&& src) noexcept;
	~ImageList() = default;

	uint32_t getWidth() const;
	uint32_t getHeight() const;
	uint32_t getLayerCount() const;
	VkImageSubresourceRange getSubresourceRange() const;
	
	void cmdCreateMipmaps(VkCommandBuffer cb, VkImageLayout currentListLayout);

	void cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, uint32_t regionCount, VkDeviceSize* bufferOffset, uint32_t* width, uint32_t* height, uint32_t* dstImageLayerIndex, uint32_t* mipLevel = nullptr);

	bool getLayer(uint16_t& layerIndex);
	void freeLayer(uint16_t freedLayerIndex);

};


class ImageCubeMap : public ImageBase
{
private:
	uint32_t m_sideLength{};

	VkSampler m_sampler{};

public:
	ImageCubeMap() = delete;
	ImageCubeMap(VkDevice device, VkFormat format, uint32_t sideLength, VkImageUsageFlags usageFlags);
	ImageCubeMap(ImageCubeMap&& src) noexcept;
	~ImageCubeMap();

	VkSampler getSampler() const { return m_sampler; }

	void cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, uint32_t regionCount, uint32_t sideLength, VkDeviceSize* bufferOffset, uint32_t* dstImageLayerIndex);
};

#endif