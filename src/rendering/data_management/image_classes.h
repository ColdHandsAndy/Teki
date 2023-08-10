#ifndef IMAGE_LIST_CLASS_HEADER
#define IMAGE_LIST_CLASS_HEADER

#include <list>
#include <deque>

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include "src/rendering/data_management/memory_manager.h"

class ImageBase
{
protected:
	VkImage m_imageHandle{};
	VkImageView m_imageViewHandle{};

	VkFormat m_format{};
	uint32_t m_mipLevelCount{};

	std::list<VmaAllocation>::const_iterator m_imageAllocIter{};

	bool m_invalid{ false };

	inline static MemoryManager* m_memoryManager{ nullptr };

	ImageBase() = default;

public:
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
	uint32_t getMipLevelCount() const
	{
		return m_mipLevelCount;
	}

	static void assignGlobalMemoryManager(MemoryManager& memManager)
	{
		m_memoryManager = &memManager;
	}

};


class Image : public ImageBase
{
private:
	uint32_t m_width{};
	uint32_t m_height{};
	VkImageAspectFlags m_aspects{};

public:
	Image() = delete;
	explicit Image(VkDevice device, VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usageFlags, VkImageAspectFlags imageAspects, bool allocateMips = false);
	Image(Image&& src) noexcept;
	~Image() = default;

	uint32_t getWidth() const;
	uint32_t getHeight() const;
	VkImageSubresourceRange getSubresourceRange() const;

	void cmdCreateMipmaps(VkCommandBuffer cb, VkImageLayout currentImageLayout);

	void cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, VkDeviceSize bufferOffset, int xOffset, int yOffset, uint32_t width, uint32_t height, uint32_t mipLevel = 0);

};

class ImageMS : public ImageBase
{
private:
	uint32_t m_width{};
	uint32_t m_height{};
	VkImageAspectFlags m_aspects{};

public:
	ImageMS() = delete;
	explicit ImageMS(VkDevice device, VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usageFlags, VkImageAspectFlags imageAspects, VkSampleCountFlagBits sampleCount);
	ImageMS(ImageMS&& src) noexcept;
	~ImageMS() = default;

	uint32_t getWidth() const;
	uint32_t getHeight() const;
	VkImageSubresourceRange getSubresourceRange() const;

};

class ImageList : public ImageBase
{
private:
	uint32_t m_width{};
	uint32_t m_height{};
	uint32_t m_arrayLayerCount{};

	VkImageAspectFlags m_aspects{};

	std::deque<uint16_t> m_freeLayers{};

public:
	ImageList() = delete;
	explicit ImageList(VkDevice device, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usageFlags, bool allocateMips = false, uint32_t layercount = 0, VkImageAspectFlags aspects = VK_IMAGE_ASPECT_COLOR_BIT); //if layerCount is zero default layerCount is used
	ImageList(ImageList&& src) noexcept;
	~ImageList() = default;
	ImageList& operator=(ImageList&& src) noexcept;


	uint32_t getWidth() const;
	uint32_t getHeight() const;
	uint32_t getLayerCount() const;
	VkImageSubresourceRange getSubresourceRange() const;
	
	void cmdCreateMipmaps(VkCommandBuffer cb, VkImageLayout currentListLayout);

	void cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, uint32_t regionCount, VkDeviceSize* bufferOffset, uint32_t* width, uint32_t* height, uint32_t* dstImageLayerIndex, uint32_t* mipLevel = nullptr);
	void cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, uint32_t regionCount, VkDeviceSize* bufferOffset, uint32_t* dstImageLayerIndex, uint32_t* mipLevel = nullptr);

	void cmdTransitionLayoutFromUndefined(VkCommandBuffer cb, VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask, VkImageLayout dstLayout);

	bool getLayer(uint16_t& layerIndex);
	void freeLayer(uint16_t freedLayerIndex);

	friend class ImageListContainer;
};


class ImageCubeMap : public ImageBase
{
private:
	uint32_t m_sideLength{};
	uint32_t m_arrayLayerCount{};

	VkImageAspectFlags m_aspects{};
	VkSampler m_sampler{};

public:
	ImageCubeMap() = delete;
	explicit ImageCubeMap(VkDevice device, VkFormat format, uint32_t sideLength, VkImageUsageFlags usageFlags, int mipLevels = -1, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
	ImageCubeMap(ImageCubeMap&& src) noexcept;
	~ImageCubeMap();

	VkSampler getSampler() const { return m_sampler; }
	uint32_t getSideLength() const { return m_sideLength; };
	VkImageSubresourceRange getSubresourceRange() const { return VkImageSubresourceRange{ .aspectMask = m_aspects, .baseMipLevel = 0, .levelCount = m_mipLevelCount, .baseArrayLayer = 0, .layerCount = m_arrayLayerCount }; };

	void cmdTransitionLayoutFromUndefined(VkCommandBuffer cb, VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask, VkImageLayout dstLayout);
	void cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, uint32_t sideLength, uint32_t regionCount, VkDeviceSize* bufferOffset, uint32_t* dstImageLayerIndex, uint32_t* mipLevel = nullptr);
};



//TODO: If new list is added after associated ResourceSet already initialised, need to update ResourceSet payload
class ImageListContainer
{
private:
	VkDevice m_device{};
	VkImageUsageFlags m_listsUsage{};
	bool m_allocateMipmaps{};
	VkSampler m_sampler{};
	VkImageAspectFlags m_aspects{};
	struct ImageListAndAvailability
	{
		ImageList list;
		bool available;
	};
	std::vector<ImageListAndAvailability> m_imageLists{};

	const uint32_t m_listLayerCount{};

public:
	struct ImageListContainerIndices
	{
		uint16_t listIndex;
		uint16_t layerIndex;
	};

public:
	explicit ImageListContainer(VkDevice device, VkImageUsageFlags usageFlags, bool allocateMips, VkSamplerCreateInfo samplerCI, uint32_t listLayerCount = 4, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);
	~ImageListContainer();

	int getImageListCount() const { return m_imageLists.size(); }
	VkImageSubresourceRange getImageListSubresourceRange(uint32_t index) const { return m_imageLists[index].list.getSubresourceRange();  };
	void getImageListResolution(uint32_t listIndex, uint32_t& width, uint32_t& height) { width = m_imageLists[listIndex].list.getWidth(); height = m_imageLists[listIndex].list.getHeight(); };

	[[nodiscard]] ImageListContainerIndices getNewImage(int width, int height, VkFormat format);
	void freeImage(ImageListContainerIndices indices);
	VkSampler getSampler() const;
	VkImage getImageHandle(uint16_t listIndex) const;
	VkImageView getImageViewHandle(uint16_t listIndex) const;
	void cmdTransitionLayoutsFromUndefined(VkCommandBuffer cb, VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask, VkImageLayout dstLayout);
	void cmdCreateImageMipmaps(VkCommandBuffer cb, ImageListContainerIndices indices, VkImageLayout currentLayout);
	void cmdCreateListMipmaps(VkCommandBuffer cb, uint16_t listIndex, VkImageLayout currentLayout);
	void cmdCreateMipmaps(VkCommandBuffer cb, VkImageLayout currentLayout);
	void cmdCopyDataFromBuffer(VkCommandBuffer cb, uint32_t listIndex, VkBuffer srcBuffer, uint32_t regionCount, VkDeviceSize* bufferOffset, uint32_t* width, uint32_t* height, uint32_t* dstImageLayerIndex, uint32_t* mipLevel = nullptr);
	void cmdCopyDataFromBuffer(VkCommandBuffer cb, uint32_t listIndex, VkBuffer srcBuffer, uint32_t regionCount, VkDeviceSize* bufferOffset, uint32_t* dstImageLayerIndex, uint32_t* mipLevel = nullptr);

};

#endif