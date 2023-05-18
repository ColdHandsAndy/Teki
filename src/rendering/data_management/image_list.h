#ifndef IMAGE_LIST_CLASS_HEADER
#define IMAGE_LIST_CLASS_HEADER

#include <list>
#include <stack>

#include "vulkan/vulkan.h"
#include "vma/vk_mem_alloc.h"

#include "src/rendering/data_management/memory_manager.h"

class ImageList
{
private:
	VkImage m_imageHandle{};
	VkImageView m_imageViewHandle{};

	VkFormat m_format{};
	uint32_t m_width{};
	uint32_t m_height{};
	uint32_t m_mipLevelCount{};
	uint32_t m_arrayLayerCount{};

	std::stack<uint16_t> m_freeLayers{};

	std::list<VmaAllocation>::const_iterator m_imageAllocIter{};

	inline static MemoryManager* m_memoryManager{ nullptr };

public:
	ImageList() = delete;
	ImageList(VkDevice device, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usageFlags, bool allocateMips = false, uint32_t layercount = 0); //if layerCount is zero default layerCount is used
	~ImageList();

	VkImage getImageHandle();
	VkImageView getImageView();
	VkFormat getFormat();
	uint32_t getWidth();
	uint32_t getHeight();
	uint32_t getLayerCount();
	VkImageSubresourceRange getSubresourceRange();

	static void assignGlobalMemoryManager(MemoryManager& memManager);

	void cmdCopyDataFromBuffer(VkCommandBuffer cb, VkBuffer srcBuffer, uint32_t regionCount, VkDeviceSize* bufferOffset, uint32_t* width, uint32_t* height, uint32_t* dstImageLayerIndex, uint32_t* mipLevel = nullptr);

	bool getLayer(uint16_t& layerIndex);
	void freeLayer(uint16_t freedLayerIndex);

};

#endif
