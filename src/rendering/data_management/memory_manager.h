#ifndef MEMORY_MANAGER_CLASS_HEADER
#define MEMORY_MANAGER_CLASS_HEADER

#include <memory>
#include <cassert>
#include <cstdint>
#include <list>

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"

class ResourceSet;

class [[nodiscard]] MemoryManager
{
private:
	VmaAllocator m_allocator{};
	std::list<VmaAllocation> m_allocations{};

	VkDevice m_device{};
	const VkPhysicalDeviceLimits& m_physDevLimits{};

	uint32_t m_sharedMemoryTypeMask{ 0 };

	uint32_t m_graphicsQueueFamilyIndex{};
	uint32_t m_computeQueueFamilyIndex{};
	uint32_t m_transferQueueFamilyIndex{};

public:
	MemoryManager() = delete;
	MemoryManager(const VulkanObjectHandler& vulkanObjects);
	~MemoryManager();

private:
	VmaAllocator getAllocator();
	std::list<VmaAllocation>::iterator addAllocation();
	void destroyBuffer(VkBuffer buffer, std::list<VmaAllocation>::const_iterator allocIter);
	void destroyImage(VkImage image, std::list<VmaAllocation>::const_iterator allocIter);

	friend class BufferBase;
	friend class BufferBaseHostInaccessible;
	friend class BufferBaseHostAccessible;
	friend class ImageBase;
	friend class Image;
	friend class ImageList;
	friend class ImageCubeMap;
	friend class DescriptorManager;
	friend class ResourceSet;
};

#endif