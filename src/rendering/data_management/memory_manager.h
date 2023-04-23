#ifndef MEMORY_MANAGER_CLASS_HEADER
#define MEMORY_MANAGER_CLASS_HEADER

#include <memory>
#include <cassert>
#include <cstdint>
#include <list>

#include "vulkan/vulkan.h"
#include "vma/vk_mem_alloc.h"

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"

class BufferBaseHostInaccessible;
class BufferBaseHostAccessible;

class [[nodiscard]] MemoryManager
{
private:
	VmaAllocator m_allocator{};
	std::list<VmaAllocation> m_allocations{};

public:
	MemoryManager() = delete;
	MemoryManager(const VulkanObjectHandler& vulkanObjects);
	~MemoryManager();

	void destroyBuffer(VkBuffer buffer, std::list<VmaAllocation>::const_iterator allocIter);
	
	//create pool
	//pool allocation
	
//private:
public:
	VmaAllocator getAllocator();
	std::list<VmaAllocation>::iterator addAllocation();

	friend class BufferBaseHostInaccessible;
	friend class BufferBaseHostAccessible;
};

#endif