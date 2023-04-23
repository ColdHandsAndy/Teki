#include "memory_manager.h"
#include "src/tools/asserter.h"

MemoryManager::MemoryManager(const VulkanObjectHandler& vulkanObjects)
{
	VmaAllocatorCreateInfo createInfo{};
	createInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	createInfo.instance = vulkanObjects.getInstance();
	createInfo.physicalDevice = vulkanObjects.getPhysicalDevice();
	createInfo.device = vulkanObjects.getLogicalDevice();
	createInfo.vulkanApiVersion = VK_API_VERSION_1_3;

	ASSERT_ALWAYS(vmaCreateAllocator(&createInfo, &m_allocator) == VK_SUCCESS, "VMA", "Allocator creation failed.")
}

MemoryManager::~MemoryManager()
{
	vmaDestroyAllocator(m_allocator);
}

VmaAllocator MemoryManager::getAllocator()
{
	return m_allocator;
}

std::list<VmaAllocation>::iterator MemoryManager::addAllocation()
{
	return m_allocations.insert(m_allocations.end(), VmaAllocation{});
}

void MemoryManager::destroyBuffer(VkBuffer buffer, std::list<VmaAllocation>::const_iterator allocIter)
{
	vmaDestroyBuffer(m_allocator, buffer, *(allocIter));

	m_allocations.erase(allocIter);
}
