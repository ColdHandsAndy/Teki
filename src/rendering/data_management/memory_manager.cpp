#include "memory_manager.h"
#include "src/tools/asserter.h"

MemoryManager::MemoryManager(const VulkanObjectHandler& vulkanObjects) : m_physDevLimits{ vulkanObjects.getPhysDevLimits() }
{
	VmaAllocatorCreateInfo createInfo{};
	createInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	createInfo.instance = vulkanObjects.getInstance();
	createInfo.physicalDevice = vulkanObjects.getPhysicalDevice();
	createInfo.device = vulkanObjects.getLogicalDevice();
	createInfo.vulkanApiVersion = VK_API_VERSION_1_3;

	EASSERT(vmaCreateAllocator(&createInfo, &m_allocator) == VK_SUCCESS, "VMA", "Allocator creation failed.")

	m_device = vulkanObjects.getLogicalDevice();

	m_graphicsQueueFamilyIndex = vulkanObjects.getGraphicsFamilyIndex();
	m_computeQueueFamilyIndex = vulkanObjects.getComputeFamilyIndex();
	m_transferQueueFamilyIndex = vulkanObjects.getTransferFamilyIndex();
}

MemoryManager::~MemoryManager()
{
	m_descriptorBufferDestruction();
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

void MemoryManager::destroyImage(VkImage image, std::list<VmaAllocation>::const_iterator allocIter)
{
	vmaDestroyImage(m_allocator, image, *(allocIter));

	m_allocations.erase(allocIter);
}
