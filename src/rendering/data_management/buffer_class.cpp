#include "buffer_class.h"
#include "src/tools/asserter.h"

Buffer::Buffer(VkBufferCreateInfo& bufferCreateInfo, int flags)
{
	VmaAllocationCreateInfo allocCreateInfo{};
	if (flags)
	{
		allocCreateInfo.flags = flags;
	}
	allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	m_bufferByteSize = bufferCreateInfo.size;

	std::list<VmaAllocation>::iterator allocationIter{ m_memoryManager->addAllocation() };
	m_allocationIter = allocationIter;

	ASSERT_DEBUG(vmaCreateBuffer(m_memoryManager->getAllocator(), &bufferCreateInfo, &allocCreateInfo, &m_bufferHandle, &(*allocationIter), nullptr) == VK_SUCCESS, "VMA", "Buffer creation failed.")
}

Buffer::Buffer(Buffer&& srcBuffer)
{
	this->m_bufferHandle = srcBuffer.m_bufferHandle;
	this->m_bufferByteSize = srcBuffer.m_bufferByteSize;
	this->m_allocationIter = srcBuffer.m_allocationIter;
	srcBuffer.m_invalid = true;
}

VkDeviceAddress Buffer::getBufferDeviceAddress(VkDevice device) const
{
	VkBufferDeviceAddressInfo addressInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = m_bufferHandle };
	return vkGetBufferDeviceAddress(device, &addressInfo);
}

Buffer::~Buffer()
{
	if (!m_invalid)
	{
		m_memoryManager->destroyBuffer(this->m_bufferHandle, this->m_allocationIter);
	}
}

void Buffer::assignGlobalMemoryManager(MemoryManager& memManager)
{
	m_memoryManager = &memManager;
}

VkBuffer Buffer::getBufferHandle() const
{
	return m_bufferHandle;
}

VkDeviceSize Buffer::getBufferByteSize() const
{
	return m_bufferByteSize;
}

void Buffer::cmdCopyTo(VkCommandBuffer cmdBuffer, VkBuffer dstBufferHandle, uint32_t regionCount, const VkBufferCopy* regions)
{
	vkCmdCopyBuffer(cmdBuffer, this->m_bufferHandle, dstBufferHandle, regionCount, regions);
}

void Buffer::cmdCopyFrom(VkCommandBuffer cmdBuffer, VkBuffer srcBufferHandle, uint32_t regionCount, const VkBufferCopy* regions)
{
	vkCmdCopyBuffer(cmdBuffer, srcBufferHandle, this->m_bufferHandle, regionCount, regions);
}

Buffer& Buffer::operator=(Buffer&& srcBuffer) noexcept
{
	this->m_bufferHandle = srcBuffer.m_bufferHandle;
	this->m_bufferByteSize = srcBuffer.m_bufferByteSize;
	this->m_allocationIter = srcBuffer.m_allocationIter;
	srcBuffer.m_invalid = true;

	return *this;
}



BufferMappable::BufferMappable(VkBufferCreateInfo& bufferCreateInfo, int flags, bool memoryIsCached, bool useSharedMemory)
{
	VmaAllocationCreateInfo allocCreateInfo{};
	
	if (memoryIsCached && !useSharedMemory)
	{
		allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	}
	else
	{
		allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
	}
	if (useSharedMemory)
		allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	else
		allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

	allocCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	m_bufferByteSize = bufferCreateInfo.size;

	std::list<VmaAllocation>::iterator allocationIter{ m_memoryManager->addAllocation() };
	m_allocationIter = allocationIter;

	ASSERT_DEBUG(vmaCreateBuffer(m_memoryManager->getAllocator(), &bufferCreateInfo, &allocCreateInfo, &m_bufferHandle, &(*allocationIter), nullptr) == VK_SUCCESS, "VMA", "Buffer creation failed.")
}

BufferMappable::BufferMappable(BufferMappable&& srcBuffer)
{
	this->m_bufferHandle = srcBuffer.m_bufferHandle;
	this->m_bufferByteSize = srcBuffer.m_bufferByteSize;
	this->m_allocationIter = srcBuffer.m_allocationIter;
	this->m_mappedMemoryPointer = srcBuffer.m_mappedMemoryPointer;
	srcBuffer.m_invalid = true;
}

BufferMappable::~BufferMappable()
{
	if (!m_invalid)
	{
		if (m_mappedMemoryPointer != nullptr)
		{
			unmapMemory();
		}
	}
}

void* BufferMappable::getData()
{
	if (m_mappedMemoryPointer == nullptr)
	{
		mapMemory();
	}

	return m_mappedMemoryPointer;
}

void BufferMappable::mapMemory()
{
	vmaMapMemory(m_memoryManager->getAllocator(), *m_allocationIter, &m_mappedMemoryPointer);
}

void BufferMappable::unmapMemory()
{
	vmaUnmapMemory(m_memoryManager->getAllocator(), *m_allocationIter);
	m_mappedMemoryPointer = nullptr;
}

BufferMappable& BufferMappable::operator=(BufferMappable&& srcBuffer) noexcept
{
	this->m_bufferHandle = srcBuffer.m_bufferHandle;
	this->m_bufferByteSize = srcBuffer.m_bufferByteSize;
	this->m_allocationIter = srcBuffer.m_allocationIter;
	this->m_mappedMemoryPointer = srcBuffer.m_mappedMemoryPointer;
	srcBuffer.m_invalid = true;

	return *this;
}

/////////////////////////////////////////////////////// REFACTORED BUFFER CLASSES /////////////////////////////////////////////////////////

//BufferBase::BufferBase(VkDevice device, const VkBufferCreateInfo& bufferCI, bool sharedMem, bool mappable, bool cached, int flags = NULL_FLAG)
//{
//	VmaAllocationCreateInfo allocCI{};
//	if (sharedMem)
//	{
//		allocCI.flags = flags;
//		allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
//		allocCI.usage = VMA_MEMORY_USAGE_AUTO;
//	}
//	else if (mappable && cached)
//	{
//		allocCI.flags = flags;
//		allocCI.requiredFlags =  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
//		allocCI.usage = VMA_MEMORY_USAGE_AUTO;
//	}
//	else if (mappable && !cached)
//	{
//		allocCI.flags = flags;
//		allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
//		allocCI.usage = VMA_MEMORY_USAGE_AUTO;
//	}
//	else
//	{
//		allocCI.flags = flags;
//		allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
//		allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
//	}
//
//	VmaVirtualBlockCreateInfo virtBlockCI{};
//	virtBlockCI.size = bufferCI.size;
//
//	ASSERT_DEBUG((m_memoryManager != nullptr), "App", "Memory manager is not assigned.")
//	ASSERT_DEBUG(vmaCreateVirtualBlock(&virtBlockCI, &m_memoryProxy) == VK_SUCCESS, "VMA", "Virtual block creation failed.")
//
//	std::list<VmaAllocation>::iterator allocationIter{ m_memoryManager->addAllocation() };
//	m_bufferAllocIter = allocationIter;
//
//	ASSERT_DEBUG(vmaCreateBuffer(m_memoryManager->getAllocator(), &bufferCI, &allocCI, &m_bufferHandle, &(*allocationIter), nullptr) == VK_SUCCESS, "VMA", "Buffer creation failed.")
//
//	m_memoryByteSize = bufferCI.size;
//	VkBufferDeviceAddressInfo devicAddrInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = m_bufferHandle};
//	m_bufferMemoryAddress = vkGetBufferDeviceAddress(device, &devicAddrInfo);
//	VkMemoryRequirements memReqs{};
//	vkGetBufferMemoryRequirements(device, m_bufferHandle, &memReqs);
//	m_bufferAlignment = memReqs.alignment;
//}
//BufferBase::~BufferBase()
//{
//	if (!m_invalid)
//	{
//		vmaClearVirtualBlock(m_memoryProxy);
//		vmaDestroyVirtualBlock(m_memoryProxy);
//		m_memoryManager->destroyBuffer(this->m_bufferHandle, this->m_bufferAllocIter);
//	}
//}
//
//void BufferBase::allocateFromBuffer(VkDeviceSize allocSize, VmaVirtualAllocation& allocation, VkDeviceSize& inBufferOffset)
//{
//	VmaVirtualAllocationCreateInfo allocCI{ .size = allocSize, .alignment = m_bufferAlignment };
//	vmaVirtualAllocate(m_memoryProxy, &allocCI, &allocation, &inBufferOffset);
//}
//void BufferBase::freeBufferAllocation(VmaVirtualAllocation allocation)
//{
//	vmaVirtualFree(m_memoryProxy, allocation);
//}
//VkBuffer BufferBase::getBufferHandle() const
//{
//	return m_bufferHandle;
//}
//VkDeviceSize BufferBase::getBufferByteSize() const
//{
//	return m_memoryByteSize;
//}
//VkDeviceAddress BufferBase::getBufferDeviceAddress() const
//{
//	return m_bufferMemoryAddress;
//}
//
//void BufferBase::assignGlobalMemoryManager(MemoryManager& memManager)
//{
//	m_memoryManager = &memManager;
//}
//
//
//BufferBaseHostInaccessible::BufferBaseHostInaccessible(VkDevice device, const VkBufferCreateInfo& bufferCreateInfo, int flags)
//	: BufferBase(device, bufferCreateInfo, false, false, false, flags)
//{
//}
//BufferBaseHostInaccessible::~BufferBaseHostInaccessible()
//{
//}
//
//Buffer::Buffer(BufferBaseHostInaccessible& motherBuffer, VkDeviceSize size) : m_motherBuffer{ motherBuffer }
//{
//	m_bufferHandle = motherBuffer.getBufferHandle();
//	m_bufferSize = size;
//	motherBuffer.allocateFromBuffer(size, m_allocation, m_bufferOffset);
//	m_deviceAddress = motherBuffer.getBufferDeviceAddress() + m_bufferOffset;
//}
//Buffer::~Buffer()
//{
//	m_motherBuffer.freeBufferAllocation(m_allocation);
//}
//VkBuffer Buffer::getBufferHandle()
//{
//	return m_bufferHandle;
//}
//VkDeviceAddress Buffer::getDeviceAddress()
//{
//	return m_deviceAddress;
//}
//
//
//BufferBaseHostAccessible::BufferBaseHostAccessible(VkDevice device, VkBufferCreateInfo bufferCreateInfo, int flags, bool useSharedMemory = false, bool memoryIsCached = false)
//	: BufferBase(device, bufferCreateInfo, useSharedMemory, true, memoryIsCached, flags)
//{
//
//}
//BufferBaseHostAccessible::~BufferBaseHostAccessible()
//{
//	if (!m_invalid)
//	{
//		if (m_mappedMemoryPointer != nullptr)
//		{
//			unmapMemory();
//		}
//	}
//}
//
//void BufferBaseHostAccessible::mapMemory()
//{
//	vmaMapMemory(m_memoryManager->getAllocator(), *m_bufferAllocIter, &m_mappedMemoryPointer);
//}
//void BufferBaseHostAccessible::unmapMemory()
//{
//	vmaUnmapMemory(m_memoryManager->getAllocator(), *m_bufferAllocIter);
//	m_mappedMemoryPointer = nullptr;
//}
//void* BufferBaseHostAccessible::getData()
//{
//	if (m_mappedMemoryPointer == nullptr)
//	{
//		mapMemory();
//	}
//
//	return m_mappedMemoryPointer;
//}
//
//BufferMapped::BufferMapped(BufferBaseHostAccessible& motherBuffer, VkDeviceSize size) : m_motherBuffer{motherBuffer}
//{
//	m_bufferHandle = motherBuffer.getBufferHandle();
//	m_bufferSize = size;
//	motherBuffer.allocateFromBuffer(size, m_allocation, m_bufferOffset);
//	m_deviceAddress = motherBuffer.getBufferDeviceAddress() + m_bufferOffset;
//	m_dataPtr = reinterpret_cast<void*>(reinterpret_cast<uint8_t>(motherBuffer.getData()) + m_bufferOffset);
//}
//BufferMapped::~BufferMapped()
//{
//	m_motherBuffer.freeBufferAllocation(m_allocation);
//}
//VkBuffer BufferMapped::getBufferHandle()
//{
//	return m_bufferHandle;
//}
//void* BufferMapped::getData()
//{
//	return m_dataPtr;
//}
//VkDeviceAddress BufferMapped::getDeviceAddress()
//{
//	return m_deviceAddress;
//}