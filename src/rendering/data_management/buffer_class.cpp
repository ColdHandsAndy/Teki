#include "buffer_class.h"
#include "src/tools/logging.h"
#include "src/tools/asserter.h"

namespace BufferTools
{
	void cmdBufferCopy(VkCommandBuffer cmdBuffer, VkBuffer srcBufferHandle, VkBuffer dstBufferHandle, uint32_t regionCount, const VkBufferCopy* regions)
	{
		vkCmdCopyBuffer(cmdBuffer, srcBufferHandle, dstBufferHandle, regionCount, regions);
	}
}

BufferBase::BufferBase(VkDevice device, const VkBufferCreateInfo& bufferCI, bool sharedMem, bool mappable, bool cached, int allocFlags)
{
	VmaAllocationCreateInfo allocCI{};
	if (sharedMem)
	{
		allocCI.flags = allocFlags | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		allocCI.usage = VMA_MEMORY_USAGE_AUTO;
	}
	else if (mappable && cached)
	{
		allocCI.flags = allocFlags | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
		allocCI.requiredFlags =  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
		allocCI.usage = VMA_MEMORY_USAGE_AUTO;
	}
	else if (mappable && !cached)
	{
		allocCI.flags = allocFlags | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
		allocCI.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		allocCI.usage = VMA_MEMORY_USAGE_AUTO;
	}
	else
	{
		allocCI.flags = allocFlags;
		allocCI.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		allocCI.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	}

	VmaVirtualBlockCreateInfo virtBlockCI{};
	virtBlockCI.size = bufferCI.size;

	ASSERT_DEBUG((m_memoryManager != nullptr), "App", "Memory manager is not assigned.")
	ASSERT_DEBUG(vmaCreateVirtualBlock(&virtBlockCI, &m_memoryProxy) == VK_SUCCESS, "VMA", "Virtual block creation failed.")

	std::list<VmaAllocation>::iterator allocationIter{ m_memoryManager->addAllocation() };
	m_bufferAllocIter = allocationIter;

	ASSERT_DEBUG(vmaCreateBuffer(m_memoryManager->getAllocator(), &bufferCI, &allocCI, &m_bufferHandle, &(*allocationIter), nullptr) == VK_SUCCESS, "VMA", "Buffer creation failed.")

	m_memoryByteSize = bufferCI.size;
	if (bufferCI.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
	{
		VkBufferDeviceAddressInfo devicAddrInfo{ .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = m_bufferHandle };
		m_bufferMemoryAddress = vkGetBufferDeviceAddress(device, &devicAddrInfo);
	}
	else
	{
		m_bufferMemoryAddress = UINT64_MAX;
	}
	VkMemoryRequirements memReqs{};
	vkGetBufferMemoryRequirements(device, m_bufferHandle, &memReqs);
	m_bufferAlignment = memReqs.alignment;
}
BufferBase::BufferBase(BufferBase&& bufBase) noexcept
{
	if (&bufBase == this)
		return;

	this->m_bufferAllocIter		=	bufBase.m_bufferAllocIter;
	this->m_memoryProxy			=	bufBase.m_memoryProxy;
	this->m_bufferHandle		=	bufBase.m_bufferHandle;
	this->m_bufferMemoryAddress =	bufBase.m_bufferMemoryAddress;
	this->m_memoryByteSize		=	bufBase.m_memoryByteSize;
	this->m_bufferAlignment		=	bufBase.m_bufferAlignment;

	bufBase.m_invalid = true;
}
BufferBase::~BufferBase()
{
	if (!m_invalid)
	{
		vmaClearVirtualBlock(m_memoryProxy);
		vmaDestroyVirtualBlock(m_memoryProxy);
		m_memoryManager->destroyBuffer(this->m_bufferHandle, this->m_bufferAllocIter);
	}
}

void BufferBase::allocateFromBuffer(VkDeviceSize allocSize, VmaVirtualAllocation& allocation, VkDeviceSize& inBufferOffset)
{
	VmaVirtualAllocationCreateInfo allocCI{ .size = allocSize, .alignment = m_bufferAlignment };
	LOG_IF_WARNING(vmaVirtualAllocate(m_memoryProxy, &allocCI, &allocation, &inBufferOffset) == VK_ERROR_OUT_OF_DEVICE_MEMORY, "{}", "Not enough buffer memory for suballocation. Need mechanism to handle the situation")
}
void BufferBase::freeBufferAllocation(VmaVirtualAllocation allocation)
{
	vmaVirtualFree(m_memoryProxy, allocation);
}
VkBuffer BufferBase::getBufferHandle() const
{
	return m_bufferHandle;
}
VkDeviceSize BufferBase::getBufferByteSize() const
{
	return m_memoryByteSize;
}
VkDeviceAddress BufferBase::getBufferDeviceAddress() const
{
	LOG_IF_WARNING(m_bufferMemoryAddress == UINT64_MAX, "Buffer was not created with \"VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO\"")
	return m_bufferMemoryAddress;
}

void BufferBase::assignGlobalMemoryManager(MemoryManager& memManager)
{
	m_memoryManager = &memManager;
}


BufferBaseHostInaccessible::BufferBaseHostInaccessible(VkDevice device, const VkBufferCreateInfo& bufferCreateInfo, int allocFlags)
	: BufferBase(device, bufferCreateInfo, false, false, false, allocFlags)
{
}
BufferBaseHostInaccessible::BufferBaseHostInaccessible(VkDevice device, VkDeviceSize bufferSize, VkBufferUsageFlags usageFlags, std::span<const uint32_t> queueFamilyIndices, int allocFlags)
	: BufferBase(device, 
		VkBufferCreateInfo
		{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 
			.size = bufferSize, 
				.usage = usageFlags, 
					.sharingMode = (queueFamilyIndices.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE), 
						.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size()), 
							.pQueueFamilyIndices = queueFamilyIndices.data()
		},
		false, 
		false, 
		false, 
		allocFlags)
{

}
BufferBaseHostInaccessible::~BufferBaseHostInaccessible()
{
}

Buffer::Buffer(BufferBaseHostInaccessible& motherBuffer, VkDeviceSize size) : m_motherBuffer{ motherBuffer }
{
	m_bufferHandle = motherBuffer.getBufferHandle();
	m_bufferSize = size;
	motherBuffer.allocateFromBuffer(size, m_allocation, m_bufferOffset);
	m_deviceAddress = motherBuffer.getBufferDeviceAddress() + m_bufferOffset;
}
Buffer::~Buffer()
{
	m_motherBuffer.freeBufferAllocation(m_allocation);
}
VkBuffer Buffer::getBufferHandle() const
{
	return m_bufferHandle;
}
VkDeviceSize Buffer::getSize() const
{
	return m_bufferSize;
}
VkDeviceSize Buffer::getOffset() const
{
	return m_bufferOffset;
}
VkDeviceAddress Buffer::getDeviceAddress() const
{
	return m_deviceAddress;
}


BufferBaseHostAccessible::BufferBaseHostAccessible(VkDevice device, const VkBufferCreateInfo& bufferCreateInfo, int allocFlags, bool useSharedMemory, bool memoryIsCached)
	: BufferBase(device, bufferCreateInfo, useSharedMemory, true, memoryIsCached, allocFlags)
{

}
BufferBaseHostAccessible::BufferBaseHostAccessible(VkDevice device, VkDeviceSize bufferSize, VkBufferUsageFlags usageFlags, std::span<const uint32_t> queueFamilyIndices, int allocFlags, bool useSharedMemory, bool memoryIsCached)
	: BufferBase(device,
		VkBufferCreateInfo
		{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = bufferSize,
				.usage = usageFlags,
					.sharingMode = (queueFamilyIndices.size() > 1 ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE),
						.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size()),
							.pQueueFamilyIndices = queueFamilyIndices.data()
		},
		useSharedMemory,
		true,
		memoryIsCached,
		allocFlags)
{

}
BufferBaseHostAccessible::BufferBaseHostAccessible(BufferBaseHostAccessible&& srcBuffer) noexcept
{
	if (&srcBuffer == this)
		return;

	this->m_bufferAllocIter = srcBuffer.m_bufferAllocIter;
	this->m_memoryProxy = srcBuffer.m_memoryProxy;
	this->m_bufferHandle = srcBuffer.m_bufferHandle;
	this->m_bufferMemoryAddress = srcBuffer.m_bufferMemoryAddress;
	this->m_memoryByteSize = srcBuffer.m_memoryByteSize;
	this->m_bufferAlignment = srcBuffer.m_bufferAlignment;
	this->m_mappedMemoryPointer = srcBuffer.m_mappedMemoryPointer;

	srcBuffer.m_invalid = true;
}
BufferBaseHostAccessible::~BufferBaseHostAccessible()
{
	if (!m_invalid)
	{
		if (m_mappedMemoryPointer != nullptr)
		{
			unmapMemory();
		}
	}
}

void BufferBaseHostAccessible::mapMemory()
{
	if (m_mappedMemoryPointer == nullptr)
	{
		vmaMapMemory(m_memoryManager->getAllocator(), *m_bufferAllocIter, &m_mappedMemoryPointer);
	}
}
void BufferBaseHostAccessible::unmapMemory()
{
	vmaUnmapMemory(m_memoryManager->getAllocator(), *m_bufferAllocIter);
	m_mappedMemoryPointer = nullptr;
}
void* BufferBaseHostAccessible::getData()
{
	if (m_mappedMemoryPointer == nullptr)
	{
		mapMemory();
	}

	return m_mappedMemoryPointer;
}

BufferMapped::BufferMapped(BufferBaseHostAccessible& motherBuffer, VkDeviceSize size) : m_motherBuffer{motherBuffer}
{
	m_bufferHandle = motherBuffer.getBufferHandle();
	m_bufferSize = size;
	motherBuffer.allocateFromBuffer(size, m_allocation, m_bufferOffset);
	m_deviceAddress = motherBuffer.getBufferDeviceAddress() + m_bufferOffset;

	m_dataPtr = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(motherBuffer.getData()) + m_bufferOffset);
}
BufferMapped::~BufferMapped()
{
	m_motherBuffer.freeBufferAllocation(m_allocation);
}
VkBuffer BufferMapped::getBufferHandle() const
{
	return m_bufferHandle;
}
VkDeviceSize BufferMapped::getSize() const
{
	return m_bufferSize;
}
VkDeviceSize BufferMapped::getOffset() const
{
	return m_bufferOffset;
}
void* BufferMapped::getData() const
{
	return m_dataPtr;
}
VkDeviceAddress BufferMapped::getDeviceAddress() const
{
	return m_deviceAddress;
}