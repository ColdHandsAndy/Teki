#ifndef BUFFER_CLASS_HEADER
#define BUFFER_CLASS_HEADER

#include <memory>
#include <list>
#include <cassert>

#include "vulkan/vulkan.h"
#include "vma/vk_mem_alloc.h"

#include "memory_manager.h"

namespace BufferOperations
{
	void cmdBufferCopy(VkCommandBuffer cmdBuffer, VkBuffer srcBufferHandle, VkBuffer dstBufferHandle, uint32_t regionCount, const VkBufferCopy* regions);
}

class BufferBase
{
protected:
	VkBuffer m_bufferHandle{};
	VkDeviceAddress m_bufferMemoryAddress{};
	VmaVirtualBlock m_memoryProxy{};
	VkDeviceSize m_memoryByteSize{};
	VkDeviceSize m_bufferAlignment{};

	std::list<VmaAllocation>::const_iterator m_bufferAllocIter{};
	inline static MemoryManager* m_memoryManager{ nullptr };

	bool m_invalid{ false };

public:
	VkBuffer getBufferHandle() const;
	VkDeviceSize getBufferByteSize() const;
	VkDeviceAddress getBufferDeviceAddress() const;

	static void assignGlobalMemoryManager(MemoryManager& memManager);

	enum MemoryFlags
	{
		NULL_FLAG = 0,
		DEDICATED_FLAG = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
		ALIASING_FLAG = VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT,
		FAST_ALLOCATION_FLAG = VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT,
		EFFICENT_ALLOCATION_FLAG = VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT
	};

protected:
	BufferBase() = default;
	BufferBase(VkDevice device, const VkBufferCreateInfo& bufferCI, bool sharedMem, bool mappable, bool cached, int flags = NULL_FLAG);
	BufferBase(BufferBase&& bufBase) noexcept;
	~BufferBase();

	void allocateFromBuffer(VkDeviceSize allocSize, VmaVirtualAllocation& allocation, VkDeviceSize& inBufferOffset);
	void freeBufferAllocation(VmaVirtualAllocation allocation);

	friend class Buffer;
	friend class BufferMapped;
};



class BufferBaseHostInaccessible : public BufferBase
{
public:
	BufferBaseHostInaccessible(VkDevice device, const VkBufferCreateInfo& bufferCreateInfo, int flags = NULL_FLAG);
	BufferBaseHostInaccessible(BufferBaseHostInaccessible&& srcBuffer) = default;
	~BufferBaseHostInaccessible();

protected:

	friend class BufferMapped;
};

class Buffer
{
private:
	VkBuffer m_bufferHandle{};
	VkDeviceAddress m_deviceAddress{};
	VkDeviceSize m_bufferOffset{};
	VkDeviceSize m_bufferSize{};

	BufferBaseHostInaccessible& m_motherBuffer;
	VmaVirtualAllocation m_allocation{};

public:
	Buffer(BufferBaseHostInaccessible& motherBuffer, VkDeviceSize size);
	~Buffer();

	VkBuffer getBufferHandle();
	VkDeviceAddress getDeviceAddress();
};


class BufferBaseHostAccessible : public BufferBase
{
private:
	void* m_mappedMemoryPointer{ nullptr };

public:
	BufferBaseHostAccessible(VkDevice device, VkBufferCreateInfo bufferCreateInfo, int flags = NULL_FLAG, bool useSharedMemory = false, bool memoryIsCached = false);
	BufferBaseHostAccessible(BufferBaseHostAccessible&& srcBuffer);
	~BufferBaseHostAccessible();

	void mapMemory();
	void unmapMemory();

	void* getData();

	friend class BufferMapped;
};

class BufferMapped
{
private:
	VkBuffer m_bufferHandle{};
	VkDeviceAddress m_deviceAddress{};
	VkDeviceSize m_bufferOffset{};
	VkDeviceSize m_bufferSize{};
	void* m_dataPtr{ nullptr };

	BufferBaseHostAccessible& m_motherBuffer;
	VmaVirtualAllocation m_allocation{};

public:
	BufferMapped(BufferBaseHostAccessible& motherBuffer, VkDeviceSize size);
	~BufferMapped();

	VkBuffer getBufferHandle();
	void* getData();
	VkDeviceAddress getDeviceAddress();
};

#endif