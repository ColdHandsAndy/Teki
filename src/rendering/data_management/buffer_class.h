#ifndef BUFFER_CLASS_HEADER
#define BUFFER_CLASS_HEADER

#include <memory>
#include <list>
#include <cassert>
#include <span>

#include "vulkan/vulkan.h"
#include "vma/vk_mem_alloc.h"

#include "memory_manager.h"

namespace BufferTools
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
	VkDeviceSize getOffset() const;
	VkDeviceSize getSize() const;
	VkDeviceAddress getDeviceAddress() const;
	VkDeviceSize getAlignment() const;

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
	BufferBase(VkDevice device, const VkBufferCreateInfo& bufferCI, bool sharedMem, bool mappable, bool cached, int allocFlags = NULL_FLAG);
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
	BufferBaseHostInaccessible(VkDevice device, const VkBufferCreateInfo& bufferCreateInfo, int allocFlags = NULL_FLAG);
	BufferBaseHostInaccessible(VkDevice device, VkDeviceSize bufferSize, VkBufferUsageFlags usageFlags, int allocFlags = NULL_FLAG);
	BufferBaseHostInaccessible(VkDevice device, VkDeviceSize bufferSize, VkBufferUsageFlags usageFlags, std::span<const uint32_t> queueFamilyIndices, int allocFlags = NULL_FLAG);
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

	BufferBaseHostInaccessible* m_motherBuffer{ nullptr };
	VmaVirtualAllocation m_allocation{};

	bool m_invalid{ true };

public:
	Buffer();
	Buffer(BufferBaseHostInaccessible& motherBuffer);
	Buffer(BufferBaseHostInaccessible& motherBuffer, VkDeviceSize size);
	Buffer(Buffer&& srcBuffer) noexcept;
	~Buffer();

	VkBuffer getBufferHandle() const;
	VkDeviceSize getSize() const;
	VkDeviceSize getOffset() const;
	VkDeviceAddress getDeviceAddress() const;
	VkDeviceSize getAlignment() const;

	void initialize(BufferBaseHostInaccessible& motherBuffer, VkDeviceSize size);
	void initialize(VkDeviceSize size);

	void reset();

	void operator=(Buffer&) = delete;
};


class BufferBaseHostAccessible : public BufferBase
{
private:
	void* m_mappedMemoryPointer{ nullptr };

public:
	BufferBaseHostAccessible(VkDevice device, const VkBufferCreateInfo& bufferCreateInfo, int allocFlags = NULL_FLAG, bool useSharedMemory = false, bool memoryIsCached = false);
	BufferBaseHostAccessible(VkDevice device, VkDeviceSize bufferSize, VkBufferUsageFlags usageFlags, int allocFlags = NULL_FLAG, bool useSharedMemory = false, bool memoryIsCached = false);
	BufferBaseHostAccessible(VkDevice device, VkDeviceSize bufferSize, VkBufferUsageFlags usageFlags, std::span<const uint32_t> queueFamilyIndices, int allocFlags = NULL_FLAG, bool useSharedMemory = false, bool memoryIsCached = false);
	BufferBaseHostAccessible(BufferBaseHostAccessible&& srcBuffer) noexcept;
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

	BufferBaseHostAccessible* m_motherBuffer{ nullptr };
	VmaVirtualAllocation m_allocation{};

	bool m_invalid{ true };

public:
	BufferMapped();
	BufferMapped(BufferBaseHostAccessible& motherBuffer);
	BufferMapped(BufferBaseHostAccessible& motherBuffer, VkDeviceSize size);
	BufferMapped(BufferMapped&& srcBuffer) noexcept;
	~BufferMapped();

	VkBuffer getBufferHandle() const;
	VkDeviceSize getSize() const;
	VkDeviceSize getOffset() const;
	void* getData() const;
	VkDeviceAddress getDeviceAddress() const;
	VkDeviceSize getAlignment() const;

	void initialize(BufferBaseHostAccessible& motherBuffer, VkDeviceSize size);
	void initialize(VkDeviceSize size);

	void reset();

	void operator=(BufferMapped&) = delete;
};

#endif