#ifndef COMMAND_MANAGEMENT_HEADER
#define COMMAND_MANAGEMENT_HEADER

#include <vector>
#include <thread>
#include <cassert>
#include <stack>
#include <utility>

#include <vulkan/vulkan.h>

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/tools/asserter.h"

#define TRANSIENT_COMMAND_BUFFER_DEFAULT_NUM uint32_t(3)

class CommandBufferSet
{
private:
	VkDevice m_device{};

	VkCommandPool m_mainPool{};
	std::vector<VkCommandPool> m_threadCommandPools{ std::thread::hardware_concurrency(), VkCommandPool{} };
	VkCommandPool m_asyncComputePool{};
	VkCommandPool m_asyncTransferPool{};
	VkCommandPool m_transientPool{};

	VkCommandBuffer m_mainCB{};
	VkCommandBuffer m_asyncComputeCB{};
	VkCommandBuffer m_asyncTransferCB{};
	std::vector<VkCommandBuffer> m_transientCBs{ TRANSIENT_COMMAND_BUFFER_DEFAULT_NUM, VkCommandBuffer{} };
	std::stack<uint32_t> m_transientFreeIndices{};
	std::stack<uint32_t> m_buffersToResetIndices{};
	std::vector<VkCommandBuffer> m_perThreadCBs{ std::thread::hardware_concurrency(), VkCommandBuffer{} };

	VkCommandPool m_interchangeableMainCB{};
	VkCommandPool m_interchangeableAsyncComputeCB{};
	VkCommandPool m_interchangeableAsyncTransferCB{};
	std::vector<std::vector<VkCommandBuffer>> m_interchangeableCBs{};

public:
	CommandBufferSet(const VulkanObjectHandler& vulkanObjects);
	~CommandBufferSet();

	enum CommandBufferType
	{
		MAIN_CB,
		ASYNC_COMPUTE_CB,
		ASYNC_TRANSFER_CB
	};
	enum CommandPoolType
	{
		MAIN_POOL,
		COMPUTE_POOL,
		TRANSFER_POOL,
		TRANSIENT_POOL
	};

	[[nodiscard]] uint32_t createInterchangeableSet(uint32_t cbCount, CommandBufferType type);

	[[nodiscard]] VkCommandBuffer beginRecording(CommandBufferType type);
	[[nodiscard]] VkCommandBuffer beginTransientRecording();
	[[nodiscard]] VkCommandBuffer beginPerThreadRecording(int32_t index);
	[[nodiscard]] VkCommandBuffer beginInterchangeableRecording(uint32_t indexToSet, uint32_t commandBufferIndex);

	void endRecording(VkCommandBuffer recordedBuffer);

	void resetInterchangeable(uint32_t indexToSet, uint32_t commandBufferIndex);
	void resetAll();
	void resetPool(CommandPoolType type);

	CommandBufferSet() = delete;
	void operator=(CommandBufferSet) = delete;
private:
	VkCommandPool createCommandPool(VkDevice deviceHandle, VkCommandPoolCreateFlags flags, uint32_t queueFamilyIndex);
	void allocateBuffers(VkCommandBuffer* buffers, VkCommandPool pool, VkCommandBufferLevel level, uint32_t count);
};

#endif