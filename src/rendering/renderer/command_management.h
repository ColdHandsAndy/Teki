#ifndef COMMAND_MANAGEMENT_HEADER
#define COMMAND_MANAGEMENT_HEADER

#include <vector>
#include <thread>
#include <cassert>
#include <stack>
#include <utility>

#include "vulkan/vulkan.h"

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/tools/asserter.h"

#define TRANSIENT_COMMAND_BUFFER_DEFAULT_NUM uint32_t(3)

class FrameCommandBufferSet;

class FrameCommandPoolSet
{
private:
	VkDevice m_deviceHandle{};

	VkCommandPool m_mainPool{};
	std::vector<VkCommandPool> m_threadCommandPools{ std::thread::hardware_concurrency() - 1, VkCommandPool{} };
	VkCommandPool m_asyncComputePool{};
	VkCommandPool m_asyncTransferPool{};
	VkCommandPool m_transientPool{};

public:
	FrameCommandPoolSet(const VulkanObjectHandler& vulkanObjects);
	~FrameCommandPoolSet();

	const VkCommandPool getMainPool() const;
	const VkCommandPool getComputePool() const;
	const VkCommandPool getTransferPool() const;
	const VkCommandPool getTransientPool() const;
	const std::vector<VkCommandPool>& getPerThreadPools() const;

	FrameCommandPoolSet() = delete;
	void operator=(FrameCommandPoolSet) = delete;

private:
	VkCommandPool createCommandPool(VkDevice deviceHandle, VkCommandPoolCreateFlags flags, uint32_t queueFamilyIndex);

	friend FrameCommandBufferSet;
};

class FrameCommandBufferSet
{
private:
	VkDevice m_deviceHandle{};

	FrameCommandPoolSet* m_associatedPoolSet{ nullptr };

	VkCommandBuffer m_mainCB;
	VkCommandBuffer m_asyncComputeCB;
	VkCommandBuffer m_asyncTransferCB;
	std::vector<VkCommandBuffer> m_transientCBs{ TRANSIENT_COMMAND_BUFFER_DEFAULT_NUM, VkCommandBuffer{} };
	std::stack<uint32_t> m_transientFreeIndices{};
	std::stack<uint32_t> m_buffersToResetIndices{};
	//std::vector<bool> m_transientFreeBuffers{ TRANSIENT_COMMAND_BUFFER_DEFAULT_NUM, true };
	std::vector<VkCommandBuffer> m_perThreadCBs{ std::thread::hardware_concurrency() - 1, VkCommandBuffer{} };

public:
	FrameCommandBufferSet(FrameCommandPoolSet& poolSet);
	~FrameCommandBufferSet();

	enum CommandBufferType
	{
		MAIN_CB,
		ASYNC_COMPUTE_CB,
		ASYNC_TRANSFER_CB
	};

	void allocateBuffers(VkCommandBuffer* buffers, VkCommandPool pool, VkCommandBufferLevel level, uint32_t count);

	[[nodiscard]] VkCommandBuffer beginRecording(CommandBufferType type);
	[[nodiscard]] VkCommandBuffer beginTransientRecording();
	[[nodiscard]] VkCommandBuffer beginPerThreadRecording(int32_t index);

	void endRecording(VkCommandBuffer recordedBuffer);
	void resetCommandBuffer(VkCommandBuffer commandBuffer, bool releaseResources = false);

	void resetBuffers();
};

#endif