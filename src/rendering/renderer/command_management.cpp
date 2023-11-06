#include "command_management.h"

CommandBufferSet::CommandBufferSet(const VulkanObjectHandler& vulkanObjects)
{
	m_device = vulkanObjects.getLogicalDevice();

	uint32_t graphicsFamilyIndex{ vulkanObjects.getGraphicsFamilyIndex() };
	uint32_t computeFamilyIndex{ vulkanObjects.getComputeFamilyIndex() };
	uint32_t transferFamilyIndex{ vulkanObjects.getTransferFamilyIndex() };

	m_mainPool = createCommandPool(m_device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, graphicsFamilyIndex);

	for (auto& pool : m_threadCommandPools)
	{
		pool = createCommandPool(m_device, NULL, graphicsFamilyIndex);
	}

	m_transientPool = createCommandPool(m_device, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, graphicsFamilyIndex);
	m_asyncComputePool = createCommandPool(m_device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, computeFamilyIndex);
	m_asyncTransferPool = createCommandPool(m_device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, transferFamilyIndex);

	m_interchangeableMainCB = createCommandPool(m_device, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, graphicsFamilyIndex);
	m_interchangeableAsyncComputeCB = createCommandPool(m_device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, computeFamilyIndex);
	m_interchangeableAsyncTransferCB = createCommandPool(m_device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, transferFamilyIndex);


	allocateBuffers(&m_mainCB, m_mainPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	allocateBuffers(&m_asyncComputeCB, m_asyncComputePool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	allocateBuffers(&m_asyncTransferCB, m_asyncTransferPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	for (int i{ 0 }; i < m_perThreadCBs.size(); ++i)
	{
		allocateBuffers(&m_perThreadCBs[i], m_threadCommandPools[i], VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	}
	allocateBuffers(m_transientCBs.data(), m_transientPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, m_transientCBs.size());
	for (int32_t i{ 0 }; i < m_transientCBs.size(); ++i)
	{
		m_transientFreeIndices.push(i);
	}
}
CommandBufferSet::~CommandBufferSet()
{
	vkFreeCommandBuffers(m_device, m_transientPool, m_transientCBs.size(), m_transientCBs.data());
	for (int i{ 0 }; i < m_perThreadCBs.size(); ++i)
	{
		vkFreeCommandBuffers(m_device, m_threadCommandPools[i], 1, &m_perThreadCBs[i]);
	}
	vkFreeCommandBuffers(m_device, m_asyncTransferPool, 1, &m_asyncTransferCB);
	vkFreeCommandBuffers(m_device, m_asyncComputePool, 1, &m_asyncComputeCB);
	vkFreeCommandBuffers(m_device, m_mainPool, 1, &m_mainCB);

	vkDestroyCommandPool(m_device, m_interchangeableAsyncTransferCB, nullptr);
	vkDestroyCommandPool(m_device, m_interchangeableAsyncComputeCB, nullptr);
	vkDestroyCommandPool(m_device, m_interchangeableMainCB, nullptr);
	vkDestroyCommandPool(m_device, m_asyncTransferPool, nullptr);
	vkDestroyCommandPool(m_device, m_asyncComputePool, nullptr);
	vkDestroyCommandPool(m_device, m_transientPool, nullptr);
	for (auto& pool : m_threadCommandPools)
	{
		vkDestroyCommandPool(m_device, pool, nullptr);
	}
	vkDestroyCommandPool(m_device, m_mainPool, nullptr);
}

[[nodiscard]] uint32_t CommandBufferSet::createInterchangeableSet(uint32_t cbCount, CommandBufferType type)
{
	uint32_t index{ static_cast<uint32_t>(m_interchangeableCBs.size()) };
	auto& cbs{ m_interchangeableCBs.emplace_back() };
	cbs.resize(cbCount);
	VkCommandPool pool{};
	switch (type)
	{
	case MAIN_CB:
		pool = m_interchangeableMainCB;
		break;
	case ASYNC_COMPUTE_CB:
		pool = m_interchangeableAsyncComputeCB;
		break;
	case ASYNC_TRANSFER_CB:
		pool = m_interchangeableAsyncTransferCB;
		break;
	default:
		EASSERT(false, "App", "Unknown command buffer type. || Should never happen.");
	}
	allocateBuffers(cbs.data(), pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, cbCount);
	return index;
}

[[nodiscard]] VkCommandBuffer CommandBufferSet::beginRecording(CommandBufferType type)
{
	VkCommandBuffer cBuffer{};
	switch (type)
	{
	case MAIN_CB:
		cBuffer = m_mainCB;
		break;
	case ASYNC_COMPUTE_CB:
		cBuffer = m_asyncComputeCB;
		break;
	case ASYNC_TRANSFER_CB:
		cBuffer = m_asyncTransferCB;
		break;
	}

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	EASSERT(vkBeginCommandBuffer(cBuffer, &beginInfo) == VK_SUCCESS, "Vulkan", "Couldn't begin a command buffer");
	return cBuffer;
}
[[nodiscard]] VkCommandBuffer CommandBufferSet::beginTransientRecording()
{
	uint32_t index{};
	if (!m_transientFreeIndices.empty())
	{
		index = m_transientFreeIndices.top();
		m_buffersToResetIndices.push(index);
		m_transientFreeIndices.pop();
	}
	else
	{
		m_buffersToResetIndices.push(m_transientCBs.size());
		index = m_transientCBs.size();
		VkCommandBuffer newCB{};
		allocateBuffers(&newCB, m_transientPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		m_transientCBs.push_back(newCB);
	}

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	EASSERT(vkBeginCommandBuffer(m_transientCBs[index], &beginInfo) == VK_SUCCESS, "Vulkan", "Couldn't begin a command buffer");
	return m_transientCBs[index];
}
[[nodiscard]] VkCommandBuffer CommandBufferSet::beginPerThreadRecording(int32_t index)
{
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	EASSERT(vkBeginCommandBuffer(m_perThreadCBs.at(index), &beginInfo) == VK_SUCCESS, "Vulkan", "Couldn't begin a command buffer");
	return m_perThreadCBs.at(index);
}
[[nodiscard]] VkCommandBuffer CommandBufferSet::beginInterchangeableRecording(uint32_t indexToSet, uint32_t commandBufferIndex)
{
	VkCommandBuffer cb{ m_interchangeableCBs[indexToSet][commandBufferIndex] };
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	EASSERT(vkBeginCommandBuffer(cb, &beginInfo) == VK_SUCCESS, "Vulkan", "Couldn't begin a command buffer");
	return cb;
}

void CommandBufferSet::endRecording(VkCommandBuffer recordedBuffer)
{
	EASSERT(vkEndCommandBuffer(recordedBuffer) == VK_SUCCESS, "Vulkan", "Couldn't end a command buffer");
}

void CommandBufferSet::resetInterchangeable(uint32_t indexToSet, uint32_t commandBufferIndex)
{
	EASSERT(vkResetCommandBuffer(m_interchangeableCBs[indexToSet][commandBufferIndex], 0) == VK_SUCCESS, "Vulkan", "Command buffer reset failed.");
}

void CommandBufferSet::resetAll()
{
	while (!m_buffersToResetIndices.empty())
	{
		m_transientFreeIndices.push(m_buffersToResetIndices.top());
		m_buffersToResetIndices.pop();
	}
	vkResetCommandPool(m_device, m_transientPool, 0);

	resetPoolsOnThreads();

	m_interchangeableCBs.clear();

	vkResetCommandPool(m_device, m_interchangeableMainCB, 0);
	vkResetCommandPool(m_device, m_interchangeableAsyncComputeCB, 0);
	vkResetCommandPool(m_device, m_interchangeableAsyncTransferCB, 0);

	vkResetCommandPool(m_device, m_asyncTransferPool, 0);
	vkResetCommandPool(m_device, m_asyncComputePool, 0);
	vkResetCommandPool(m_device, m_mainPool, 0);
}

void CommandBufferSet::resetPool(CommandPoolType type)
{
	VkCommandPool pool{};
	switch (type)
	{
	case MAIN_POOL:
		pool = m_mainPool;
		break;
	case COMPUTE_POOL:
		pool = m_asyncComputePool;
		break;
	case TRANSFER_POOL:
		pool = m_asyncTransferPool;
		break;
	case TRANSIENT_POOL:
		pool = m_transientPool;
		while (!m_buffersToResetIndices.empty())
		{
			m_transientFreeIndices.push(m_buffersToResetIndices.top());
			m_buffersToResetIndices.pop();
		}
		break;
	default:
		EASSERT(false, "App", "Unknown command buffer type. || Should never happen.");
	}
	vkResetCommandPool(m_device, pool, 0);
}
void CommandBufferSet::resetPoolsOnThreads()
{
	for (auto& pool : m_threadCommandPools)
		vkResetCommandPool(m_device, pool, 0);
}


VkCommandPool CommandBufferSet::createCommandPool(VkDevice deviceHandle, VkCommandPoolCreateFlags flags, uint32_t queueFamilyIndex)
{
	VkCommandPoolCreateInfo commandPoolCI{};
	commandPoolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	if (flags)
	{
		commandPoolCI.flags = flags;
	}
	commandPoolCI.queueFamilyIndex = queueFamilyIndex;
	VkCommandPool commandPool{};
	EASSERT(vkCreateCommandPool(deviceHandle, &commandPoolCI, nullptr, &commandPool) == VK_SUCCESS, "Vulkan", "Command pool creation failed.");
	return commandPool;
}
void CommandBufferSet::allocateBuffers(VkCommandBuffer* buffers, VkCommandPool pool, VkCommandBufferLevel level, uint32_t count)
{
	VkCommandBufferAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocateInfo.commandPool = pool;
	allocateInfo.level = level;
	allocateInfo.commandBufferCount = count;

	EASSERT(vkAllocateCommandBuffers(m_device, &allocateInfo, buffers) == VK_SUCCESS, "Vulkan", "Command buffer allocation failed");
}