#include "command_management.h"

VkCommandPool FrameCommandPoolSet::createCommandPool(VkDevice deviceHandle, VkCommandPoolCreateFlags flags, uint32_t queueFamilyIndex)
{
	VkCommandPoolCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	if (flags)
	{
		createInfo.flags = flags;
	}
	createInfo.queueFamilyIndex = queueFamilyIndex;
	VkCommandPool commandPool{};
	ASSERT_ALWAYS(vkCreateCommandPool(deviceHandle, &createInfo, nullptr, &commandPool) == VK_SUCCESS, "Vulkan", "Command pool creation failed.")
	return commandPool;
}

FrameCommandPoolSet::FrameCommandPoolSet(const VulkanObjectHandler& vulkanObjects)
{
	m_deviceHandle = vulkanObjects.getLogicalDevice();

	uint32_t graphicsFamilyIndex{ vulkanObjects.getGraphicsFamilyIndex() };
	uint32_t computeFamilyIndex{ vulkanObjects.getComputeFamilyIndex() };
	uint32_t transferFamilyIndex{ vulkanObjects.getTransferFamilyIndex()};

	m_mainPool = createCommandPool(m_deviceHandle, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, graphicsFamilyIndex);
	
	for (auto& pool: m_threadCommandPools)
	{
		pool = createCommandPool(m_deviceHandle, NULL, graphicsFamilyIndex);
	}

	m_transientPool = createCommandPool(m_deviceHandle, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, graphicsFamilyIndex);

	m_asyncComputePool = createCommandPool(m_deviceHandle, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, computeFamilyIndex);

	m_asyncTransferPool = createCommandPool(m_deviceHandle, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, transferFamilyIndex);
}

FrameCommandPoolSet::~FrameCommandPoolSet()
{
	vkDestroyCommandPool(m_deviceHandle, m_asyncTransferPool, nullptr);
	vkDestroyCommandPool(m_deviceHandle, m_asyncComputePool, nullptr);
	vkDestroyCommandPool(m_deviceHandle, m_transientPool, nullptr);
	for (auto& pool : m_threadCommandPools)
	{
		vkDestroyCommandPool(m_deviceHandle, pool, nullptr);
	}
	vkDestroyCommandPool(m_deviceHandle, m_mainPool, nullptr);
}

const VkCommandPool FrameCommandPoolSet::getMainPool() const
{
	return m_mainPool;
}

const VkCommandPool FrameCommandPoolSet::getComputePool() const
{
	return m_asyncComputePool;
}

const VkCommandPool FrameCommandPoolSet::getTransferPool() const
{
	return m_asyncTransferPool;
}

const VkCommandPool FrameCommandPoolSet::getTransientPool() const
{
	return m_transientPool;
}

const std::vector<VkCommandPool>& FrameCommandPoolSet::getPerThreadPools() const
{
	return m_threadCommandPools;
}



void FrameCommandBufferSet::allocateBuffers(VkCommandBuffer* buffers, VkCommandPool pool, VkCommandBufferLevel level, uint32_t count)
{
	VkCommandBufferAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocateInfo.commandPool = pool;
	allocateInfo.level = level;
	allocateInfo.commandBufferCount = count;

	ASSERT_DEBUG(vkAllocateCommandBuffers(m_deviceHandle, &allocateInfo, buffers) == VK_SUCCESS, "Vulkan", "Command buffer allocation failed");
}

FrameCommandBufferSet::FrameCommandBufferSet(FrameCommandPoolSet& poolSet)
{
	m_deviceHandle = poolSet.m_deviceHandle;
	m_associatedPoolSet = &poolSet;

	allocateBuffers(&m_mainCB, poolSet.getMainPool(), VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	allocateBuffers(&m_asyncComputeCB, poolSet.getComputePool(), VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	allocateBuffers(&m_asyncTransferCB, poolSet.getTransferPool(), VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	for (int i{0} ; i < m_perThreadCBs.size(); ++i)
	{
		allocateBuffers(&m_perThreadCBs[i], poolSet.getPerThreadPools()[i], VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	}
	allocateBuffers(m_transientCBs.data(), poolSet.getTransientPool(), VK_COMMAND_BUFFER_LEVEL_PRIMARY, m_transientCBs.size());
	for (int32_t i{0}; i < m_transientCBs.size(); ++i) //Using stack to get free buffers' indices 
	{
		m_transientFreeIndices.push(i);
	}
}

FrameCommandBufferSet::~FrameCommandBufferSet()
{
	vkFreeCommandBuffers(m_deviceHandle, m_associatedPoolSet->getTransientPool(), m_transientCBs.size(), m_transientCBs.data());
	for (int i{ 0 }; i < m_perThreadCBs.size(); ++i)
	{
		vkFreeCommandBuffers(m_deviceHandle, m_associatedPoolSet->getPerThreadPools()[i], 1, &m_perThreadCBs[i]);
	}
	vkFreeCommandBuffers(m_deviceHandle, m_associatedPoolSet->getTransferPool(), 1, &m_asyncTransferCB);
	vkFreeCommandBuffers(m_deviceHandle, m_associatedPoolSet->getComputePool(), 1, &m_asyncComputeCB);
	vkFreeCommandBuffers(m_deviceHandle, m_associatedPoolSet->getMainPool(), 1, &m_mainCB);
}
VkCommandBuffer FrameCommandBufferSet::beginRecording(CommandBufferType type)
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

	ASSERT_DEBUG(vkBeginCommandBuffer(cBuffer, &beginInfo) == VK_SUCCESS, "Vulkan", "Couldn't begin a command buffer");
	return cBuffer;
}

void FrameCommandBufferSet::endRecording(VkCommandBuffer recordedBuffer)
{
	ASSERT_DEBUG(vkEndCommandBuffer(recordedBuffer) == VK_SUCCESS, "Vulkan", "Couldn't end a command buffer");
}

void FrameCommandBufferSet::resetCommandBuffer(VkCommandBuffer commandBuffer, bool releaseResources)
{
	VkCommandBufferResetFlags flags{};
	if (releaseResources)
		flags |= VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT;
	ASSERT_DEBUG(vkResetCommandBuffer(commandBuffer, flags) == VK_SUCCESS, "Vulkan", "Couldn't reset a command buffer");
}

//Function doesn't support multithreading
VkCommandBuffer FrameCommandBufferSet::beginTransientRecording()
{
	//int n = m_transientCBs.size();
	//for (int i{ 0 }; i <= n; ++i)
	//{
	//	if (i == n)
	//	{
	//		VkCommandBuffer newCB{};
	//		allocateBuffers(&newCB, m_associatedPoolSet->getTransientPool(), VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	//		m_transientCBs.push_back(newCB);
	//		index = i;
	//		break;
	//	}
	//	if (m_transientFreeBuffers[i])
	//	{
	//		index = i;
	//		m_transientFreeBuffers[index] = false;
	//		break;
	//	}
	//}
	uint32_t index{};
	if (!m_transientFreeIndices.empty())
	{
		index = m_transientFreeIndices.top();
		m_buffersToResetIndices.push(index);
		m_transientFreeIndices.pop();
	}
	else
	{
		m_buffersToResetIndices.push(m_transientFreeIndices.size());
		index = m_transientFreeIndices.size();
		VkCommandBuffer newCB{};
		allocateBuffers(&newCB, m_associatedPoolSet->getTransientPool(), VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		m_transientCBs.push_back(newCB);
	}

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	ASSERT_DEBUG(vkBeginCommandBuffer(m_transientCBs[index], &beginInfo) == VK_SUCCESS, "Vulkan", "Couldn't begin a command buffer");
	return m_transientCBs[index];
}

VkCommandBuffer FrameCommandBufferSet::beginPerThreadRecording(int32_t index)
{
	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

	ASSERT_DEBUG(vkBeginCommandBuffer(m_perThreadCBs.at(index), &beginInfo) == VK_SUCCESS, "Vulkan", "Couldn't begin a command buffer");
	return m_perThreadCBs.at(index);
}

void FrameCommandBufferSet::resetBuffers()
{
	while (!m_buffersToResetIndices.empty())
	{
		m_transientFreeIndices.push(m_buffersToResetIndices.top());
		m_buffersToResetIndices.pop();
	}
	vkResetCommandPool(m_deviceHandle, m_associatedPoolSet->getTransientPool(), 0);

	for(auto& pool : m_associatedPoolSet->getPerThreadPools())
	{
		//May be parallelised
		vkResetCommandPool(m_deviceHandle, pool, 0);
	}

	vkResetCommandPool(m_deviceHandle, m_associatedPoolSet->getTransferPool(), 0);
	vkResetCommandPool(m_deviceHandle, m_associatedPoolSet->getComputePool(), 0);
	vkResetCommandPool(m_deviceHandle, m_associatedPoolSet->getMainPool(), 0);
}