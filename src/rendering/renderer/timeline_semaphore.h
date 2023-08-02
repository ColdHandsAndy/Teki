#ifndef TIMELINE_SEMAPHORE_CLASS
#define TIMELINE_SEMAPHORE_CLASS

#include <vulkan/vulkan.h>

class TimelineSemaphore
{
private:
	VkDevice m_device{};
	VkSemaphore m_semaphore{};

	uint64_t m_currentVal{};

public:
	TimelineSemaphore(VkDevice device, uint64_t initValue = 0) : m_device{ device }, m_currentVal{ initValue }
	{
		VkSemaphoreTypeCreateInfo timelineCI;
		timelineCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
		timelineCI.pNext = nullptr;
		timelineCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		timelineCI.initialValue = initValue;

		VkSemaphoreCreateInfo semaphoreCI;
		semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		semaphoreCI.pNext = &timelineCI;
		semaphoreCI.flags = 0;

		vkCreateSemaphore(device, &semaphoreCI, nullptr, &m_semaphore);
	}
	~TimelineSemaphore()
	{
		vkDestroySemaphore(m_device, m_semaphore, nullptr);
	}

	void signal(uint64_t signalValue)
	{
		VkSemaphoreSignalInfo signalInfo{};
		signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
		signalInfo.pNext = nullptr;
		signalInfo.semaphore = m_semaphore;
		signalInfo.value = signalValue;

		vkSignalSemaphore(m_device, &signalInfo);

		m_currentVal = signalValue;
	}

	void wait(uint64_t waitValue)
	{
		VkSemaphoreWaitInfo waitInfo{};
		waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
		waitInfo.pNext = nullptr;
		waitInfo.flags = 0;
		waitInfo.semaphoreCount = 1;
		waitInfo.pSemaphores = &m_semaphore;
		waitInfo.pValues = &waitValue;

		vkWaitSemaphores(m_device, &waitInfo, UINT64_MAX);
	}

	static VkTimelineSemaphoreSubmitInfo getSubmitInfo(uint32_t waitCount, uint64_t* waitValues, uint32_t signalCount, uint64_t* signalValues)
	{
		VkTimelineSemaphoreSubmitInfo timelineInfo{};
		timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
		timelineInfo.pNext = NULL;
		timelineInfo.waitSemaphoreValueCount = waitCount;
		timelineInfo.pWaitSemaphoreValues = waitValues;
		timelineInfo.signalSemaphoreValueCount = signalCount;
		timelineInfo.pSignalSemaphoreValues = signalValues;
		return timelineInfo;
	}

	void newValue(uint64_t value)
	{
		m_currentVal = value;
	}

	uint64_t getValue()
	{
		return m_currentVal;
	}

	VkSemaphore getHandle()
	{
		return m_semaphore;
	}
};

#endif