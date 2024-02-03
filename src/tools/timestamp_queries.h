#ifndef TIMESTAMP_QUERIES_HEADER
#define TIMESTAMP_QUERIES_HEADER

#include <cstdint>

#include <vulkan/vulkan.h>
#include <LegitProfiler/ImGuiProfilerRenderer.h>

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/data_management/buffer_class.h"

template<uint32_t QueryNum>
class TimestampQueries
{
public:
	struct Query
	{
		uint64_t startTime{};
		uint64_t availabilityStart{};
		uint64_t endTime{};
		uint64_t availabilityEnd{};
	};

private:
	VkDevice m_device{};

	BufferMapped m_queries{};

	VkQueryPool m_pool{};

	double m_timeScaleMS{};

public:
	TimestampQueries(VulkanObjectHandler& vulkanObjectHandler, BufferBaseHostAccessible& baseBuffer) : 
		m_device{ vulkanObjectHandler.getLogicalDevice() },
		m_timeScaleMS{ vulkanObjectHandler.getPhysDevLimits().timestampPeriod / 1000000000.0 },
		m_queries{ baseBuffer, sizeof(Query) * QueryNum }
	{
		VkQueryPoolCreateInfo queryPoolCI{};
		queryPoolCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
		queryPoolCI.queryCount = QueryNum * 2;
		vkCreateQueryPool(vulkanObjectHandler.getLogicalDevice(), &queryPoolCI, nullptr, &m_pool);
	}
	~TimestampQueries()
	{
		vkDestroyQueryPool(m_device, m_pool, nullptr);
	}

	void cmdReset(VkCommandBuffer cb, uint32_t offset, uint32_t count)
	{
		vkCmdResetQueryPool(cb, m_pool, offset * 2, count * 2);
	}

	void cmdWriteStart(VkCommandBuffer cb, uint32_t queryIndex)
	{
		vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_pool, queryIndex * 2);
	}
	void cmdWriteEnd(VkCommandBuffer cb, uint32_t queryIndex)
	{
		vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_pool, queryIndex * 2 + 1);
	}

	void cmdUpdateResults(VkCommandBuffer cb, uint32_t firstQuery, uint32_t queryCount)
	{
		vkCmdCopyQueryPoolResults(cb, 
			m_pool, 
			firstQuery * 2, queryCount * 2,
			m_queries.getBufferHandle(), m_queries.getOffset() + sizeof(Query) * firstQuery, 
			sizeof(uint64_t) * 2, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
	}

	void updateResults()
	{
		vkGetQueryPoolResults(m_device, m_pool, 0, QueryNum * 2, QueryNum * sizeof(Query), &m_queries.getData(), sizeof(uint64_t) * 2, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
	}

	void uploadQueryDataToProfilerTasks(legit::ProfilerTask* tasks, uint32_t count, uint32_t queryOffset = 0)
	{
		Query* queries{ reinterpret_cast<Query*>(m_queries.getData()) };
		double timeS = 0.0;
		for (int i{ static_cast<int>(queryOffset) }; i < QueryNum || i < count; ++i)
		{
			if (queries[i].availabilityStart != 0 && queries[i].availabilityEnd != 0)
			{
				(tasks + i)->startTime = timeS;
				timeS += (queries[i].endTime - queries[i].startTime) * m_timeScaleMS;
				(tasks + i)->endTime = timeS;
			}
			else
			{
				double temp = timeS;
				timeS += ((tasks + i)->endTime - (tasks + i)->startTime);
				(tasks + i)->startTime = temp;
				(tasks + i)->endTime = timeS;
			}
		}
	}
};

#endif