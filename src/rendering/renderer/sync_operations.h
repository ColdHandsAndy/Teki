#ifndef BARRIER_OPERATIONS_HEADER
#define BARRIER_OPERATIONS_HEADER

#include <cstdint>
#include <string>
#include <array>
#include <span>

#include <vulkan/vulkan.h>

#include "src/tools/asserter.h"
#include "src/tools/arraysize.h"

namespace SyncOperations
{
    constexpr VkMemoryBarrier2 constructMemoryBarrier(VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask)
    {
        return VkMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask };
    }
    constexpr VkBufferMemoryBarrier2 constructBufferBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        uint32_t srcQueueFamilyIndex,
        uint32_t dstQueueFamilyIndex,
        VkBuffer buffer,
        VkDeviceSize offset,
        VkDeviceSize size)
    {
        return VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .srcQueueFamilyIndex = srcQueueFamilyIndex,
            .dstQueueFamilyIndex = dstQueueFamilyIndex,
            .buffer = buffer,
            .offset = offset,
            .size = size };
    }
    constexpr VkBufferMemoryBarrier2 constructBufferBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        VkBuffer buffer,
        VkDeviceSize offset,
        VkDeviceSize size)
    {
        return VkBufferMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask,
            .buffer = buffer,
            .offset = offset,
            .size = size };
    }
    constexpr VkImageMemoryBarrier2 constructImageBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        uint32_t srcQueueFamilyIndex,
        uint32_t dstQueueFamilyIndex,
        VkImage image,
        const VkImageSubresourceRange& subresourceRange)
    {
        return VkImageMemoryBarrier2{
           .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
           .srcStageMask = srcStageMask,
           .srcAccessMask = srcAccessMask,
           .dstStageMask = dstStageMask,
           .dstAccessMask = dstAccessMask,
           .oldLayout = oldLayout,
           .newLayout = newLayout,
           .srcQueueFamilyIndex = srcQueueFamilyIndex,
           .dstQueueFamilyIndex = dstQueueFamilyIndex,
           .image = image,
           .subresourceRange = subresourceRange };
    }
    constexpr VkImageMemoryBarrier2 constructImageBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkImage image,
        const VkImageSubresourceRange& subresourceRange)
    {
        return VkImageMemoryBarrier2{
           .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
           .srcStageMask = srcStageMask,
           .srcAccessMask = srcAccessMask,
           .dstStageMask = dstStageMask,
           .dstAccessMask = dstAccessMask,
           .oldLayout = oldLayout,
           .newLayout = newLayout,
           .image = image,
           .subresourceRange = subresourceRange };
    }


    void cmdExecuteBarrier(VkCommandBuffer cb, const VkDependencyInfo& dependInfo);
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkMemoryBarrier2> memBarriers, std::span<const VkBufferMemoryBarrier2> bufBarriers, std::span<const VkImageMemoryBarrier2> imageBarriers);
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkMemoryBarrier2> memBarriers);
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkBufferMemoryBarrier2> bufBarriers);
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkImageMemoryBarrier2> imageBarriers);

    VkDependencyInfo createDependencyInfo(std::span<const VkMemoryBarrier2> memBarriers, std::span<const VkBufferMemoryBarrier2> bufBarriers, std::span<const VkImageMemoryBarrier2> imageBarriers);
    VkDependencyInfo createDependencyInfo(std::span<const VkMemoryBarrier2> memBarriers);
    VkDependencyInfo createDependencyInfo(std::span<const VkBufferMemoryBarrier2> bufBarriers);
    VkDependencyInfo createDependencyInfo(std::span<const VkImageMemoryBarrier2> imageBarriers);

    //Creates only device events
    template<size_t MaxEvents>
    class EventHolder
    {
    private:
        VkDevice m_device{};
        VkEvent m_eventHandles[MaxEvents]{};

    public:
        EventHolder(VkDevice device) : m_device{ device }
        {
            VkEventCreateInfo eventCI{};
            eventCI.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
            eventCI.flags = VK_EVENT_CREATE_DEVICE_ONLY_BIT;

            for (auto& handle : m_eventHandles)
                vkCreateEvent(device, &eventCI, nullptr, &handle);
        }
        ~EventHolder()
        {
            for (auto& handle : m_eventHandles)
                vkDestroyEvent(m_device, handle, nullptr);
        }

        void cmdSet(VkCommandBuffer cb, uint32_t index, const VkDependencyInfo& dependInfo)
        {
            EASSERT(index >= 0 && index < ARRAYSIZE(m_eventHandles), "App", "Invalid index.");

            vkCmdSetEvent2(cb, m_eventHandles[index], &dependInfo);
        }
        void cmdWait(VkCommandBuffer cb, uint32_t count, const uint32_t* indices, const VkDependencyInfo* dependInfos)
        {
            VkEvent events[MaxEvents]{};

            for (int i{ 0 }; i < count; ++i)
            {
                EASSERT(indices[i] >= 0 && indices[i] < ARRAYSIZE(m_eventHandles), "App", "Invalid index" + std::to_string(indices[i]) + '.');
                events[i] = m_eventHandles[indices[i]];
            }

            vkCmdWaitEvents2(cb, count, events, dependInfos);
        }
        void cmdReset(VkCommandBuffer cb, uint32_t index, VkPipelineStageFlags2 firstScopeStages)
        {
            EASSERT(index >= 0 && index < ARRAYSIZE(m_eventHandles), "App", "Invalid index.");

            vkCmdResetEvent2(cb, m_eventHandles[index], firstScopeStages);
        }
        void cmdResetAll(VkCommandBuffer cb, VkPipelineStageFlags2 firstScopeStages)
        {
            for (auto& handle : m_eventHandles)
                vkCmdResetEvent2(cb, handle, firstScopeStages);
        }
    };
}

#endif