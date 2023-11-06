#include "src/rendering/renderer/sync_operations.h"

namespace SyncOperations
{
    void cmdExecuteBarrier(VkCommandBuffer cb, const VkDependencyInfo& dependInfo)
    {
        vkCmdPipelineBarrier2(cb, &dependInfo);
    }
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkMemoryBarrier2> memBarriers, std::span<const VkBufferMemoryBarrier2> bufBarriers, std::span<const VkImageMemoryBarrier2> imageBarriers)
    {
        VkDependencyInfo dependInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        if (!memBarriers.empty())
        {
            dependInfo.memoryBarrierCount = memBarriers.size();
            dependInfo.pMemoryBarriers = memBarriers.data();
        }
        if (!bufBarriers.empty())
        {
            dependInfo.bufferMemoryBarrierCount = bufBarriers.size();
            dependInfo.pBufferMemoryBarriers = bufBarriers.data();
        }
        if (!imageBarriers.empty())
        {
            dependInfo.imageMemoryBarrierCount = imageBarriers.size();
            dependInfo.pImageMemoryBarriers = imageBarriers.data();
        }

        vkCmdPipelineBarrier2(cb, &dependInfo);
    }
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkMemoryBarrier2> memBarriers)
    {
        if (memBarriers.empty())
        {
            return;
        }
        VkDependencyInfo dependInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependInfo.memoryBarrierCount = memBarriers.size();
        dependInfo.pMemoryBarriers = memBarriers.data();
        vkCmdPipelineBarrier2(cb, &dependInfo);
    }
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkBufferMemoryBarrier2> bufBarriers)
    {
        if (bufBarriers.empty())
        {
            return;
        }
        VkDependencyInfo dependInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependInfo.bufferMemoryBarrierCount = bufBarriers.size();
        dependInfo.pBufferMemoryBarriers = bufBarriers.data();
        vkCmdPipelineBarrier2(cb, &dependInfo);
    }
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkImageMemoryBarrier2> imageBarriers)
    {
        if (imageBarriers.empty())
        {
            return;
        }
        VkDependencyInfo dependInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dependInfo.imageMemoryBarrierCount = imageBarriers.size();
        dependInfo.pImageMemoryBarriers = imageBarriers.data();
        vkCmdPipelineBarrier2(cb, &dependInfo);
    }

    VkDependencyInfo createDependencyInfo(std::span<const VkMemoryBarrier2> memBarriers, std::span<const VkBufferMemoryBarrier2> bufBarriers, std::span<const VkImageMemoryBarrier2> imageBarriers)
    {
        VkDependencyInfo dependInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        if (!memBarriers.empty())
        {
            dependInfo.memoryBarrierCount = memBarriers.size();
            dependInfo.pMemoryBarriers = memBarriers.data();
        }
        if (!bufBarriers.empty())
        {
            dependInfo.bufferMemoryBarrierCount = bufBarriers.size();
            dependInfo.pBufferMemoryBarriers = bufBarriers.data();
        }
        if (!imageBarriers.empty())
        {
            dependInfo.imageMemoryBarrierCount = imageBarriers.size();
            dependInfo.pImageMemoryBarriers = imageBarriers.data();
        }
        return dependInfo;
    }
    VkDependencyInfo createDependencyInfo(std::span<const VkMemoryBarrier2> memBarriers)
    {
        VkDependencyInfo dependInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        if (!memBarriers.empty())
        {
            dependInfo.memoryBarrierCount = memBarriers.size();
            dependInfo.pMemoryBarriers = memBarriers.data();
        }
        return dependInfo;
    }
    VkDependencyInfo createDependencyInfo(std::span<const VkBufferMemoryBarrier2> bufBarriers)
    {
        VkDependencyInfo dependInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        if (!bufBarriers.empty())
        {
            dependInfo.bufferMemoryBarrierCount = bufBarriers.size();
            dependInfo.pBufferMemoryBarriers = bufBarriers.data();
        }
        return dependInfo;
    }
    VkDependencyInfo createDependencyInfo(std::span<const VkImageMemoryBarrier2> imageBarriers)
    {
        VkDependencyInfo dependInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        if (!imageBarriers.empty())
        {
            dependInfo.imageMemoryBarrierCount = imageBarriers.size();
            dependInfo.pImageMemoryBarriers = imageBarriers.data();
        }
        return dependInfo;
    }
}