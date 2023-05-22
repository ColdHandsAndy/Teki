#include "src/rendering/renderer/barrier_operations.h"

namespace BarrierOperations
{

    VkMemoryBarrier2 constructMemoryBarrier(VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask)
    {
        return VkMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = srcStageMask,
            .srcAccessMask = srcAccessMask,
            .dstStageMask = dstStageMask,
            .dstAccessMask = dstAccessMask };
    }
    VkBufferMemoryBarrier2 constructBufferBarrier(VkPipelineStageFlags2 srcStageMask,
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
    VkBufferMemoryBarrier2 constructBufferBarrier(VkPipelineStageFlags2 srcStageMask,
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
    VkImageMemoryBarrier2 constructImageBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        uint32_t srcQueueFamilyIndex,
        uint32_t dstQueueFamilyIndex,
        VkImage image,
        VkImageSubresourceRange subresourceRange)
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
    VkImageMemoryBarrier2 constructImageBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkImage image,
        VkImageSubresourceRange subresourceRange)
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
}