#ifndef BARRIER_OPERATIONS_HEADER
#define BARRIER_OPERATIONS_HEADER

#include <cstdint>
#include <span>

#include <vulkan/vulkan.h>

namespace BarrierOperations
{
    VkMemoryBarrier2 constructMemoryBarrier(VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask, VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask);
    VkBufferMemoryBarrier2 constructBufferBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        uint32_t srcQueueFamilyIndex,
        uint32_t dstQueueFamilyIndex,
        VkBuffer buffer,
        VkDeviceSize offset,
        VkDeviceSize size);
    VkBufferMemoryBarrier2 constructBufferBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        VkBuffer buffer,
        VkDeviceSize offset,
        VkDeviceSize size);
    VkImageMemoryBarrier2 constructImageBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        uint32_t srcQueueFamilyIndex,
        uint32_t dstQueueFamilyIndex,
        VkImage image,
        VkImageSubresourceRange subresourceRange);
    VkImageMemoryBarrier2 constructImageBarrier(VkPipelineStageFlags2 srcStageMask,
        VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask,
        VkAccessFlags2 dstAccessMask,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        VkImage image,
        VkImageSubresourceRange subresourceRange);


    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkMemoryBarrier2> memBarriers, std::span<const VkBufferMemoryBarrier2> bufBarriers, std::span<const VkImageMemoryBarrier2> imageBarriers);
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkMemoryBarrier2> memBarriers);
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkBufferMemoryBarrier2> bufBarriers);
    void cmdExecuteBarrier(VkCommandBuffer cb, std::span<const VkImageMemoryBarrier2> imageBarriers);
}

#endif