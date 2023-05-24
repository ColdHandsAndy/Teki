#ifndef CUBEMAP_LOADER_HEADER
#define CUBEMAP_LOADER_HEADER

#include <filesystem>

#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>
#include <tbb/parallel_for.h>

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/renderer/barrier_operations.h"
#include "src/tools/asserter.h"

namespace fs = std::filesystem;

inline ImageCubeMap loadCubemap(std::shared_ptr<VulkanObjectHandler> vulkanObjects,
								FrameCommandBufferSet& commandBufferSet,
								BufferBaseHostAccessible& stagingBase,
								fs::path folderpath)
{
	constexpr uint32_t cubemapFaceCount{ 6 };

	std::array<fs::path, cubemapFaceCount> paths{
		fs::path{ folderpath / "px.hdr" },
		fs::path{ folderpath / "nx.hdr" },
		fs::path{ folderpath / "py.hdr" },
		fs::path{ folderpath / "ny.hdr" },
		fs::path{ folderpath / "pz.hdr" },
		fs::path{ folderpath / "nz.hdr" }
	};

	std::array<OIIO::ImageBuf, cubemapFaceCount> imBuffers{};
	uint32_t sideLength{ 0 };
	for (int i{ 0 }; i < cubemapFaceCount; ++i)
	{
		OIIO::ImageBuf& curBuffer{ imBuffers[i] };
		curBuffer = OIIO::ImageBuf{ paths[i].generic_string() };
		curBuffer = OIIO::ImageBufAlgo::channels(curBuffer, 4, {0, 1, 2, -1}, 0.0);
		ASSERT_ALWAYS(sideLength ? ((sideLength == curBuffer.roi().width()) && (curBuffer.roi().width() == curBuffer.roi().height())) : true, "App", "Cubemap images are not equal in size.");
		sideLength = curBuffer.roi().width();
	}
	uint64_t imageByteSize{ static_cast<uint64_t>(sideLength) * sideLength * 4 * 2 };

	BufferMapped staging{ stagingBase, cubemapFaceCount * imageByteSize };
	VkDeviceSize stagingOffsets[cubemapFaceCount]{};
	uint32_t cubemapLayers[cubemapFaceCount]{};

	oneapi::tbb::parallel_for(0u, 6u,
		[&imBuffers, &staging, &cubemapLayers, &stagingOffsets, imageByteSize](int i)
		{
			cubemapLayers[i] = i;
			stagingOffsets[i] = i * imageByteSize;
			OIIO::ImageSpec spec{ imBuffers[i].spec() };
			imBuffers[i].get_pixels(imBuffers[i].roi(), OIIO::TypeDesc::HALF, reinterpret_cast<uint8_t*>(staging.getData()) + stagingOffsets[i], OIIO::AutoStride, OIIO::AutoStride, OIIO::AutoStride);
		}
	);

	for (int i{ 0 }; i < cubemapFaceCount; ++i)
	{
		stagingOffsets[i] += staging.getOffset();
	}


	ImageCubeMap cubemap{ vulkanObjects->getLogicalDevice(), VK_FORMAT_R16G16B16A16_SFLOAT, sideLength, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};

	VkCommandBuffer cb{ commandBufferSet.beginTransientRecording() };
	BarrierOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
												.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
												.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
												.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
												.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
												.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
												.image = cubemap.getImageHandle(),
												.subresourceRange =
													{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
													.baseMipLevel = 0,
													.levelCount = 1,
													.baseArrayLayer = 0,
													.layerCount = cubemapFaceCount }}}
													});
	cubemap.cmdCopyDataFromBuffer(cb, staging.getBufferHandle(), cubemapFaceCount, sideLength, stagingOffsets, cubemapLayers);
	BarrierOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
												.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
												.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
												.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
												.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
												.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
												.image = cubemap.getImageHandle(),
												.subresourceRange =
													{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
													.baseMipLevel = 0,
													.levelCount = 1,
													.baseArrayLayer = 0,
													.layerCount = cubemapFaceCount }}}
													});
	commandBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(vulkanObjects->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(vulkanObjects->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));

	return cubemap;
}

#endif