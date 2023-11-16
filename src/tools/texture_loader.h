#ifndef TEXTURE_LOADER_HEADER
#define TEXTURE_LOADER_HEADER

#include <filesystem>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <OpenImageIO/imageio.h>
#include <ktx.h>
#include <ktxvulkan.h>

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/renderer/sync_operations.h"
#include "src/tools/asserter.h"
#include "src/tools/logging.h"

namespace fs = std::filesystem;
namespace TextureLoaders
{

inline ImageListContainer::ImageListContainerIndices loadTexture(const VulkanObjectHandler& vulkanObjects,
		CommandBufferSet& commandBufferSet,
		BufferBaseHostAccessible& stagingBase,
		ImageListContainer& imageContainer,
		fs::path filepath)
	{
		ktxTexture2* textureKTX{};

		EASSERT(ktxTexture2_CreateFromNamedFile(filepath.generic_string().c_str(), KTX_TEXTURE_CREATE_NO_FLAGS, &textureKTX) == KTX_SUCCESS, "ktx", "Could not load image.");
		bool needsTranscoding = ktxTexture2_NeedsTranscoding(textureKTX);
		if (needsTranscoding)
			EASSERT(ktxTexture2_TranscodeBasis(textureKTX, KTX_TTF_BC7_RGBA, 0) == KTX_SUCCESS, "ktx", "Transcoding failed.");

		//VkFormat format{ static_cast<VkFormat>(textureKTX->vkFormat) };
		VkFormat format{ VK_FORMAT_BC7_UNORM_BLOCK };
		uint64_t totalByteSize{ textureKTX->dataSize };
		uint32_t mipLevelCount{ textureKTX->numLevels };
		uint32_t width{ textureKTX->baseWidth };
		uint32_t height{ textureKTX->baseHeight };

		std::vector<uint64_t> stagingOffsets{};

		BufferMapped staging{ stagingBase, totalByteSize };

		for (int i{ 0 }; i < mipLevelCount; ++i)
		{
			stagingOffsets.push_back(0);
			EASSERT(ktxTexture_GetImageOffset((ktxTexture*)textureKTX, i, 0, 0, &stagingOffsets.back()) == KTX_SUCCESS, "ktx", "Could not get offset.");
			stagingOffsets.back() += staging.getOffset();
		}

		if (needsTranscoding)
			std::memcpy(staging.getData(), textureKTX->pData, totalByteSize);
		else
			EASSERT(ktxTexture_LoadImageData((ktxTexture*)textureKTX, (ktx_uint8_t*)staging.getData(), totalByteSize) == KTX_SUCCESS, "ktx", "Could not load data into buffer.");

		ktxTexture_Destroy((ktxTexture*)textureKTX);

		ImageListContainer::ImageListContainerIndices imageIndices{ imageContainer.getNewImage(width, height, format) };

		VkImage imageHandle{ imageContainer.getImageHandle(imageIndices.listIndex) };
		uint32_t mipCount{ imageContainer.getImageListSubresourceRange(imageIndices.listIndex).levelCount };
		
		VkCommandBuffer cb{ commandBufferSet.beginTransientRecording() };
		SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
													.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.image = imageHandle,
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = mipCount,
														.baseArrayLayer = imageIndices.layerIndex,
														.layerCount = 1 }}}
			});

		imageContainer.cmdCopyDataFromBufferAllMips(cb, imageIndices.listIndex, staging.getBufferHandle(), imageIndices.layerIndex, stagingOffsets.size(), stagingOffsets.data());

		SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
													.image = imageHandle,
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = mipCount,
														.baseArrayLayer = imageIndices.layerIndex,
														.layerCount = 1 }}}
			});
		commandBufferSet.endRecording(cb);

		VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
		vkQueueSubmit(vulkanObjects.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(vulkanObjects.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));

		commandBufferSet.resetAllTransient();

		return imageIndices;
	}

	inline Image loadTexture(const VulkanObjectHandler& vulkanObjects,
		CommandBufferSet& commandBufferSet,
		BufferBaseHostAccessible& stagingBase,
		fs::path filepath,
		VkImageUsageFlags usageFlags,
		VkImageAspectFlags aspect)
	{
		ktxTexture2* textureKTX{};

		EASSERT(ktxTexture2_CreateFromNamedFile(filepath.generic_string().c_str(), KTX_TEXTURE_CREATE_NO_FLAGS, &textureKTX) == KTX_SUCCESS, "ktx", "Could not load image.");
		bool needsTranscoding = ktxTexture2_NeedsTranscoding(textureKTX);
		if (needsTranscoding)
			EASSERT(ktxTexture2_TranscodeBasis(textureKTX, KTX_TTF_BC7_RGBA, 0) == KTX_SUCCESS, "ktx", "Transcoding failed.");

		//VkFormat format{ static_cast<VkFormat>(textureKTX->vkFormat) };
		VkFormat format{ VK_FORMAT_BC7_UNORM_BLOCK };
		uint64_t totalByteSize{ textureKTX->dataSize };
		uint32_t mipLevelCount{ textureKTX->numLevels };
		uint32_t width{ textureKTX->baseWidth };
		uint32_t height{ textureKTX->baseHeight };

		std::vector<uint64_t> stagingOffsets{};

		BufferMapped staging{ stagingBase, totalByteSize };

		for (int i{ 0 }; i < mipLevelCount; ++i)
		{
			stagingOffsets.push_back(0);
			EASSERT(ktxTexture_GetImageOffset((ktxTexture*)textureKTX, i, 0, 0, &stagingOffsets.back()) == KTX_SUCCESS, "ktx", "Could not get offset.");
			stagingOffsets.back() += staging.getOffset();
		}

		if (needsTranscoding)
			std::memcpy(staging.getData(), textureKTX->pData, totalByteSize);
		else
			EASSERT(ktxTexture_LoadImageData((ktxTexture*)textureKTX, (ktx_uint8_t*)staging.getData(), totalByteSize) == KTX_SUCCESS, "ktx", "Could not load data into buffer.");

		ktxTexture_Destroy((ktxTexture*)textureKTX);

		Image texture{ vulkanObjects.getLogicalDevice(), format, width, height, usageFlags, aspect, mipLevelCount > 1 ? true : false };

		VkCommandBuffer cb{ commandBufferSet.beginTransientRecording() };
		SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
													.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.image = texture.getImageHandle(),
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = texture.getMipLevelCount(),
														.baseArrayLayer = 0,
														.layerCount = 0 }}}
			});

		texture.cmdCopyDataFromBuffer(cb, staging.getBufferHandle(), stagingOffsets.size(), stagingOffsets.data());

		SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
													.image = texture.getImageHandle(),
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = texture.getMipLevelCount(),
														.baseArrayLayer = 0,
														.layerCount = 0 }}}
			});
		commandBufferSet.endRecording(cb);

		VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
		vkQueueSubmit(vulkanObjects.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(vulkanObjects.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));

		commandBufferSet.resetAllTransient();

		return texture;
	}

	inline ImageCubeMap loadCubemap(const VulkanObjectHandler& vulkanObjects,
									CommandBufferSet& commandBufferSet,
									BufferBaseHostAccessible& stagingBase,
									fs::path filepath)
	{
		constexpr int cubemapFaceCount{ 6 };

		ktxTexture2* texture{};

		EASSERT(ktxTexture2_CreateFromNamedFile(filepath.generic_string().c_str(), KTX_TEXTURE_CREATE_NO_FLAGS, &texture) == KTX_SUCCESS, "ktx", "Could not load cubemap.");
		EASSERT(texture->isCubemap, "App", "Cubemap image has more than six faces");
		bool needsTranscoding = ktxTexture2_NeedsTranscoding(texture);
		if (needsTranscoding)
			EASSERT(ktxTexture2_TranscodeBasis(texture, KTX_TTF_BC7_RGBA, 0) == KTX_SUCCESS, "ktx", "Transcoding failed.");

		VkFormat format{ static_cast<VkFormat>(texture->vkFormat) };
		uint64_t totalByteSize{ texture->dataSize };
		uint32_t mipLevelCount{ texture->numLevels };
		uint32_t sideLength{ texture->baseWidth };

		std::vector<uint64_t> stagingOffsets{};
		std::vector<uint32_t> faceIndices{};
		std::vector<uint32_t> mipIndices{};

		BufferMapped staging{ stagingBase, totalByteSize };

		for (int i{ 0 }; i < mipLevelCount; ++i)
		{
			for (int j{ 0 }; j < cubemapFaceCount; ++j)
			{
				stagingOffsets.push_back(0);
				EASSERT(ktxTexture_GetImageOffset((ktxTexture*)texture, i, 0, j, &stagingOffsets.back()) == KTX_SUCCESS, "ktx", "Could not get offset.");
				stagingOffsets.back() += staging.getOffset();
				mipIndices.push_back(i);
				faceIndices.push_back(j);
			}
		}
		if (needsTranscoding)
			std::memcpy(staging.getData(), texture->pData, totalByteSize);
		else
			EASSERT(ktxTexture_LoadImageData((ktxTexture*)texture, (ktx_uint8_t*)staging.getData(), totalByteSize) == KTX_SUCCESS, "ktx", "Could not load data into buffer.");
		ktxTexture_Destroy((ktxTexture*)texture);

		ImageCubeMap cubemap{ vulkanObjects.getLogicalDevice(), format, static_cast<uint32_t>(sideLength), VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, static_cast<int>(mipLevelCount) };

		VkCommandBuffer cb{ commandBufferSet.beginTransientRecording() };
		SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
													.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.image = cubemap.getImageHandle(),
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = cubemap.getMipLevelCount(),
														.baseArrayLayer = 0,
														.layerCount = cubemapFaceCount }}}
			});
		cubemap.cmdCopyDataFromBuffer(cb, staging.getBufferHandle(), sideLength, stagingOffsets.size(), stagingOffsets.data(), faceIndices.data(), mipIndices.data());
		SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
													.image = cubemap.getImageHandle(),
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = cubemap.getMipLevelCount(),
														.baseArrayLayer = 0,
														.layerCount = cubemapFaceCount }}}
			});
		commandBufferSet.endRecording(cb);

		VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
		vkQueueSubmit(vulkanObjects.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(vulkanObjects.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));

		commandBufferSet.resetAllTransient();

		return cubemap;
	}



	inline Image loadImage(std::shared_ptr<VulkanObjectHandler> vulkanObjects,
						   CommandBufferSet& commandBufferSet,
						   BufferBaseHostAccessible& stagingBase,
						   fs::path filepath,
						   VkImageUsageFlags usageFlags,
						   int channelNum, 
						   OIIO::TypeDesc::BASETYPE oiioFormat,				   
						   VkFormat vulkanFormat,
						   bool genMipmap = false)
	{
		OIIO::ImageSpec config{};
		config["oiio:UnassociatedAlpha"] = 1;
		auto imInp = OIIO::ImageInput::open(filepath, &config);
		OIIO::ImageSpec spec{ imInp->spec() };
		EASSERT(spec.depth == 1, "App", "Multi-layered image is not supported yet");
		LOG_IF_WARNING(spec.format.basetype != oiioFormat, "Image native format is not the same as required format. {}", "Data will be converted.");

		uint64_t totalByteSize{ spec.image_bytes() };
		
		BufferMapped staging{ stagingBase, totalByteSize };

		imInp->read_image(0, channelNum, oiioFormat, staging.getData());
		imInp->close();

		Image image{ vulkanObjects->getLogicalDevice(), vulkanFormat, static_cast<uint32_t>(spec.width), static_cast<uint32_t>(spec.height), usageFlags, VK_IMAGE_ASPECT_COLOR_BIT, genMipmap };
		VkCommandBuffer cb{ commandBufferSet.beginTransientRecording() };
		SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
													.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.image = image.getImageHandle(),
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = image.getMipLevelCount(),
														.baseArrayLayer = 0,
														.layerCount = 1 }}}
			});
		image.cmdCopyDataFromBuffer(cb, staging.getBufferHandle(), staging.getOffset(), 0, 0, image.getWidth(), image.getHeight());
		SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
													.image = image.getImageHandle(),
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = image.getMipLevelCount(),
														.baseArrayLayer = 0,
														.layerCount = 1 }}}
			});
		if (genMipmap)
			image.cmdCreateMipmaps(cb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		else
		{
			SyncOperations::cmdExecuteBarrier(cb, { {VkImageMemoryBarrier2{
													.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
													.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
													.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
													.dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
													.dstAccessMask = VK_ACCESS_NONE,
													.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
													.image = image.getImageHandle(),
													.subresourceRange =
														{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
														.baseMipLevel = 0,
														.levelCount = image.getMipLevelCount(),
														.baseArrayLayer = 0,
														.layerCount = 1 }}}
				});
		}
		commandBufferSet.endRecording(cb);

		VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
		vkQueueSubmit(vulkanObjects->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(vulkanObjects->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));

		commandBufferSet.resetAllTransient();

		return image;
	}

}

#endif