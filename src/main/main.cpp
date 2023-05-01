#include <iostream>
#include <vector>
#include <span>
#include <tuple>
#include <filesystem>
#include <fstream>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "vulkan/vulkan.h"

#define VMA_IMPLEMENTATION
#include "vma/vk_mem_alloc.h"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/transform.hpp>

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/data_management/memory_manager.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_list.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/window/window.h"
#include "src/world_state_class/world_state.h"

#include "src/tools/asserter.h"

#include "obj loader/obj_loader.h"

#define WINDOW_WIDTH_DEFAULT  1280
#define WINDOW_HEIGHT_DEFAULT 720

#define GENERAL_BUFFER_DEFAULT_SIZE 67108864
#define STAGING_BUFFER_DEFAULT_SIZE 8388608

namespace fs = std::filesystem;

std::shared_ptr<VulkanObjectHandler> initializeVulkan(const Window& window);

struct PipelineStuff
{
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout layout;
};
PipelineStuff createDummyPipeline(VkDevice device, VkDescriptorSetLayout layout);
struct DepthBufferImageStuff
{
	VkImage depthImage;
	VkImageView depthImageView;
	VkDeviceMemory depthbufferMemory;
};
DepthBufferImageStuff createDepthBuffer(VkPhysicalDevice physDevice, VkDevice device);
void destroyDepthBuffer(VkDevice device, DepthBufferImageStuff depthBuffer);
std::vector<char> getShaderCode(fs::path filepath);
VkShaderModule createModule(VkDevice device, std::vector<char>& code);

int main()
{
	ASSERT_ALWAYS(glfwInit(), "GLFW", "GLFW was not initialized.")
	Window window(WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT, "Engine");

	std::shared_ptr<VulkanObjectHandler> vulkanObjectHandler{ initializeVulkan(window) };

	MemoryManager memManager{ *vulkanObjectHandler };
	BufferBase::assignGlobalMemoryManager(memManager);
	ImageList::assignGlobalMemoryManager(memManager);

	FrameCommandPoolSet poolSet{ *vulkanObjectHandler };
	FrameCommandBufferSet bufferSet{ poolSet };

	DescriptorManager descriptorManager{ *vulkanObjectHandler };
	ResourceSetSharedData::initializeResourceManagement(*vulkanObjectHandler, descriptorManager);


	VkDevice device{ vulkanObjectHandler->getLogicalDevice() };

	uint32_t gfIndex{ vulkanObjectHandler->getGraphicsFamilyIndex() };
	uint32_t cfIndex{ vulkanObjectHandler->getComputeFamilyIndex() };

	BufferBaseHostInaccessible baseDeviceBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, {{gfIndex}} };
	BufferBaseHostAccessible baseHostBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, {{gfIndex}} };
	BufferBaseHostAccessible stagingBaseBuffer{ device, STAGING_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, {{gfIndex, cfIndex}} };


	// START: Dummy rendering test

	BufferMapped uniformBuffer{ baseHostBuffer, sizeof(glm::mat4) * 3 };

	glm::mat4* mvpMatrices{ reinterpret_cast<glm::mat4*>(uniformBuffer.getData()) };
	mvpMatrices[0] = glm::mat4{ 1.0 };
	mvpMatrices[1] = glm::lookAt(glm::vec3{ -1.0, 2.0, -5.0 }, glm::vec3{ 0.0, 2.0, 0.0 }, glm::vec3{ 0.0, 1.0, 0.0 });
	mvpMatrices[2] = glm::perspective(glm::radians(45.0), (double)window.getWidth() / window.getHeight(), 0.1, 100.0);

	VkDescriptorSetLayoutBinding binding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT addressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = uniformBuffer.getDeviceAddress(), .range = uniformBuffer.getSize() };

	std::vector<char> vertCode{ getShaderCode("shaders/cmpld/shader_vert.spv") };
	std::vector<char> fragCode{ getShaderCode("shaders/cmpld/shader_frag.spv") };
	std::vector<ShaderStage> shaderStages{ ShaderStage{createModule(device, vertCode), VK_SHADER_STAGE_VERTEX_BIT, "main"}, ShaderStage{createModule(device, fragCode), VK_SHADER_STAGE_FRAGMENT_BIT, "main"}};
	std::vector<ResourceSet> resourceSets{ /*ResourceSet{device, 0, VkDescriptorSetLayoutCreateFlags{}, {{binding}}, 1, {{{.pUniformBuffer = &addressinfo}}}}*/ };
	resourceSets.emplace_back( device, 0, VkDescriptorSetLayoutCreateFlags{}, std::span<const VkDescriptorSetLayoutBinding>{{binding}}, 1, std::span<const VkDescriptorDataEXT>{{{.pUniformBuffer = &addressinfo}}} );
	Pipeline pipeline{ device, shaderStages, resourceSets, {{PosTexVertex::getBindingDescription()}}, {PosTexVertex::getAttributeDescriptions()} };
	vertCode.clear();
	fragCode.clear();
	
	DepthBufferImageStuff depthBuffer{ createDepthBuffer(vulkanObjectHandler->getPhysicalDevice(), vulkanObjectHandler->getLogicalDevice()) };

	objl::Loader objLoader{};
	//objLoader.LoadFile("D:/Projects/Engine/obj loader/meshes/bunny10k.obj");
	objLoader.LoadFile("D:/Projects/Engine/obj loader/meshes/greek_helmet.obj");
	std::vector<PosTexVertex> vertices{};
	for (auto& vert : objLoader.LoadedMeshes[0].Vertices)
	{
		vertices.push_back({ {vert.Position.X, vert.Position.Y, vert.Position.Z}, {vert.TextureCoordinate.X, vert.TextureCoordinate.Y} });
	}
	vertices.shrink_to_fit();
	std::vector<uint32_t> indices{ objLoader.LoadedMeshes[0].Indices };

	//ImageList listImage{ device, 1024, 1024, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT };
	Buffer vertexData{ baseDeviceBuffer, vertices.size() * sizeof(PosTexVertex) };
	Buffer indexData{ baseDeviceBuffer, indices.size() * sizeof(uint32_t) };
	VkBuffer bufferHandle{ vertexData.getBufferHandle() };
	VkDeviceSize vertexOffset{ vertexData.getOffset() };
	VkDeviceSize indexOffset{ indexData.getOffset() };

	void* ptr{ stagingBaseBuffer.getData() };
	std::memcpy(ptr, vertices.data(), vertexData.getSize());
	std::memcpy(reinterpret_cast<uint8_t*>(ptr) + vertexData.getSize(), indices.data(), indexData.getSize());

	VkCommandBuffer CB{ bufferSet.beginRecording(FrameCommandBufferSet::MAIN_CB) };
	{
		VkBufferCopy regions[2]{ {.srcOffset = 0, .dstOffset = vertexData.getOffset(), .size = vertexData.getSize()}, {.srcOffset = vertexData.getSize(), .dstOffset = indexData.getOffset(), .size = indexData.getSize()} };
		BufferTools::cmdBufferCopy(CB, stagingBaseBuffer.getBufferHandle(), vertexData.getBufferHandle(), 2, regions);
	}
	bufferSet.endRecording(CB);
	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &CB };
	ASSERT_ALWAYS(vkQueueSubmit(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS, "Vulkan", "Queue submission failed");
	ASSERT_ALWAYS(vkQueueWaitIdle(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) == VK_SUCCESS, "Vulkan", "Wait idle failed");



	VkSemaphore swapchainSemaphore{};
	VkSemaphoreCreateInfo semCI1{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	vkCreateSemaphore(device, &semCI1, nullptr, &swapchainSemaphore);

	VkSubmitInfo submitInfoRender{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &swapchainSemaphore, .commandBufferCount = 1, .pCommandBuffers = &CB };

	uint32_t swapchainIndex{};
	VkRenderingAttachmentInfo attachmentInfo{};
	attachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	attachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	attachmentInfo.clearValue = VkClearValue{ .color{.float32{0.4f, 1.0f, 0.8f}} };

	VkRenderingAttachmentInfo depthAttachmentInfo{};
	depthAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	depthAttachmentInfo.imageView = depthBuffer.depthImageView;
	depthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
	depthAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachmentInfo.clearValue = { .depthStencil = {.depth = 1.0f, .stencil = 0} };


	VkImageMemoryBarrier image_memory_barrier1
	{
	.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	.subresourceRange = {
	  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	  .baseMipLevel = 0,
	  .levelCount = 1,
	  .baseArrayLayer = 0,
	  .layerCount = 1,
	}
	};
	VkImageMemoryBarrier image_memory_barrier
	{
	.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
	.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	.subresourceRange = {
	  .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
	  .baseMipLevel = 0,
	  .levelCount = 1,
	  .baseArrayLayer = 0,
	  .layerCount = 1,
	}
	};

	VkImageMemoryBarrier depthBufferBarrier
	{
	.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	.srcAccessMask = 0,
	.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
	.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	.newLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
	.image = depthBuffer.depthImage,
	.subresourceRange = {
	  .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
	  .baseMipLevel = 0,
	  .levelCount = 1,
	  .baseArrayLayer = 0,
	  .layerCount = 1,
	}
	};

	VkFence fence{};
	VkFenceCreateInfo fenceCI{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
	vkCreateFence(device, &fenceCI, nullptr, &fence);

	WorldState::initialize();
	std::tuple<VkImage, VkImageView, uint32_t> swapchainImageData{};
	double angle{0.0};
	while (!glfwWindowShouldClose(window))
	{
		vkWaitForFences(device, 1, &fence, true, UINT64_MAX);
		WorldState::refreshFrameTime();
		angle += glm::radians(90.0f) * WorldState::getDeltaTime();

		//mvpMatrices[0] = glm::rotate(static_cast<float>(angle), glm::vec3{ 0.0, 1.0, 0.0 }) * glm::scale(glm::vec3{19.0f});
		mvpMatrices[0] = glm::rotate(static_cast<float>(angle), glm::vec3{ 0.0, 1.0, 0.0 }) * glm::scale(glm::vec3{ 0.5f }) * glm::translate(glm::vec3{ 0.0f, 5.0f, 0.0f });

		vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex);
		swapchainImageData = vulkanObjectHandler->getSwapchainImageData(swapchainIndex);
		attachmentInfo.imageView = std::get<1>(swapchainImageData);
		image_memory_barrier1.image = std::get<0>(swapchainImageData);
		image_memory_barrier.image = std::get<0>(swapchainImageData);
		vkQueueWaitIdle(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
		

		VkCommandBuffer CBloop{ bufferSet.beginRecording(FrameCommandBufferSet::MAIN_CB) };

		descriptorManager.cmdSubmitPipelineResources(CBloop, pipeline.getResourceSets(), pipeline.getPipelineLayoutHandle());

		vkCmdPipelineBarrier(
			CBloop,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&depthBufferBarrier
		);

		vkCmdPipelineBarrier(
			CBloop,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,  // srcStageMask
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, // dstStageMask
			0,
			0,
			nullptr,
			0,
			nullptr,
			1, // imageMemoryBarrierCount
			&image_memory_barrier1 // pImageMemoryBarriers
		);


		pipeline.cmdBind(CBloop);
		vkCmdBindVertexBuffers(CBloop, 0, 1, &bufferHandle, &vertexOffset);
		vkCmdBindIndexBuffer(CBloop, bufferHandle, indexOffset, VK_INDEX_TYPE_UINT32);
		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = { .offset{0,0}, .extent{.width = WINDOW_WIDTH_DEFAULT, .height = WINDOW_HEIGHT_DEFAULT} };
		renderInfo.layerCount = 1;
		renderInfo.pDepthAttachment = &depthAttachmentInfo;
		renderInfo.colorAttachmentCount = 1;
		renderInfo.pColorAttachments = &attachmentInfo;
		
		vkCmdBeginRendering(CBloop, &renderInfo);
		vkCmdDrawIndexed(CBloop, indices.size(), 1, 0, 0, 0);
		vkCmdEndRendering(CBloop);


		vkCmdPipelineBarrier(
			CBloop,
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0,
			0,
			nullptr,
			0,
			nullptr,
			1,
			&image_memory_barrier
		);

		bufferSet.endRecording(CBloop);
		vkResetFences(device, 1, &fence);
		vkQueueSubmit(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfoRender, fence);
		vkQueueWaitIdle(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));

		bufferSet.resetCommandBuffer(CBloop);

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		VkSwapchainKHR swapChains[] = { vulkanObjectHandler->getSwapchain() };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;

		presentInfo.pImageIndices = &std::get<2>(swapchainImageData);

		ASSERT_ALWAYS(vkQueuePresentKHR(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), &presentInfo) == VK_SUCCESS, "Vulkan", "Present didn't work");

		glfwPollEvents();
	}

	// END: Dummy rendering test


	destroyDepthBuffer(device, depthBuffer);
	vkDestroyFence(device, fence, nullptr);
	vkDestroySemaphore(device, swapchainSemaphore, nullptr);
	glfwTerminate();
	return 0;
}

std::shared_ptr<VulkanObjectHandler> initializeVulkan(const Window& window)
{
	VulkanCreateInfo info{};
	info.windowPtr = window;
	return std::shared_ptr<VulkanObjectHandler>{ std::make_shared<VulkanObjectHandler>(info) };
}

std::vector<char> getShaderCode(fs::path filepath)
{
	std::ifstream stream{ filepath, std::ios::ate | std::ios::binary };
	ASSERT_ALWAYS(stream.is_open(), "App", "Could not open shader file");

	size_t streamSize{ static_cast<size_t>(stream.tellg()) };
	size_t codeSize{ streamSize };
	if (static_cast<size_t>(streamSize) % 4 != 0)
		codeSize += (4 - static_cast<size_t>(streamSize) % 4);

	std::vector<char> code(codeSize);
	stream.seekg(std::ios_base::beg);
	stream.read(code.data(), streamSize);
	return code;
}

VkShaderModule createModule(VkDevice device, std::vector<char>& code)
{
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.codeSize = code.size();
	createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	ASSERT_ALWAYS(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) == VK_SUCCESS, "Vulkan", "Shader module creation failed.");
	return shaderModule;
}

DepthBufferImageStuff createDepthBuffer(VkPhysicalDevice physDevice, VkDevice device)
{
	DepthBufferImageStuff dbStuff{};

	VkImageCreateInfo imageInfo{};
	imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width = WINDOW_WIDTH_DEFAULT;
	imageInfo.extent.height = WINDOW_HEIGHT_DEFAULT;
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.format = VK_FORMAT_D32_SFLOAT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	ASSERT_ALWAYS(vkCreateImage(device, &imageInfo, nullptr, &dbStuff.depthImage) == VK_SUCCESS, "Vulkan", "Image creation failed")

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(device, dbStuff.depthImage, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;

	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(physDevice, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) 
	{
		if ((memRequirements.memoryTypeBits & (1 << i)) && memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) 
		{
			allocInfo.memoryTypeIndex = i;
			break;
		}
	}

	ASSERT_ALWAYS(vkAllocateMemory(device, &allocInfo, nullptr, &dbStuff.depthbufferMemory) == VK_SUCCESS, "Vulkan", "Memory allocation failed")

	vkBindImageMemory(device, dbStuff.depthImage, dbStuff.depthbufferMemory, 0);


	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = dbStuff.depthImage;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = VK_FORMAT_D32_SFLOAT;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	ASSERT_ALWAYS(vkCreateImageView(device, &viewInfo, nullptr, &dbStuff.depthImageView) == VK_SUCCESS, "Vulkan", "Image view creation failed")

	return dbStuff;
}

void destroyDepthBuffer(VkDevice device, DepthBufferImageStuff depthBuffer)
{
	vkDestroyImageView(device, depthBuffer.depthImageView, nullptr);
	vkDestroyImage(device, depthBuffer.depthImage, nullptr);
	vkFreeMemory(device, depthBuffer.depthbufferMemory, nullptr);
}