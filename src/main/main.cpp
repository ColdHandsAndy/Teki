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

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/transform.hpp>

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/data_management/memory_manager.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/window/window.h"
#include "src/world_state_class/world_state.h"

#include "src/tools/asserter.h"

#define WINDOW_HEIGHT_DEFAULT 720
#define WINDOW_WIDTH_DEFAULT  1280

namespace fs = std::filesystem;

struct PipelineStuff
{
	VkPipeline pipeline;
	VkPipelineLayout pipelineLayout;
	VkDescriptorSetLayout layout;
};
PipelineStuff createDummyPipeline(VkDevice device, VkDescriptorSetLayout layout);

PFN_vkGetDescriptorSetLayoutSizeEXT lvkGetDescriptorSetLayoutSizeEXT;
PFN_vkGetDescriptorSetLayoutBindingOffsetEXT lvkGetDescriptorSetLayoutBindingOffsetEXT;
PFN_vkCmdBindDescriptorBuffersEXT lvkCmdBindDescriptorBuffersEXT;
PFN_vkCmdSetDescriptorBufferOffsetsEXT lvkCmdSetDescriptorBufferOffsetsEXT;
PFN_vkGetDescriptorEXT lvkGetDescriptorEXT;
PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT lvkCmdBindDescriptorBufferEmbeddedSamplersEXT;

int main()
{
	ASSERT_ALWAYS(glfwInit(), "GLFW", "GLFW was not initialized.")
	Window window(WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT, "Engine");

	VulkanCreateInfo info{};
	info.windowPtr = window;
	std::shared_ptr<VulkanObjectHandler> vulkanObjectHandler{ std::make_shared<VulkanObjectHandler>(info) };

	MemoryManager memManager{ *vulkanObjectHandler };
	BufferBaseHostInaccessible::assignGlobalMemoryManager(memManager);

	FrameCommandPoolSet poolSet{ *vulkanObjectHandler };
	FrameCommandBufferSet bufferSet{ poolSet };

	DescriptorManager descriptorManager{ *vulkanObjectHandler };


	// START: Descriptor buffer test
	VkDevice device{ vulkanObjectHandler->getLogicalDevice() };

	lvkGetDescriptorSetLayoutSizeEXT = reinterpret_cast<PFN_vkGetDescriptorSetLayoutSizeEXT>(vkGetDeviceProcAddr(device, "vkGetDescriptorSetLayoutSizeEXT"));
	lvkGetDescriptorSetLayoutBindingOffsetEXT = reinterpret_cast<PFN_vkGetDescriptorSetLayoutBindingOffsetEXT>(vkGetDeviceProcAddr(device, "vkGetDescriptorSetLayoutBindingOffsetEXT"));
	lvkCmdBindDescriptorBuffersEXT = reinterpret_cast<PFN_vkCmdBindDescriptorBuffersEXT>(vkGetDeviceProcAddr(device, "vkCmdBindDescriptorBuffersEXT"));
	lvkGetDescriptorEXT = reinterpret_cast<PFN_vkGetDescriptorEXT>(vkGetDeviceProcAddr(device, "vkGetDescriptorEXT"));
	lvkCmdBindDescriptorBufferEmbeddedSamplersEXT = reinterpret_cast<PFN_vkCmdBindDescriptorBufferEmbeddedSamplersEXT>(vkGetDeviceProcAddr(device, "vkCmdBindDescriptorBufferEmbeddedSamplersEXT"));
	lvkCmdSetDescriptorBufferOffsetsEXT = reinterpret_cast<PFN_vkCmdSetDescriptorBufferOffsetsEXT>(vkGetDeviceProcAddr(device, "vkCmdSetDescriptorBufferOffsetsEXT"));



	uint32_t gfIndex{ vulkanObjectHandler->getGraphicsFamilyIndex() };
	VkBufferCreateInfo uniformBufferCI{};
	uniformBufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	uniformBufferCI.size = sizeof(glm::mat4) * 3;
	uniformBufferCI.queueFamilyIndexCount = 1;
	uniformBufferCI.pQueueFamilyIndices = &gfIndex;
	uniformBufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	uniformBufferCI.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	BufferBaseHostAccessible uniformBuffer{ device, uniformBufferCI };
	glm::mat4* mvpMatrices{ reinterpret_cast<glm::mat4*>(uniformBuffer.getData()) };
	mvpMatrices[0] = glm::mat4{1.0};
	mvpMatrices[1] = glm::lookAt(glm::vec3{ -1.0, 3.0, -5.0 }, glm::vec3{ 0.0, 0.0, 0.0 }, glm::vec3{ 0.0, 1.0, 0.0 });
	mvpMatrices[2] = glm::perspective(glm::radians(45.0), (double)window.getWidth() / window.getHeight(), 0.1, 100.0);

	ResourceSetSharedData::initializeResourceManagement(*vulkanObjectHandler, descriptorManager);
	VkDescriptorSetLayoutBinding binding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };

	VkDescriptorAddressInfoEXT addressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = uniformBuffer.getBufferDeviceAddress(), .range = uniformBuffer.getBufferByteSize() };
	VkDescriptorDataEXT descData[1]{ {.pUniformBuffer = &addressinfo} };
	std::vector<VkDescriptorSetLayoutBinding> bindings{ binding };
	ResourceSet resourceSet{ device, 0, VkDescriptorSetLayoutCreateFlags{}, bindings, 1 , std::span<VkDescriptorDataEXT>{descData} };

	PipelineStuff dummyPipeline{ createDummyPipeline(vulkanObjectHandler->getLogicalDevice(), resourceSet.getSetLayout()) };

	// END: Descriptor buffer test

	// START: Dummy rendering test

	
	float vertexPos[8 * 3] =
	{
		-1, -1, -1,
		1, -1, -1,
		1, 1, -1,
		-1, 1, -1,
		-1, -1, 1,
		1, -1, 1,
		1, 1, 1,
		-1, 1, 1
	};

	int indices[6 * 6] =
	{
		0, 1, 3, 3, 1, 2,
		1, 5, 2, 2, 5, 6,
		5, 4, 6, 6, 4, 7,
		4, 0, 7, 7, 0, 3,
		3, 2, 7, 7, 2, 6,
		4, 5, 0, 0, 5, 1
	};

	uint32_t graphicFamilyIndex{ vulkanObjectHandler->getGraphicsFamilyIndex() };
	VkBufferCreateInfo vbCI{};
	vbCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vbCI.size = sizeof(vertexPos) + sizeof(indices);
	vbCI.queueFamilyIndexCount = 1;
	vbCI.pQueueFamilyIndices = &graphicFamilyIndex;
	vbCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	vbCI.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	BufferBaseHostInaccessible vertexData{ device, vbCI, BufferBase::NULL_FLAG };

	VkBufferCreateInfo stagingBufCI{};
	stagingBufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufCI.size = sizeof(vertexPos) + sizeof(indices);
	stagingBufCI.queueFamilyIndexCount = 1;
	stagingBufCI.pQueueFamilyIndices = &graphicFamilyIndex;
	stagingBufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	stagingBufCI.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	BufferBaseHostAccessible stagingBuf{ device, stagingBufCI, BufferBase::NULL_FLAG };
	void* ptr{ stagingBuf.getData() };

	std::memcpy(ptr, vertexPos, sizeof(vertexPos));
	std::memcpy(reinterpret_cast<char*>(ptr) + sizeof(vertexPos), indices, sizeof(indices));

	VkCommandBuffer CB{ bufferSet.beginRecording(FrameCommandBufferSet::MAIN_CB) };
	VkBufferCopy region{ .srcOffset = 0, .dstOffset = 0, .size = stagingBufCI.size };
	BufferOperations::cmdBufferCopy(CB, stagingBuf.getBufferHandle(), vertexData.getBufferHandle(), 1, &region);
	bufferSet.endRecording(CB);
	VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &CB };
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
	VkBuffer bufferHandle{ vertexData.getBufferHandle() };
	VkDeviceSize vertexOffset{ 0 };
	VkDeviceSize indexOffset{ sizeof(vertexPos) };
	
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
		angle += glm::radians(180.0f) * WorldState::getDeltaTime();

		mvpMatrices[0] = glm::rotate(static_cast<float>(angle), glm::vec3{ 0.0, 1.0, 0.0 }) * glm::mat4{1.0f};

		vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex);
		swapchainImageData = vulkanObjectHandler->getSwapchainImageData(swapchainIndex);
		attachmentInfo.imageView = std::get<1>(swapchainImageData);
		attachmentInfo.imageView = std::get<1>(swapchainImageData);
		image_memory_barrier1.image = std::get<0>(swapchainImageData);
		image_memory_barrier.image = std::get<0>(swapchainImageData);
		vkQueueWaitIdle(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
		

		VkCommandBuffer CBloop{ bufferSet.beginRecording(FrameCommandBufferSet::MAIN_CB) };

		descriptorManager.cmdSubmitResource(CBloop, dummyPipeline.pipelineLayout, resourceSet);

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

		vkCmdBindPipeline(CBloop, VK_PIPELINE_BIND_POINT_GRAPHICS, dummyPipeline.pipeline);
		vkCmdBindVertexBuffers(CBloop, 0, 1, &bufferHandle, &vertexOffset);
		vkCmdBindIndexBuffer(CBloop, bufferHandle, indexOffset, VK_INDEX_TYPE_UINT32);
		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = { .offset{0,0}, .extent{.width = WINDOW_WIDTH_DEFAULT, .height = WINDOW_HEIGHT_DEFAULT} };
		renderInfo.layerCount = 1;
		renderInfo.colorAttachmentCount = 1;
		renderInfo.pColorAttachments = &attachmentInfo;
		vkCmdBeginRendering(CBloop, &renderInfo);
		vkCmdDrawIndexed(CBloop, sizeof(indices) / sizeof(indices[0]), 1, 0, 0, 0);
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


	vkDestroyFence(device, fence, nullptr);
	vkDestroySemaphore(device, swapchainSemaphore, nullptr);
	vkDestroyPipeline(device, dummyPipeline.pipeline, nullptr);
	vkDestroyPipelineLayout(device, dummyPipeline.pipelineLayout, nullptr);
	glfwTerminate();
	return 0;
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

PipelineStuff createDummyPipeline(VkDevice device, VkDescriptorSetLayout layout)
{
	VkPipelineLayout pipelineLayout{};

	VkPipelineLayoutCreateInfo pipelineLayoutCI{};
	pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

	pipelineLayoutCI.setLayoutCount = 1;
	pipelineLayoutCI.pSetLayouts = &layout;
	ASSERT_ALWAYS(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &pipelineLayout) == VK_SUCCESS, "Vulkan", "Pipeline layout creation failed.");

	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssembly.primitiveRestartEnable = VK_FALSE;

	VkPipelineRasterizationStateCreateInfo rasterizer{};
	rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizer.depthClampEnable = VK_FALSE;
	rasterizer.rasterizerDiscardEnable = VK_FALSE;
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.lineWidth = 1.0f;
	//rasterizer.cullMode = VK_CULL_MODE_NONE_BIT;
	//rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
	rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.depthBiasEnable = VK_FALSE;

	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;
	colorBlending.blendConstants[0] = 0.0f;
	colorBlending.blendConstants[1] = 0.0f;
	colorBlending.blendConstants[2] = 0.0f;
	colorBlending.blendConstants[3] = 0.0f;

	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencil.depthTestEnable = VK_FALSE;
	depthStencil.depthWriteEnable = VK_FALSE;
	depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencil.depthBoundsTestEnable = VK_FALSE;
	depthStencil.stencilTestEnable = VK_FALSE;

	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	VkViewport viewport{ .x = 0, .y = WINDOW_HEIGHT_DEFAULT, .width = WINDOW_WIDTH_DEFAULT, .height = -WINDOW_HEIGHT_DEFAULT };
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	VkRect2D rect{ .offset{0, 0}, .extent{.width = WINDOW_WIDTH_DEFAULT, .height = WINDOW_HEIGHT_DEFAULT} };
	viewportState.pScissors = &rect;

	VkPipelineMultisampleStateCreateInfo multisampling{};
	multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisampling.sampleShadingEnable = VK_FALSE;
	multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;

	VkPipelineShaderStageCreateInfo shaderStages[2]{};
	std::vector<char> vertCode{getShaderCode("shaders/cmpld/shader_vert.spv")};
	VkShaderModule vertModule{createModule(device, vertCode)};
	shaderStages[0] = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertModule, .pName = "main"};
	std::vector<char> fragCode{getShaderCode("shaders/cmpld/shader_frag.spv")};
	VkShaderModule fragModule{createModule(device, fragCode)};
	shaderStages[1] = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragModule, .pName = "main" };

	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	auto bindingDescription = PosOnlyVertex::getBindingDescription();
	auto attributeDescriptions = PosOnlyVertex::getAttributeDescriptions();
	vertexInputInfo.vertexBindingDescriptionCount = 1;
	vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
	vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

	VkGraphicsPipelineCreateInfo pipelineCI{};
	pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCI.layout = pipelineLayout;
	pipelineCI.pInputAssemblyState = &inputAssembly;
	pipelineCI.pRasterizationState = &rasterizer;
	pipelineCI.pColorBlendState = &colorBlending;
	pipelineCI.pMultisampleState = &multisampling;
	pipelineCI.pViewportState = &viewportState;
	pipelineCI.pDepthStencilState = &depthStencil;
	pipelineCI.pDynamicState = &dynamicState;
	pipelineCI.stageCount = 2;
	pipelineCI.pStages = shaderStages;
	pipelineCI.pVertexInputState = &vertexInputInfo;
	pipelineCI.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;


	VkPipeline pipeline{};
	ASSERT_ALWAYS(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline) == VK_SUCCESS, "Vulkan", "Pipeline creation failed.");

	vkDestroyShaderModule(device, vertModule, nullptr);
	vkDestroyShaderModule(device, fragModule, nullptr);
	return { pipeline, pipelineLayout, layout };
}