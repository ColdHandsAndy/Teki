#include <iostream>
#include <vector>
#include <span>
#include <tuple>
#include <filesystem>
#include <fstream>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_USE_CPP14
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma warning(suppress : 4996)
#include "tiny_gltf.h"
#undef TINYGLTF_IMPLEMENTATION
#undef STB_IMAGE_IMPLEMENTATION
#undef STB_IMAGE_WRITE_IMPLEMENTATION

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/shader_management/shader_operations.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/data_management/memory_manager.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/renderer/barrier_operations.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/data_abstraction/mesh.h"
#include "src/rendering/data_abstraction/runit.h"

#include "src/window/window.h"
#include "src/world_state_class/world_state.h"

#include "src/tools/alignment.h"
#include "src/tools/model_loader.h"
#include "src/tools/cubemap_loader.h"
#include "src/tools/asserter.h"


#define WINDOW_WIDTH_DEFAULT  1280
#define WINDOW_HEIGHT_DEFAULT 720

#define GENERAL_BUFFER_DEFAULT_SIZE 67108864
#define STAGING_BUFFER_DEFAULT_SIZE 8388608
//#define DEVICE_BUFFER_DEFAULT_VALUE 2147483648ll

namespace fs = std::filesystem;

std::shared_ptr<VulkanObjectHandler> initializeVulkan(const Window& window);

Pipeline createForwardSimplePipeline(PipelineAssembler& assembler, BufferMapped& spaceTransformDataUB, BufferMapped& transformDataUB, BufferMapped& perDrawDataIndicesUB, std::vector<ImageList>& imageLists, VkSampler sampler);
Pipeline createSpaceLinesPipeline(PipelineAssembler& assembler, BufferMapped& spaceTransformDataUB);
Pipeline createSkyboxPipeline(PipelineAssembler& assembler, const ImageCubeMap& cubemapImages, const BufferMapped& skyboxTransformUB);

uint32_t uploadLineVertices(fs::path filepath, Buffer& vertexBuffer, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue);
void uploadSkyboxVertexData(Buffer& skyboxData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue);


struct DepthBufferImageStuff
{
	VkImage depthImage;
	VkImageView depthImageView;
	VkDeviceMemory depthbufferMemory;
};
DepthBufferImageStuff createDepthBuffer(VkPhysicalDevice physDevice, VkDevice device);
void destroyDepthBuffer(VkDevice device, DepthBufferImageStuff depthBuffer);
VkSampler createUniversalSampler(VulkanObjectHandler& vulkanObjHandler);



int main()
{
	ASSERT_ALWAYS(glfwInit(), "GLFW", "GLFW was not initialized.")
	Window window(WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT, "Engine");

	std::shared_ptr<VulkanObjectHandler> vulkanObjectHandler{ initializeVulkan(window) };

	MemoryManager memManager{ *vulkanObjectHandler };
	BufferBase::assignGlobalMemoryManager(memManager);
	ImageList::assignGlobalMemoryManager(memManager);
	
	FrameCommandPoolSet cmdPoolSet{ *vulkanObjectHandler };
	FrameCommandBufferSet cmdBufferSet{ cmdPoolSet };
	
	DescriptorManager descriptorManager{ *vulkanObjectHandler };
	ResourceSetSharedData::initializeResourceManagement(*vulkanObjectHandler, descriptorManager);


	VkDevice device{ vulkanObjectHandler->getLogicalDevice() };

	uint32_t gfIndex{ vulkanObjectHandler->getGraphicsFamilyIndex() };
	uint32_t cfIndex{ vulkanObjectHandler->getComputeFamilyIndex() };
	uint32_t tfIndex{ vulkanObjectHandler->getTransferFamilyIndex() };

	BufferBaseHostInaccessible baseDeviceBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostCachedBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, BufferBase::DEDICATED_FLAG, false, true };
	


	Buffer vertexData{ baseDeviceBuffer };
	Buffer indexData{ baseDeviceBuffer };
	BufferMapped indirectCmdBuffer{ baseHostCachedBuffer };
	std::vector<ImageList> imageLists{};
	std::vector<StaticMesh> staticMeshes(loadStaticMeshes(vulkanObjectHandler, descriptorManager, cmdBufferSet, vertexData, indexData, indirectCmdBuffer, imageLists,
		{
			//"A:/Models/gltf/sci-fi_personal_space_pod_shipweekly_challenge/scene.gltf",
			//"A:/Models/gltf/sci-fi_personal_space_pod_shipweekly_challenge/scene.gltf",
			"A:/Models/gltf/skull_trophy/scene.gltf",
			//"A:/Models/gltf/hoverbike_on_service/scene.gltf"
		}));
	uint32_t drawCount{ static_cast<uint32_t>(indirectCmdBuffer.getSize() / sizeof(VkDrawIndexedIndirectCommand)) };


	PipelineAssembler assembler{ device };
	
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DEFAULT);
	VkSampler sampler{ createUniversalSampler(*vulkanObjectHandler) };
	BufferMapped spaceTransformDataUB{ baseHostBuffer, sizeof(glm::mat4) * 2 + sizeof(glm::vec4) * 2 };
	BufferMapped transformDataUB{ baseHostBuffer, sizeof(glm::mat4) * 8 };
	BufferMapped perDrawDataIndicesSSBO{ baseHostBuffer, sizeof(uint8_t) * 12 * drawCount };
	Pipeline forwardSimplePipeline{ createForwardSimplePipeline(assembler, spaceTransformDataUB, transformDataUB, perDrawDataIndicesSSBO, imageLists, sampler) };

	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0f, VK_CULL_MODE_NONE);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_SKYBOX);
	Buffer skyboxData{ baseDeviceBuffer };
	uploadSkyboxVertexData(skyboxData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
	BufferMapped skyboxTransformUB{ baseHostBuffer, sizeof(glm::mat4) * 2 };
	ImageCubeMap cubemapSkybox{ loadCubemap(vulkanObjectHandler, cmdBufferSet, baseHostBuffer, "A:\\cubemapsHDR\\green")};
	Pipeline skyboxPipeline{ createSkyboxPipeline(assembler, cubemapSkybox, skyboxTransformUB) };
	
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_LINE_DRAWING);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 0.6f);
	Buffer spaceLinesVertexData{ baseDeviceBuffer };
	uint32_t lineVertNum{ uploadLineVertices("D:/Projects/Engine/bins/space_lines.bin", spaceLinesVertexData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) };
	Pipeline spaceLinesPipeline{ createSpaceLinesPipeline(assembler, spaceTransformDataUB) };


	// START: Dummy rendering test
	
	//Resource filling
	glm::vec3 cameraPos{ glm::vec3{0.0, 3.0, -9.0} };
	glm::vec3 viewDir{ glm::normalize(-cameraPos) };
	glm::vec3 upVec{ glm::vec3{0.0, -1.0, 0.0} };
	glm::vec3 lightPos{ glm::vec3{0.0, 1.0, 0.0} };

	glm::mat4* vpMatrices{ reinterpret_cast<glm::mat4*>(spaceTransformDataUB.getData()) };
	vpMatrices[0] = glm::lookAt(cameraPos, viewDir, upVec);
	vpMatrices[1] = glm::perspective(glm::radians(45.0), (double)window.getWidth() / window.getHeight(), 0.1, 100.0);
	glm::vec3* lightInfo{ reinterpret_cast<glm::vec3*>(&vpMatrices[2]) };
	lightInfo[0] = lightPos;
	glm::vec3* camPosData{ reinterpret_cast<glm::vec3*>(reinterpret_cast<uint8_t*>(spaceTransformDataUB.getData()) + sizeof(glm::mat4) * 2 + sizeof(glm::vec4)) };
	*camPosData = cameraPos;
	glm::vec3* lightPosData{&lightInfo[0]};

	glm::mat4* transformMatrices{ reinterpret_cast<glm::mat4*>(transformDataUB.getData()) };
	transformMatrices[0] = glm::translate(glm::vec3{-8.0f, 0.0f, 5.0f}) * glm::scale(glm::vec3{ 3.0f });
	transformMatrices[1] = glm::translate(glm::vec3{-4.0f, 0.0f, 5.0f}) * glm::scale(glm::vec3{ 3.0f });
	transformMatrices[2] = glm::translate(glm::vec3{-2.0f, 0.0f, 5.0f}) * glm::scale(glm::vec3{ 3.0f });
	transformMatrices[3] = glm::translate(glm::vec3{-1.0f, 0.0f, 5.0f}) * glm::scale(glm::vec3{ 3.0f });
	transformMatrices[4] = glm::translate(glm::vec3{1.0f, 0.0f, 5.0f}) * glm::scale(glm::vec3{ 3.0f });
	transformMatrices[5] = glm::translate(glm::vec3{2.0f, 0.0f, 5.0f}) * glm::scale(glm::vec3{ 3.0f });
	transformMatrices[6] = glm::translate(glm::vec3{4.0f, 0.0f, 5.0f}) * glm::scale(glm::vec3{ 3.0f });
	transformMatrices[7] = glm::translate(glm::vec3{8.0f, 0.0f, 5.0f}) * glm::scale(glm::vec3{ 3.0f });
	
	uint8_t* drawDataIndices{ reinterpret_cast<uint8_t*>(perDrawDataIndicesSSBO.getData()) };
	//Per mesh indices
	uint32_t transMatIndex{ 0 };
	uint32_t drawNum{ static_cast<uint32_t>(staticMeshes[transMatIndex].getRUnits().size()) };
	for (uint32_t i{ 0 }; i < drawCount; ++i)
	{
		if (i == drawNum)
		{
			drawNum += staticMeshes[++transMatIndex].getRUnits().size();
		}
		*(drawDataIndices + i * 12 + 0) = transMatIndex;
		*(drawDataIndices + i * 12 + 1) = 0;
		*(drawDataIndices + i * 12 + 2) = 0;
		*(drawDataIndices + i * 12 + 3) = 0;
	}
	//Per unit indices
	for (auto& mesh : staticMeshes)
	{
		for (auto& rUnit : mesh.getRUnits())
		{
			auto indices{rUnit.getMaterialIndices()};
			*(drawDataIndices + 4) = indices[0].first;
			*(drawDataIndices + 5) = indices[0].second;
			*(drawDataIndices + 6) = indices[1].first;
			*(drawDataIndices + 7) = indices[1].second;
			*(drawDataIndices + 8) = indices[2].first;
			*(drawDataIndices + 9) = indices[2].second;
			*(drawDataIndices + 10) = indices[3].first;
			*(drawDataIndices + 11) = indices[3].second;
			drawDataIndices += sizeof(uint8_t) * 12;
		}
	}
	//end
	glm::mat4* skyboxTransform{ reinterpret_cast<glm::mat4*>(skyboxTransformUB.getData()) };
	skyboxTransform[0] = vpMatrices[0];
	skyboxTransform[0][3] = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
	skyboxTransform[1] = vpMatrices[1];


	DepthBufferImageStuff depthBuffer{ createDepthBuffer(vulkanObjectHandler->getPhysicalDevice(), vulkanObjectHandler->getLogicalDevice()) };
	
	VkSemaphore swapchainSemaphore{};
	VkSemaphore readyToPresentSemaphore{};
	VkSemaphoreCreateInfo semCI1{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	vkCreateSemaphore(device, &semCI1, nullptr, &swapchainSemaphore);
	vkCreateSemaphore(device, &semCI1, nullptr, &readyToPresentSemaphore);
	
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

	VkFence renderCompleteFence{};
	VkFenceCreateInfo fenceCI{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
	vkCreateFence(device, &fenceCI, nullptr, &renderCompleteFence);

	std::tuple<VkImage, VkImageView, uint32_t> swapchainImageData{};
	WorldState::initialize();
	vkDeviceWaitIdle(device);
	while (!glfwWindowShouldClose(window))
	{
		WorldState::refreshFrameTime();
		double angle{ glfwGetTime() * 0.5 };
		double mplier{ std::sin(glfwGetTime() * 0.5) * 0.5 + 0.5 };

		vpMatrices[0] = glm::lookAt(glm::vec3(glm::rotate((float)angle * 0.3f, glm::vec3(0.0, 1.0, 0.0)) * glm::vec4(cameraPos, 1.0)), viewDir, upVec);
		skyboxTransform[0] = vpMatrices[0];
		skyboxTransform[0][3] = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };

		glm::dmat4 transMatr{ glm::rotate(angle, glm::dvec3(0.0, 1.0, 0.0)) };
		glm::dvec4 lightPos{ 4.0, 0.0, 0.0, 1.0 };
		transformMatrices[0] = glm::translate(glm::dvec3{ 0.0f, 0.0f, 0.0f } * mplier) * glm::scale(glm::dvec3{ 0.002 });
		//transformMatrices[0] = glm::translate(glm::dvec3{ 0.0f, 0.0f, 0.0f } * mplier)* glm::scale(glm::dvec3{ 1.5 });
		transformMatrices[1] = glm::translate((glm::vec3)(transMatr * lightPos)) * glm::scale(glm::vec3{ 0.2 });
		*lightPosData = glm::vec3(transMatr * lightPos);
		//transformMatrices[0] = glm::translate(glm::dvec3{ 3.0f, 0.0f, 0.0f } * mplier) * glm::scale(glm::dvec3{ 0.5 });
		//transformMatrices[1] = glm::translate(glm::dvec3{ 0.0f, 3.0f, 0.0f } * mplier) * glm::scale(glm::dvec3{ 0.5 });
		//transformMatrices[2] = glm::translate(glm::dvec3{ 0.0f, 0.0f, 3.0f } * mplier) * glm::scale(glm::dvec3{ 0.5 });
		
		
		ASSERT_ALWAYS(vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex) == VK_SUCCESS, "Vulkan", "Could not acquire swapchain image.");
		swapchainImageData = vulkanObjectHandler->getSwapchainImageData(swapchainIndex);
		attachmentInfo.imageView = std::get<1>(swapchainImageData);


		VkCommandBuffer CBloop{ cmdBufferSet.beginRecording(FrameCommandBufferSet::MAIN_CB) };


		BarrierOperations::cmdExecuteBarrier(CBloop, std::span<const VkImageMemoryBarrier2>{
			{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 
				VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				0,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 
				std::get<0>(swapchainImageData),
				{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1}),
			BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				0,
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				depthBuffer.depthImage,
				{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1})}
			});

		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = { .offset{0,0}, .extent{.width = WINDOW_WIDTH_DEFAULT, .height = WINDOW_HEIGHT_DEFAULT} };
		renderInfo.layerCount = 1;
		renderInfo.pDepthAttachment = &depthAttachmentInfo;
		renderInfo.colorAttachmentCount = 1;
		renderInfo.pColorAttachments = &attachmentInfo;
		

		vkCmdBeginRendering(CBloop, &renderInfo);
		descriptorManager.cmdSubmitPipelineResources(CBloop, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardSimplePipeline.getResourceSets(), forwardSimplePipeline.getResourceSetsInUse(), forwardSimplePipeline.getPipelineLayoutHandle());
		VkBuffer vertexBindings[1]{ vertexData.getBufferHandle() };
		VkDeviceSize vertexBindingOffsets[1]{ vertexData.getOffset()};
		vkCmdBindVertexBuffers(CBloop, 0, 1, vertexBindings, vertexBindingOffsets);
		vkCmdBindIndexBuffer(CBloop, indexData.getBufferHandle(), indexData.getOffset(), VK_INDEX_TYPE_UINT32);
		forwardSimplePipeline.cmdBind(CBloop);
		vkCmdDrawIndexedIndirect(CBloop, indirectCmdBuffer.getBufferHandle(), indirectCmdBuffer.getOffset(), drawCount, sizeof(VkDrawIndexedIndirectCommand));
		
		VkBuffer lineVertexBindings[1]{ spaceLinesVertexData.getBufferHandle() };
		VkDeviceSize lineVertexBindingOffsets[1]{ spaceLinesVertexData.getOffset() };
		vkCmdBindVertexBuffers(CBloop, 0, 1, lineVertexBindings, lineVertexBindingOffsets);
		spaceLinesPipeline.cmdBind(CBloop);
		vkCmdDraw(CBloop, lineVertNum, 1, 0, 0);

		descriptorManager.cmdSubmitPipelineResources(CBloop, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline.getResourceSets(), skyboxPipeline.getResourceSetsInUse(), skyboxPipeline.getPipelineLayoutHandle());
		VkBuffer skyboxVertexBinding[1]{ skyboxData.getBufferHandle() };
		VkDeviceSize skyboxVertexOffsets[1]{ skyboxData.getOffset() };
		vkCmdBindVertexBuffers(CBloop, 0, 1, skyboxVertexBinding, skyboxVertexOffsets);
		skyboxPipeline.cmdBind(CBloop);
		vkCmdDraw(CBloop, 36, 1, 0, 0);

		vkCmdEndRendering(CBloop);



		BarrierOperations::cmdExecuteBarrier(CBloop, std::span<const VkImageMemoryBarrier2>{
			{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				0,
				VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
				std::get<0>(swapchainImageData),
				{
				.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1 })}
			});

		cmdBufferSet.endRecording(CBloop);
		vkResetFences(device, 1, &renderCompleteFence);
		VkPipelineStageFlags stagesToWaitOn{ VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSubmitInfo submitInfoRender{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &swapchainSemaphore, .pWaitDstStageMask = &stagesToWaitOn, .commandBufferCount = 1, .pCommandBuffers = &CBloop, .signalSemaphoreCount =1, .pSignalSemaphores = &readyToPresentSemaphore };
		vkQueueSubmit(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfoRender, renderCompleteFence);
		
		
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		VkSwapchainKHR swapChains[] = { vulkanObjectHandler->getSwapchain() }; 
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &std::get<2>(swapchainImageData);
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &readyToPresentSemaphore;

		ASSERT_ALWAYS(vkQueuePresentKHR(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), &presentInfo) == VK_SUCCESS, "Vulkan", "Present didn't work.");
		vkWaitForFences(device, 1, &renderCompleteFence, true, UINT64_MAX);
		cmdBufferSet.resetCommandBuffer(CBloop);

		glfwPollEvents();
	}
	// END: Dummy rendering test

	ASSERT_ALWAYS(vkDeviceWaitIdle(device) == VK_SUCCESS, "Vulkan", "Device wait failed.");
	vkDestroySampler(device, sampler, nullptr);
	destroyDepthBuffer(device, depthBuffer);
	vkDestroyFence(device, renderCompleteFence, nullptr);
	vkDestroySemaphore(device, readyToPresentSemaphore, nullptr);
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

Pipeline createForwardSimplePipeline(PipelineAssembler& assembler, BufferMapped& spaceTransformDataUB, BufferMapped& transformDataUB, BufferMapped& perDrawDataIndicesSSBO, std::vector<ImageList>& imageLists, VkSampler sampler) 
{
	VkDescriptorSetLayoutBinding uniformViewProjBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT uniformViewProjAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = spaceTransformDataUB.getDeviceAddress(), .range = spaceTransformDataUB.getSize() };

	VkDescriptorSetLayoutBinding uniformTransMatBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT uniformTransMatAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = transformDataUB.getDeviceAddress(), .range = transformDataUB.getSize() };

	VkDescriptorSetLayoutBinding imageListsBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 64, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	std::vector<VkDescriptorImageInfo> storageImageData(imageLists.size());
	std::vector<VkDescriptorDataEXT> imageListsDescData(imageLists.size());
	for (uint32_t i{ 0 }; i < imageListsDescData.size(); ++i)
	{
		storageImageData[i] = { .sampler = sampler, .imageView = imageLists[i].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		imageListsDescData[i].pStorageImage = &storageImageData[i];
	}


	VkDescriptorSetLayoutBinding uniformDrawIndicesBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT uniformDrawIndicesAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = perDrawDataIndicesSSBO.getDeviceAddress(), .range = perDrawDataIndicesSSBO.getSize() };

	std::vector<ResourceSet> resourceSets{};
	VkDevice device{ assembler.getDevice() };
	resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformViewProjBinding},  {}, {{{.pUniformBuffer = &uniformViewProjAddressinfo}}} });
	resourceSets.push_back({ device, 1, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformTransMatBinding}, {{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }}, {{{.pUniformBuffer = &uniformTransMatAddressinfo}}} });
	resourceSets.push_back({ device, 2, VkDescriptorSetLayoutCreateFlags{}, 1, {imageListsBinding}, {{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }}, {imageListsDescData} });
	resourceSets.push_back({ device, 3, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformDrawIndicesBinding}, {}, {{{.pUniformBuffer = &uniformDrawIndicesAddressinfo}}} });
	
	return 	Pipeline{ assembler, 
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "D:/Projects/Engine/shaders/cmpld/shader_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "D:/Projects/Engine/shaders/cmpld/shader_frag.spv"}}}, 
		resourceSets, 
		{{StaticVertex::getBindingDescription()}}, 
		{StaticVertex::getAttributeDescriptions()} };
}

Pipeline createSpaceLinesPipeline(PipelineAssembler& assembler, BufferMapped& spaceTransformDataUB)
{
	std::vector<ResourceSet> resourceSets{};
	VkDescriptorSetLayoutBinding uniformViewProjBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT uniformViewProjAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = spaceTransformDataUB.getDeviceAddress(), .range = sizeof(glm::mat4) * 2};
	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformViewProjBinding},  {}, {{{.pUniformBuffer = &uniformViewProjAddressinfo}}}});
	return Pipeline{ assembler, 
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "D:/Projects/Engine/shaders/cmpld/shader_space_lines_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "D:/Projects/Engine/shaders/cmpld/shader_space_lines_frag.spv"}}},
		resourceSets,
		{{PosColorVertex::getBindingDescription()}},
		{PosColorVertex::getAttributeDescriptions()} };
}

uint32_t uploadLineVertices(fs::path filepath, Buffer& vertexBuffer, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue)
{
	std::ifstream istream{ filepath, std::ios::binary };
	istream.seekg(0, std::ios::beg);
	
	uint32_t vertNum{};
	istream.read(reinterpret_cast<char*>(&vertNum), sizeof(vertNum));
	istream.seekg(sizeof(vertNum), std::ios::beg);

	uint32_t dataSize{ vertNum * sizeof(PosColorVertex) };

	vertexBuffer.initialize(dataSize);
	BufferMapped staging{ stagingBase, dataSize };

	istream.read(reinterpret_cast<char*>(staging.getData()), dataSize);

	VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };

	VkBufferCopy copy{ .srcOffset = staging.getOffset(), .dstOffset = vertexBuffer.getOffset(), .size = dataSize };
	BufferTools::cmdBufferCopy(cb, staging.getBufferHandle(), vertexBuffer.getBufferHandle(), 1, &copy);

	cmdBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);
	
	return vertNum;
}

Pipeline createSkyboxPipeline(PipelineAssembler& assembler, const ImageCubeMap& cubemapImages, const BufferMapped& skyboxTransformUB)
{
	std::vector<ResourceSet> resourceSets{};
	VkDescriptorSetLayoutBinding uniformViewTranslateBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT uniformViewTranslateAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = skyboxTransformUB.getDeviceAddress(), .range = sizeof(glm::mat4) * 2 };
	VkDescriptorSetLayoutBinding cubeTextureBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo cubeTextureAddressInfo{ .sampler = cubemapImages.getSampler(), .imageView = cubemapImages.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformViewTranslateBinding, cubeTextureBinding},  {}, {{{.pUniformBuffer = &uniformViewTranslateAddressinfo}}, {{.pCombinedImageSampler = &cubeTextureAddressInfo}}} });
	return Pipeline{ assembler,
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "D:/Projects/Engine/shaders/cmpld/shader_skybox_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "D:/Projects/Engine/shaders/cmpld/shader_skybox_frag.spv"}}},
		resourceSets,
		{{PosOnlyVertex::getBindingDescription()}},
		{PosOnlyVertex::getAttributeDescriptions()} };
}

void uploadSkyboxVertexData(Buffer& skyboxData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue)
{
	float vertices[] =
	{
		-1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		-1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f
	};
	skyboxData.initialize(sizeof(vertices));
	BufferMapped staging{stagingBase, skyboxData.getSize()};
	std::memcpy(staging.getData(), vertices, skyboxData.getSize());
	
	VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };
	VkBufferCopy copy{ .srcOffset = staging.getOffset(), .dstOffset = skyboxData.getOffset(), .size = skyboxData.getSize() };
	BufferTools::cmdBufferCopy(cb, staging.getBufferHandle(), skyboxData.getBufferHandle(), 1, &copy);
	cmdBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);
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

VkSampler createUniversalSampler(VulkanObjectHandler& vulkanObjHandler)
{
	VkSampler sampler{};
	VkSamplerCreateInfo samplerCI{};
	samplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerCI.magFilter = VK_FILTER_LINEAR;
	samplerCI.minFilter = VK_FILTER_LINEAR;
	samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.anisotropyEnable = VK_TRUE;
	samplerCI.maxAnisotropy = vulkanObjHandler.getPhysDevLimits().maxSamplerAnisotropy;
	samplerCI.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerCI.unnormalizedCoordinates = VK_FALSE;
	samplerCI.compareEnable = VK_FALSE;
	samplerCI.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCI.minLod = 0.0f;
	samplerCI.maxLod = 11.0f;
	samplerCI.mipLodBias = 0.0f;
	vkCreateSampler(vulkanObjHandler.getLogicalDevice(), &samplerCI, nullptr, &sampler);
	return sampler;
}