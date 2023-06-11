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
#define GLM_FORCE_LEFT_HANDED
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
#include <tiny_gltf.h>
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
#include "src/rendering/lighting/light_types.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/data_abstraction/mesh.h"
#include "src/rendering/data_abstraction/runit.h"

#include "src/window/window.h"
#include "src/world_state_class/world_state.h"

#include "src/tools/asserter.h"
#include "src/tools/alignment.h"
#include "src/tools/texture_loader.h"
#include "src/tools/gltf_loader.h"
#include "src/tools/obj_loader.h"

#define WINDOW_WIDTH_DEFAULT  1280
#define WINDOW_HEIGHT_DEFAULT 720

#define GENERAL_BUFFER_DEFAULT_SIZE 134217728ll
#define STAGING_BUFFER_DEFAULT_SIZE 8388608ll
//#define DEVICE_BUFFER_DEFAULT_VALUE 2147483648ll

namespace fs = std::filesystem;

std::shared_ptr<VulkanObjectHandler> initializeVulkan(const Window& window);

Pipeline createForwardSimplePipeline(PipelineAssembler& assembler,
	BufferMapped& viewprojDataUB,
	BufferMapped& modelTransformDataUB,
	BufferMapped& perDrawDataIndicesSSBO,
	ImageListContainer& imageLists,
	const ImageCubeMap& skybox,
	const ImageCubeMap& radiance,
	const ImageCubeMap& irradiance,
	const Image& brdfLUT,
	VkSampler univSampler,
	BufferMapped& directionalLightSSBO,
	BufferMapped& pointLightsSSBO,
	BufferMapped& spotLightsSSBO);
Pipeline createSpaceLinesPipeline(PipelineAssembler& assembler, BufferMapped& spaceTransformDataUB);
Pipeline createSkyboxPipeline(PipelineAssembler& assembler, const ImageCubeMap& cubemapImages, const BufferMapped& skyboxTransformUB);
Pipeline createSphereBoundPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, BufferMapped& instanceData, std::span<LightTypes::PointLight> pointlights);
Pipeline createConeBoundPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, BufferMapped& instanceData, std::span<LightTypes::SpotLight> spotlights);
Pipeline createSphereTestPBRPipeline(PipelineAssembler& assembler,
	BufferMapped& viewprojDataUB, 
	BufferMapped& perInstanceDataSSBO,
	const ImageCubeMap& skybox,
	const ImageCubeMap& radiance, 
	const ImageCubeMap& irradiance, 
	VkSampler uSampler, 
	const Image& brdfLUT,
	int instCount);

uint32_t uploadLineVertices(fs::path filepath, Buffer& vertexBuffer, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue);
void uploadSkyboxVertexData(Buffer& skyboxData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue);
uint32_t uploadSphereBoundVertexData(fs::path filepath, Buffer& sphereBoundsData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue);
uint32_t uploadConeBoundVertexData(fs::path filepath, Buffer& coneBoundsData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue);
uint32_t uploadSphereVertexData(fs::path filepath, Buffer& sphereVertexData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue);



struct DepthBufferImageStuff
{
	VkImage depthImage;
	VkImageView depthImageView;
	VkDeviceMemory depthbufferMemory;
};
DepthBufferImageStuff createDepthBuffer(VkPhysicalDevice physDevice, VkDevice device);
void destroyDepthBuffer(VkDevice device, DepthBufferImageStuff depthBuffer);

VkSampler createUniversalSampler(VkDevice device, float maxAnisotropy)
{
	VkSamplerCreateInfo samplerCI{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = maxAnisotropy,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = 128.0f,
			.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE };
	VkSampler sampler{};
	vkCreateSampler(device, &samplerCI, nullptr, &sampler);
	return sampler;
}

int main()
{
	ASSERT_ALWAYS(glfwInit(), "GLFW", "GLFW was not initialized.")
	Window window(WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT, "Engine");

	std::shared_ptr<VulkanObjectHandler> vulkanObjectHandler{ initializeVulkan(window) };

	MemoryManager memManager{ *vulkanObjectHandler };
	BufferBase::assignGlobalMemoryManager(memManager);
	ImageBase::assignGlobalMemoryManager(memManager);
	
	FrameCommandPoolSet cmdPoolSet{ *vulkanObjectHandler };
	FrameCommandBufferSet cmdBufferSet{ cmdPoolSet };
	
	DescriptorManager descriptorManager{ *vulkanObjectHandler };
	ResourceSetSharedData::initializeResourceManagement(*vulkanObjectHandler, descriptorManager);

	VkDevice device{ vulkanObjectHandler->getLogicalDevice() };

	BufferBaseHostInaccessible baseDeviceBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostCachedBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, BufferBase::DEDICATED_FLAG, false, true };
	

	ImageListContainer materialTextures{ device, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, true, 
		VkSamplerCreateInfo{ 
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, 
			.magFilter = VK_FILTER_LINEAR, 
			.minFilter = VK_FILTER_LINEAR, 
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR, 
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = vulkanObjectHandler->getPhysDevLimits().maxSamplerAnisotropy,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = 128.0f,
			.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE } };

	Buffer vertexData{ baseDeviceBuffer };
	Buffer indexData{ baseDeviceBuffer };
	BufferMapped indirectCmdBuffer{ baseHostCachedBuffer };
	std::vector<StaticMesh> staticMeshes{ loadStaticMeshes(vertexData, indexData, indirectCmdBuffer, materialTextures, 
		{
			//"A:/Models/gltf/sci-fi_personal_space_pod_shipweekly_challenge/scene.gltf",
			//"A:/Models/gltf/skull_trophy/scene.gltf",
			//"A:/Models/gltf/fontaine_de_saint_michel_-_pbr/scene.gltf",
			"A:/Models/gltf/flightHelmet/FlightHelmet.gltf"
		},
		vulkanObjectHandler, descriptorManager, cmdBufferSet)
	};

	uint32_t drawCount{ static_cast<uint32_t>(indirectCmdBuffer.getSize() / sizeof(VkDrawIndexedIndirectCommand)) };

	VkSampler universalSampler{ createUniversalSampler(device, vulkanObjectHandler->getPhysDevLimits().maxSamplerAnisotropy) };
	ImageCubeMap cubemapSkybox{ TextureLoaders::loadCubemap(vulkanObjectHandler, cmdBufferSet, baseHostBuffer, "A:/materials/distant_probes_cubemaps/room/skybox/skybox.ktx2")};
	ImageCubeMap cubemapSkyboxRadiance{ TextureLoaders::loadCubemap(vulkanObjectHandler, cmdBufferSet, baseHostBuffer, "A:/materials/distant_probes_cubemaps/room/radiance/radiance.ktx2") };
	ImageCubeMap cubemapSkyboxIrradiance{ TextureLoaders::loadCubemap(vulkanObjectHandler, cmdBufferSet, baseHostBuffer, "A:/materials/distant_probes_cubemaps/room/irradiance/irradiance.ktx2") };

	Image brdfLUT{ TextureLoaders::loadImage(vulkanObjectHandler, cmdBufferSet, baseHostBuffer,
											 "A:/materials/brdfLUT/brdfLUT_MS.exr",
											 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
											 4, OIIO::TypeDesc::HALF, VK_FORMAT_R16G16B16A16_SFLOAT) };

	PipelineAssembler assembler{ device };
	
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
	BufferMapped viewprojDataUB{ baseHostBuffer, sizeof(glm::mat4) * 2 };
	BufferMapped modelTransformDataUB{ baseHostBuffer, sizeof(glm::mat4) * 8 };
	BufferMapped perDrawDataIndicesSSBO{ baseHostBuffer, sizeof(uint8_t) * 12 * drawCount };
	
	BufferMapped directionalLightSSBO{ baseHostBuffer, LightTypes::DirectionalLight::getDataByteSize() };
	LightTypes::DirectionalLight dirLight{ {1.0f, 1.0f, 1.0f}, 1.8f, {0.0f, -1.0f, 0.0f} };
	dirLight.plantData(directionalLightSSBO.getData());
	
	BufferMapped pointLightsSSBO{ baseHostBuffer, sizeof(float) * 4 + LightTypes::PointLight::getDataByteSize() * 64 };
	std::array<LightTypes::PointLight, 0> pointlights{};
	*reinterpret_cast<uint32_t*>(pointLightsSSBO.getData()) = pointlights.size();
	for (int i{0}; i < pointlights.size(); ++i)
	{
		pointlights[i].plantData((reinterpret_cast<uint8_t*>(pointLightsSSBO.getData()) + 16 + LightTypes::PointLight::getDataByteSize() * i));
	}
	
	BufferMapped spotLightsSSBO{ baseHostBuffer, sizeof(float) * 4 + LightTypes::SpotLight::getDataByteSize() * 64 };
	std::array<LightTypes::SpotLight, 0> spotlights{};
	*reinterpret_cast<uint32_t*>(spotLightsSSBO.getData()) = spotlights.size();
	for (int i{ 0 }; i < spotlights.size(); ++i)
	{
		spotlights[i].plantData((reinterpret_cast<uint8_t*>(spotLightsSSBO.getData()) + 16 + LightTypes::SpotLight::getDataByteSize() * i));
	}

	Pipeline forwardSimplePipeline{ 
		createForwardSimplePipeline(assembler, 
		viewprojDataUB, modelTransformDataUB, perDrawDataIndicesSSBO, 
		materialTextures, 
		cubemapSkybox, cubemapSkyboxRadiance, cubemapSkyboxIrradiance, brdfLUT, universalSampler, 
		directionalLightSSBO, pointLightsSSBO, spotLightsSSBO) };

	//Buffer sphereTestPBRdata{ baseDeviceBuffer };
	//uint32_t sphereTestPBRVertNum{ uploadSphereVertexData("A:/Models/obj/sphere.obj", sphereTestPBRdata, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) };
	//BufferMapped perInstPBRTestSSBO{ baseHostBuffer };
	//int sphereInstCount{ 25 };
	//Pipeline sphereTestPBRPipeline{ createSphereTestPBRPipeline(assembler, viewprojDataUB, perInstPBRTestSSBO, cubemapSkybox, cubemapSkyboxRadiance, cubemapSkyboxIrradiance, universalSampler, brdfLUT, sphereInstCount) };

	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0f, VK_CULL_MODE_NONE);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_SKYBOX);
	Buffer skyboxData{ baseDeviceBuffer };
	uploadSkyboxVertexData(skyboxData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
	BufferMapped skyboxTransformUB{ baseHostBuffer, sizeof(glm::mat4) * 2 };
	Pipeline skyboxPipeline{ createSkyboxPipeline(assembler, cubemapSkybox, skyboxTransformUB) };
	
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_LINE_POLYGONS, 1.0f, VK_CULL_MODE_NONE);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	Buffer sphereBoundVertexData{ baseDeviceBuffer };
	uint32_t sphereBoundVertNum{ uploadSphereBoundVertexData("D:/Projects/Engine/bins/sphere_vertices.bin", sphereBoundVertexData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE))};
	BufferMapped sphereBoundInstanceData{ baseHostBuffer };
	if (pointlights.size() > 0)		sphereBoundInstanceData.initialize(sizeof(glm::vec4) * 2 * pointlights.size());
	Pipeline sphereBoundPipeline{ createSphereBoundPipeline(assembler, viewprojDataUB, sphereBoundInstanceData, pointlights) };
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_TRIANGLE_FAN_DRAWING);
	Buffer coneBoundVertexData{ baseDeviceBuffer };
	uint32_t coneBoundVertNum{ uploadConeBoundVertexData("D:/Projects/Engine/bins/light_cone_vertices.bin", coneBoundVertexData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) };
	BufferMapped coneBoundInstanceData{ baseHostBuffer };
	if (spotlights.size() > 0)		coneBoundInstanceData.initialize(sizeof(glm::vec4) * 2 * spotlights.size());
	Pipeline coneBoundPipeline{ createConeBoundPipeline(assembler, viewprojDataUB, coneBoundInstanceData, spotlights) };
	
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_LINE_DRAWING);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.6f);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DEFAULT);
	Buffer spaceLinesVertexData{ baseDeviceBuffer };
	uint32_t lineVertNum{ uploadLineVertices("D:/Projects/Engine/bins/space_lines_vertices.bin", spaceLinesVertexData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) };
	Pipeline spaceLinesPipeline{ createSpaceLinesPipeline(assembler, viewprojDataUB) };



	// START: Dummy rendering test
	    
	//Resource filling 
	glm::vec3 cameraPos{ glm::vec3{0.0, 2.0, 24.0} };
	glm::vec3 viewDir{ glm::normalize(-cameraPos) };
	glm::vec3 upVec{ glm::vec3{0.0, 1.0, 0.0} };

	glm::mat4* vpMatrices{ reinterpret_cast<glm::mat4*>(viewprojDataUB.getData()) };
	vpMatrices[0] = glm::lookAt(cameraPos, viewDir, upVec);
	vpMatrices[1] = glm::infinitePerspective(glm::radians(60.0), (double)window.getWidth() / window.getHeight(), 0.1);

	glm::mat4* transformMatrices{ reinterpret_cast<glm::mat4*>(modelTransformDataUB.getData()) };
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
	glm::mat4* skyboxTransform{ reinterpret_cast<glm::mat4*>(skyboxTransformUB.getData()) };
	skyboxTransform[0] = vpMatrices[0];
	skyboxTransform[0][3] = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
	skyboxTransform[1] = vpMatrices[1];

	//uint8_t* perInstDataPtr{ reinterpret_cast<uint8_t*>(perInstPBRTestSSBO.getData()) };
	//for (int i{ 0 }; i < sphereInstCount; ++i)
	//{
	//	glm::vec4* color_metalnessData{ reinterpret_cast<glm::vec4*>(perInstDataPtr) };
	//	glm::vec4* wPos_roughnessData{ color_metalnessData + 1 };
	//
	//	int row{ i / 5 };
	//	int column{ i % 5 };
	//
	//	glm::vec3 sphereHorDistance{ 4.0, 0.0, 0.0 };
	//	glm::vec3 sphereVerDistance{ 0.0, 4.0, 0.0 };
	//
	//	glm::vec3 color{ 1.0f, 1.0f, 1.0f };
	//	float metalness{ 0.25f * row };
	//
	//	glm::vec3 position{ -8.0, -8.0, 0.0 };
	//	position += (sphereHorDistance * static_cast<float>(column) + sphereVerDistance * static_cast<float>(row));
	//	float roughness{ 0.25f * column };
	//
	//	*color_metalnessData = glm::vec4{ color, metalness };
	//	*wPos_roughnessData = glm::vec4{ position, roughness };
	//	perInstDataPtr += sizeof(glm::vec4) * 2;
	//}
	//end   


	DepthBufferImageStuff depthBuffer{ createDepthBuffer(vulkanObjectHandler->getPhysicalDevice(), vulkanObjectHandler->getLogicalDevice()) };
	
	VkSemaphore swapchainSemaphore{};
	VkSemaphore readyToPresentSemaphore{};
	VkSemaphoreCreateInfo semCI1{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	vkCreateSemaphore(device, &semCI1, nullptr, &swapchainSemaphore);
	vkCreateSemaphore(device, &semCI1, nullptr, &readyToPresentSemaphore);
	
	uint32_t swapchainIndex{};
	VkRenderingAttachmentInfo colorAttachmentInfo{};
	colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	colorAttachmentInfo.clearValue = VkClearValue{ .color{.float32{0.4f, 1.0f, 0.8f}} };
	VkRenderingAttachmentInfo depthAttachmentInfo{};
	depthAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	depthAttachmentInfo.imageView = depthBuffer.depthImageView;
	depthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
	depthAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachmentInfo.clearValue = { .depthStencil = {.depth = 1.0f, .stencil = 0} };
	std::array<VkClearAttachment, 2> attachmentClears{
		VkClearAttachment{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0, .clearValue = {.color{.float32{0.4f, 1.0f, 0.8f}}} }, 
			VkClearAttachment{.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .clearValue = {.depthStencil = {.depth = 1.0f, .stencil = 0}}}};
	std::array<VkClearRect, 2> attachmentClearRects{
		VkClearRect{.rect = {.offset = {.x = 0, .y = 0}, .extent = {.width = WINDOW_WIDTH_DEFAULT, .height = WINDOW_HEIGHT_DEFAULT}}, .baseArrayLayer = 0, .layerCount = 1}, 
			VkClearRect{.rect = {.offset = {.x = 0, .y = 0}, .extent = {.width = WINDOW_WIDTH_DEFAULT, .height = WINDOW_HEIGHT_DEFAULT}}, .baseArrayLayer = 0, .layerCount = 1}};

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

		glm::vec3 curCamPos{ glm::vec3(glm::rotate((float)angle * 0.5f, glm::vec3(0.0, 1.0, 0.0)) * glm::vec4(0.0f, 10.0f, -28.0f, 1.0f)) };
		//glm::vec3 curCamPos{ glm::vec3(0.0f, 2.0f, -26.0f) };
		vpMatrices[0] = glm::lookAt(curCamPos, glm::vec3{0.0f, 9.0f, 0.0f }, upVec);
		skyboxTransform[0] = vpMatrices[0];
		skyboxTransform[0][3] = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };

		//transformMatrices[0] = glm::translate(glm::dvec3{ 0.0f, 9.0f, 0.0f }) * glm::scale(glm::dvec3{ 0.5 });
		//transformMatrices[0] = glm::translate(glm::dvec3{ 0.0f, -10.0f, 0.0f } * 1.0) * glm::scale(glm::dvec3{ 2.3 });
		//transformMatrices[0] = glm::translate(glm::dvec3{ 5.0 } * 1.0) * glm::scale(glm::dvec3{ 0.8 });
		transformMatrices[0] = glm::translate(glm::dvec3{ 0.0f, 0.0f, 0.0f } * 1.0)* glm::scale(glm::dvec3{ 30.5 });
		transformMatrices[1] = glm::translate(glm::dvec3{ 0.0f, 10.0f, 0.0f } * mplier) * glm::scale(glm::dvec3{ 10.5 });
		transformMatrices[2] = glm::translate(glm::dvec3{ 0.0f, 0.0f, 10.0f } * mplier) * glm::scale(glm::dvec3{ 10.5 });
		
		
		ASSERT_ALWAYS(vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex) == VK_SUCCESS, "Vulkan", "Could not acquire swapchain image.");
		swapchainImageData = vulkanObjectHandler->getSwapchainImageData(swapchainIndex);
		colorAttachmentInfo.imageView = std::get<1>(swapchainImageData);

		//Begin recording
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
			renderInfo.pColorAttachments = &colorAttachmentInfo;
				

			vkCmdBeginRendering(CBloop, &renderInfo);

				vkCmdClearAttachments(CBloop, attachmentClears.size(), attachmentClears.data(), attachmentClearRects.size(), attachmentClearRects.data());
			
				descriptorManager.cmdSubmitPipelineResources(CBloop, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardSimplePipeline.getResourceSets(), forwardSimplePipeline.getResourceSetsInUse(), forwardSimplePipeline.getPipelineLayoutHandle());
				VkBuffer vertexBindings[1]{ vertexData.getBufferHandle() };
				VkDeviceSize vertexBindingOffsets[1]{ vertexData.getOffset()};
				vkCmdBindVertexBuffers(CBloop, 0, 1, vertexBindings, vertexBindingOffsets);
				vkCmdBindIndexBuffer(CBloop, indexData.getBufferHandle(), indexData.getOffset(), VK_INDEX_TYPE_UINT32);
				forwardSimplePipeline.cmdBind(CBloop);
				vkCmdPushConstants(CBloop, forwardSimplePipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &curCamPos);
				vkCmdDrawIndexedIndirect(CBloop, indirectCmdBuffer.getBufferHandle(), indirectCmdBuffer.getOffset(), drawCount, sizeof(VkDrawIndexedIndirectCommand));
				
				if (pointlights.size() > 0)
				{
					VkBuffer sphereBoundsVertexBindings[2]{ sphereBoundVertexData.getBufferHandle(), sphereBoundInstanceData.getBufferHandle() };
					VkDeviceSize sphereBoundsVertexBindingOffsets[2]{ sphereBoundVertexData.getOffset(), sphereBoundInstanceData.getOffset() };
					vkCmdBindVertexBuffers(CBloop, 0, 2, sphereBoundsVertexBindings, sphereBoundsVertexBindingOffsets);
					sphereBoundPipeline.cmdBind(CBloop);
					vkCmdDraw(CBloop, sphereBoundVertNum, pointlights.size(), 0, 0);
				}
				if (spotlights.size() > 0)
				{
					VkBuffer coneBoundsVertexBindings[2]{ coneBoundVertexData.getBufferHandle(), coneBoundInstanceData.getBufferHandle() };
					VkDeviceSize coneBoundsVertexBindingOffsets[2]{ coneBoundVertexData.getOffset(), coneBoundInstanceData.getOffset() };
					vkCmdBindVertexBuffers(CBloop, 0, 2, coneBoundsVertexBindings, coneBoundsVertexBindingOffsets);
					coneBoundPipeline.cmdBind(CBloop);
					vkCmdDraw(CBloop, coneBoundVertNum, spotlights.size(), 0, 0);
				}

				descriptorManager.cmdSubmitPipelineResources(CBloop, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline.getResourceSets(), skyboxPipeline.getResourceSetsInUse(), skyboxPipeline.getPipelineLayoutHandle());
				VkBuffer skyboxVertexBinding[1]{ skyboxData.getBufferHandle() };
				VkDeviceSize skyboxVertexOffsets[1]{ skyboxData.getOffset() };
				vkCmdBindVertexBuffers(CBloop, 0, 1, skyboxVertexBinding, skyboxVertexOffsets);
				skyboxPipeline.cmdBind(CBloop);
				vkCmdDraw(CBloop, 36, 1, 0, 0);

				/*descriptorManager.cmdSubmitPipelineResources(CBloop, VK_PIPELINE_BIND_POINT_GRAPHICS, sphereTestPBRPipeline.getResourceSets(), sphereTestPBRPipeline.getResourceSetsInUse(), sphereTestPBRPipeline.getPipelineLayoutHandle());
				VkBuffer sphereTestPBRVertexBinding[1]{ sphereTestPBRdata.getBufferHandle() };
				VkDeviceSize sphereTestPBRVertexOffsets[1]{ sphereTestPBRdata.getOffset() };
				vkCmdBindVertexBuffers(CBloop, 0, 1, sphereTestPBRVertexBinding, sphereTestPBRVertexOffsets);
				sphereTestPBRPipeline.cmdBind(CBloop);
				vkCmdPushConstants(CBloop, sphereTestPBRPipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &curCamPos);
				vkCmdDraw(CBloop, sphereTestPBRVertNum, sphereInstCount, 0, 0);*/


			vkCmdEndRendering(CBloop);
			
			BarrierOperations::cmdExecuteBarrier(CBloop, std::span<const VkMemoryBarrier2>{
				{BarrierOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
					VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT)}
			});
			
			vkCmdBeginRendering(CBloop, &renderInfo);

				descriptorManager.cmdSubmitPipelineResources(CBloop, VK_PIPELINE_BIND_POINT_GRAPHICS, spaceLinesPipeline.getResourceSets(), spaceLinesPipeline.getResourceSetsInUse(), spaceLinesPipeline.getPipelineLayoutHandle());
				VkBuffer lineVertexBindings[1]{ spaceLinesVertexData.getBufferHandle() };
				VkDeviceSize lineVertexBindingOffsets[1]{ spaceLinesVertexData.getOffset() };
				vkCmdBindVertexBuffers(CBloop, 0, 1, lineVertexBindings, lineVertexBindingOffsets); 
				spaceLinesPipeline.cmdBind(CBloop);
				vkCmdPushConstants(CBloop, spaceLinesPipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &curCamPos);
				vkCmdDraw(CBloop, lineVertNum, 1, 0, 0);

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
		//End recording

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
		cmdBufferSet.resetBuffers();
		
		glfwPollEvents();
	}
	// END: Dummy rendering test

	ASSERT_ALWAYS(vkDeviceWaitIdle(device) == VK_SUCCESS, "Vulkan", "Device wait failed.");
	destroyDepthBuffer(device, depthBuffer);
	vkDestroySampler(device, universalSampler, nullptr);
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

Pipeline createForwardSimplePipeline(PipelineAssembler& assembler,
	BufferMapped& viewprojDataUB,
	BufferMapped& modelTransformDataUB,
	BufferMapped& perDrawDataIndicesSSBO,
	ImageListContainer& imageLists,
	const ImageCubeMap& skybox, 
	const ImageCubeMap& radiance, 
	const ImageCubeMap& irradiance,
	const Image& brdfLUT,
	VkSampler univSampler,
	BufferMapped& directionalLightSSBO,
	BufferMapped& pointLightsSSBO,
	BufferMapped& spotLightsSSBO)
{
	VkDescriptorSetLayoutBinding viewprojBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT viewprojAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = viewprojDataUB.getSize() };

	VkDescriptorSetLayoutBinding imageListsBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 64, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	std::vector<VkDescriptorImageInfo> storageImageData(imageLists.getImageListCount());
	std::vector<VkDescriptorDataEXT> imageListsDescData(imageLists.getImageListCount());
	for (uint32_t i{ 0 }; i < imageListsDescData.size(); ++i)
	{
		storageImageData[i] = { .sampler = imageLists.getSampler(), .imageView = imageLists.getImageViewHandle(i), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
		imageListsDescData[i].pStorageImage = &storageImageData[i];
	}
	VkDescriptorSetLayoutBinding modelTransformBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT modelTransformAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = modelTransformDataUB.getDeviceAddress(), .range = modelTransformDataUB.getSize() };

	VkDescriptorSetLayoutBinding uniformDrawIndicesBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT uniformDrawIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = perDrawDataIndicesSSBO.getDeviceAddress(), .range = perDrawDataIndicesSSBO.getSize() };

	VkDescriptorSetLayoutBinding directionalLightBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT directionalLightAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = directionalLightSSBO.getDeviceAddress(), .range = directionalLightSSBO.getSize() };
	VkDescriptorSetLayoutBinding pointLightsBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT pointLightsAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = pointLightsSSBO.getDeviceAddress(), .range = pointLightsSSBO.getSize() };
	VkDescriptorSetLayoutBinding spotLightsBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT spotLightsAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = spotLightsSSBO.getDeviceAddress(), .range = spotLightsSSBO.getSize() };

	VkDescriptorSetLayoutBinding skyboxBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo skyboxAddressInfo{ .sampler = skybox.getSampler(), .imageView = skybox.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding radianceBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo radianceAddressInfo{ .sampler = radiance.getSampler(), .imageView = radiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding irradianceBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo irradianceAddressInfo{ .sampler = irradiance.getSampler(), .imageView = irradiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding lutBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo lutAddressInfo{ .sampler = univSampler, .imageView = brdfLUT.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	std::vector<ResourceSet> resourceSets{};
	VkDevice device{ assembler.getDevice() };
	resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, 1, 
		{viewprojBinding},  {}, 
			{{{.pUniformBuffer = &viewprojAddressInfo}}} });
	resourceSets.push_back({ device, 1, VkDescriptorSetLayoutCreateFlags{}, 1, 
		{imageListsBinding, modelTransformBinding}, {{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }, {}}, 
			{imageListsDescData, {{.pUniformBuffer = &modelTransformAddressInfo}}} });
	resourceSets.push_back({ device, 2, VkDescriptorSetLayoutCreateFlags{}, 1, 
		{uniformDrawIndicesBinding}, {}, 
			{{{.pStorageBuffer = &uniformDrawIndicesAddressInfo}}} });
	resourceSets.push_back({ device, 3, VkDescriptorSetLayoutCreateFlags{}, 1, 
		{directionalLightBinding, pointLightsBinding, spotLightsBinding}, {}, 
			{{{.pUniformBuffer = &directionalLightAddressInfo}}, {{.pUniformBuffer = &pointLightsAddressInfo}}, {{.pUniformBuffer = &spotLightsAddressInfo}}} });
	resourceSets.push_back({ device, 4, VkDescriptorSetLayoutCreateFlags{}, 1,
		{skyboxBinding, radianceBinding, irradianceBinding, lutBinding}, {}, 
			{{{.pCombinedImageSampler = &skyboxAddressInfo}}, {{.pCombinedImageSampler = &radianceAddressInfo}}, {{.pCombinedImageSampler = &irradianceAddressInfo}}, {{.pCombinedImageSampler = &lutAddressInfo}}} });

	return 	Pipeline{ assembler, 
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "D:/Projects/Engine/shaders/cmpld/shader_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "D:/Projects/Engine/shaders/cmpld/shader_frag.spv"}}}, 
		resourceSets, 
		{{StaticVertex::getBindingDescription()}}, 
		{StaticVertex::getAttributeDescriptions()}, 
		{{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(glm::vec3)}}} };
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

Pipeline createSpaceLinesPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB)
{
	std::vector<ResourceSet> resourceSets{};
	VkDescriptorSetLayoutBinding uniformViewProjBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT uniformViewProjAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = sizeof(glm::mat4) * 2};
	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformViewProjBinding},  {}, {{{.pUniformBuffer = &uniformViewProjAddressinfo}}}});
	return Pipeline{ assembler, 
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "D:/Projects/Engine/shaders/cmpld/shader_space_lines_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "D:/Projects/Engine/shaders/cmpld/shader_space_lines_frag.spv"}}},
		resourceSets,
		{{PosColorVertex::getBindingDescription()}},
		{PosColorVertex::getAttributeDescriptions()},
		{{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(glm::vec3)}}} };
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

Pipeline createSphereBoundPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, BufferMapped& instanceData, std::span<LightTypes::PointLight> pointlights)
{
	std::vector<ResourceSet> resourceSets{};
	VkDescriptorSetLayoutBinding uniformViewProjBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT uniformViewProjAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = sizeof(glm::mat4) * 2 };
	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformViewProjBinding},  {}, {{{.pUniformBuffer = &uniformViewProjAddressinfo}}} });

	glm::vec4* instanceDataPtr{ reinterpret_cast<glm::vec4*>(instanceData.getData()) };
	for (auto& light : pointlights)
	{
		*(instanceDataPtr++) = glm::vec4{light.getPosition(), light.getRadius()};
	}

	std::array<VkVertexInputBindingDescription, PosOnlyVertex::getBindingDescriptionCount() + 1> bindingDescriptions{};
	bindingDescriptions[0] = PosOnlyVertex::getBindingDescription();
	bindingDescriptions[1] = VkVertexInputBindingDescription{ .binding = 1, .stride = sizeof(glm::vec4), .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE };

	std::array<VkVertexInputAttributeDescription, PosOnlyVertex::getAttributeDescriptionCount() + 1> attributeDescriptions{};
	std::copy_n(PosOnlyVertex::getAttributeDescriptions().begin(), PosOnlyVertex::getAttributeDescriptionCount(), attributeDescriptions.begin());
	attributeDescriptions[PosOnlyVertex::getAttributeDescriptionCount()] = VkVertexInputAttributeDescription{ .location = 1, .binding = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 0 };

	return Pipeline{ assembler,
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "D:/Projects/Engine/shaders/cmpld/light_sphere_bounds_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "D:/Projects/Engine/shaders/cmpld/light_sphere_bounds_frag.spv"}}},
		resourceSets,
		{bindingDescriptions},
		{attributeDescriptions} };
}
uint32_t uploadSphereBoundVertexData(fs::path filepath, Buffer& sphereBoundsData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue)
{
	std::ifstream istream{ filepath, std::ios::binary };
	istream.seekg(0, std::ios::beg);

	uint32_t vertNum{};
	istream.read(reinterpret_cast<char*>(&vertNum), sizeof(vertNum));
	istream.seekg(sizeof(vertNum), std::ios::beg);

	uint32_t dataSize{ vertNum * sizeof(glm::vec3) };

	sphereBoundsData.initialize(dataSize);
	BufferMapped staging{ stagingBase, dataSize };

	istream.read(reinterpret_cast<char*>(staging.getData()), dataSize);

	VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };

	VkBufferCopy copy{ .srcOffset = staging.getOffset(), .dstOffset = sphereBoundsData.getOffset(), .size = dataSize };
	BufferTools::cmdBufferCopy(cb, staging.getBufferHandle(), sphereBoundsData.getBufferHandle(), 1, &copy);

	cmdBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	return vertNum;
}

Pipeline createConeBoundPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, BufferMapped& instanceData, std::span<LightTypes::SpotLight> spotlights)
{
	std::vector<ResourceSet> resourceSets{};
	VkDescriptorSetLayoutBinding uniformViewProjBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT uniformViewProjAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = sizeof(glm::mat4) * 2 };
	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformViewProjBinding}, {}, {{{.pUniformBuffer = &uniformViewProjAddressinfo}}} });
	
	glm::vec4* instanceDataPtr{ reinterpret_cast<glm::vec4*>(instanceData.getData()) };
	for (auto& light : spotlights)
	{
		*(instanceDataPtr++) = glm::vec4{light.getDirection(), light.getLength()};
		*(instanceDataPtr++) = glm::vec4{light.getPosition(), light.getAngle()};
	}

	std::array<VkVertexInputBindingDescription, PosOnlyVertex::getBindingDescriptionCount() + 1> bindingDescriptions{};
	bindingDescriptions[0] = PosOnlyVertex::getBindingDescription();
	bindingDescriptions[1] = VkVertexInputBindingDescription{ .binding = 1, .stride = sizeof(glm::vec4) * 2, .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE };

	std::array<VkVertexInputAttributeDescription, PosOnlyVertex::getAttributeDescriptionCount() + 2> attributeDescriptions{};
	std::copy_n(PosOnlyVertex::getAttributeDescriptions().begin(), PosOnlyVertex::getAttributeDescriptionCount(), attributeDescriptions.begin());
	attributeDescriptions[PosOnlyVertex::getAttributeDescriptionCount()] = VkVertexInputAttributeDescription{ .location = 1, .binding = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = 0 };
	attributeDescriptions[PosOnlyVertex::getAttributeDescriptionCount() + 1] = VkVertexInputAttributeDescription{ .location = 2, .binding = 1, .format = VK_FORMAT_R32G32B32A32_SFLOAT, .offset = sizeof(glm::vec4) };

	return Pipeline{ assembler,
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "D:/Projects/Engine/shaders/cmpld/spotlight_cone_bounds_vert.spv"}, 
		ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "D:/Projects/Engine/shaders/cmpld/spotlight_cone_bounds_frag.spv"}}},
		resourceSets,
		{bindingDescriptions},
		{attributeDescriptions} };
}
uint32_t uploadConeBoundVertexData(fs::path filepath, Buffer& coneBoundsData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue)
{
	std::ifstream istream{ filepath, std::ios::binary };
	istream.seekg(0, std::ios::beg);

	uint32_t vertNum{};
	istream.read(reinterpret_cast<char*>(&vertNum), sizeof(vertNum));
	istream.seekg(sizeof(vertNum), std::ios::beg);

	uint32_t dataSize{ vertNum * sizeof(glm::vec3) };

	coneBoundsData.initialize(dataSize);
	BufferMapped staging{ stagingBase, dataSize };

	istream.read(reinterpret_cast<char*>(staging.getData()), dataSize);

	VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };

	VkBufferCopy copy{ .srcOffset = staging.getOffset(), .dstOffset = coneBoundsData.getOffset(), .size = dataSize };
	BufferTools::cmdBufferCopy(cb, staging.getBufferHandle(), coneBoundsData.getBufferHandle(), 1, &copy);

	cmdBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	return vertNum;
}

Pipeline createSphereTestPBRPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, BufferMapped& perInstanceDataSSBO, const ImageCubeMap& skybox, const ImageCubeMap& radiance, const ImageCubeMap& irradiance, VkSampler univSampler, const Image& brdfLUT, int instCount)
{
	std::vector<ResourceSet> resourceSets{};
	VkDescriptorSetLayoutBinding uniformViewProjBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT uniformViewProjAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = sizeof(glm::mat4) * 2 };

	perInstanceDataSSBO.initialize(sizeof(glm::vec4) * 2 * instCount);
	VkDescriptorSetLayoutBinding uniformInstDataBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT uniformInstDataAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = perInstanceDataSSBO.getDeviceAddress(), .range = sizeof(glm::vec4) * 2 * instCount };

	VkDescriptorSetLayoutBinding skyboxBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo skyboxAddressInfo{ .sampler = skybox.getSampler(), .imageView = skybox.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	VkDescriptorSetLayoutBinding radianceBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo radianceAddressInfo{ .sampler = radiance.getSampler(), .imageView = radiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	VkDescriptorSetLayoutBinding irradianceBinding{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo irradianceAddressInfo{ .sampler = irradiance.getSampler(), .imageView = irradiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	VkDescriptorSetLayoutBinding lutBinding{ .binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo lutAddressInfo{ .sampler = univSampler, .imageView = brdfLUT.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1, 
		{uniformViewProjBinding, 
			uniformInstDataBinding, 
				skyboxBinding,
					radianceBinding,
						irradianceBinding,
							lutBinding},
		{}, 
		{{{.pUniformBuffer = &uniformViewProjAddressinfo}}, 
			{{.pStorageBuffer = &uniformInstDataAddressinfo}}, 
				{{.pCombinedImageSampler = &skyboxAddressInfo}},
					{{.pCombinedImageSampler = &radianceAddressInfo}},
						{{.pCombinedImageSampler = &irradianceAddressInfo}},
							{{.pCombinedImageSampler = &lutAddressInfo}}} });

	return Pipeline{ assembler,
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "D:/Projects/Engine/shaders/cmpld/pbr_spheres_test_vert.spv"},
		ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "D:/Projects/Engine/shaders/cmpld/pbr_spheres_test_frag.spv"}}},
		resourceSets,
		{{PosTexVertex::getBindingDescription()}},
		{PosTexVertex::getAttributeDescriptions()},
		{{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(glm::vec3)}}} };
}
uint32_t uploadSphereVertexData(fs::path filepath, Buffer& sphereVertexData, BufferBaseHostAccessible& stagingBase, FrameCommandBufferSet& cmdBufferSet, VkQueue queue)
{
	BufferMapped staging{ stagingBase };

	LoaderOBJ::loadOBJfile(filepath, staging, LoaderOBJ::POS_VERT | LoaderOBJ::TEXC_VERT);

	sphereVertexData.initialize(staging.getSize());

	VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };

	VkBufferCopy copy{ .srcOffset = staging.getOffset(), .dstOffset = sphereVertexData.getOffset(), .size = staging.getSize() };
	BufferTools::cmdBufferCopy(cb, staging.getBufferHandle(), sphereVertexData.getBufferHandle(), 1, &copy);

	cmdBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	return staging.getSize() / (5 * sizeof(float));
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