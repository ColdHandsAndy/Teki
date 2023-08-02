#include <iostream>
#include <vector>
#include <span>
#include <tuple>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <bitset>
#include <intrin.h>

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
#include <glm/gtx/rotate_vector.hpp>

#define WINDOW_WIDTH_DEFAULT  1280
#define WINDOW_HEIGHT_DEFAULT 720
#define HBAO_WIDTH_DEFAULT  1280
#define HBAO_HEIGHT_DEFAULT 720

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/shader_management/shader_operations.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/data_management/memory_manager.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/renderer/barrier_operations.h"
#include "src/rendering/renderer/timeline_semaphore.h"
#include "src/rendering/lighting/light_types.h"
#include "src/rendering/renderer/clusterer.h"
#include "src/rendering/renderer/culling.h"
#include "src/rendering/scene/parse_scene.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/data_abstraction/mesh.h"
#include "src/rendering/data_abstraction/runit.h"
#include "src/rendering/renderer/depth_buffer.h"
#include "src/rendering/renderer/HBAO.h"
#include "src/rendering/renderer/FXAA.h"
#include "src/rendering/data_abstraction/BB.h"
#include "src/rendering/UI/UI.h"

#include "src/window/window.h"
#include "src/world_state/world_state.h"

#include "src/tools/tools.h"

#define GENERAL_BUFFER_DEFAULT_SIZE 134217728ll
#define SHARED_BUFFER_DEFAULT_SIZE 8388608ll
#define DEVICE_BUFFER_DEFAULT_SIZE  268435456ll

namespace fs = std::filesystem;

struct CamInfo
{
	//glm::vec3 camPos{ 9.305, 3.871, 0.228 };
	//glm::vec3 camFront{ -0.997, -0.077, 0.02 };
	glm::vec3 camPos{ 0.0, 0.0, -10.0 };
	glm::vec3 camFront{ 0.0, 0.0, 1.0 };
	double speed{ 10.0 };
	double sensetivity{ 1.9 };
	glm::vec2 lastCursorP{};
} camInfo;

struct InputAccessedData
{
	CamInfo* cameraInfo{ &camInfo };
} inputAccessedData;

std::shared_ptr<VulkanObjectHandler> initializeVulkan(const Window& window);

glm::mat4 getProjection(float FOV, float aspect, float zNear, float zFar);

Pipeline createForwardClusteredPipeline(PipelineAssembler& assembler,
	const BufferMapped& viewprojDataUB,
	const BufferMapped& modelTransformDataUB,
	const BufferMapped& drawDataSSBO,
	const Buffer& drawDataIndicesSSBO,
	const ImageListContainer& imageLists,
	const ImageCubeMap& skybox,
	const ImageCubeMap& radiance,
	const ImageCubeMap& irradiance,
	const Image& brdfLUT,
	const Image& imageAO,
	VkSampler univSampler,
	const BufferMapped& directionalLightUB,
	const BufferMapped& sortedLightsDataUB,
	const BufferBaseHostAccessible& typeDataUB,
	const BufferBaseHostInaccessible& tileDataSSBO,
	const BufferMapped& zBinDataUB);
Pipeline createPointBVPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, const BufferMapped& instancePointLightIndexData, const BufferMapped& sortedLightsDataUB);
Pipeline createSpotBVPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, const BufferMapped& instanceSpotLightIndexData, const BufferMapped& sortedLightsDataUB);
Pipeline createSpaceLinesPipeline(PipelineAssembler& assembler, BufferMapped& spaceTransformDataUB);
Pipeline createSkyboxPipeline(PipelineAssembler& assembler, const ImageCubeMap& cubemapImages, const BufferMapped& skyboxTransformUB);
Pipeline createSphereTestPBRPipeline(PipelineAssembler& assembler,
	BufferMapped& viewprojDataUB, 
	BufferMapped& perInstanceDataSSBO,
	const ImageCubeMap& skybox,
	const ImageCubeMap& radiance, 
	const ImageCubeMap& irradiance, 
	VkSampler uSampler, 
	const Image& brdfLUT,
	int instCount);

uint32_t uploadLineVertices(fs::path filepath, Buffer& vertexBuffer, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue);
void uploadSkyboxVertexData(Buffer& skyboxData, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue);
uint32_t uploadSphereVertexData(fs::path filepath, Buffer& sphereVertexData, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue);

void loadDefaultTextures(ImageListContainer& imageLists, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue);
void transformOBBs(OBBs& boundingBoxes, std::vector<StaticMesh>& staticMeshes, int drawCount, const std::vector<glm::mat4>& modelMatrices);
void getBoundingSpheres(BufferMapped& indirectDataBuffer, const OBBs& boundingBoxes);

void fillFrustumData(glm::mat4* vpMatrices, Window& window, Clusterer& clusterer, HBAO& hbao, FrustumInfo& frustumInfo);
void fillModelMatrices(const BufferMapped& modelTransformDataUB, const std::vector<glm::mat4>& modelMatrices);
void fillDrawDataIndices(const BufferMapped& perDrawDataIndicesSSBO, std::vector<StaticMesh>& staticMeshes, int drawCount);
void fillSkyboxTransformData(glm::mat4* vpMatrices, glm::mat4* skyboxTransform);
void fillPBRSphereTestInstanceData(const BufferMapped& perInstPBRTestSSBO, int sphereInstCount);

void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

VkSampler createUniversalSampler(VkDevice device, float maxAnisotropy);

int main()
{
	EASSERT(glfwInit(), "GLFW", "GLFW was not initialized.")

	Window window{ "Teki" };
	//Window window{ WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT, "Teki" };
	glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GLFW_TRUE);
	glfwSetCursorPosCallback(window, mouseCallback);
	glfwSetKeyCallback(window, keyCallback);
	glfwSetWindowUserPointer(window, &inputAccessedData);

	std::shared_ptr<VulkanObjectHandler> vulkanObjectHandler{ initializeVulkan(window) };

	MemoryManager memManager{ *vulkanObjectHandler };
	BufferBase::assignGlobalMemoryManager(memManager);
	ImageBase::assignGlobalMemoryManager(memManager);
	
	CommandBufferSet cmdBufferSet{ *vulkanObjectHandler };

	UI ui{ window, *vulkanObjectHandler, cmdBufferSet };
	
	DescriptorManager descriptorManager{ *vulkanObjectHandler };
	ResourceSetSharedData::initializeResourceManagement(*vulkanObjectHandler, descriptorManager);

	VkDevice device{ vulkanObjectHandler->getLogicalDevice() };

	BufferBaseHostInaccessible baseDeviceBuffer{ device, DEVICE_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostCachedBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, BufferBase::DEDICATED_FLAG, false, true };
	BufferBaseHostAccessible baseSharedBuffer{ device, 8388608ll,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, BufferBase::NULL_FLAG, true };

	BufferMapped viewprojDataUB{ baseHostBuffer, sizeof(glm::mat4) * 2 };

	Clusterer clusterer{ device, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), window.getWidth(), window.getHeight(), viewprojDataUB};
	LightTypes::LightBase::assignGlobalClusterer(clusterer);

	HBAO hbao{ device, HBAO_WIDTH_DEFAULT, HBAO_HEIGHT_DEFAULT };

	Image framebufferImage{ device, vulkanObjectHandler->getSwapchainFormat(), window.getWidth(), window.getHeight(), VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT };
	FXAA fxaa{ device, window.getWidth(), window.getHeight(), framebufferImage };

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
	loadDefaultTextures(materialTextures, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));

	Buffer vertexData{ baseDeviceBuffer };
	Buffer indexData{ baseDeviceBuffer };
	BufferMapped indirectDataBuffer{ baseHostCachedBuffer };
	
	std::vector<fs::path> modelPaths{};
	std::vector<glm::mat4> modelMatrices{};
	fs::path envPath{};
	Scene::parseSceneData("internal/scene_info.json", modelPaths, modelMatrices, envPath);

	FrustumInfo frustumInfo{};

	OBBs rUnitOBBs{ 256 };
	uint32_t drawCount{};

	std::vector<StaticMesh> staticMeshes{ loadStaticMeshes(vertexData, indexData, 
		indirectDataBuffer, drawCount,
		rUnitOBBs,
		materialTextures, 
		modelPaths,
		vulkanObjectHandler, descriptorManager, cmdBufferSet)
	};

	transformOBBs(rUnitOBBs, staticMeshes, drawCount, modelMatrices);
	getBoundingSpheres(indirectDataBuffer, rUnitOBBs);

	cmdBufferSet.resetPool(CommandBufferSet::TRANSIENT_POOL);

	VkSampler universalSampler{ createUniversalSampler(device, vulkanObjectHandler->getPhysDevLimits().maxSamplerAnisotropy) };
	ImageCubeMap cubemapSkybox{ TextureLoaders::loadCubemap(vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "skybox/skybox.ktx2") };
	ImageCubeMap cubemapSkyboxRadiance{ TextureLoaders::loadCubemap(vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "radiance/radiance.ktx2") };
	ImageCubeMap cubemapSkyboxIrradiance{ TextureLoaders::loadCubemap(vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "irradiance/irradiance.ktx2") };

	Image brdfLUT{ TextureLoaders::loadImage(vulkanObjectHandler, cmdBufferSet, baseHostBuffer,
											 "internal/brdfLUT/brdfLUT.exr",
											 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
											 4, OIIO::TypeDesc::HALF, VK_FORMAT_R16G16B16A16_SFLOAT) };

	DepthBuffer depthBuffer{ device, window.getWidth(), window.getHeight() };

	Culling culling{ device, drawCount, 0.01, viewprojDataUB, indirectDataBuffer, depthBuffer, vulkanObjectHandler->getComputeFamilyIndex(), vulkanObjectHandler->getGraphicsFamilyIndex() };

	cmdBufferSet.resetPool(CommandBufferSet::TRANSIENT_POOL);

	struct
	{
		bool drawBVs{ false };
		bool drawSpaceLines{ false };
		int hiZmipLevel{ 0 };
		uint32_t frustumCulledCount{ 0 };
		BufferMapped finalDrawCount;
	} renderingData;
	renderingData.finalDrawCount.initialize(baseHostBuffer, sizeof(uint32_t));

	/*std::random_device dev{};
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> dist{};

	std::vector<LightTypes::PointLight> points{};
	int pointLightsX{ 24 };
	int pointLightsY{ 8 };
	for (int i{ 0 }; i < pointLightsX * pointLightsY; ++i)
	{
		points.emplace_back(glm::vec3{0.0}, glm::vec3{ dist(rng), dist(rng), dist(rng) }, 22.0, 1.0);
	}
	std::vector<LightTypes::SpotLight> spots{};
	float spotLightAngle{ glm::radians(30.0) };
	glm::vec3 spotsLeftDir{0.0, -std::cos(spotLightAngle), std::sin(spotLightAngle)};
	glm::vec3 spotsRightDir{0.0, -std::cos(spotLightAngle), -std::sin(spotLightAngle)};
	glm::vec3 spotsLeftRotDir{0.0, -std::cos(spotLightAngle * 0.5), std::sin(spotLightAngle * 0.5)};
	glm::vec3 spotsRightRotDir{0.0, -std::cos(spotLightAngle * 0.5), -std::sin(spotLightAngle * 0.5)};
	for (int i{ 0 }; i < 4; ++i)
	{
		spots.emplace_back(glm::vec3{8.0f * i - 8.0f * i / 2.0f, 8.0, -4.0}, glm::vec3{ dist(rng), dist(rng), dist(rng) }, 900.0, 10.0, spotsLeftDir, glm::radians(9.0), glm::radians(19.0));
		spots.emplace_back(glm::vec3{8.0f * i - 8.0f * i / 2.0f, 8.0, -10.0}, glm::vec3{ dist(rng), dist(rng), dist(rng) }, 900.0, 10.0, spotsRightDir, glm::radians(9.0), glm::radians(19.0));
	}*/
	clusterer.submitPointLight(glm::vec3{0.5f, 3.2f, -2.0f}, glm::vec3{0.8f, 0.4f, 0.2f}, 80.0f, 6.0f);



	PipelineAssembler assembler{ device };
	
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, window.getWidth(), window.getHeight());
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DEFAULT);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);
	BufferMapped modelTransformDataUB{ baseHostBuffer, sizeof(glm::mat4) * modelMatrices.size() };
	BufferMapped drawDataSSBO{ baseHostBuffer, sizeof(uint8_t) * 12 * drawCount };
	
	BufferMapped directionalLightUB{ baseHostBuffer, LightTypes::DirectionalLight::getDataByteSize() };
	LightTypes::DirectionalLight dirLight{ {1.0f, 1.0f, 1.0f}, 0.0f, {0.0f, -1.0f, 0.0f} };
	dirLight.plantData(directionalLightUB.getData());

	Pipeline forwardClusteredPipeline( 
		createForwardClusteredPipeline(assembler, 
			viewprojDataUB, modelTransformDataUB, drawDataSSBO, culling.getDrawDataIndexBuffer(),
			materialTextures,
			cubemapSkybox, cubemapSkyboxRadiance, cubemapSkyboxIrradiance, brdfLUT, hbao.getAO(), universalSampler,
			directionalLightUB,
			clusterer.getSortedLightsUB(), clusterer.getSortedTypeDataUB(), clusterer.getTileDataSSBO(), clusterer.getZBinUB())
	);

	/*Buffer sphereTestPBRdata{ baseDeviceBuffer };
	uint32_t sphereTestPBRVertNum{ uploadSphereVertexData("A:/Models/obj/sphere.obj", sphereTestPBRdata, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) };
	BufferMapped perInstPBRTestSSBO{ baseHostBuffer };
	int sphereInstCount{ 25 };
	Pipeline sphereTestPBRPipeline{ createSphereTestPBRPipeline(assembler, viewprojDataUB, perInstPBRTestSSBO, cubemapSkybox, cubemapSkyboxRadiance, cubemapSkyboxIrradiance, universalSampler, brdfLUT, sphereInstCount) };*/

	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_LINE_POLYGONS, 1.4f, VK_CULL_MODE_NONE);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);
	Pipeline pointBVPipeline{ createPointBVPipeline(assembler, viewprojDataUB, clusterer.getPointIndicesUB(), clusterer.getSortedLightsUB())};
	Pipeline spotBVPipeline{ createSpotBVPipeline(assembler, viewprojDataUB, clusterer.getSpotIndicesUB(), clusterer.getSortedLightsUB())};

	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0f, VK_CULL_MODE_NONE);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_SKYBOX);
	Buffer skyboxData{ baseDeviceBuffer };
	uploadSkyboxVertexData(skyboxData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
	BufferMapped skyboxTransformUB{ baseHostBuffer, sizeof(glm::mat4) * 2 };
	Pipeline skyboxPipeline{ createSkyboxPipeline(assembler, cubemapSkybox, skyboxTransformUB) };
	
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_LINE_DRAWING);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.5f);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DEFAULT);
	Buffer spaceLinesVertexData{ baseDeviceBuffer };
	uint32_t lineVertNum{ uploadLineVertices("internal/spaceLinesMesh/space_lines_vertices.bin", spaceLinesVertexData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) };
	Pipeline spaceLinesPipeline{ createSpaceLinesPipeline(assembler, viewprojDataUB) };

	hbao.acquireDepthPassData(modelTransformDataUB, drawDataSSBO);
	hbao.fiilRandomRotationImage(cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));

	//Resource filling 
	glm::mat4* vpMatrices{ reinterpret_cast<glm::mat4*>(viewprojDataUB.getData()) };
	glm::mat4* skyboxTransform{ reinterpret_cast<glm::mat4*>(skyboxTransformUB.getData()) };
	fillFrustumData(vpMatrices, window, clusterer, hbao, frustumInfo);
	fillModelMatrices(modelTransformDataUB, modelMatrices);
	fillDrawDataIndices(drawDataSSBO, staticMeshes, drawCount);
	fillSkyboxTransformData(vpMatrices, skyboxTransform);
	//fillPBRSphereTestInstanceData(perInstPBRTestSSBO, sphereInstCount);
	//end   

	TimelineSemaphore semaphore{ device };
	TimelineSemaphore semaphoreHiZ{ device };
	VkSemaphore swapchainSemaphore{};
	VkSemaphore readyToPresentSemaphore{};
	VkSemaphoreCreateInfo semCI{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	vkCreateSemaphore(device, &semCI, nullptr, &swapchainSemaphore);
	vkCreateSemaphore(device, &semCI, nullptr, &readyToPresentSemaphore);
	
	uint32_t swapchainIndex{};
	VkRenderingAttachmentInfo colorAttachmentInfo{};
	colorAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	colorAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
	colorAttachmentInfo.clearValue = VkClearValue{ .color{.float32{0.4f, 1.0f, 0.8f}} };
	VkRenderingAttachmentInfo depthAttachmentInfo{};
	depthAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	depthAttachmentInfo.imageView = depthBuffer.getImageView();
	depthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
	depthAttachmentInfo.clearValue = { .depthStencil = {.depth = 0.0f, .stencil = 0} };

	VkRenderingInfo renderInfo{};
	renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	renderInfo.renderArea = { .offset{0,0}, .extent{.width = window.getWidth(), .height = window.getHeight()} };
	renderInfo.layerCount = 1;
	renderInfo.colorAttachmentCount = 1;
	renderInfo.pDepthAttachment = &depthAttachmentInfo;
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	VkSwapchainKHR swapChains[] = { vulkanObjectHandler->getSwapchain() };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &readyToPresentSemaphore;
	std::tuple<VkImage, VkImageView, uint32_t> swapchainImageData{};

	uint32_t cbSetIndex{ cmdBufferSet.createInterchangeableSet(2, CommandBufferSet::ASYNC_COMPUTE_CB) };

	WorldState::initialize();
	vkDeviceWaitIdle(device);
	while (!glfwWindowShouldClose(window))
	{
		//
		TIME_MEASURE_START(100, 0);
		//

		/*double timeModif{ glfwGetTime() };
		oneapi::tbb::parallel_for(0, pointLightsY,
			[&points, timeModif, pointLightsX](int i)
			{
				float yPos{ 0.3f + 1.4f * i };

				for (int j{ 0 }; j < pointLightsX; ++j)
				{
					float xPos{ static_cast<float>(std::sqrt(j + 1.0f) * 2.4f) };
					float cosA{ std::cos(static_cast<float>(timeModif) + j * 1.82f + i * 0.52f) };
					float sinA{ std::sin(static_cast<float>(timeModif) + j * 1.82f + i * 0.52f) };
					points[pointLightsX * i + j].changePosition(glm::vec3{sinA * xPos, yPos, cosA * xPos});
				}
			});
		for (int i{ 0 }; i < 4; ++i)
		{
			float xPos{ 3.0f * i - 12.0f / 2.0f };
			float spotLightAngle{ glm::radians(30.0) };

			spots[2 * i + 0].changeDirection(glm::rotate(spotsLeftDir, static_cast<float>(timeModif), spotsLeftRotDir));
			spots[2 * i + 1].changeDirection(glm::rotate(spotsRightDir, static_cast<float>(timeModif), spotsRightRotDir));
		}*/

		WorldState::refreshFrameTime();

		vpMatrices[0] = glm::lookAt(camInfo.camPos, camInfo.camPos + camInfo.camFront, glm::vec3{0.0, 1.0, 0.0});
		skyboxTransform[0] = vpMatrices[0];
		skyboxTransform[0][3] = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };

		hbao.submitViewMatrix(vpMatrices[0]);
		clusterer.submitViewMatrix(vpMatrices[0]);

		clusterer.startClusteringProcess();
		renderingData.frustumCulledCount = culling.cullAgainstFrustum(rUnitOBBs, frustumInfo, vpMatrices[0]);

		if (!vulkanObjectHandler->checkSwapchain(vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex)))
		{
			vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex);
			swapChains[0] = vulkanObjectHandler->getSwapchain();
		}
		swapchainImageData = vulkanObjectHandler->getSwapchainImageData(swapchainIndex);
		colorAttachmentInfo.imageView = framebufferImage.getImageView();
		renderInfo.pColorAttachments = &colorAttachmentInfo;

		VkCommandBuffer cbPreprocessing{ cmdBufferSet.beginTransientRecording() };

			culling.cmdCullOccluded(cbPreprocessing, descriptorManager);
			clusterer.cmdPassConductTileTest(cbPreprocessing, descriptorManager);

		cmdBufferSet.endRecording(cbPreprocessing);

		VkCommandBuffer cbDraw{ cmdBufferSet.beginRecording(CommandBufferSet::MAIN_CB) };
			
			{
				VkBufferCopy copy{ .srcOffset = culling.getDrawCountBufferOffset(), .dstOffset = renderingData.finalDrawCount.getOffset(), .size = renderingData.finalDrawCount.getSize() };
				BufferTools::cmdBufferCopy(cbDraw, culling.getDrawCountBufferHandle(), renderingData.finalDrawCount.getBufferHandle(), 1, &copy);
			}

			hbao.cmdPassCalcHBAO(cbDraw, descriptorManager, culling, vertexData, indexData, culling.getMaxDrawCount());

			BarrierOperations::cmdExecuteBarrier(cbDraw, std::span<const VkMemoryBarrier2>{
				{BarrierOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_READ_BIT)}
			});
			BarrierOperations::cmdExecuteBarrier(cbDraw, std::span<const VkImageMemoryBarrier2>{
				{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					framebufferImage.getImageHandle(),
					{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 }),
				BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				0, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
				depthBuffer.getImageHandle(),
					{
					.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 })}
			});
			
			colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			depthAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			vkCmdBeginRendering(cbDraw, &renderInfo);
			
				if (renderingData.drawBVs)
					clusterer.cmdDrawBVs(cbDraw, descriptorManager, pointBVPipeline, spotBVPipeline);
				
				descriptorManager.cmdSubmitPipelineResources(cbDraw, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline.getResourceSets(), skyboxPipeline.getResourceSetsInUse(), skyboxPipeline.getPipelineLayoutHandle());
				VkBuffer skyboxVertexBinding[1]{ skyboxData.getBufferHandle() };
				VkDeviceSize skyboxVertexOffsets[1]{ skyboxData.getOffset() };
				vkCmdBindVertexBuffers(cbDraw, 0, 1, skyboxVertexBinding, skyboxVertexOffsets);
				skyboxPipeline.cmdBind(cbDraw);
				vkCmdDraw(cbDraw, 36, 1, 0, 0);
				
				descriptorManager.cmdSubmitPipelineResources(cbDraw, VK_PIPELINE_BIND_POINT_GRAPHICS,
					forwardClusteredPipeline.getResourceSets(), forwardClusteredPipeline.getResourceSetsInUse(), forwardClusteredPipeline.getPipelineLayoutHandle());
				VkBuffer vertexBindings[1]{ vertexData.getBufferHandle() };
				VkDeviceSize vertexBindingOffsets[1]{ vertexData.getOffset() };
				vkCmdBindVertexBuffers(cbDraw, 0, 1, vertexBindings, vertexBindingOffsets);
				vkCmdBindIndexBuffer(cbDraw, indexData.getBufferHandle(), indexData.getOffset(), VK_INDEX_TYPE_UINT32);
				forwardClusteredPipeline.cmdBind(cbDraw);
				struct { glm::vec3 camInfo{}; float binWidth{}; glm::vec2 resolutionAO{}; uint32_t widthTiles{}; } pushConstData;
				pushConstData = { glm::vec3(camInfo.camPos.x, camInfo.camPos.y, camInfo.camPos.z), clusterer.getCurrentBinWidth(), glm::vec2{window.getWidth(), window.getHeight()}, clusterer.getWidthInTiles() };
				vkCmdPushConstants(cbDraw, forwardClusteredPipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pushConstData), &pushConstData);
				vkCmdDrawIndexedIndirectCount(cbDraw,
					culling.getDrawCommandBufferHandle(), culling.getDrawCommandBufferOffset(),
					culling.getDrawCountBufferHandle(), culling.getDrawCountBufferOffset(),
					culling.getMaxDrawCount(), culling.getDrawCommandBufferStride());
				
				
				//PBR sphere test
				/*descriptorManager.cmdSubmitPipelineResources(cbDraw, VK_PIPELINE_BIND_POINT_GRAPHICS, sphereTestPBRPipeline.getResourceSets(), sphereTestPBRPipeline.getResourceSetsInUse(), sphereTestPBRPipeline.getPipelineLayoutHandle());
				VkBuffer sphereTestPBRVertexBinding[1]{ sphereTestPBRdata.getBufferHandle() };
				VkDeviceSize sphereTestPBRVertexOffsets[1]{ sphereTestPBRdata.getOffset() };
				vkCmdBindVertexBuffers(cbDraw, 0, 1, sphereTestPBRVertexBinding, sphereTestPBRVertexOffsets);
				sphereTestPBRPipeline.cmdBind(cbDraw);
				vkCmdPushConstants(cbDraw, sphereTestPBRPipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &camInfo.camPos);
				vkCmdDraw(cbDraw, sphereTestPBRVertNum, sphereInstCount, 0, 0);*/
				//
			
			
			vkCmdEndRendering(cbDraw);
			
			if (renderingData.drawSpaceLines)
			{
				BarrierOperations::cmdExecuteBarrier(cbDraw, std::span<const VkMemoryBarrier2>{
					{BarrierOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
						VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
						VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT)}
				});
			
				colorAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
				colorAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				depthAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
				depthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				vkCmdBeginRendering(cbDraw, &renderInfo);
			
					descriptorManager.cmdSubmitPipelineResources(cbDraw, VK_PIPELINE_BIND_POINT_GRAPHICS, spaceLinesPipeline.getResourceSets(), spaceLinesPipeline.getResourceSetsInUse(), spaceLinesPipeline.getPipelineLayoutHandle());
					VkBuffer lineVertexBindings[1]{ spaceLinesVertexData.getBufferHandle() };
					VkDeviceSize lineVertexBindingOffsets[1]{ spaceLinesVertexData.getOffset() };
					vkCmdBindVertexBuffers(cbDraw, 0, 1, lineVertexBindings, lineVertexBindingOffsets);
					spaceLinesPipeline.cmdBind(cbDraw);
					vkCmdPushConstants(cbDraw, spaceLinesPipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &camInfo.camPos);
					vkCmdDraw(cbDraw, lineVertNum, 1, 0, 0);
			
				vkCmdEndRendering(cbDraw);
			}
			
			BarrierOperations::cmdExecuteBarrier(cbDraw, std::span<const VkImageMemoryBarrier2>{
				{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
					VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					framebufferImage.getImageHandle(),
					{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 })}
			});

		cmdBufferSet.endRecording(cbDraw);

		VkCommandBuffer cbPostprocessing{ cmdBufferSet.beginTransientRecording() };

			BarrierOperations::cmdExecuteBarrier(cbPostprocessing, std::span<const VkImageMemoryBarrier2>{
				{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
					VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
					std::get<0>(swapchainImageData),
					{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 })}
			});
			
			fxaa.cmdPassFXAA(cbPostprocessing, descriptorManager, std::get<1>(swapchainImageData));
			
			static bool hiZvis{ false };
			if (hiZvis)
			{
				depthBuffer.cmdVisualizeHiZ(cbPostprocessing, descriptorManager, std::get<0>(swapchainImageData), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, renderingData.hiZmipLevel);
			}
			
			ui.startUIPass(cbPostprocessing, std::get<1>(swapchainImageData));
			ImGui::Begin("Settings");
			if (ImGui::TreeNode("Stats"))
			{
				ImGui::Text("Frustum culled meshes - %u", renderingData.frustumCulledCount);
				ImGui::Text("Occlusion culled meshes - %u", drawCount - *reinterpret_cast<uint32_t*>(renderingData.finalDrawCount.getData()) - renderingData.frustumCulledCount);
				ImGui::TreePop();
			}
			ImGui::Checkbox("Enable Hi-Z visualiztion", &hiZvis);
			if (hiZvis)
			{
				ImGui::SliderInt("Mip-chain level", &renderingData.hiZmipLevel, 0, depthBuffer.getMipLevelCountHiZ() - 1);
			}
			ImGui::Checkbox("Space lines", &renderingData.drawSpaceLines);
			ImGui::Checkbox("Light bounding volumes", &renderingData.drawBVs);
			ImGui::End();
			ui.endUIPass(cbPostprocessing);
			
			
			BarrierOperations::cmdExecuteBarrier(cbPostprocessing, std::span<const VkImageMemoryBarrier2>{
				{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
					VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
					std::get<0>(swapchainImageData),
					{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 })}
			});
			
		cmdBufferSet.endRecording(cbPostprocessing);

		static uint32_t currentCBindex{ 1 };
		currentCBindex = currentCBindex ? 0 : 1;
		VkCommandBuffer cbCompute{ cmdBufferSet.beginInterchangeableRecording(cbSetIndex, currentCBindex)};

			depthBuffer.cmdCalcHiZ(cbCompute, descriptorManager);

		cmdBufferSet.endRecording(cbCompute);

		{
			VkSubmitInfo submitInfos[4]{};
			VkTimelineSemaphoreSubmitInfo semaphoreSubmit[4]{};
			uint64_t timelineVal{ semaphore.getValue() };
			uint64_t timelineValHiZ{ semaphoreHiZ.getValue() };

			uint64_t waitValues0[]{ timelineValHiZ };
			uint64_t signalValues0[]{ ++timelineVal };
			VkSemaphore waitSemaphores0[]{ semaphoreHiZ.getHandle()};
			VkSemaphore signalSemaphores0[]{ semaphore.getHandle() };
			VkPipelineStageFlags stageFlags0[]{ VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
			semaphoreSubmit[0] = TimelineSemaphore::getSubmitInfo(ARRAYSIZE(waitValues0), waitValues0, ARRAYSIZE(signalValues0), signalValues0);
			submitInfos[0] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit,
				.waitSemaphoreCount = ARRAYSIZE(waitSemaphores0), .pWaitSemaphores = waitSemaphores0, .pWaitDstStageMask = stageFlags0,
				.commandBufferCount = 1, .pCommandBuffers = &cbPreprocessing,
				.signalSemaphoreCount = ARRAYSIZE(signalSemaphores0), .pSignalSemaphores = signalSemaphores0 };
			static bool firstIt{ true }; if (firstIt) { firstIt = false; submitInfos[0].waitSemaphoreCount = 0; }

			uint64_t waitValues1[]{ timelineVal };
			uint64_t signalValues1[]{ ++timelineVal };
			VkSemaphore waitSemaphores1[]{ semaphore.getHandle() };
			VkSemaphore signalSemaphores1[]{ semaphore.getHandle() };
			VkPipelineStageFlags stageFlags1[]{ VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT};
			semaphoreSubmit[1] = TimelineSemaphore::getSubmitInfo(ARRAYSIZE(waitValues1), waitValues1, ARRAYSIZE(signalValues1), signalValues1);
			submitInfos[1] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit + 1,
				.waitSemaphoreCount = ARRAYSIZE(waitSemaphores1), .pWaitSemaphores = waitSemaphores1, .pWaitDstStageMask = stageFlags1,
				.commandBufferCount = 1, .pCommandBuffers = &cbDraw,
				.signalSemaphoreCount = ARRAYSIZE(signalSemaphores1), .pSignalSemaphores = signalSemaphores1 };

			uint64_t waitValues2[]{ timelineVal, 0 };
			uint64_t signalValues2[]{ ++timelineVal, 0 };
			VkSemaphore waitSemaphores2[]{ semaphore.getHandle(), swapchainSemaphore };
			VkSemaphore signalSemaphores2[]{ semaphore.getHandle(), readyToPresentSemaphore };
			VkPipelineStageFlags stageFlags2[]{ VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
			semaphoreSubmit[2] = TimelineSemaphore::getSubmitInfo(ARRAYSIZE(waitValues2), waitValues2, ARRAYSIZE(signalValues2), signalValues2);
			submitInfos[2] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit + 2,
				.waitSemaphoreCount = ARRAYSIZE(waitSemaphores2), .pWaitSemaphores = waitSemaphores2, .pWaitDstStageMask = stageFlags2,
				.commandBufferCount = 1, .pCommandBuffers = &cbPostprocessing,
				.signalSemaphoreCount = ARRAYSIZE(signalSemaphores2), .pSignalSemaphores = signalSemaphores2 };


			uint64_t waitValues3[]{ waitValues1[0]};
			uint64_t signalValues3[]{ ++timelineValHiZ };
			VkSemaphore waitSemaphores3[]{ semaphore.getHandle() };
			VkSemaphore signalSemaphores3[]{ semaphoreHiZ.getHandle()};
			VkPipelineStageFlags stageFlags3{ VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
			semaphoreSubmit[3] = TimelineSemaphore::getSubmitInfo(ARRAYSIZE(waitValues3), waitValues3, ARRAYSIZE(signalValues3), signalValues3);
			submitInfos[3] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit + 3,
				.waitSemaphoreCount = ARRAYSIZE(waitSemaphores3), .pWaitSemaphores = waitSemaphores3, .pWaitDstStageMask = &stageFlags3,
				.commandBufferCount = 1, .pCommandBuffers = &cbCompute,
				.signalSemaphoreCount = ARRAYSIZE(signalSemaphores3), .pSignalSemaphores = signalSemaphores3 };


			clusterer.waitClusteringProcess();
			vkQueueSubmit(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 3, submitInfos, VK_NULL_HANDLE);
			vkQueueSubmit(vulkanObjectHandler->getQueue(VulkanObjectHandler::COMPUTE_QUEUE_TYPE), 1, submitInfos + 3, VK_NULL_HANDLE);

			presentInfo.pImageIndices = &std::get<2>(swapchainImageData);

			if (!vulkanObjectHandler->checkSwapchain(vkQueuePresentKHR(vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), &presentInfo)))
				swapChains[0] = vulkanObjectHandler->getSwapchain();

			semaphore.wait(timelineVal);
			semaphore.newValue(timelineVal);
			semaphoreHiZ.newValue(timelineValHiZ);
		}
		cmdBufferSet.resetPool(CommandBufferSet::MAIN_POOL);
		cmdBufferSet.resetPool(CommandBufferSet::TRANSIENT_POOL);
		cmdBufferSet.resetInterchangeable(cbSetIndex, currentCBindex ? 0 : 1);

		glfwPollEvents();

		//
		TIME_MEASURE_END(100, 0);
		//
	}

	EASSERT(vkDeviceWaitIdle(device) == VK_SUCCESS, "Vulkan", "Device wait failed.");
	vkDestroySampler(device, universalSampler, nullptr);
	vkDestroySemaphore(device, swapchainSemaphore, nullptr);
	vkDestroySemaphore(device, readyToPresentSemaphore, nullptr);
	glfwTerminate();
	return 0;
}

std::shared_ptr<VulkanObjectHandler> initializeVulkan(const Window& window)
{
	VulkanCreateInfo info{};
	info.windowPtr = window;
	return std::shared_ptr<VulkanObjectHandler>{ std::make_shared<VulkanObjectHandler>(info) };
}

glm::mat4 getProjection(float FOV, float aspect, float zNear, float zFar)
{
	float h{ static_cast<float>(1.0 / std::tan((FOV * 0.5))) };
	float w{ h / aspect };
	float a{ -zNear / (zFar - zNear) };
	float b{ (zNear * zFar) / (zFar - zNear) };

	glm::mat4 mat{
		glm::vec4(w,   0.0,	0.0, 0.0),
		glm::vec4(0.0,	-h, 0.0, 0.0),
		glm::vec4(0.0, 0.0,	  a, 1.0),
		glm::vec4(0.0, 0.0,	  b, 0.0 )
	};
	return mat;
}

void loadDefaultTextures(ImageListContainer& imageLists, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue)
{
	uint8_t data[16]
	{
		uint8_t(0), uint8_t(0), uint8_t(0), uint8_t(255), //BaseColor	
		uint8_t(127), uint8_t(255), uint8_t(127), uint8_t(0), //Normal	
		uint8_t(255), uint8_t(127), uint8_t(10), uint8_t(0),  //AO_Rough_Met 
		uint8_t(0), uint8_t(0), uint8_t(0), uint8_t(0)		  //Emissive
	};

	ImageListContainer::ImageListContainerIndices indices0{ imageLists.getNewImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM) };
	ImageListContainer::ImageListContainerIndices indices1{ imageLists.getNewImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM) };
	ImageListContainer::ImageListContainerIndices indices2{ imageLists.getNewImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM) };
	ImageListContainer::ImageListContainerIndices indices3{ imageLists.getNewImage(1, 1, VK_FORMAT_R8G8B8A8_UNORM) };

	BufferMapped staging{ stagingBase, sizeof(uint8_t) * 4 * 4 };
	std::memcpy(staging.getData(), &data, sizeof(data));

	VkDeviceSize offsets[4]{ 0 + staging.getOffset(), sizeof(uint8_t) * 4 + staging.getOffset(), sizeof(uint8_t) * 4 * 2 + staging.getOffset(), sizeof(uint8_t) * 4 * 3 + staging.getOffset() };
	uint32_t layerIndices[4]{ 0, 1, 2, 3 };

	VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };

	BarrierOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
		{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0, VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			imageLists.getImageHandle(0),
			{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 4 })}
	});
	imageLists.cmdCopyDataFromBuffer(cb, 0, staging.getBufferHandle(), 4, offsets, layerIndices);
	BarrierOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
		{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
			0, 0,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			imageLists.getImageHandle(0),
			{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 4 })}
	});

	cmdBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos)
{
	static bool mouseWasCaptured{ false };
	if (ImGui::GetIO().WantCaptureMouse == true)
	{
		mouseWasCaptured = true;
		return;
	}

	CamInfo* camInfo{ reinterpret_cast<InputAccessedData*>(glfwGetWindowUserPointer(window))->cameraInfo };

	double xOffs{};
	double yOffs{};
	static bool firstCall{ true };
	static int width{};
	static int height{};

	if (firstCall)
	{
		xOffs = 0.0f;
		yOffs = 0.0f;
		glfwGetWindowSize(window, &width, &height);
		firstCall = false;
	}
	else
	{
		xOffs = (xpos - camInfo->lastCursorP.x) / width;
		yOffs = (ypos - camInfo->lastCursorP.y ) / height;
	}

	glm::vec3 upWVec{ 0.0, 1.0, 0.0 };
	glm::vec3 sideVec{ glm::normalize(glm::cross(upWVec, camInfo->camFront)) };
	glm::vec3 upRelVec{ glm::normalize(glm::cross(camInfo->camFront, sideVec)) };

	int stateL = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (stateL == GLFW_PRESS && mouseWasCaptured != true)
	{
		camInfo->camFront = glm::rotate(camInfo->camFront, static_cast<float>(xOffs * camInfo->sensetivity), upWVec);
		glm::vec3 newFront{ glm::rotate(camInfo->camFront, static_cast<float>(yOffs* camInfo->sensetivity), sideVec) };
		if (!(glm::abs(glm::dot(upWVec, newFront)) > 0.999))
			camInfo->camFront = newFront;
	}

	int stateR = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
	if (stateR == GLFW_PRESS && mouseWasCaptured != true)
	{
		camInfo->camPos += static_cast<float>(-xOffs * camInfo->speed) * sideVec + static_cast<float>(yOffs * camInfo->speed) * upRelVec;
	}

	int stateM = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);
	if (stateM == GLFW_PRESS && mouseWasCaptured != true)
	{
		camInfo->camPos += static_cast<float>(-yOffs * camInfo->speed) * camInfo->camFront;
	}

	camInfo->lastCursorP = {xpos, ypos};

	mouseWasCaptured = false;
}
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
}

Pipeline createForwardClusteredPipeline(PipelineAssembler& assembler,
	const BufferMapped& viewprojDataUB,
	const BufferMapped& modelTransformDataUB,
	const BufferMapped& drawDataSSBO,
	const Buffer& drawDataIndicesSSBO,
	const ImageListContainer& imageLists,
	const ImageCubeMap& skybox,
	const ImageCubeMap& radiance,
	const ImageCubeMap& irradiance,
	const Image& brdfLUT,
	const Image& imageAO,
	VkSampler univSampler,
	const BufferMapped& directionalLightUB, 
	const BufferMapped& sortedLightsDataUB,
	const BufferBaseHostAccessible& typeDataUB,
	const BufferBaseHostInaccessible& tileDataSSBO,
	const BufferMapped& zBinDataUB)
{
	VkDescriptorSetLayoutBinding viewprojBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT viewprojAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = viewprojDataUB.getSize() };

	VkDescriptorSetLayoutBinding imageListsBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 64, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	std::vector<VkDescriptorImageInfo> storageImageData(imageLists.getImageListCount());
	std::vector<VkDescriptorDataEXT> imageListsDescData(imageLists.getImageListCount());
	for (uint32_t i{ 0 }; i < imageListsDescData.size(); ++i)
	{
		storageImageData[i] = { .sampler = imageLists.getSampler(), .imageView = imageLists.getImageViewHandle(i), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		imageListsDescData[i].pStorageImage = &storageImageData[i];
	}
	VkDescriptorSetLayoutBinding modelTransformBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT modelTransformAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = modelTransformDataUB.getDeviceAddress(), .range = modelTransformDataUB.getSize() };

	VkDescriptorSetLayoutBinding drawDataBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT drawDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = drawDataSSBO.getDeviceAddress(), .range = drawDataSSBO.getSize() };
	VkDescriptorSetLayoutBinding drawDataIndicesBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT drawDataIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = drawDataIndicesSSBO.getDeviceAddress(), .range = drawDataIndicesSSBO.getSize() };

	VkDescriptorSetLayoutBinding sortedLightsBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT sortedLightsAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = sortedLightsDataUB.getDeviceAddress(), .range = sortedLightsDataUB.getSize() };
	VkDescriptorSetLayoutBinding typesBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT typesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = typeDataUB.getDeviceAddress(), .range = typeDataUB.getSize() };
	VkDescriptorSetLayoutBinding tileDataBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT tileDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = tileDataSSBO.getDeviceAddress(), .range = tileDataSSBO.getSize() };
	VkDescriptorSetLayoutBinding zBinDataBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT zBinDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = zBinDataUB.getDeviceAddress(), .range = zBinDataUB.getSize() };

	VkDescriptorSetLayoutBinding skyboxBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo skyboxImageInfo{ .sampler = skybox.getSampler(), .imageView = skybox.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding radianceBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo radianceImageInfo{ .sampler = radiance.getSampler(), .imageView = radiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding irradianceBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo irradianceImageInfo{ .sampler = irradiance.getSampler(), .imageView = irradiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding lutBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo lutImageInfo{ .sampler = univSampler, .imageView = brdfLUT.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding aoBinding{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo aoImageInfo{ .sampler = univSampler, .imageView = imageAO.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	VkDescriptorSetLayoutBinding directionalLightBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT directionalLightAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = directionalLightUB.getDeviceAddress(), .range = directionalLightUB.getSize() };

	std::vector<ResourceSet> resourceSets{};
	VkDevice device{ assembler.getDevice() };


	resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
		{viewprojBinding},  {},
			{{{.pUniformBuffer = &viewprojAddressInfo}}}, false });

	resourceSets.push_back({ device, 1, VkDescriptorSetLayoutCreateFlags{}, 1,
		{imageListsBinding, modelTransformBinding}, {{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }, {}},
			{imageListsDescData, {{.pStorageBuffer = &modelTransformAddressInfo}}}, true });

	resourceSets.push_back({ device, 2, VkDescriptorSetLayoutCreateFlags{}, 1,
		{drawDataBinding, drawDataIndicesBinding}, {},
			{{{.pStorageBuffer = &drawDataAddressInfo}}, {{.pStorageBuffer = &drawDataIndicesAddressInfo}}}, false });

	resourceSets.push_back({ device, 3, VkDescriptorSetLayoutCreateFlags{}, 1,
		{sortedLightsBinding, typesBinding, tileDataBinding, zBinDataBinding},  {},
			{{{.pUniformBuffer = &sortedLightsAddressInfo}}, 
			{{.pUniformBuffer = &typesAddressInfo}}, 
			{{.pStorageBuffer = &tileDataAddressInfo}}, 
			{{.pUniformBuffer = &zBinDataAddressInfo}}}, false });

	resourceSets.push_back({ device, 4, VkDescriptorSetLayoutCreateFlags{}, 1,
		{skyboxBinding, radianceBinding, irradianceBinding, lutBinding, aoBinding}, {},
			{{{.pCombinedImageSampler = &skyboxImageInfo}}, 
			{{.pCombinedImageSampler = &radianceImageInfo}}, 
			{{.pCombinedImageSampler = &irradianceImageInfo}},
			{{.pCombinedImageSampler = &lutImageInfo}}, 
			{{.pCombinedImageSampler = &aoImageInfo}}}, true });

	resourceSets.push_back({ device, 5, VkDescriptorSetLayoutCreateFlags{}, 1,
		{directionalLightBinding}, {},
			{{{.pUniformBuffer = &directionalLightAddressInfo}}}, false });


	return	Pipeline{ assembler,
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/shader_vert.spv"},
		ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/shader_frag.spv"}}},
		resourceSets,
		{{StaticVertex::getBindingDescription()}},
		{StaticVertex::getAttributeDescriptions()},
		{{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(glm::vec3) + sizeof(float) + sizeof(glm::vec2) + sizeof(uint32_t)}}} };
}

Pipeline createSkyboxPipeline(PipelineAssembler& assembler, const ImageCubeMap& cubemapImages, const BufferMapped& skyboxTransformUB)
{
	std::vector<ResourceSet> resourceSets{};
	VkDescriptorSetLayoutBinding uniformViewTranslateBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT uniformViewTranslateAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = skyboxTransformUB.getDeviceAddress(), .range = sizeof(glm::mat4) * 2 };
	VkDescriptorSetLayoutBinding cubeTextureBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo cubeTextureAddressInfo{ .sampler = cubemapImages.getSampler(), .imageView = cubemapImages.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1,
		{uniformViewTranslateBinding, cubeTextureBinding},  {},
		{{{.pUniformBuffer = &uniformViewTranslateAddressinfo}}, {{.pCombinedImageSampler = &cubeTextureAddressInfo}}}, true });
	return Pipeline{ assembler,
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/shader_skybox_vert.spv"}, 
		ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/shader_skybox_frag.spv"}}},
		resourceSets,
		{{PosOnlyVertex::getBindingDescription()}},
		{PosOnlyVertex::getAttributeDescriptions()} };
}
void uploadSkyboxVertexData(Buffer& skyboxData, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue)
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
	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformViewProjBinding},  {}, {{{.pUniformBuffer = &uniformViewProjAddressinfo}}}, false});
	return Pipeline{ assembler, 
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/shader_space_lines_vert.spv"}, 
		ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/shader_space_lines_frag.spv"}}},
		resourceSets,
		{{PosColorVertex::getBindingDescription()}},
		{PosColorVertex::getAttributeDescriptions()},
		{{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(glm::vec3)}}} };
}
uint32_t uploadLineVertices(fs::path filepath, Buffer& vertexBuffer, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue)
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

Pipeline createPointBVPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, const BufferMapped& instancePointLightIndexData, const BufferMapped& sortedLightsDataUB)
{
	//Binding 0
	//viewproj matrices
	VkDescriptorSetLayoutBinding viewprojBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT viewprojAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = viewprojDataUB.getSize() };
	//Binding 2 Point
	//Light indices
	VkDescriptorSetLayoutBinding pointLightIndicesBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT pointLightIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = instancePointLightIndexData.getDeviceAddress(), .range = instancePointLightIndexData.getSize() };
	//Binding 3
	//Light data
	VkDescriptorSetLayoutBinding lightDataBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT lightDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = sortedLightsDataUB.getDeviceAddress(), .range = sortedLightsDataUB.getSize() };

	std::vector<ResourceSet> resourceSets{};

	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1,
	{viewprojBinding, pointLightIndicesBinding, lightDataBinding},  {},
		{{{.pUniformBuffer = &viewprojAddressInfo}},
		{{.pStorageBuffer = &pointLightIndicesAddressInfo}},
		{{.pStorageBuffer = &lightDataAddressInfo}}}, false });

	return Pipeline{ assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/point_BV_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/BV_frag.spv"} } },
		resourceSets,
		{ {PosOnlyVertex::getBindingDescription()} },
		{ PosOnlyVertex::getAttributeDescriptions() } };
}
Pipeline createSpotBVPipeline(PipelineAssembler& assembler, BufferMapped& viewprojDataUB, const BufferMapped& instanceSpotLightIndexData, const BufferMapped& sortedLightsDataUB)
{
	//Binding 0
	//viewproj matrices
	VkDescriptorSetLayoutBinding viewprojBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT viewprojAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = viewprojDataUB.getSize() };

	//Binding 2 Spot
	//Light indices
	VkDescriptorSetLayoutBinding spotLightIndicesBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT spotLightIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = instanceSpotLightIndexData.getDeviceAddress(), .range = instanceSpotLightIndexData.getSize() };
	//Binding 3
	//Light data
	VkDescriptorSetLayoutBinding lightDataBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT lightDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = sortedLightsDataUB.getDeviceAddress(), .range = sortedLightsDataUB.getSize() };

	std::vector<ResourceSet> resourceSets{};

	resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1,
	{viewprojBinding, spotLightIndicesBinding, lightDataBinding},  {},
		{{{.pUniformBuffer = &viewprojAddressInfo}},
		{{.pStorageBuffer = &spotLightIndicesAddressInfo}},
		{{.pStorageBuffer = &lightDataAddressInfo}}}, false });

	return Pipeline{ assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/cone_BV_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/BV_frag.spv"} } },
		resourceSets,
		{},
		{} };
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
							{{.pCombinedImageSampler = &lutAddressInfo}}}, true });

	return Pipeline{ assembler,
		{{ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/pbr_spheres_test_vert.spv"},
		ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/pbr_spheres_test_frag.spv"}}},
		resourceSets,
		{{PosTexVertex::getBindingDescription()}},
		{PosTexVertex::getAttributeDescriptions()},
		{{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(glm::vec3)}}} };
}
uint32_t uploadSphereVertexData(fs::path filepath, Buffer& sphereVertexData, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue)
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


void transformOBBs(OBBs& boundingBoxes, std::vector<StaticMesh>& staticMeshes, int drawCount, const std::vector<glm::mat4>& modelMatrices)
{
	uint32_t transMatIndex{ 0 };
	uint32_t drawNum{ static_cast<uint32_t>(staticMeshes[transMatIndex].getRUnits().size()) };
	for (uint32_t i{ 0 }; i < drawCount; ++i)
	{
		if (i == drawNum)
		{
			drawNum += staticMeshes[++transMatIndex].getRUnits().size();
		}
		boundingBoxes.transformOBB(i, modelMatrices[transMatIndex]);
	}
}
void getBoundingSpheres(BufferMapped& indirectDataBuffer, const OBBs& boundingBoxes)
{
	IndirectData* data{ reinterpret_cast<IndirectData*>(indirectDataBuffer.getData()) };

	for (uint32_t i{ 0 }; i < boundingBoxes.getBBCount(); ++i)
	{
		boundingBoxes.getBoundingSphere(i, data[i].bsPos, &(data[i].bsRad));
	}
}

void fillFrustumData(glm::mat4* vpMatrices, Window& window, Clusterer& clusterer, HBAO& hbao, FrustumInfo& frustumInfo)
{
	float nearPlane{ 0.1 };
	float farPlane{ 100.0 };
	float aspect{ static_cast<float>(static_cast<double>(window.getWidth()) / window.getHeight()) };
	float FOV{ glm::radians(80.0) };

	vpMatrices[1] = getProjection(FOV, aspect, nearPlane, farPlane);
	clusterer.submitFrustum(nearPlane, farPlane, aspect, FOV);
	hbao.submitFrustum(nearPlane, farPlane, aspect, FOV);

	double FOV_X{ glm::atan(glm::tan(FOV / 2) * aspect) * 2 };
	frustumInfo.planes[0] = { 0.0f, 0.0f, -1.0f, nearPlane }; //near plane
	frustumInfo.planes[1] = glm::vec4{ glm::rotate(glm::dvec3{-1.0, 0.0, 0.0}, FOV_X / 2.0, glm::dvec3{0.0, -1.0, 0.0}), 0.0 }; //left plane
	frustumInfo.planes[2] = glm::vec4{ glm::rotate(glm::dvec3{1.0, 0.0, 0.0}, FOV_X / 2.0, glm::dvec3{0.0, 1.0, 0.0}), 0.0 }; //right plane
	frustumInfo.planes[3] = glm::vec4{ glm::rotate(glm::dvec3{0.0, 1.0, 0.0}, FOV / 2.0, glm::dvec3{-1.0, 0.0, 0.0}), 0.0 }; //up plane
	frustumInfo.planes[4] = glm::vec4{ glm::rotate(glm::dvec3{0.0, -1.0, 0.0}, FOV / 2.0, glm::dvec3{1.0, 0.0, 0.0}), 0.0 }; //down plane
	frustumInfo.planes[5] = { 0.0f, 0.0f, 1.0f, -farPlane }; //far plane

	double fovXScale{ glm::tan(FOV_X / 2) };
	double fovYScale{ glm::tan(FOV / 2) };

	frustumInfo.points[0] = { -(nearPlane)*fovXScale, -(nearPlane)*fovYScale, nearPlane }; //near bot left
	frustumInfo.points[1] = { -(nearPlane)*fovXScale, (nearPlane)*fovYScale, nearPlane }; //near top left
	frustumInfo.points[2] = { (nearPlane)*fovXScale, (nearPlane)*fovYScale, nearPlane }; //near top right
	frustumInfo.points[3] = { (nearPlane)*fovXScale, -(nearPlane)*fovYScale, nearPlane }; //near bot right
	frustumInfo.points[4] = { -(farPlane)*fovXScale, -(farPlane)*fovYScale, farPlane }; //far bot left
	frustumInfo.points[5] = { -(farPlane)*fovXScale, (farPlane)*fovYScale, farPlane }; //far top left
	frustumInfo.points[6] = { (farPlane)*fovXScale, (farPlane)*fovYScale, farPlane }; //far top right
	frustumInfo.points[7] = { (farPlane)*fovXScale, -(farPlane)*fovYScale, farPlane }; //far bot right
}
void fillModelMatrices(const BufferMapped& modelTransformDataUB, const std::vector<glm::mat4>& modelMatrices)
{
	glm::mat4* transformMatrices{ reinterpret_cast<glm::mat4*>(modelTransformDataUB.getData()) };
	for (int i{ 0 }; i < modelMatrices.size(); ++i)
	{
		transformMatrices[i] = modelMatrices[i];
	}
}
void fillDrawDataIndices(const BufferMapped& perDrawDataIndicesSSBO, std::vector<StaticMesh>& staticMeshes, int drawCount)
{
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
			auto indices{ rUnit.getMaterialIndices() };
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
}
void fillSkyboxTransformData(glm::mat4* vpMatrices, glm::mat4* skyboxTransform)
{
	skyboxTransform[0] = vpMatrices[0];
	skyboxTransform[0][3] = glm::vec4{ 0.0f, 0.0f, 0.0f, 1.0f };
	skyboxTransform[1] = vpMatrices[1];
}
void fillPBRSphereTestInstanceData(const BufferMapped& perInstPBRTestSSBO, int sphereInstCount)
{
	uint8_t* perInstDataPtr{ reinterpret_cast<uint8_t*>(perInstPBRTestSSBO.getData()) };
	for (int i{ 0 }; i < sphereInstCount; ++i)
	{
		glm::vec4* color_metalnessData{ reinterpret_cast<glm::vec4*>(perInstDataPtr) };
		glm::vec4* wPos_roughnessData{ color_metalnessData + 1 };

		int row{ i / 5 };
		int column{ i % 5 };

		glm::vec3 sphereHorDistance{ 4.0, 0.0, 0.0 };
		glm::vec3 sphereVerDistance{ 0.0, 4.0, 0.0 };

		glm::vec3 color{ 1.0f };
		float metalness{ 0.25f * row };

		glm::vec3 position{ -8.0, -8.0, 0.0 };
		position += (sphereHorDistance * static_cast<float>(column) + sphereVerDistance * static_cast<float>(row));
		float roughness{ 0.25f * column };

		*color_metalnessData = glm::vec4{ color, metalness };
		*wPos_roughnessData = glm::vec4{ position, roughness };
		perInstDataPtr += sizeof(glm::vec4) * 2;
	}
}

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
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE };
	VkSampler sampler{};
	vkCreateSampler(device, &samplerCI, nullptr, &sampler);
	return sampler;
}