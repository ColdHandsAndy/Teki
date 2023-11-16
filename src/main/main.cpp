#include <iostream>
#include <vector>
#include <span>
#include <initializer_list>
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

#include <tbb/flow_graph.h>

#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/shader_management/shader_operations.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/renderer/deferred_lighting.h"
#include "src/rendering/data_management/memory_manager.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/renderer/sync_operations.h"
#include "src/rendering/renderer/timeline_semaphore.h"
#include "src/rendering/lighting/light_types.h"
#include "src/rendering/lighting/shadows.h"
#include "src/rendering/renderer/clusterer.h"
#include "src/rendering/renderer/culling.h"
#include "src/rendering/scene/parse_scene.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/data_abstraction/mesh.h"
#include "src/rendering/data_abstraction/runit.h"
#include "src/rendering/renderer/depth_buffer.h"
#include "src/rendering/renderer/HBAO.h"
#include "src/rendering/renderer/TAA.h"
#include "src/rendering/data_abstraction/BB.h"
#include "src/rendering/renderer/world_transform.h"
#include "src/rendering/UI/UI.h"

#include "src/window/window.h"
#include "src/world_state/world_state.h"

#include "src/tools/tools.h"

#define WINDOW_WIDTH_DEFAULT  1920u
#define WINDOW_HEIGHT_DEFAULT 1080u
#define HBAO_WIDTH_DEFAULT  1280u
#define HBAO_HEIGHT_DEFAULT 720u

#define MAX_INDIRECT_DRAWS 512
#define MAX_TRANSFORM_MATRICES 64

#define NEAR_PLANE 0.1
#define FAR_PLANE  10000.0

#define GENERAL_BUFFER_DEFAULT_SIZE 134217728ll
#define SHARED_BUFFER_DEFAULT_SIZE 8388608ll
#define DEVICE_BUFFER_DEFAULT_SIZE 268435456ll * 2

namespace fs = std::filesystem;

struct CamInfo
{
	glm::vec3 camPos{ 0.0, 0.0, -10.0 };
	glm::vec3 camFront{ 0.0, 0.0, 1.0 };
	double speed{ 10.0 };
	double sensetivity{ 1.9 };
	glm::vec2 lastCursorP{};
	bool camPosChanged{ false };
} camInfo;

struct InputAccessedData
{
	CamInfo* cameraInfo{ &camInfo };
} inputAccessedData;

std::shared_ptr<VulkanObjectHandler> initializeVulkan(const Window& window);

Pipeline createSkyboxPipeline(PipelineAssembler& assembler, const ResourceSet& viewprojRS, const ResourceSet& skyboxLightingRS);
Pipeline createSpaceLinesPipeline(PipelineAssembler& assembler, const ResourceSet& viewprojRS);

void createResourceSets(VkDevice device,
	ResourceSet& transformMatricesRS,
	const BufferMapped& transformMatrices,
	ResourceSet& materialsTexturesRS,
	const ImageListContainer& imageLists,
	ResourceSet& skyboxLightingRS,
	const ImageCubeMap& skybox,
	const ImageCubeMap& radiance,
	const ImageCubeMap& irradiance);
void createDrawDataResourceSet(VkDevice device,
	ResourceSet& drawDataRS,
	const BufferMapped& drawData,
	const Buffer& drawDataIndices);
void createShadowMapResourceSet(VkDevice device,
	ResourceSet& shadowMapsRS,
	const ImageListContainer& shadowMapLists,
	const std::vector<ImageList>& shadowCubeMapLists,
	const BufferBaseHostAccessible& shadowMapViewMatrices);
void createPBRResourceSet(VkDevice device,
	VkSampler generalSampler,
	ResourceSet& pbrRS,
	const Image& brdfLUT,
	const Image& aoImage);
void createDirecLightingResourceSet(VkDevice device,
	ResourceSet& directLightingRS,
	const BufferMapped& directionalLight,
	const BufferMapped& sortedLightsData,
	const BufferBaseHostAccessible& typeData,
	const BufferBaseHostInaccessible& tileData,
	const BufferMapped& zBinData);

uint32_t uploadLineVertices(fs::path filepath, Buffer& vertexBuffer, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue);
void uploadSkyboxVertexData(Buffer& skyboxData, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue);

void loadDefaultTextures(ImageListContainer& imageLists, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue);
void transformOBBs(OBBs& boundingBoxes, std::vector<StaticMesh>& staticMeshes, int drawCount, const std::vector<glm::mat4>& modelMatrices);
void getBoundingSpheres(BufferMapped& indirectDataBuffer, const OBBs& boundingBoxes);

void fillFrustumData(CoordinateTransformation& coordinateTransformation, Window& window, Clusterer& clusterer, HBAO& hbao, FrustumInfo& frustumInfo, ShadowCaster& caster, DeferredLighting& deferredLighting);
void fillModelMatrices(const BufferMapped& modelTransformDataSSBO, const std::vector<glm::mat4>& modelMatrices);
void fillDrawData(const BufferMapped& perDrawDataIndicesSSBO, std::vector<StaticMesh>& staticMeshes, int drawCount);

void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

VkSampler createGeneralSampler(VkDevice device, float maxAnisotropy);

void submitAndWait(VulkanObjectHandler& vulkanObjectHandler,
	CommandBufferSet& cmdBufferSet, VkCommandBuffer cbPreprocessing, VkCommandBuffer cbDraw, VkCommandBuffer cbPostprocessing, VkCommandBuffer cbCompute,
	uint32_t indexToCBSet, uint32_t currentCommandBufferIndex,
	TimelineSemaphore& semaphore, TimelineSemaphore& semaphoreCompute, VkSemaphore swapchainSemaphore, VkSemaphore readyToPresentSemaphore,
	VkPresentInfoKHR& presentInfo, uint32_t swapchainIndex, VkSwapchainKHR& swapChain);

int main()
{
	EASSERT(glfwInit(), "GLFW", "GLFW was not initialized.")

	Window window{ WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT, "Teki", true };
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
	
	ResourceSet::initializeDescriptorBuffers(*vulkanObjectHandler, memManager);

	VkDevice device{ vulkanObjectHandler->getLogicalDevice() };

	BufferBaseHostInaccessible baseDeviceBuffer{ device, DEVICE_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostCachedBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, BufferBase::DEDICATED_FLAG, false, true };

	std::vector<fs::path> modelPaths{};
	std::vector<glm::mat4> modelMatrices{};
	fs::path envPath{};
	Scene::parseSceneData("internal/scene_info.json", modelPaths, modelMatrices, envPath);

	
	Buffer vertexData{ baseDeviceBuffer };
	Buffer indexData{ baseDeviceBuffer };
	Buffer skyboxData{ baseDeviceBuffer };
	uploadSkyboxVertexData(skyboxData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
	Buffer spaceLinesVertexData{ baseDeviceBuffer };
	uint32_t lineVertNum{ uploadLineVertices("internal/spaceLinesMesh/space_lines_vertices.bin", spaceLinesVertexData, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) };
	BufferMapped indirectDrawCmdData{ baseHostCachedBuffer, sizeof(IndirectData) * MAX_INDIRECT_DRAWS };
	BufferMapped drawData{ baseHostBuffer, sizeof(uint8_t) * 12 * MAX_INDIRECT_DRAWS };
	BufferMapped transformMatrices{ baseHostBuffer, sizeof(glm::mat4) * MAX_TRANSFORM_MATRICES };
	BufferMapped directionalLight{ baseHostBuffer, LightTypes::DirectionalLight::getDataByteSize() };
	VkSampler generalSampler{ createGeneralSampler(device, vulkanObjectHandler->getPhysDevLimits().maxSamplerAnisotropy) };
	Image brdfLUT{ TextureLoaders::loadImage(vulkanObjectHandler, cmdBufferSet, baseHostBuffer,
											 "internal/brdfLUT/brdfLUT.exr",
											 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
											 4, OIIO::TypeDesc::HALF, VK_FORMAT_R16G16B16A16_SFLOAT) };
	ImageCubeMap cubemapSkybox{ TextureLoaders::loadCubemap(*vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "skybox/skybox.ktx2") };
	ImageCubeMap cubemapSkyboxRadiance{ TextureLoaders::loadCubemap(*vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "radiance/radiance.ktx2") };
	ImageCubeMap cubemapSkyboxIrradiance{ TextureLoaders::loadCubemap(*vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "irradiance/irradiance.ktx2") };
	ImageListContainer materialsTextures{ device, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, true,
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
	loadDefaultTextures(materialsTextures, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
	ImageListContainer shadowMaps{ device, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, false,
		VkSamplerCreateInfo{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = 128.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE }, 
			2, VK_IMAGE_ASPECT_DEPTH_BIT };
	std::vector<ImageList> shadowCubeMaps{};
	FrustumInfo frustumInfo{};
	OBBs rUnitOBBs{ 8192 };
	uint32_t drawCount{};
	UiData renderingData{};
	renderingData.finalDrawCount.initialize(baseHostBuffer, sizeof(uint32_t));
	CoordinateTransformation coordinateTransformation{ device, baseHostCachedBuffer };
	coordinateTransformation.updateScreenDimensions(window.getWidth(), window.getHeight());

	std::vector<StaticMesh> staticMeshes{ loadStaticMeshes(vertexData, indexData, 
		indirectDrawCmdData, drawCount,
		rUnitOBBs,
		materialsTextures, 
		modelPaths,
		*vulkanObjectHandler, cmdBufferSet)
	};
	transformOBBs(rUnitOBBs, staticMeshes, drawCount, modelMatrices);
	getBoundingSpheres(indirectDrawCmdData, rUnitOBBs);


	ResourceSet transformMatricesRS{};
	ResourceSet materialsTexturesRS{};
	ResourceSet shadowMapsRS{};
	ResourceSet skyboxLightingRS{};
	ResourceSet drawDataRS{};
	ResourceSet pbrRS{};
	ResourceSet directLightingRS{};
	createResourceSets(device,
		transformMatricesRS, transformMatrices, 
		materialsTexturesRS, materialsTextures, 
		skyboxLightingRS, cubemapSkybox, cubemapSkyboxRadiance, cubemapSkyboxIrradiance);
	DepthBuffer depthBuffer{ device, window.getWidth(), window.getHeight() };
	Clusterer clusterer{ device, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), window.getWidth(), window.getHeight(), coordinateTransformation.getResourceSet() };
	ShadowCaster caster{ device, clusterer, shadowMaps, shadowCubeMaps, indirectDrawCmdData, transformMatrices, drawData, rUnitOBBs };
	LightTypes::LightBase::assignGlobalClusterer(clusterer);
	LightTypes::LightBase::assignGlobalShadowCaster(caster);

	LightTypes::DirectionalLight dirLight{ {1.0f, 1.0f, 1.0f}, 0.0f, {0.0f, -1.0f, 0.0f} };
	dirLight.plantData(directionalLight.getData());
	LightTypes::PointLight pLight(glm::vec3{4.5f, 3.2f, 0.0f}, glm::vec3{0.8f, 0.4f, 0.2f}, 200.0f, 10.0f, 1024, 0.0);
	LightTypes::SpotLight sLight(glm::vec3{-8.5f, 14.0f, -3.5f}, glm::vec3{0.6f, 0.5f, 0.7f}, 1000.0f, 24.0f, glm::vec3{0.0, -1.0, 0.3}, glm::radians(25.0), glm::radians(30.0), 2048, 0.0);

	Culling culling{ device, MAX_INDIRECT_DRAWS, NEAR_PLANE, coordinateTransformation.getResourceSet(), indirectDrawCmdData, depthBuffer, vulkanObjectHandler->getComputeFamilyIndex(), vulkanObjectHandler->getGraphicsFamilyIndex()};
	HBAO hbao{ device, HBAO_WIDTH_DEFAULT, HBAO_HEIGHT_DEFAULT, depthBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE) };
	createDrawDataResourceSet(device, drawDataRS, drawData, culling.getDrawDataIndexBuffer());
	createShadowMapResourceSet(device, shadowMapsRS, shadowMaps, shadowCubeMaps, caster.getShadowViewMatrices());
	createPBRResourceSet(device, generalSampler, pbrRS, brdfLUT, hbao.getAO());
	createDirecLightingResourceSet(device, directLightingRS, directionalLight, clusterer.getSortedLights(), clusterer.getSortedTypeData(), clusterer.getTileData(), clusterer.getZBin());
	DeferredLighting deferredLighting{device, window.getWidth(), window.getHeight(), 
		depthBuffer, 
		hbao,
		coordinateTransformation.getResourceSet(), transformMatricesRS, 
		materialsTexturesRS, shadowMapsRS, skyboxLightingRS, 
		drawDataRS, pbrRS, directLightingRS, generalSampler};
	deferredLighting.updateTileWidth(clusterer.getWidthInTiles());
	TAA taa{ device, depthBuffer, deferredLighting.getFramebuffer(), coordinateTransformation.getResourceSet(), cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE) };

	 
	 
	PipelineAssembler assembler{ device };
	
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, window.getWidth(), window.getHeight());
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0f, VK_CULL_MODE_NONE);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_SKYBOX, VK_COMPARE_OP_EQUAL);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT, VK_FORMAT_R16G16B16A16_SFLOAT);
	Pipeline skyboxPipeline{ createSkyboxPipeline(assembler, coordinateTransformation.getResourceSet(), skyboxLightingRS) };
	
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_LINE_DRAWING);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.5f);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DEFAULT);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);
	Pipeline spaceLinesPipeline{ createSpaceLinesPipeline(assembler, coordinateTransformation.getResourceSet()) };


	fillFrustumData(coordinateTransformation, window, clusterer, hbao, frustumInfo, caster, deferredLighting);
	fillModelMatrices(transformMatrices, modelMatrices);
	fillDrawData(drawData, staticMeshes, drawCount);

	

	VkBuffer vertexBindings[1]{ vertexData.getBufferHandle() };
	VkDeviceSize vertexBindingOffsets[1]{ vertexData.getOffset() };

	TimelineSemaphore semaphore{ device };
	TimelineSemaphore semaphoreCompute{ device };
	VkSemaphore swapchainSemaphore{};
	VkSemaphore readyToPresentSemaphore{};
	VkSemaphoreCreateInfo semCI{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
	vkCreateSemaphore(device, &semCI, nullptr, &swapchainSemaphore);
	vkCreateSemaphore(device, &semCI, nullptr, &readyToPresentSemaphore);
	
	uint32_t swapchainIndex{};

	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
	VkSwapchainKHR swapChains[] = { vulkanObjectHandler->getSwapchain() };
	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = swapChains;
	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &readyToPresentSemaphore;
	std::tuple<VkImage, VkImageView, uint32_t> swapchainImageData{};

	SyncOperations::EventHolder<3> events{ device };

	VkCommandBuffer cbPreprocessing{};
	VkCommandBuffer cbDraw{};
	VkCommandBuffer cbPostprocessing{};
	VkCommandBuffer cbCompute{};
	uint32_t cbSetIndex{ cmdBufferSet.createInterchangeableSet(2, CommandBufferSet::ASYNC_COMPUTE_CB) };
	uint32_t currentCBindex{ 1 };
	
	typedef oneapi::tbb::flow::continue_node<oneapi::tbb::flow::continue_msg> node_t;
	typedef const oneapi::tbb::flow::continue_msg& msg_t;
	oneapi::tbb::flow::graph flowGraph{};
	node_t nodeAcquireSwapchainUpdateViewMatrices{ flowGraph, [&](msg_t)
		{
			if (!vulkanObjectHandler->checkSwapchain(vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex)))
			{
				vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex);
				swapChains[0] = vulkanObjectHandler->getSwapchain();
			}
			swapchainImageData = vulkanObjectHandler->getSwapchainImageData(swapchainIndex);

			deferredLighting.updateCameraPosition(camInfo.camPos);

			coordinateTransformation.updateProjectionMatrixJitter();
			coordinateTransformation.updateViewMatrix(camInfo.camPos, camInfo.camPos + camInfo.camFront, glm::vec3{ 0.0, 1.0, 0.0 });
			clusterer.submitViewMatrix(coordinateTransformation.getViewMatrix());
		} };
	node_t nodeFrustumCulling{ flowGraph, [&](msg_t)
		{
			renderingData.frustumCulledCount = culling.cullAgainstFrustum(rUnitOBBs, frustumInfo, coordinateTransformation.getViewMatrix());
		} };
	node_t nodePrepareDataForShadowMapRender{ flowGraph, [&](msg_t)
		{
			caster.prepareDataForShadowMapRendering();
		} };
	node_t nodePreprocessCB1{ flowGraph, [&](msg_t)
		{
			cbPreprocessing = cmdBufferSet.beginPerThreadRecording(0);

			culling.cmdTransferSetDrawCountToZero(cbPreprocessing);
			SyncOperations::cmdExecuteBarrier(cbPreprocessing, culling.getDependency());
			culling.cmdDispatchCullOccluded(cbPreprocessing);
			clusterer.cmdTransferClearTileBuffer(cbPreprocessing);
			events.cmdSet(cbPreprocessing, 0, clusterer.getDependency()); //Event 1 set
		} };
	node_t nodePreprocessCB2{ flowGraph, [&](msg_t)
		{
			hbao.cmdTransferClearBuffers(cbPreprocessing);
			caster.cmdRenderShadowMaps(cbPreprocessing, vertexData, indexData);
			uint32_t indices[]{ 0 };
			events.cmdWait(cbPreprocessing, 1, indices, &clusterer.getDependency()); //Event 1 wait
		} };
	node_t nodePreprocessCB3{ flowGraph, [&](msg_t)
		{
			clusterer.cmdPassConductTileTest(cbPreprocessing);
		} };
	node_t nodePreprocessCB4{ flowGraph, [&](msg_t)
		{
			VkBufferCopy copy{.srcOffset = culling.getDrawCountBufferOffset(), .dstOffset = renderingData.finalDrawCount.getOffset(), .size = renderingData.finalDrawCount.getSize() };
			BufferTools::cmdBufferCopy(cbPreprocessing, culling.getDrawCountBufferHandle(), renderingData.finalDrawCount.getBufferHandle(), 1, &copy);

			cmdBufferSet.endRecording(cbPreprocessing);
		} };
	node_t nodeDrawCB{ flowGraph, [&](msg_t)
		{
			cbDraw = cmdBufferSet.beginPerThreadRecording(1);

			SyncOperations::cmdExecuteBarrier(cbDraw,
				{ {SyncOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
					VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT)} });

				
				deferredLighting.cmdPassDrawToUVBuffer(cbDraw, culling, vertexData, indexData);

				SyncOperations::cmdExecuteBarrier(cbDraw,
					{ {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
						VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
						depthBuffer.getImageHandle(), depthBuffer.getDepthBufferSubresourceRange())} });

				hbao.cmdDispatchHBAO(cbDraw);
				SyncOperations::cmdExecuteBarrier(cbDraw, hbao.getDependency());
				hbao.cmdDispatchHBAOBlur(cbDraw);

				SyncOperations::cmdExecuteBarrier(cbDraw, deferredLighting.getDependency());

				deferredLighting.updateLightBinWidth(clusterer.getCurrentBinWidth());
				deferredLighting.cmdDispatchLightingCompute(cbDraw);
				
				SyncOperations::cmdExecuteBarrier(cbDraw, 
					{ {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
						VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
						VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
						depthBuffer.getImageHandle(), depthBuffer.getDepthBufferSubresourceRange())} });
				
				VkRenderingAttachmentInfo colorAttachmentInfoSkybox{};
				colorAttachmentInfoSkybox.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
				colorAttachmentInfoSkybox.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
				colorAttachmentInfoSkybox.clearValue = VkClearValue{ .color{.float32{0.4f, 1.0f, 0.8f}} };
				colorAttachmentInfoSkybox.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
				colorAttachmentInfoSkybox.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				colorAttachmentInfoSkybox.imageView = deferredLighting.getFramebufferImageView();
				VkRenderingAttachmentInfo depthAttachmentInfoSkybox{};
				depthAttachmentInfoSkybox.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
				depthAttachmentInfoSkybox.imageView = depthBuffer.getImageView();
				depthAttachmentInfoSkybox.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
				depthAttachmentInfoSkybox.clearValue = { .depthStencil = {.depth = 0.0f, .stencil = 0} };
				depthAttachmentInfoSkybox.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
				depthAttachmentInfoSkybox.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
				VkRenderingInfo renderInfoSkybox{};
				renderInfoSkybox.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
				renderInfoSkybox.renderArea = { .offset{0,0}, .extent{.width = window.getWidth(), .height = window.getHeight()} };
				renderInfoSkybox.layerCount = 1;
				renderInfoSkybox.colorAttachmentCount = 1;
				renderInfoSkybox.pColorAttachments = &colorAttachmentInfoSkybox;
				renderInfoSkybox.pDepthAttachment = &depthAttachmentInfoSkybox;
				vkCmdBeginRendering(cbDraw, &renderInfoSkybox);
					VkBuffer skyboxVertexBinding[1]{ skyboxData.getBufferHandle() };
					VkDeviceSize skyboxVertexOffsets[1]{ skyboxData.getOffset() };
					vkCmdBindVertexBuffers(cbDraw, 0, 1, skyboxVertexBinding, skyboxVertexOffsets);
					skyboxPipeline.cmdBindResourceSets(cbDraw);
					skyboxPipeline.cmdBind(cbDraw);
					vkCmdDraw(cbDraw, 36, 1, 0, 0);
				vkCmdEndRendering(cbDraw);

			cmdBufferSet.endRecording(cbDraw);
		} };
	node_t nodePostprocessCB{ flowGraph, [&](msg_t)
		{
			cbPostprocessing = cmdBufferSet.beginPerThreadRecording(2);

			SyncOperations::cmdExecuteBarrier(cbPostprocessing,
				{ {SyncOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
					VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT)} });

			SyncOperations::cmdExecuteBarrier(cbPostprocessing, 
				{{SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0, VK_ACCESS_SHADER_WRITE_BIT,
					VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
					std::get<0>(swapchainImageData),
					{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 })} });

			taa.adjustSmoothingFactor(WorldState::deltaTime, camInfo.speed, camInfo.camPosChanged);
			taa.updateJitterValue(coordinateTransformation.getCurrentJitter());
			taa.cmdDispatchTAA(cbPostprocessing, std::get<1>(swapchainImageData));
			
			SyncOperations::cmdExecuteBarrier(cbPostprocessing,
				{{SyncOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
					VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)}});

			VkRenderingAttachmentInfo colorAttachmentInfoDirectDraw{};
			colorAttachmentInfoDirectDraw.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			colorAttachmentInfoDirectDraw.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			colorAttachmentInfoDirectDraw.clearValue = VkClearValue{ .color{.float32{0.4f, 1.0f, 0.8f}} };
			colorAttachmentInfoDirectDraw.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			colorAttachmentInfoDirectDraw.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachmentInfoDirectDraw.imageView = std::get<1>(swapchainImageData);
			VkRenderingAttachmentInfo depthAttachmentInfoDirectDraw{};
			depthAttachmentInfoDirectDraw.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			depthAttachmentInfoDirectDraw.imageView = depthBuffer.getImageView();
			depthAttachmentInfoDirectDraw.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			depthAttachmentInfoDirectDraw.clearValue = { .depthStencil = {.depth = 0.0f, .stencil = 0} };
			depthAttachmentInfoDirectDraw.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			depthAttachmentInfoDirectDraw.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			VkRenderingInfo renderInfoDirectDraw{};
			renderInfoDirectDraw.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfoDirectDraw.renderArea = { .offset{0,0}, .extent{.width = window.getWidth(), .height = window.getHeight()} };
			renderInfoDirectDraw.layerCount = 1;
			renderInfoDirectDraw.colorAttachmentCount = 1;
			renderInfoDirectDraw.pColorAttachments = &colorAttachmentInfoDirectDraw;
			renderInfoDirectDraw.pDepthAttachment = &depthAttachmentInfoDirectDraw;
			vkCmdBeginRendering(cbPostprocessing, &renderInfoDirectDraw);
				if (renderingData.drawBVs)
					clusterer.cmdDrawBVs(cbPostprocessing);
				if (renderingData.drawLightProxies)
					clusterer.cmdDrawProxies(cbPostprocessing);

				if (renderingData.drawSpaceLines)
				{
					VkBuffer lineVertexBindings[1]{ spaceLinesVertexData.getBufferHandle() };
					VkDeviceSize lineVertexBindingOffsets[1]{ spaceLinesVertexData.getOffset() };
					vkCmdBindVertexBuffers(cbPostprocessing, 0, 1, lineVertexBindings, lineVertexBindingOffsets);
					spaceLinesPipeline.cmdBindResourceSets(cbPostprocessing);
					spaceLinesPipeline.cmdBind(cbPostprocessing);
					vkCmdPushConstants(cbPostprocessing, spaceLinesPipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &camInfo.camPos);
					vkCmdDraw(cbPostprocessing, lineVertNum, 1, 0, 0);
				}
			vkCmdEndRendering(cbPostprocessing);


			ui.startUIPass(cbPostprocessing, std::get<1>(swapchainImageData));
			ui.begin("Settings");
			ui.stats(renderingData, drawCount);
			ui.lightSettings(renderingData, pLight, sLight);
			ui.misc(renderingData);
			ui.end();
			ui.endUIPass(cbPostprocessing);

			SyncOperations::cmdExecuteBarrier(cbPostprocessing, 
				{{SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_NONE,
					VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, 0,
					VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
					std::get<0>(swapchainImageData),
					{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 })}});

			cmdBufferSet.endRecording(cbPostprocessing);
		} };
	node_t nodeComputeCB{ flowGraph, [&](msg_t)
		{
			currentCBindex = currentCBindex ? 0 : 1;
			cbCompute = cmdBufferSet.beginInterchangeableRecording(cbSetIndex, currentCBindex);

			depthBuffer.cmdCalcHiZ(cbCompute);

			cmdBufferSet.endRecording(cbCompute);
		} };

	oneapi::tbb::flow::make_edge(nodeAcquireSwapchainUpdateViewMatrices, nodePostprocessCB);
	oneapi::tbb::flow::make_edge(nodeAcquireSwapchainUpdateViewMatrices, nodeComputeCB);
	oneapi::tbb::flow::make_edge(nodeAcquireSwapchainUpdateViewMatrices, nodeFrustumCulling);
	oneapi::tbb::flow::make_edge(nodeAcquireSwapchainUpdateViewMatrices, nodeDrawCB);
	oneapi::tbb::flow::make_edge(nodeFrustumCulling, nodePreprocessCB1);
	oneapi::tbb::flow::make_edge(nodePreprocessCB1, nodePreprocessCB2);
	oneapi::tbb::flow::make_edge(nodePrepareDataForShadowMapRender, nodePreprocessCB2);
	oneapi::tbb::flow::make_edge(nodePreprocessCB2, nodePreprocessCB3);
	oneapi::tbb::flow::make_edge(nodePreprocessCB3, nodePreprocessCB4);
	clusterer.connectToFlowGraph(flowGraph, nodeAcquireSwapchainUpdateViewMatrices, nodePrepareDataForShadowMapRender, nodePreprocessCB3);

	vkDeviceWaitIdle(device);
	WorldState::initialize();
	while (!glfwWindowShouldClose(window))
	{
		TIME_MEASURE_START(100, 0);

		WorldState::refreshFrameTime();

		nodeAcquireSwapchainUpdateViewMatrices.try_put(oneapi::tbb::flow::continue_msg{});
		flowGraph.wait_for_all();

		submitAndWait(*vulkanObjectHandler, 
			cmdBufferSet, cbPreprocessing, cbDraw, cbPostprocessing, cbCompute, cbSetIndex, currentCBindex, 
			semaphore, semaphoreCompute, swapchainSemaphore, readyToPresentSemaphore, 
			presentInfo, 
			std::get<2>(swapchainImageData), swapChains[0]);

		glfwPollEvents();

		TIME_MEASURE_END(100, 0);
	}
	
	EASSERT(vkDeviceWaitIdle(device) == VK_SUCCESS, "Vulkan", "Device wait failed.");
	vkDestroySampler(device, generalSampler, nullptr);
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

void loadDefaultTextures(ImageListContainer& imageLists, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue)
{
	uint8_t data[16]
	{
		uint8_t(0), uint8_t(0), uint8_t(0), uint8_t(255), //BaseColor	
		uint8_t(127), uint8_t(127), uint8_t(255), uint8_t(0), //Normal	
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

	SyncOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
		{SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
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
	SyncOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
		{SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
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

	cmdBufferSet.resetAll();
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

	bool sideMoved{ false };
	int stateR = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
	if (stateR == GLFW_PRESS && mouseWasCaptured != true)
	{
		camInfo->camPos += static_cast<float>(-xOffs * camInfo->speed) * sideVec + static_cast<float>(yOffs * camInfo->speed) * upRelVec;
		sideMoved = true;
	}

	bool frontMoved{ false };
	int stateM = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);
	if (stateM == GLFW_PRESS && mouseWasCaptured != true)
	{
		camInfo->camPos += static_cast<float>(-yOffs * camInfo->speed) * camInfo->camFront;
		frontMoved = true;
	}

	camInfo->lastCursorP = {xpos, ypos};

	camInfo->camPosChanged = sideMoved || frontMoved;

	mouseWasCaptured = false;
}
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
}

void createResourceSets(VkDevice device,
	ResourceSet& transformMatricesRS, 
	const BufferMapped& transformMatrices,
	ResourceSet& materialsTexturesRS, 
	const ImageListContainer& imageLists,
	ResourceSet& skyboxLightingRS, 
	const ImageCubeMap& skybox,
	const ImageCubeMap& radiance,
	const ImageCubeMap& irradiance)
{
	VkDescriptorSetLayoutBinding transformMatricesBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT transformMatricesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = transformMatrices.getDeviceAddress(), .range = transformMatrices.getSize() };
	transformMatricesRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ transformMatricesBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{ std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &transformMatricesAddressInfo} } },
		false);

	VkDescriptorSetLayoutBinding imageListsBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 64, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	std::vector<VkDescriptorImageInfo> storageImageData(imageLists.getImageListCount());
	std::vector<VkDescriptorDataEXT> imageListsDescData(imageLists.getImageListCount());
	for (uint32_t i{ 0 }; i < imageListsDescData.size(); ++i)
	{
		storageImageData[i] = { .sampler = imageLists.getSampler(), .imageView = imageLists.getImageViewHandle(i), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		imageListsDescData[i].pCombinedImageSampler = &storageImageData[i];
	}
	materialsTexturesRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ imageListsBinding },
		std::array<VkDescriptorBindingFlags, 1>{ {VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT} },
		std::vector<std::vector<VkDescriptorDataEXT>>{ imageListsDescData },
		true);

	VkDescriptorSetLayoutBinding skyboxBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo skyboxImageInfo{ .sampler = skybox.getSampler(), .imageView = skybox.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding radianceBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo radianceImageInfo{ .sampler = radiance.getSampler(), .imageView = radiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding irradianceBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo irradianceImageInfo{ .sampler = irradiance.getSampler(), .imageView = irradiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	skyboxLightingRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ skyboxBinding, radianceBinding, irradianceBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{ 
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &skyboxImageInfo} } ,
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &radianceImageInfo} } ,
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &irradianceImageInfo} } },
		true);
}

void createDrawDataResourceSet(VkDevice device,
	ResourceSet& drawDataRS,
	const BufferMapped& drawData,
	const Buffer& drawDataIndices)
{
	VkDescriptorSetLayoutBinding drawDataBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT drawDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = drawData.getDeviceAddress(), .range = drawData.getSize() };
	VkDescriptorSetLayoutBinding drawDataIndicesBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT drawDataIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = drawDataIndices.getDeviceAddress(), .range = drawDataIndices.getSize() };
	drawDataRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ drawDataBinding, drawDataIndicesBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &drawDataAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &drawDataIndicesAddressInfo} } },
		false);
}

void createShadowMapResourceSet(VkDevice device,
	ResourceSet& shadowMapsRS,
	const ImageListContainer& shadowMapLists,
	const std::vector<ImageList>& shadowCubeMapLists, 
	const BufferBaseHostAccessible& shadowMapViewMatrices)
{
	VkDescriptorSetLayoutBinding shadowMapsBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 64, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	std::vector<VkDescriptorImageInfo> shadowMapsImageData(shadowMapLists.getImageListCount());
	std::vector<VkDescriptorDataEXT> shadowMapsDescData(shadowMapLists.getImageListCount());
	for (uint32_t i{ 0 }; i < shadowMapsDescData.size(); ++i)
	{
		shadowMapsImageData[i] = { .imageView = shadowMapLists.getImageViewHandle(i), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		shadowMapsDescData[i].pSampledImage = &shadowMapsImageData[i];
	}
	VkDescriptorSetLayoutBinding shadowCubeMapsBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 64, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	std::vector<VkDescriptorImageInfo> shadowCubeMapsImageData(shadowCubeMapLists.size());
	std::vector<VkDescriptorDataEXT> shadowCubeMapsDescData(shadowCubeMapLists.size());
	for (uint32_t i{ 0 }; i < shadowCubeMapsDescData.size(); ++i)
	{
		shadowCubeMapsImageData[i] = { .imageView = shadowCubeMapLists[i].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
		shadowCubeMapsDescData[i].pSampledImage = &shadowCubeMapsImageData[i];
	}
	VkDescriptorSetLayoutBinding samplerBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkSampler samplerData{ shadowMapLists.getSampler() };
	VkDescriptorSetLayoutBinding viewMatricesBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT viewMatricesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = shadowMapViewMatrices.getDeviceAddress(), .range = shadowMapViewMatrices.getSize() };

	shadowMapsRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ shadowMapsBinding, shadowCubeMapsBinding, samplerBinding, viewMatricesBinding },
		std::array<VkDescriptorBindingFlags, 4>{ VkDescriptorBindingFlags{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }, VkDescriptorBindingFlags{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }, 0 , 0},
		std::vector<std::vector<VkDescriptorDataEXT>>{ 
			shadowMapsDescData, 
			shadowCubeMapsDescData, 
			std::vector<VkDescriptorDataEXT>{VkDescriptorDataEXT{ .pSampler = &samplerData }},
			std::vector<VkDescriptorDataEXT>{VkDescriptorDataEXT{ .pStorageBuffer = &viewMatricesAddressInfo }} },
		true);
}

void createPBRResourceSet(VkDevice device,
	VkSampler generalSampler,
	ResourceSet& pbrRS,
	const Image& brdfLUT,
	const Image& aoImage)
{
	VkDescriptorSetLayoutBinding lutBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo lutImageInfo{ .sampler = generalSampler, .imageView = brdfLUT.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding aoImageBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo aoImageInfo{ .sampler = generalSampler, .imageView = aoImage.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	pbrRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ lutBinding, aoImageBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &lutImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &aoImageInfo} }},
		true);
}

void createDirecLightingResourceSet(VkDevice device,
	ResourceSet& directLightingRS,
	const BufferMapped& directionalLight,
	const BufferMapped& sortedLightsData,
	const BufferBaseHostAccessible& typeData,
	const BufferBaseHostInaccessible& tileData,
	const BufferMapped& zBinData)
{
	VkDescriptorSetLayoutBinding directionalLightBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT directionalLightAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = directionalLight.getDeviceAddress(), .range = directionalLight.getSize() };
	VkDescriptorSetLayoutBinding sortedLightsBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT sortedLightsAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = sortedLightsData.getDeviceAddress(), .range = sortedLightsData.getSize() };
	VkDescriptorSetLayoutBinding typesBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT typesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = typeData.getDeviceAddress(), .range = typeData.getSize() };
	VkDescriptorSetLayoutBinding tileDataBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT tileDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = tileData.getDeviceAddress(), .range = tileData.getSize() };
	VkDescriptorSetLayoutBinding zBinDataBinding{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT zBinDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = zBinData.getDeviceAddress(), .range = zBinData.getSize() };
	directLightingRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ directionalLightBinding, sortedLightsBinding, typesBinding, tileDataBinding, zBinDataBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pUniformBuffer = &directionalLightAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &sortedLightsAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &typesAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &tileDataAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &zBinDataAddressInfo} }},
		false);
}

Pipeline createSkyboxPipeline(PipelineAssembler& assembler, const ResourceSet& viewprojRS, const ResourceSet& skyboxLightingRS)
{
	std::array<std::reference_wrapper<const ResourceSet>, 2> resourceSets{ viewprojRS, skyboxLightingRS };
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

	cmdBufferSet.resetAll();
}

Pipeline createSpaceLinesPipeline(PipelineAssembler& assembler, const ResourceSet& viewprojRS)
{
	std::array<std::reference_wrapper<const ResourceSet>, 1> resourceSets{ viewprojRS };
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

	cmdBufferSet.resetAll();

	return vertNum;
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

void fillFrustumData(CoordinateTransformation& coordinateTransformation, Window& window, Clusterer& clusterer, HBAO& hbao, FrustumInfo& frustumInfo, ShadowCaster& caster, DeferredLighting& deferredLighting)
{
	float nearPlane{ NEAR_PLANE };
	float farPlane{ FAR_PLANE };
	float aspect{ static_cast<float>(static_cast<double>(window.getWidth()) / window.getHeight()) };
	float FOV{ glm::radians(80.0) };

	coordinateTransformation.updateProjectionMatrix(FOV, aspect, nearPlane, farPlane);
	clusterer.submitFrustum(nearPlane, farPlane, aspect, FOV);
	hbao.submitFrustum(nearPlane, farPlane, aspect, FOV);
	caster.submitFrustum(nearPlane, farPlane);
	deferredLighting.updateProjectionData(nearPlane, farPlane);

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
void fillModelMatrices(const BufferMapped& modelTransformData, const std::vector<glm::mat4>& modelMatrices)
{
	glm::mat4* transformMatrices{ reinterpret_cast<glm::mat4*>(modelTransformData.getData()) };
	for (int i{ 0 }; i < modelMatrices.size(); ++i)
	{
		transformMatrices[i] = modelMatrices[i];
	}
}
void fillDrawData(const BufferMapped& perDrawDataIndices, std::vector<StaticMesh>& staticMeshes, int drawCount)
{
	uint8_t* drawDataIndices{ reinterpret_cast<uint8_t*>(perDrawDataIndices.getData()) };
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

VkSampler createGeneralSampler(VkDevice device, float maxAnisotropy)
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

void submitAndWait(VulkanObjectHandler& vulkanObjectHandler,
	CommandBufferSet& cmdBufferSet, VkCommandBuffer cbPreprocessing, VkCommandBuffer cbDraw, VkCommandBuffer cbPostprocessing, VkCommandBuffer cbCompute,
	uint32_t indexToCBSet, uint32_t currentCommandBufferIndex,
	TimelineSemaphore& semaphore, TimelineSemaphore& semaphoreCompute, VkSemaphore swapchainSemaphore, VkSemaphore readyToPresentSemaphore,
	VkPresentInfoKHR& presentInfo, uint32_t swapchainIndex, VkSwapchainKHR& swapChain)
{
	VkSubmitInfo submitInfos[3]{};
	VkTimelineSemaphoreSubmitInfo semaphoreSubmit[3]{};
	uint64_t timelineVal{ semaphore.getValue() };
	uint64_t timelineValCompute{ semaphoreCompute.getValue() };

	uint64_t waitValues0[]{ timelineValCompute };
	uint64_t signalValues0[]{ ++timelineVal };
	VkSemaphore waitSemaphores0[]{ semaphoreCompute.getHandle() };
	VkSemaphore signalSemaphores0[]{ semaphore.getHandle() };
	VkPipelineStageFlags stageFlags0[]{ VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
	VkCommandBuffer cbs0[]{ cbPreprocessing, cbDraw };
	semaphoreSubmit[0] = TimelineSemaphore::getSubmitInfo(ARRAYSIZE(waitValues0), waitValues0, ARRAYSIZE(signalValues0), signalValues0);
	submitInfos[0] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit,
		.waitSemaphoreCount = ARRAYSIZE(waitSemaphores0), .pWaitSemaphores = waitSemaphores0, .pWaitDstStageMask = stageFlags0,
		.commandBufferCount = ARRAYSIZE(cbs0),.pCommandBuffers = cbs0,
		.signalSemaphoreCount = ARRAYSIZE(signalSemaphores0), .pSignalSemaphores = signalSemaphores0 };
	static bool firstIt{ true }; if (firstIt) { firstIt = false; submitInfos[0].waitSemaphoreCount = 0; }

	uint64_t signalValues1[]{ ++timelineVal, 0 };
	VkSemaphore waitSemaphores1[]{ swapchainSemaphore };
	VkSemaphore signalSemaphores1[]{ semaphore.getHandle(), readyToPresentSemaphore };
	VkPipelineStageFlags stageFlags1[]{ VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };
	semaphoreSubmit[1] = TimelineSemaphore::getSubmitInfo(0, nullptr, ARRAYSIZE(signalValues1), signalValues1);
	submitInfos[1] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit + 1,
		.waitSemaphoreCount = ARRAYSIZE(waitSemaphores1), .pWaitSemaphores = waitSemaphores1, .pWaitDstStageMask = stageFlags1,
		.commandBufferCount = 1, .pCommandBuffers = &cbPostprocessing,
		.signalSemaphoreCount = ARRAYSIZE(signalSemaphores1), .pSignalSemaphores = signalSemaphores1 };


	uint64_t waitValues3[]{ signalValues0[0] };
	uint64_t signalValues3[]{ ++timelineValCompute };
	VkSemaphore waitSemaphores3[]{ semaphore.getHandle() };
	VkSemaphore signalSemaphores3[]{ semaphoreCompute.getHandle() };
	VkPipelineStageFlags stageFlags3{ VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
	semaphoreSubmit[2] = TimelineSemaphore::getSubmitInfo(ARRAYSIZE(waitValues3), waitValues3, ARRAYSIZE(signalValues3), signalValues3);
	submitInfos[2] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit + 2,
		.waitSemaphoreCount = ARRAYSIZE(waitSemaphores3), .pWaitSemaphores = waitSemaphores3, .pWaitDstStageMask = &stageFlags3,
		.commandBufferCount = 1, .pCommandBuffers = &cbCompute,
		.signalSemaphoreCount = ARRAYSIZE(signalSemaphores3), .pSignalSemaphores = signalSemaphores3 };

	vkQueueSubmit(vulkanObjectHandler.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 2, submitInfos, VK_NULL_HANDLE);
	vkQueueSubmit(vulkanObjectHandler.getQueue(VulkanObjectHandler::COMPUTE_QUEUE_TYPE), 1, submitInfos + 2, VK_NULL_HANDLE);

	presentInfo.pImageIndices = &swapchainIndex;
	if (!vulkanObjectHandler.checkSwapchain(vkQueuePresentKHR(vulkanObjectHandler.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), &presentInfo)))
		swapChain = vulkanObjectHandler.getSwapchain();

	semaphore.wait(timelineVal);
	semaphore.newValue(timelineVal);
	semaphoreCompute.newValue(timelineValCompute);

	cmdBufferSet.resetPoolsOnThreads();
	cmdBufferSet.resetInterchangeable(indexToCBSet, currentCommandBufferIndex ? 0 : 1);
}