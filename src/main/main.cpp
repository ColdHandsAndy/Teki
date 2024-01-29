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
#include "src/rendering/renderer/GI.h"
#include "src/rendering/lighting/light_types.h"
#include "src/rendering/lighting/shadows.h"
#include "src/rendering/renderer/clusterer.h"
#include "src/rendering/renderer/culling.h"
#include "src/rendering/scene/parse_scene.h"
#include "src/rendering/scene/camera.h"
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

#define WINDOW_WIDTH_DEFAULT  1600u
#define WINDOW_HEIGHT_DEFAULT 900u
#define HBAO_WIDTH_DEFAULT  1280u
#define HBAO_HEIGHT_DEFAULT 720u

#define MAX_INDIRECT_DRAWS 4096
#define MAX_TRANSFORM_MATRICES 64

#define NEAR_PLANE 0.1
#define FAR_PLANE  10000.0

#define GENERAL_BUFFER_DEFAULT_SIZE 134217728ll
#define SHARED_BUFFER_DEFAULT_SIZE 8388608ll
#define DEVICE_BUFFER_DEFAULT_SIZE 268435456ll

namespace fs = std::filesystem;

std::shared_ptr<VulkanObjectHandler> initializeVulkan(const Window& window);

Pipeline createSkyboxPipeline(PipelineAssembler& assembler, const ResourceSet& viewprojRS, const ResourceSet& skyboxLightingRS);
Pipeline createSpaceLinesPipeline(PipelineAssembler& assembler, const ResourceSet& viewprojRS);

void createResourceSets(VkDevice device,
	ResourceSet& transformMatricesRS,
	const BufferMapped& transformMatrices,
	ResourceSet& materialsTexturesRS,
	const ImageListContainer& imageLists,
	ResourceSet& skyboxRS,
	const ImageCubeMap& skybox,
	ResourceSet& distantProbeRS,
	const ImageCubeMap& radiance);
void createDrawDataResourceSet(VkDevice device,
	ResourceSet& drawDataRS,
	const BufferMapped& drawData,
	const Buffer& drawDataIndices);
void createShadowMapResourceSet(VkDevice device,
	ResourceSet& shadowMapsRS,
	const ImageListContainer& shadowMapLists,
	const std::vector<ImageList>& shadowCubeMapLists,
	const BufferBaseHostAccessible& shadowMapViewMatrices,
	VkSampler nearestSampler,
	Image& shadowSamplingRotationTexture,
	BufferBaseHostAccessible& stagingBase,
	CommandBufferSet& cmdBufferSet,
	VkQueue queue);
void createBRDFLUTResourceSet(VkDevice device,
	VkSampler generalSampler,
	ResourceSet& brdfLUTRS,
	const Image& brdfLUT);
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

void fillFrustumData(CoordinateTransformation& coordinateTransformation, Camera& camera, Clusterer& clusterer, HBAO& hbao, FrustumInfo& frustumInfo, ShadowCaster& caster, DeferredLighting& deferredLighting);
void fillModelMatrices(const BufferMapped& modelTransformDataSSBO, const std::vector<glm::mat4>& modelMatrices);
void fillDrawData(const BufferMapped& perDrawDataIndicesSSBO, std::vector<StaticMesh>& staticMeshes, int drawCount);

void processInput(const Window& window, Camera& camera, float deltaTime, bool disableCursor);

void voxelize(GI& gi, CommandBufferSet& cmdBufferSet, VkQueue queue, const BufferMapped& indirectDrawCmdData, const Buffer& vertexData, const Buffer& indexData, uint32_t drawCmdCount, uint32_t drawCmdOffset, uint32_t drawCmdStride);

VkSampler createLinearSampler(VkDevice device, float maxAnisotropy);
VkSampler createNearestSampler(VkDevice device, float maxAnisotropy);

void submitAndWait(VulkanObjectHandler& vulkanObjectHandler,
	CommandBufferSet& cmdBufferSet, VkCommandBuffer cbPreprocessing, VkCommandBuffer cbDraw, VkCommandBuffer cbPostprocessing, VkCommandBuffer cbCompute,
	uint32_t indexToCBSet, uint32_t currentCommandBufferIndex,
	TimelineSemaphore& semaphore, TimelineSemaphore& semaphoreCompute, VkSemaphore swapchainSemaphore, VkSemaphore readyToPresentSemaphore,
	VkPresentInfoKHR& presentInfo, uint32_t swapchainIndex, VkSwapchainKHR& swapChain);

int main()
{
	EASSERT(glfwInit(), "GLFW", "GLFW was not initialized.")

	Window window{ WINDOW_WIDTH_DEFAULT, WINDOW_HEIGHT_DEFAULT, "Teki", false };
	//Window window{ "Teki" };
	Camera camera{NEAR_PLANE, FAR_PLANE, glm::radians(80.0f), static_cast<float>(window.getWidth()) / static_cast<float>(window.getHeight())};
	glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GLFW_TRUE);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);

	std::shared_ptr<VulkanObjectHandler> vulkanObjectHandler{ initializeVulkan(window) };

	MemoryManager memManager{ *vulkanObjectHandler };
	BufferBase::assignGlobalMemoryManager(memManager);
	ImageBase::assignGlobalMemoryManager(memManager);
	
	DescriptorManager descManager{ *vulkanObjectHandler };
	ResourceSet::assignGlobalDescriptorManager(descManager);

	CommandBufferSet cmdBufferSet{ *vulkanObjectHandler };

	UI ui{ window, *vulkanObjectHandler, cmdBufferSet };
	
	VkDevice device{ vulkanObjectHandler->getLogicalDevice() };

	BufferBaseHostInaccessible baseDeviceBuffer{ device, DEVICE_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };
	BufferBaseHostAccessible baseHostCachedBuffer{ device, GENERAL_BUFFER_DEFAULT_SIZE, 
		VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, BufferBase::DEDICATED_FLAG, false, true };

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
	VkSampler linearSampler{ createLinearSampler(device, vulkanObjectHandler->getPhysDevLimits().maxSamplerAnisotropy) };
	VkSampler nearestSampler{ createNearestSampler(device, vulkanObjectHandler->getPhysDevLimits().maxSamplerAnisotropy) };
	Image shadowSamplingRotationTexture{ device, VK_FORMAT_R8G8B8A8_SNORM, 2 * 32, 2 * 32, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT };
	Image brdfLUT{ TextureLoaders::loadTexture(*vulkanObjectHandler, cmdBufferSet, baseHostBuffer, "internal/brdfLUT/brdfLUT.ktx2", VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT) };
	ImageCubeMap cubemapSkybox{ TextureLoaders::loadCubemap(*vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "skybox.ktx2") };
	ImageCubeMap cubemapSkyboxRadiance{ TextureLoaders::loadCubemap(*vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "radiance.ktx2") };
	//ImageCubeMap cubemapSkyboxIrradiance{ TextureLoaders::loadCubemap(*vulkanObjectHandler, cmdBufferSet, baseHostBuffer, envPath / "irradiance.ktx2") };
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
			.compareEnable = VK_TRUE,
			.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			.minLod = 0.0f,
			.maxLod = 128.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE }, 
			2, VK_IMAGE_ASPECT_DEPTH_BIT };
	std::vector<ImageList> shadowCubeMaps{};
	FrustumInfo frustumInfo{};
	OBBs rUnitOBBs{ MAX_INDIRECT_DRAWS };
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
	ResourceSet skyboxRS{};
	ResourceSet distantProbeRS{};
	ResourceSet drawDataRS{};
	ResourceSet BRDFLUTRS{};
	ResourceSet directLightingRS{};
	createResourceSets(device,
		transformMatricesRS, transformMatrices, 
		materialsTexturesRS, materialsTextures, 
		skyboxRS, cubemapSkybox, 
		distantProbeRS, cubemapSkyboxRadiance);
	DepthBuffer depthBuffer{ device, window.getWidth(), window.getHeight() };
	Clusterer clusterer{ device, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), window.getWidth(), window.getHeight(), coordinateTransformation.getResourceSet() };
	ShadowCaster caster{ device, clusterer, shadowMaps, shadowCubeMaps, indirectDrawCmdData, transformMatrices, drawData, rUnitOBBs };
	Culling culling{ device, MAX_INDIRECT_DRAWS, NEAR_PLANE, coordinateTransformation.getResourceSet(), indirectDrawCmdData, depthBuffer, vulkanObjectHandler->getComputeFamilyIndex(), vulkanObjectHandler->getGraphicsFamilyIndex()};
	HBAO hbao{ device, HBAO_WIDTH_DEFAULT, HBAO_HEIGHT_DEFAULT, depthBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE) };
	GI gi{ device, window.getWidth(), window.getHeight(), baseHostBuffer, baseDeviceBuffer, clusterer};
	renderingData.countROM = gi.getCountROM();
	LightTypes::LightBase::assignGlobalGI(gi);
	LightTypes::LightBase::assignGlobalClusterer(clusterer);
	LightTypes::LightBase::assignGlobalShadowCaster(caster);

	LightTypes::DirectionalLight dirLight{ {1.0f, 1.0f, 1.0f}, 0.0f, {0.0f, -1.0f, 0.0f} };
	dirLight.plantData(directionalLight.getData());
	std::array<LightTypes::PointLight, 5> pointLights{ 
		LightTypes::PointLight(glm::vec3{2.560f, 5.320f, 4.760f}, glm::vec3{0.8f, 0.4f, 0.2f}, 4176.0f, 50.0f, 2048, 0.0, true),
		LightTypes::PointLight(glm::vec3{4.5f, 3.2f, 0.0f}, glm::vec3{0.8f, 0.4f, 0.2f}, 200.0f, 0.0f, 256, 0.0, true),
		LightTypes::PointLight(glm::vec3{4.5f, 3.2f, 0.0f}, glm::vec3{0.8f, 0.4f, 0.2f}, 200.0f, 0.0f, 256, 0.0, true),
		LightTypes::PointLight(glm::vec3{4.5f, 3.2f, 0.0f}, glm::vec3{0.8f, 0.4f, 0.2f}, 200.0f, 0.0f, 256, 0.0, true),
		LightTypes::PointLight(glm::vec3{4.5f, 3.2f, 0.0f}, glm::vec3{0.8f, 0.4f, 0.2f}, 200.0f, 0.0f, 256, 0.0, true),
	};
	std::array<LightTypes::SpotLight, 1> spotLights{ 
		LightTypes::SpotLight(glm::vec3{-8.5f, 14.0f, -3.5f}, glm::vec3{0.6f, 0.5f, 0.7f}, 1000.0f, 0.0f, glm::vec3{0.0, -1.0, 0.3}, glm::radians(25.0), glm::radians(30.0), 1024, 0.0, true)
	};

	createDrawDataResourceSet(device, drawDataRS, drawData, culling.getDrawDataIndexBuffer());
	createBRDFLUTResourceSet(device, linearSampler, BRDFLUTRS, brdfLUT);
	createShadowMapResourceSet(device, shadowMapsRS, shadowMaps, shadowCubeMaps, caster.getShadowViewMatrices(), nearestSampler, shadowSamplingRotationTexture, baseHostBuffer, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
	createDirecLightingResourceSet(device, directLightingRS, directionalLight, clusterer.getSortedLights(), clusterer.getSortedTypeData(), clusterer.getTileData(), clusterer.getZBin());
	gi.initialize(device, drawDataRS, transformMatricesRS, materialsTexturesRS, distantProbeRS, BRDFLUTRS, shadowMapsRS, linearSampler);
	gi.initializeDebug(device, coordinateTransformation.getResourceSet(), window.getWidth(), window.getHeight(), baseHostBuffer, linearSampler, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE));
	DeferredLighting deferredLighting{device, window.getWidth(), window.getHeight(), 
		depthBuffer, 
		hbao,
		coordinateTransformation.getResourceSet(), transformMatricesRS, 
		materialsTexturesRS, shadowMapsRS, 
		gi.getIndirectDiffuseLightingResourceSet(), gi.getIndirectSpecularLightingResourceSet(), gi.getIndirectLightingMetadataResourceSet(),
		distantProbeRS,
		drawDataRS, BRDFLUTRS, directLightingRS, linearSampler };
	deferredLighting.updateTileWidth(clusterer.getWidthInTiles());
	TAA taa{ device, depthBuffer, deferredLighting.getFramebuffer(), coordinateTransformation.getResourceSet(), cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE) };
	rUnitOBBs.initVisualizationResources(device, window.getWidth(), window.getHeight(), coordinateTransformation.getResourceSet());

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
	Pipeline skyboxPipeline{ createSkyboxPipeline(assembler, coordinateTransformation.getResourceSet(), skyboxRS) };
	
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_LINE_DRAWING);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.5f);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DEFAULT);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);
	Pipeline spaceLinesPipeline{ createSpaceLinesPipeline(assembler, coordinateTransformation.getResourceSet()) };


	fillFrustumData(coordinateTransformation, camera, clusterer, hbao, frustumInfo, caster, deferredLighting);
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


	constexpr bool profile = true;
	constexpr uint32_t queryNum = 12;
	TimestampQueries<queryNum> queries{ *vulkanObjectHandler };
	renderingData.gpuTasks.resize(queryNum);
	constexpr uint32_t gqQueryOffset = 0;
	constexpr uint32_t gqQueryCount = 8;
	constexpr uint32_t queryIndexHiZ = 0;
	constexpr uint32_t queryIndexShadowMaps = 1;
	constexpr uint32_t queryIndexTileTest = 2;
	constexpr uint32_t queryIndexUVbufferDraw = 3;
	constexpr uint32_t queryIndexLightingPass = 4;
	constexpr uint32_t queryIndexHBAO = 5;
	constexpr uint32_t queryIndexTAA = 6;
	constexpr uint32_t queryIndexGIInjectLights = 7;
	constexpr uint32_t cqQueryOffset = 8;
	constexpr uint32_t cqQueryCount = 4;
	constexpr uint32_t queryIndexGICreateROMA = 8;
	constexpr uint32_t queryIndexGITraceProbes = 9;
	constexpr uint32_t queryIndexGIComputeIrradianceAndVisibility = 10;
	constexpr uint32_t queryIndexGITraceSpecular = 11;
	renderingData.gpuTasks[queryIndexHiZ].name = "HiZ";
	renderingData.gpuTasks[queryIndexHiZ].color = legit::Colors::asbestos;
	renderingData.gpuTasks[queryIndexShadowMaps].name = "Shadow maps";
	renderingData.gpuTasks[queryIndexShadowMaps].color = legit::Colors::midnightBlue;
	renderingData.gpuTasks[queryIndexTileTest].name = "Tile test";
	renderingData.gpuTasks[queryIndexTileTest].color = legit::Colors::alizarin;
	renderingData.gpuTasks[queryIndexUVbufferDraw].name = "(Deferred) UV-buffer draw";
	renderingData.gpuTasks[queryIndexUVbufferDraw].color = legit::Colors::turqoise;
	renderingData.gpuTasks[queryIndexLightingPass].name = "(Deferred) Lighting pass";
	renderingData.gpuTasks[queryIndexLightingPass].color = legit::Colors::greenSea;
	renderingData.gpuTasks[queryIndexHBAO].name = "HBAO";
	renderingData.gpuTasks[queryIndexHBAO].color = legit::Colors::wetAsphalt;
	renderingData.gpuTasks[queryIndexTAA].name = "TAA";
	renderingData.gpuTasks[queryIndexTAA].color = legit::Colors::silver;
	renderingData.gpuTasks[queryIndexGICreateROMA].name = "(GI) Create ROMA";
	renderingData.gpuTasks[queryIndexGICreateROMA].color = legit::Colors::emerald;
	renderingData.gpuTasks[queryIndexGITraceProbes].name = "(GI) Trace probes";
	renderingData.gpuTasks[queryIndexGITraceProbes].color = legit::Colors::peterRiver;
	renderingData.gpuTasks[queryIndexGIComputeIrradianceAndVisibility].name = "(GI) Compute irradiance";
	renderingData.gpuTasks[queryIndexGIComputeIrradianceAndVisibility].color = legit::Colors::belizeHole;
	renderingData.gpuTasks[queryIndexGIInjectLights].name = "(GI) Inject lights";
	renderingData.gpuTasks[queryIndexGIInjectLights].color = legit::Colors::nephritis;
	renderingData.gpuTasks[queryIndexGITraceSpecular].name = "(GI) Trace specular";
	renderingData.gpuTasks[queryIndexGITraceSpecular].color = legit::Colors::carrot;


	SyncOperations::EventHolder<3> events{ device };

	VkCommandBuffer cbPreprocessing{};
	VkCommandBuffer cbDraw{};
	VkCommandBuffer cbPostprocessing{};
	VkCommandBuffer cbCompute{};
	uint32_t cbSetIndex{ cmdBufferSet.createInterchangeableSet(2, CommandBufferSet::ASYNC_COMPUTE_CB) };
	uint32_t currentCBindex{ 1 };
	uint32_t currentProbesIndex{ 0 };
	
	typedef oneapi::tbb::flow::continue_node<oneapi::tbb::flow::continue_msg> node_t;
	typedef const oneapi::tbb::flow::continue_msg& msg_t;
	oneapi::tbb::flow::graph flowGraph{};
	node_t nodePrepare{ flowGraph, [&](msg_t)
		{
			if (!vulkanObjectHandler->checkSwapchain(vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex)))
			{
				vkAcquireNextImageKHR(device, vulkanObjectHandler->getSwapchain(), UINT64_MAX, swapchainSemaphore, VK_NULL_HANDLE, &swapchainIndex);
				swapChains[0] = vulkanObjectHandler->getSwapchain();
			}
			swapchainImageData = vulkanObjectHandler->getSwapchainImageData(swapchainIndex);

			deferredLighting.updateCameraPosition(camera.getPosition());
			deferredLighting.updateGISceneCenter(gi.getScenePosition());
			deferredLighting.updateSkyboxState(renderingData.skyboxEnabled);

			coordinateTransformation.updateProjectionMatrixJitter();
			coordinateTransformation.updateViewMatrix(camera.getPosition(), camera.getPosition() + camera.getForwardDirection(), camera.getUpDirection());
			clusterer.submitViewMatrix(coordinateTransformation.getViewMatrix());

			currentProbesIndex = gi.getIndirectLightingCurrentSet();
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

			if (profile) queries.cmdReset(cbPreprocessing, gqQueryOffset, gqQueryCount);

			culling.cmdTransferSetDrawCountToZero(cbPreprocessing);

			if (profile) queries.cmdWriteStart(cbPreprocessing, queryIndexHiZ);
			depthBuffer.cmdCalcHiZ(cbPreprocessing);
			if (profile) queries.cmdWriteEnd(cbPreprocessing, queryIndexHiZ);

			SyncOperations::cmdExecuteBarrier(cbPreprocessing, culling.getDependency());
			culling.cmdDispatchCullOccluded(cbPreprocessing);
			clusterer.cmdTransferClearTileBuffer(cbPreprocessing);
			events.cmdSet(cbPreprocessing, 0, clusterer.getDependency()); //Event 1 set
		} };
	node_t nodePreprocessCB2{ flowGraph, [&](msg_t)
		{
			hbao.cmdTransferClearBuffers(cbPreprocessing);

			if (profile) queries.cmdWriteStart(cbPreprocessing, queryIndexShadowMaps);
			caster.cmdRenderShadowMaps(cbPreprocessing, vertexData, indexData);
			if (profile) queries.cmdWriteEnd(cbPreprocessing, queryIndexShadowMaps);

			uint32_t indices[]{ 0 };
			events.cmdWait(cbPreprocessing, 1, indices, &clusterer.getDependency()); //Event 1 wait
		} };
	node_t nodePreprocessCB3{ flowGraph, [&](msg_t)
		{
			if (profile) queries.cmdWriteStart(cbPreprocessing, queryIndexTileTest);
			clusterer.cmdPassConductTileTest(cbPreprocessing);
			if (profile) queries.cmdWriteEnd(cbPreprocessing, queryIndexTileTest);
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

				gi.cmdInjectLights(cbDraw, queries, queryIndexGIInjectLights, profile);

				if (renderingData.voxelDebug != UiData::NONE_VOXEL_DEBUG)
				{
					SyncOperations::cmdExecuteBarrier(cbDraw, { {SyncOperations::constructImageBarrier(
						VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
						VK_ACCESS_NONE, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
						VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
						depthBuffer.getImageHandle(), depthBuffer.getDepthBufferSubresourceRange())} });
				}
				else
				{
					if (profile) queries.cmdWriteStart(cbDraw, queryIndexUVbufferDraw);
					deferredLighting.cmdPassDrawToUVBuffer(cbDraw, culling, vertexData, indexData);
					if (profile) queries.cmdWriteEnd(cbDraw, queryIndexUVbufferDraw);
				}

				SyncOperations::cmdExecuteBarrier(cbDraw,
					{ {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
						VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
						VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
						depthBuffer.getImageHandle(), depthBuffer.getDepthBufferSubresourceRange())} });

				if (profile) queries.cmdWriteStart(cbDraw, queryIndexHBAO);
				hbao.cmdDispatchHBAO(cbDraw);
				SyncOperations::cmdExecuteBarrier(cbDraw, hbao.getDependency());
				hbao.cmdDispatchHBAOBlur(cbDraw);
				if (profile) queries.cmdWriteEnd(cbDraw, queryIndexHBAO);


				SyncOperations::cmdExecuteBarrier(cbDraw, deferredLighting.getDependency());

				deferredLighting.updateLightBinWidth(clusterer.getCurrentBinWidth());

				if (renderingData.voxelDebug == UiData::NONE_VOXEL_DEBUG)
				{
					if (profile) queries.cmdWriteStart(cbDraw, queryIndexLightingPass);
					deferredLighting.cmdDispatchLightingCompute(cbDraw, currentProbesIndex);
					if (profile) queries.cmdWriteEnd(cbDraw, queryIndexLightingPass);
				}

				SyncOperations::cmdExecuteBarrier(cbDraw, 
					{ {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
						VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
						VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
						depthBuffer.getImageHandle(), depthBuffer.getDepthBufferSubresourceRange())} });
				
				if (renderingData.skyboxEnabled)
				{
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
				}


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

			if (profile) queries.cmdWriteStart(cbPostprocessing, queryIndexTAA);
			taa.adjustSmoothingFactor(WorldState::deltaTime, camera.getSpeed(), camera.cameraPositionChanged());
			taa.updateJitterValue(coordinateTransformation.getCurrentJitter());
			taa.cmdDispatchTAA(cbPostprocessing, std::get<1>(swapchainImageData));
			if (profile) queries.cmdWriteEnd(cbPostprocessing, queryIndexTAA);
			
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

				if (renderingData.showOBBs)
					rUnitOBBs.cmdVisualizeOBBs(cbPostprocessing);

				if (renderingData.voxelDebug == UiData::BOM_VOXEL_DEBUG)
					gi.cmdDrawBOM(cbPostprocessing, camera.getPosition());
				else if (renderingData.voxelDebug == UiData::ROM_VOXEL_DEBUG)
					gi.cmdDrawROM(cbPostprocessing, camera.getPosition(), renderingData.indexROM);
				else if (renderingData.voxelDebug == UiData::ALBEDO_VOXEL_DEBUG)
					gi.cmdDrawAlbedo(cbPostprocessing, camera.getPosition());
				else if (renderingData.voxelDebug == UiData::METALNESS_VOXEL_DEBUG)
					gi.cmdDrawMetalness(cbPostprocessing, camera.getPosition());
				else if (renderingData.voxelDebug == UiData::ROUGHNESS_VOXEL_DEBUG)
					gi.cmdDrawRoughness(cbPostprocessing, camera.getPosition());
				else if (renderingData.voxelDebug == UiData::EMISSION_VOXEL_DEBUG)
					gi.cmdDrawEmission(cbPostprocessing, camera.getPosition());

				if (renderingData.probeDebug == UiData::RADIANCE_PROBE_DEBUG)
					gi.cmdDrawRadianceProbes(cbPostprocessing);
				else if (renderingData.probeDebug == UiData::IRRADIANCE_PROBE_DEBUG)
					gi.cmdDrawIrradianceProbes(cbPostprocessing);
				else if (renderingData.probeDebug == UiData::VISIBILITY_PROBE_DEBUG)
					gi.cmdDrawVisibilityProbes(cbPostprocessing);

				if (renderingData.drawSpaceGrid)
				{
					VkBuffer lineVertexBindings[1]{ spaceLinesVertexData.getBufferHandle() };
					VkDeviceSize lineVertexBindingOffsets[1]{ spaceLinesVertexData.getOffset() };
					vkCmdBindVertexBuffers(cbPostprocessing, 0, 1, lineVertexBindings, lineVertexBindingOffsets);
					spaceLinesPipeline.cmdBindResourceSets(cbPostprocessing);
					spaceLinesPipeline.cmdBind(cbPostprocessing);
					vkCmdPushConstants(cbPostprocessing, spaceLinesPipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(glm::vec3), &camera.getPosition());
					vkCmdDraw(cbPostprocessing, lineVertNum, 1, 0, 0);
				}
			vkCmdEndRendering(cbPostprocessing);


			ui.startUIPass(cbPostprocessing, std::get<1>(swapchainImageData));
			ui.begin("Settings");
			ui.stats(renderingData, drawCount);
			ui.lightSettings(renderingData, pointLights, spotLights);
			ui.misc(renderingData);
			ui.end();
			ui.profiler(renderingData, profile);
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

			if (profile) queries.cmdReset(cbCompute, cqQueryOffset, cqQueryCount);
			{
				gi.cmdComputeIndirect(cbCompute, 
					coordinateTransformation.getInverseViewProjectionMatrix(), 
					camera.getPosition(),
					queries, 
					queryIndexGICreateROMA, queryIndexGITraceSpecular, queryIndexGITraceProbes, queryIndexGIComputeIrradianceAndVisibility, 
					renderingData.skyboxEnabled,
					profile);
			}

			cmdBufferSet.endRecording(cbCompute);
		} };
	
	oneapi::tbb::flow::make_edge(nodePrepare, nodePostprocessCB);
	oneapi::tbb::flow::make_edge(nodePrepare, nodeComputeCB);
	oneapi::tbb::flow::make_edge(nodePrepare, nodeFrustumCulling);
	oneapi::tbb::flow::make_edge(nodePrepare, nodeDrawCB);
	oneapi::tbb::flow::make_edge(nodeFrustumCulling, nodePreprocessCB1);
	oneapi::tbb::flow::make_edge(nodePreprocessCB1, nodePreprocessCB2);
	oneapi::tbb::flow::make_edge(nodePrepareDataForShadowMapRender, nodePreprocessCB2);
	oneapi::tbb::flow::make_edge(nodePreprocessCB2, nodePreprocessCB3);
	oneapi::tbb::flow::make_edge(nodePreprocessCB3, nodePreprocessCB4);
	clusterer.connectToFlowGraph(flowGraph, nodePrepare, nodePrepareDataForShadowMapRender, nodePreprocessCB3);
	
	voxelize(gi, cmdBufferSet, vulkanObjectHandler->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), indirectDrawCmdData, vertexData, indexData, drawCount, 0, sizeof(IndirectData));

	vkDeviceWaitIdle(device);
	WorldState::initialize();
	while (!glfwWindowShouldClose(window))
	{
		WorldState::refreshFrameTime();
		glfwPollEvents();

		processInput(window, camera, WorldState::deltaTime, ui.cursorOnUI());

		nodePrepare.try_put(oneapi::tbb::flow::continue_msg{});
		flowGraph.wait_for_all();

		queries.updateResults();

		submitAndWait(*vulkanObjectHandler, 
			cmdBufferSet, cbPreprocessing, cbDraw, cbPostprocessing, cbCompute, cbSetIndex, currentCBindex, 
			semaphore, semaphoreCompute, swapchainSemaphore, readyToPresentSemaphore, 
			presentInfo, 
			std::get<2>(swapchainImageData), swapChains[0]);

		queries.uploadQueryDataToProfilerTasks(renderingData.gpuTasks.data(), renderingData.gpuTasks.size());

		//vkDeviceWaitIdle(device);
	}
	
	EASSERT(vkDeviceWaitIdle(device) == VK_SUCCESS, "Vulkan", "Device wait failed.");
	vkDestroySampler(device, linearSampler, nullptr);
	vkDestroySampler(device, nearestSampler, nullptr);
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

void createResourceSets(VkDevice device,
	ResourceSet& transformMatricesRS, 
	const BufferMapped& transformMatrices,
	ResourceSet& materialsTexturesRS, 
	const ImageListContainer& imageLists,
	ResourceSet& skyboxRS, 
	const ImageCubeMap& skybox,
	ResourceSet& distantProbeRS,
	const ImageCubeMap& radiance)
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
	skyboxRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ skyboxBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{ 
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &skyboxImageInfo} } },
		true);

	/*VkDescriptorSetLayoutBinding radianceBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo radianceImageInfo{ .sampler = radiance.getSampler(), .imageView = radiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding irradianceBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo irradianceImageInfo{ .sampler = irradiance.getSampler(), .imageView = irradiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	distantProbeRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ radianceBinding, irradianceBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &radianceImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &irradianceImageInfo} }},
		true);*/
	VkDescriptorSetLayoutBinding radianceBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo radianceImageInfo{ .sampler = radiance.getSampler(), .imageView = radiance.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	distantProbeRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ radianceBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &radianceImageInfo} } },
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
	const BufferBaseHostAccessible& shadowMapViewMatrices,
	VkSampler nearestSampler,
	Image& shadowSamplingRotationTexture,
	BufferBaseHostAccessible& stagingBase, 
	CommandBufferSet& cmdBufferSet, 
	VkQueue queue)
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
	VkDescriptorSetLayoutBinding linearSamplerBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkSampler linearSamplerData{ shadowMapLists.getSampler() };
	VkDescriptorSetLayoutBinding nearestSamplerBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkSampler nearestSamplerData{ nearestSampler };
	VkDescriptorSetLayoutBinding viewMatricesBinding{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorAddressInfoEXT viewMatricesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = shadowMapViewMatrices.getDeviceAddress(), .range = shadowMapViewMatrices.getSize() };

	VkDescriptorSetLayoutBinding offsetTextureBinding{ .binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo offsetTextureImageInfo{ .sampler = nearestSampler, .imageView = shadowSamplingRotationTexture.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL};

	shadowMapsRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ shadowMapsBinding, shadowCubeMapsBinding, linearSamplerBinding, nearestSamplerBinding, viewMatricesBinding, offsetTextureBinding },
		std::array<VkDescriptorBindingFlags, 6>{ VkDescriptorBindingFlags{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }, VkDescriptorBindingFlags{ VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }, 0 , 0, 0, 0 },
		std::vector<std::vector<VkDescriptorDataEXT>>{ 
			shadowMapsDescData, 
			shadowCubeMapsDescData, 
			std::vector<VkDescriptorDataEXT>{VkDescriptorDataEXT{ .pSampler = &linearSamplerData }},
			std::vector<VkDescriptorDataEXT>{VkDescriptorDataEXT{ .pSampler = &nearestSamplerData }},
			std::vector<VkDescriptorDataEXT>{VkDescriptorDataEXT{ .pStorageBuffer = &viewMatricesAddressInfo }},
			std::vector<VkDescriptorDataEXT>{VkDescriptorDataEXT{ .pSampledImage = &offsetTextureImageInfo }},
		},
		true);


	std::random_device rd{};
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(0.0, 1.0);
	constexpr int strataCountHor{ 4 };
	constexpr int strataCountVert{ 2 };
	constexpr float stratumHorSize{ 1.0f / 4.0f };
	constexpr float stratumVertSize{ 1.0f / 2.0f };

	BufferMapped staging{ stagingBase, sizeof(uint32_t) * 2 * 2 * 32 * 32 };
	uint32_t* data{ reinterpret_cast<uint32_t*>(staging.getData()) };

	int n{ 0 };
	for (int j{ 0 }; j < strataCountVert * 32; ++j)
	{
		for (int i{ 0 }; i < strataCountHor * 32;)
		{
			float u = dist(mt);
			float x1 = std::cos(2 * glm::pi<float>() * u);
			float y1 = std::sin(2 * glm::pi<float>() * u);
			++i;
			float v = dist(mt);
			float x2 = std::cos(2 * glm::pi<float>() * v);
			float y2 = std::sin(2 * glm::pi<float>() * v);
			++i;
			*(data + n) = glm::packSnorm4x8(glm::vec4(x1, y1, x2, y2));
			++n;
		}
	}

	VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };
		SyncOperations::cmdExecuteBarrier(cb, 
			{ {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT, 
			VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT, 
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
			shadowSamplingRotationTexture.getImageHandle(), shadowSamplingRotationTexture.getSubresourceRange())}});
		shadowSamplingRotationTexture.cmdCopyDataFromBuffer(cb, staging.getBufferHandle(), staging.getOffset(), 0, 0, shadowSamplingRotationTexture.getWidth(), shadowSamplingRotationTexture.getHeight());
		SyncOperations::cmdExecuteBarrier(cb,
			{ {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_NONE,
			VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_NONE,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
			shadowSamplingRotationTexture.getImageHandle(), shadowSamplingRotationTexture.getSubresourceRange())} });
	cmdBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	cmdBufferSet.resetAllTransient();
}

void createBRDFLUTResourceSet(VkDevice device,
	VkSampler generalSampler,
	ResourceSet& brdfLUTRS,
	const Image& brdfLUT)
{
	VkDescriptorSetLayoutBinding lutBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo lutImageInfo{ .sampler = generalSampler, .imageView = brdfLUT.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	brdfLUTRS.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ lutBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &lutImageInfo} }},
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

void fillFrustumData(CoordinateTransformation& coordinateTransformation, Camera& camera, Clusterer& clusterer, HBAO& hbao, FrustumInfo& frustumInfo, ShadowCaster& caster, DeferredLighting& deferredLighting)
{
	float nearPlane{ camera.getNear() };
	float farPlane{ camera.getFar() };
	float aspect{ camera.getAspect() };
	float FOV{ camera.getFOV() };

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

void processInput(const Window& window, Camera& camera, float deltaTime, bool disableCursor)
{
	bool cameraMoved = false;

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		camera.move(Camera::FORWARD, deltaTime);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		camera.move(Camera::LEFT, deltaTime);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		camera.move(Camera::BACK, deltaTime);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		camera.move(Camera::RIGHT, deltaTime);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
	{
		camera.move(Camera::UP, deltaTime);
		cameraMoved = true;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
	{
		camera.move(Camera::DOWN, deltaTime);
		cameraMoved = true;
	}

	double xpos{};
	double ypos{};
	glfwGetCursorPos(window, &xpos, &ypos);
	static bool invalidateLastCursorPos = false;
	if (!disableCursor)
	{
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
		{
			camera.moveFrom2DInput(glm::vec2(static_cast<float>(xpos / window.getWidth()), static_cast<float>(ypos / window.getHeight())), invalidateLastCursorPos);
			invalidateLastCursorPos = false;
		}
		else
		{
			invalidateLastCursorPos = true;
		}
	}

	if (!cameraMoved)
		camera.setCameraPositionLeftUnchanged();
}

void voxelize(GI& gi, CommandBufferSet& cmdBufferSet, VkQueue queue, const BufferMapped& indirectDrawCmdData, const Buffer& vertexData, const Buffer& indexData, uint32_t drawCmdCount, uint32_t drawCmdOffset, uint32_t drawCmdStride)
{
	VkCommandBuffer cbtr{ cmdBufferSet.beginTransientRecording() };
		gi.cmdVoxelize(cbtr, indirectDrawCmdData, vertexData, indexData, drawCmdCount, 0, sizeof(IndirectData));
	cmdBufferSet.endRecording(cbtr);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cbtr };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	cmdBufferSet.resetAllTransient();
}

VkSampler createLinearSampler(VkDevice device, float maxAnisotropy)
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
VkSampler createNearestSampler(VkDevice device, float maxAnisotropy)
{
	VkSamplerCreateInfo samplerCI{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
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
	VkSubmitInfo submitInfos[4]{};
	VkTimelineSemaphoreSubmitInfo semaphoreSubmit[3]{};
	uint64_t timelineVal{ semaphore.getValue() };
	uint64_t timelineValCompute{ semaphoreCompute.getValue() };

	submitInfos[0] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,.pCommandBuffers = &cbPreprocessing};

	uint64_t waitValues1[]{ timelineValCompute };
	uint64_t signalValues1[]{ ++timelineVal };
	VkSemaphore waitSemaphores1[]{ semaphoreCompute.getHandle() };
	VkSemaphore signalSemaphores1[]{ semaphore.getHandle() };
	VkPipelineStageFlags stageFlags1[]{ VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT };
	semaphoreSubmit[0] = TimelineSemaphore::getSubmitInfo(ARRAYSIZE(waitValues1), waitValues1, ARRAYSIZE(signalValues1), signalValues1);
	submitInfos[1] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit,
		.waitSemaphoreCount = ARRAYSIZE(waitSemaphores1), .pWaitSemaphores = waitSemaphores1, .pWaitDstStageMask = stageFlags1,
		.commandBufferCount = 1,.pCommandBuffers = &cbDraw,
		.signalSemaphoreCount = ARRAYSIZE(signalSemaphores1), .pSignalSemaphores = signalSemaphores1 };
	static bool firstIt{ true }; if (firstIt) { firstIt = false; submitInfos[1].waitSemaphoreCount = 0; }

	uint64_t signalValues2[]{ ++timelineVal, 0 };
	VkSemaphore waitSemaphores2[]{ swapchainSemaphore };
	VkSemaphore signalSemaphores2[]{ semaphore.getHandle(), readyToPresentSemaphore };
	VkPipelineStageFlags stageFlags2[]{ VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
	semaphoreSubmit[1] = TimelineSemaphore::getSubmitInfo(0, nullptr, ARRAYSIZE(signalValues2), signalValues2);
	submitInfos[2] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit + 1,
		.waitSemaphoreCount = ARRAYSIZE(waitSemaphores2), .pWaitSemaphores = waitSemaphores2, .pWaitDstStageMask = stageFlags2,
		.commandBufferCount = 1, .pCommandBuffers = &cbPostprocessing,
		.signalSemaphoreCount = ARRAYSIZE(signalSemaphores2), .pSignalSemaphores = signalSemaphores2 };


	uint64_t waitValues3[]{ signalValues1[0] };
	uint64_t signalValues3[]{ ++timelineValCompute };
	VkSemaphore waitSemaphores3[]{ semaphore.getHandle() };
	VkSemaphore signalSemaphores3[]{ semaphoreCompute.getHandle() };
	VkPipelineStageFlags stageFlags3{ VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
	semaphoreSubmit[2] = TimelineSemaphore::getSubmitInfo(ARRAYSIZE(waitValues3), waitValues3, ARRAYSIZE(signalValues3), signalValues3);
	submitInfos[3] = VkSubmitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = semaphoreSubmit + 2,
		.waitSemaphoreCount = ARRAYSIZE(waitSemaphores3), .pWaitSemaphores = waitSemaphores3, .pWaitDstStageMask = &stageFlags3,
		.commandBufferCount = 1, .pCommandBuffers = &cbCompute,
		.signalSemaphoreCount = ARRAYSIZE(signalSemaphores3), .pSignalSemaphores = signalSemaphores3 };

	vkQueueSubmit(vulkanObjectHandler.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 3, submitInfos, VK_NULL_HANDLE);
	vkQueueSubmit(vulkanObjectHandler.getQueue(VulkanObjectHandler::COMPUTE_QUEUE_TYPE), 1, submitInfos + 3, VK_NULL_HANDLE);

	presentInfo.pImageIndices = &swapchainIndex;
	if (!vulkanObjectHandler.checkSwapchain(vkQueuePresentKHR(vulkanObjectHandler.getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), &presentInfo)))
		swapChain = vulkanObjectHandler.getSwapchain();

	semaphore.wait(timelineVal);
	semaphore.newValue(timelineVal);
	semaphoreCompute.newValue(timelineValCompute);

	cmdBufferSet.resetPoolsOnThreads();
	cmdBufferSet.resetInterchangeable(indexToCBSet, currentCommandBufferIndex ? 0 : 1);
}