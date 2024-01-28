#include "src/rendering/renderer/clusterer.h"

Clusterer::Clusterer(VkDevice device, CommandBufferSet& cmdBufferSet, VkQueue queue, uint32_t windowWidth, uint32_t windowHeight, const ResourceSet& viewprojRS)
	: m_motherBufferShared{ device, CLUSTERED_BUFFERS_SIZE, 
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferBase::DEDICATED_FLAG, true },
	m_sortedTypeData{ device, MAX_LIGHTS * sizeof(uint8_t), 
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, BufferBase::NULL_FLAG, true, false },
	m_tileData{ device, TILE_DATA_SIZE * (windowWidth / TILE_PIXEL_WIDTH) * (windowHeight / TILE_PIXEL_HEIGHT), 
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, BufferBase::NULL_FLAG },
	m_constData{ device, sizeof(float) * 3,
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, BufferBase::NULL_FLAG },
	m_lightBoundingVolumeVertexData{ device, POINT_LIGHT_BV_SIZE, 
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, BufferBase::NULL_FLAG }
{
	m_device = device;
	m_widthInTiles = windowWidth / TILE_PIXEL_WIDTH;
	m_heightInTiles = windowHeight / TILE_PIXEL_HEIGHT;

	m_lightData.reserve(MAX_LIGHTS);
	m_typeData.reserve(MAX_LIGHTS);
	m_boundingSpheres.reserve(MAX_LIGHTS);
	m_nonculledLightsData = { new CulledLightData[MAX_LIGHTS] };

	m_sortedLightData.initialize(m_motherBufferShared, MAX_LIGHTS * sizeof(LightFormat)); //Allocate worst case
	m_binsMinMax.initialize(m_motherBufferShared, Z_BIN_COUNT * sizeof(uint16_t) * 2);
	m_instancePointLightIndexData.initialize(m_motherBufferShared, MAX_LIGHTS * sizeof(uint16_t));
	m_instanceSpotLightIndexData.initialize(m_motherBufferShared, MAX_LIGHTS * sizeof(uint16_t));

	createTileTestObjects(viewprojRS);
	uploadBuffersData(cmdBufferSet, queue);

	m_memBarrier = SyncOperations::constructMemoryBarrier(
		VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_SHADER_WRITE_BIT);
	m_dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
	m_dependencyInfo.memoryBarrierCount = 1;
	m_dependencyInfo.pMemoryBarriers = &m_memBarrier;

	createVisualizationPipelines(viewprojRS, windowWidth, windowHeight);
}
Clusterer::~Clusterer()
{
	delete m_visPipelines;
	delete[] m_nonculledLightsData;
	m_sortedLightData.reset();
	m_binsMinMax.reset();
	m_instancePointLightIndexData.reset();
	m_instanceSpotLightIndexData.reset();
}

void Clusterer::submitFrustum(double near, double far, double aspect, double FOV)
{
	double FOV_X{ glm::atan(glm::tan(FOV / 2) * aspect) * 2 }; //FOV provided to glm::perspective defines vertical frustum angle in radians

	m_frustumPlanes[0] = { 0.0f, 0.0f, -1.0f, near }; //near plane
	m_frustumPlanes[1] = glm::vec4{ glm::rotate(glm::dvec3{-1.0, 0.0, 0.0}, FOV_X / 2.0, glm::dvec3{0.0, -1.0, 0.0}), 0.0 }; //left plane
	m_frustumPlanes[2] = glm::vec4{ glm::rotate(glm::dvec3{1.0, 0.0, 0.0}, FOV_X / 2.0, glm::dvec3{0.0, 1.0, 0.0}), 0.0 }; //right plane
	m_frustumPlanes[3] = glm::vec4{ glm::rotate(glm::dvec3{0.0, 1.0, 0.0}, FOV / 2.0, glm::dvec3{-1.0, 0.0, 0.0}), 0.0 }; //up plane
	m_frustumPlanes[4] = glm::vec4{ glm::rotate(glm::dvec3{0.0, -1.0, 0.0}, FOV / 2.0, glm::dvec3{1.0, 0.0, 0.0}), 0.0 }; //down plane
}
void Clusterer::submitViewMatrix(const glm::mat4& viewMat)
{
	m_currentViewMat = viewMat;
}

void Clusterer::cullLights()
{
	m_nonculledLightsCount = 0;
	for (uint32_t i{ 0 }; i < m_boundingSpheres.size(); ++i)
	{
		bool testResult{ testSphereAgainstFrustum(m_boundingSpheres[i]) };
		if (testResult == true)
		{
			m_nonculledLightsData[m_nonculledLightsCount++] = { i, 0.0, 0.0 };
		}
	}
}
void Clusterer::sortLights()
{
	m_currentFurthestLight = 0.0f;
	for (uint32_t i{ 0 }; i < m_nonculledLightsCount; ++i)
	{
		computeFrontAndBack(m_lightData[m_nonculledLightsData[i].index],
			m_typeData[m_nonculledLightsData[i].index],
			m_nonculledLightsData[i].front,
			m_nonculledLightsData[i].back);
		if (m_nonculledLightsData[i].back > m_currentFurthestLight)
			m_currentFurthestLight = m_nonculledLightsData[i].back;
	}
	m_currentFurthestLight = std::max(100.0f, m_currentFurthestLight);
	oneapi::tbb::parallel_sort(m_nonculledLightsData, m_nonculledLightsData + m_nonculledLightsCount, [](const CulledLightData& data1, const CulledLightData& data2) -> bool { return data1.front < data2.front; });
}
void Clusterer::fillLightBuffers()
{
	LightFormat* sortedLightDataPtr{ reinterpret_cast<LightFormat*>(m_sortedLightData.getData()) };
	LightFormat::Types* sortedTypeDataPtr{ reinterpret_cast<LightFormat::Types*>(m_sortedTypeData.getData()) };

	m_nonculledPointLightCount = 0;
	m_nonculledSpotLightCount = 0;

	for (int i{ 0 }; i < m_nonculledLightsCount; ++i)
	{
		uint32_t index{ m_nonculledLightsData[i].index };

		sortedLightDataPtr[i] = m_lightData[index];
		sortedTypeDataPtr[i] = m_typeData[index];

		if (m_typeData[index] == LightFormat::TYPE_POINT)
		{
			*(reinterpret_cast<uint16_t*>(m_instancePointLightIndexData.getData()) + m_nonculledPointLightCount++) = static_cast<uint16_t>(i);
		}
		else
		{
			*(reinterpret_cast<uint16_t*>(m_instanceSpotLightIndexData.getData()) + m_nonculledSpotLightCount++) = static_cast<uint16_t>(i);
		}
	}
}
void Clusterer::fillZBins()
{
	float binWidth{ m_currentFurthestLight / Z_BIN_COUNT };

	static oneapi::tbb::affinity_partitioner ap{};
	oneapi::tbb::parallel_for(0u, Z_BIN_COUNT, 1u,
		[this, binWidth](uint32_t i) 
		{
			//Front is faced to zero
			float binFront{ binWidth * i };
			float binBack{ binFront + binWidth };
			uint16_t min{ UINT16_MAX };
			uint16_t max{ 0 };
			for (int j{ 0 }; j < m_nonculledLightsCount; ++j)
			{
				if (m_nonculledLightsData[j].front > binFront)
					break;
				if (m_nonculledLightsData[j].back > binFront || m_nonculledLightsData[j].front < binBack)
				{
					if (min == UINT16_MAX)
					{
						min = j;
					}
					max = j;
				}

			}
			uint16_t* minMax{ reinterpret_cast<uint16_t*>(m_binsMinMax.getData()) + i * 2 };
			minMax[0] = min;
			minMax[1] = max;
		}, ap);
}
void Clusterer::cmdTransferClearTileBuffer(VkCommandBuffer cb)
{
	vkCmdFillBuffer(cb, m_tileData.getBufferHandle(), m_tileData.getOffset(), VK_WHOLE_SIZE/*We can use it here, because this buffer is not suballocated.*/, 0);
}
const VkDependencyInfo& Clusterer::getDependency()
{
	return m_dependencyInfo;
}
void Clusterer::cmdPassConductTileTest(VkCommandBuffer cb)
{
	VkRenderingInfo renderInfo{};
	renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	renderInfo.renderArea = { .offset{0,0}, .extent{.width = m_widthInTiles, .height = m_heightInTiles} };
	renderInfo.layerCount = 1;
	renderInfo.colorAttachmentCount = 0;

	vkCmdBeginRendering(cb, &renderInfo);
		
		m_pointLightTileTestPipeline.cmdBindResourceSets(cb);
		VkBuffer vertexBinding[1]{ m_lightBoundingVolumeVertexData.getBufferHandle() };
		VkDeviceSize vertexOffsets[1]{ m_lightBoundingVolumeVertexData.getOffset() };
		vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
		m_pointLightTileTestPipeline.cmdBind(cb);
		vkCmdDraw(cb, POINT_LIGHT_BV_VERTEX_COUNT, m_nonculledPointLightCount, 0, 0);

		m_spotLightTileTestPipeline.cmdBindResourceSets(cb);
		m_spotLightTileTestPipeline.cmdBind(cb);
		vkCmdDraw(cb, SPOT_LIGHT_BV_VERTEX_COUNT, m_nonculledSpotLightCount, 0, 0);
		
	vkCmdEndRendering(cb);
}
void Clusterer::cmdDrawBVs(VkCommandBuffer cb)
{
	constexpr float pcData{ 1.0 };

	m_visPipelines->m_pointBV.cmdBindResourceSets(cb);
	VkBuffer vertexBinding[1]{ m_lightBoundingVolumeVertexData.getBufferHandle() };
	VkDeviceSize vertexOffsets[1]{ m_lightBoundingVolumeVertexData.getOffset() };
	vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
	m_visPipelines->m_pointBV.cmdBind(cb);
	vkCmdPushConstants(cb, m_visPipelines->m_pointBV.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &pcData);
	vkCmdDraw(cb, POINT_LIGHT_BV_VERTEX_COUNT, m_nonculledPointLightCount, 0, 0);

	m_visPipelines->m_spotBV.cmdBindResourceSets(cb);
	m_visPipelines->m_spotBV.cmdBind(cb);
	vkCmdPushConstants(cb, m_visPipelines->m_spotBV.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &pcData);
	vkCmdDraw(cb, SPOT_LIGHT_BV_VERTEX_COUNT, m_nonculledSpotLightCount, 0, 0);
}
void Clusterer::cmdDrawProxies(VkCommandBuffer cb)
{
	constexpr float pcData{ 0.02 };

	m_visPipelines->m_pointProxy.cmdBindResourceSets(cb);
	VkBuffer vertexBinding[1]{ m_lightBoundingVolumeVertexData.getBufferHandle() };
	VkDeviceSize vertexOffsets[1]{ m_lightBoundingVolumeVertexData.getOffset() };
	vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
	m_visPipelines->m_pointProxy.cmdBind(cb);
	vkCmdPushConstants(cb, m_visPipelines->m_pointProxy.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &pcData);
	vkCmdDraw(cb, POINT_LIGHT_BV_VERTEX_COUNT, m_nonculledPointLightCount, 0, 0);

	m_visPipelines->m_spotProxy.cmdBindResourceSets(cb);
	m_visPipelines->m_spotProxy.cmdBind(cb);
	vkCmdPushConstants(cb, m_visPipelines->m_spotProxy.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float), &pcData);
	vkCmdDraw(cb, SPOT_LIGHT_BV_VERTEX_COUNT, m_nonculledSpotLightCount, 0, 0);
}


bool Clusterer::testSphereAgainstFrustum(const glm::vec4& sphereData)
{
	float radius{ sphereData.w };

	glm::vec3 viewSphereData{ m_currentViewMat * glm::vec4(sphereData.x, sphereData.y, sphereData.z, 1.0) };

	if ((m_frustumPlanes[0].x * viewSphereData.x + m_frustumPlanes[0].y * viewSphereData.y + m_frustumPlanes[0].z * viewSphereData.z) > radius)
		return false;
	if ((m_frustumPlanes[1].x * viewSphereData.x + m_frustumPlanes[1].y * viewSphereData.y + m_frustumPlanes[1].z * viewSphereData.z) > radius)
		return false;
	if ((m_frustumPlanes[2].x * viewSphereData.x + m_frustumPlanes[2].y * viewSphereData.y + m_frustumPlanes[2].z * viewSphereData.z) > radius)
		return false;
	if ((m_frustumPlanes[3].x * viewSphereData.x + m_frustumPlanes[3].y * viewSphereData.y + m_frustumPlanes[3].z * viewSphereData.z) > radius)
		return false;
	if ((m_frustumPlanes[4].x * viewSphereData.x + m_frustumPlanes[4].y * viewSphereData.y + m_frustumPlanes[4].z * viewSphereData.z) > radius)
		return false;
	return true;
}
void Clusterer::computeFrontAndBack(const LightFormat& light, LightFormat::Types type, float& front, float& back)
{
	glm::vec4 compVec{ glm::row(m_currentViewMat, 2) };
	switch (type)
	{
	case LightFormat::TYPE_POINT:
		front = glm::dot(compVec, glm::vec4{light.position, 1.0f}) - light.length;
		back = glm::dot(compVec, glm::vec4{light.position, 1.0f}) + light.length;
		break;
	case LightFormat::TYPE_SPOT:
	{
		glm::vec3 frontDir{ 0.0f, 0.0f, -1.0f };
		glm::vec3 newDir{ glm::mat3{m_currentViewMat} *light.lightDir };

		float cosA{ light.cutoffCos };
		if (cosA < glm::dot(newDir, frontDir))
		{
			front = glm::dot(compVec, glm::vec4{light.position, 1.0f}) - light.length;
			back = glm::dot(compVec, glm::vec4{light.position, 1.0f}) + light.length;
			break;
		}
		float sinA{ std::sqrt(1 - cosA * cosA) };

		glm::vec3 normal{ newDir.z < 0.999f ? glm::normalize(glm::cross(newDir, frontDir)) : glm::vec3{ 1.0f, 0.0f, 0.0f } };

		float lightZPos{ glm::dot(compVec, glm::vec4{light.position, 1.0f}) };

		float cosWeigthTerm{ cosA * newDir.z };
		float sinWeightTerm{ sinA * (normal.x * newDir.y - normal.y * newDir.x) };

		float newDirFrontRotZ{ cosWeigthTerm + sinWeightTerm };
		front = (lightZPos + (newDirFrontRotZ > 0.0 ? 0.0 : newDirFrontRotZ * light.length));

		float newDirBackRotZ{ cosWeigthTerm - sinWeightTerm };
		back = (lightZPos + (newDirBackRotZ < 0.0 ? 0.0 : newDirBackRotZ * light.length));

		break;
	}
	default:
		EASSERT(false, "App", "Unknown unified light type. Should never happen.");
		break;
	}
}

void Clusterer::createTileTestObjects(const ResourceSet& viewprojRS)
{
	PipelineAssembler assembler{ m_device };
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, m_widthInTiles, m_heightInTiles);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_ENABLED, TILE_TEST_SAMPLE_COUNT);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0, VK_CULL_MODE_NONE);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DISABLED);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_NO_ATTACHMENT);

	//Binding 0
	//Tiling constants : tile width, tile height, max light num
	VkDescriptorSetLayoutBinding constDataBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT constDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_constData.getDeviceAddress(), .range = m_constData.getSize() };
	//Binding 1
	//Tiles light data
	VkDescriptorSetLayoutBinding tilesLightsDataBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT tilesLightsDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_tileData.getDeviceAddress(), .range = m_tileData.getSize() };
	//Binding 2 Point
	//Light indices
	VkDescriptorSetLayoutBinding pointLightIndicesBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT pointLightIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_instancePointLightIndexData.getDeviceAddress(), .range = m_instancePointLightIndexData.getSize() };
	//Binding 2 Spot
	//Light indices
	VkDescriptorSetLayoutBinding spotLightIndicesBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT spotLightIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_instanceSpotLightIndexData.getDeviceAddress(), .range = m_instanceSpotLightIndexData.getSize() };
	//Binding 3
	//Light data
	VkDescriptorSetLayoutBinding lightDataBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT lightDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_sortedLightData.getDeviceAddress(), .range = m_sortedLightData.getSize() };

	m_resourceSets[0].initializeSet(m_device, 1, VkDescriptorSetLayoutCreateFlags{},
	std::array{ constDataBinding, tilesLightsDataBinding, pointLightIndicesBinding, lightDataBinding }, std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pUniformBuffer = &constDataAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &tilesLightsDataAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &pointLightIndicesAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &lightDataAddressInfo} }}, 
		false);
	m_resourceSets[1].initializeSet(m_device, 1, VkDescriptorSetLayoutCreateFlags{},
		std::array{ constDataBinding, tilesLightsDataBinding, spotLightIndicesBinding, lightDataBinding }, std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pUniformBuffer = &constDataAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &tilesLightsDataAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &spotLightIndicesAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &lightDataAddressInfo} }},
		false);


	std::array<std::reference_wrapper<const ResourceSet>, 2> res{ viewprojRS, m_resourceSets[0] };

	m_pointLightTileTestPipeline.initializeGraphics(assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/raster_tile_point_light_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/raster_tile_frag.spv"} } },
		std::span(res.begin() + 0, res.begin() + 2),
		{ {PosOnlyVertex::getBindingDescription()} },
		{ PosOnlyVertex::getAttributeDescriptions() });
	
	res = { viewprojRS, m_resourceSets[1] };

	m_spotLightTileTestPipeline.initializeGraphics(assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/raster_tile_spot_light_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/raster_tile_frag.spv"} } },
		std::span(res.begin() + 0, res.begin() + 2),
		{},
		{});
}
void Clusterer::uploadBuffersData(CommandBufferSet& cmdBufferSet, VkQueue queue)
{
	BufferBaseHostAccessible staging{ m_device, m_lightBoundingVolumeVertexData.getSize(), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT };

	std::ifstream istream{ "internal/BVmeshes/octasphere_1sub_M.bin", std::ios::binary };
	istream.seekg(0, std::ios::beg);

	uint32_t vertNum{};
	istream.read(reinterpret_cast<char*>(&vertNum), sizeof(vertNum));
	istream.seekg(sizeof(vertNum), std::ios::beg);

	uint32_t dataSize{ vertNum * sizeof(float) * 3 };

	istream.read(reinterpret_cast<char*>(staging.getData()), dataSize);

	VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };

	VkBufferCopy copy{ .srcOffset = staging.getOffset(), .dstOffset = 0, .size = dataSize };
	BufferTools::cmdBufferCopy(cb, staging.getBufferHandle(), m_lightBoundingVolumeVertexData.getBufferHandle(), 1, &copy);

	uint32_t constData[]{ m_widthInTiles, m_heightInTiles, MAX_WORDS };
	vkCmdUpdateBuffer(cb, m_constData.getBufferHandle(), m_constData.getOffset(), m_constData.getSize(), constData);

	cmdBufferSet.endRecording(cb);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	cmdBufferSet.resetAllTransient();
}
uint32_t Clusterer::getNewLight(LightFormat** lightData, glm::vec4** boundingSphere, LightFormat::Types type)
{
	uint32_t newIndex = m_lightData.size();
	EASSERT(newIndex < MAX_LIGHTS, "App", "Number of lights exceeds the maximum.");
	*lightData = &m_lightData.emplace_back();
	*boundingSphere = &m_boundingSpheres.emplace_back();

	m_typeData.push_back(type);

	return newIndex;
}

void Clusterer::createVisualizationPipelines(const ResourceSet& viewprojRS, uint32_t windowWidth, uint32_t windowHeight)
{
	m_visPipelines = { new VisualizationPipelines };

	PipelineAssembler assembler{ m_device };
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, windowWidth, windowHeight);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_LINE_POLYGONS, 1.4f, VK_CULL_MODE_NONE);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);

	//Binding 0 Point
	//Light indices
	VkDescriptorSetLayoutBinding pointLightIndicesBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT pointLightIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_instancePointLightIndexData.getDeviceAddress(), .range = m_instancePointLightIndexData.getSize() };
	//Binding 0 Spot
	//Light indices
	VkDescriptorSetLayoutBinding spotLightIndicesBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT spotLightIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_instanceSpotLightIndexData.getDeviceAddress(), .range = m_instanceSpotLightIndexData.getSize() };
	//Binding 1
	//Light data
	VkDescriptorSetLayoutBinding lightDataBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT lightDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_sortedLightData.getDeviceAddress(), .range = m_sortedLightData.getSize() };

	m_resourceSets[2].initializeSet(m_device, 1, VkDescriptorSetLayoutCreateFlags{},
		std::array{ pointLightIndicesBinding, lightDataBinding }, std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &pointLightIndicesAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &lightDataAddressInfo} }},
		false);
	m_resourceSets[3].initializeSet(m_device, 1, VkDescriptorSetLayoutCreateFlags{},
		std::array{ spotLightIndicesBinding, lightDataBinding }, std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &spotLightIndicesAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageBuffer = &lightDataAddressInfo} }},
		false);

	std::array<std::reference_wrapper<const ResourceSet>, 2> res{ viewprojRS, m_resourceSets[2] };

	m_visPipelines->m_pointBV.initializeGraphics(assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/point_light_mesh_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/color_frag.spv"} } },
		std::span(res.begin() + 0, res.begin() + 2),
		{ {PosOnlyVertex::getBindingDescription()} },
		{ PosOnlyVertex::getAttributeDescriptions() }, {{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(float)}} });

	res = { viewprojRS, m_resourceSets[3] };

	m_visPipelines->m_spotBV.initializeGraphics(assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/cone_light_mesh_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/color_frag.spv"} } },
		std::span(res.begin() + 0, res.begin() + 2),
		{},
		{}, {{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(float)}} });


	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0, VK_CULL_MODE_NONE);


	res = { viewprojRS, m_resourceSets[2] };

	m_visPipelines->m_pointProxy.initializeGraphics(assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/point_light_mesh_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/color_frag.spv"} } },
		std::span(res.begin() + 0, res.begin() + 2),
		{ {PosOnlyVertex::getBindingDescription()} },
		{ PosOnlyVertex::getAttributeDescriptions() }, {{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(float)}} });

	res = { viewprojRS, m_resourceSets[3] };

	m_visPipelines->m_spotProxy.initializeGraphics(assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/cone_light_mesh_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/color_frag.spv"} } },
		std::span(res.begin() + 0, res.begin() + 2),
		{},
		{}, {{VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(float)}} });
}