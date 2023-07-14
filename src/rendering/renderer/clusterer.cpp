#include "src/rendering/renderer/clusterer.h"

Clusterer::Clusterer(VkDevice device, FrameCommandBufferSet& cmdBufferSet, VkQueue queue, uint32_t windowWidth, uint32_t windowHeight, const BufferMapped& viewprojDataUB)
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

	createTileTestObjects(viewprojDataUB);
	uploadBuffersData(cmdBufferSet, queue);

	oneapi::tbb::flow::make_edge(m_cullNode, m_sortNode);
	oneapi::tbb::flow::make_edge(m_sortNode, m_fillBuffersNode);
	oneapi::tbb::flow::make_edge(m_sortNode, m_fillBinsNode);
}
Clusterer::~Clusterer()
{
	delete[] m_nonculledLightsData;
	m_sortedLightData.reset();
	m_binsMinMax.reset();
	m_instancePointLightIndexData.reset();
	m_instanceSpotLightIndexData.reset();
}

void Clusterer::submitPointLight(const glm::vec3& position, const glm::vec3& color, float power, float radius)
{
	uint32_t newIndex{ static_cast<uint32_t>(m_lightData.size()) };
	LightFormat& lightData{ m_lightData.emplace_back() };
	lightData.position = position;
	lightData.spectrum = color * power;
	lightData.length = radius;

	m_boundingSpheres.push_back({ position, radius });

	m_typeData.push_back(LightFormat::TYPE_POINT);
}
void Clusterer::submitSpotLight(const glm::vec3& position, const glm::vec3& color, float power, float length, glm::vec3 lightDir, float cutoffStartAngle, float cutoffEndAngle)
{
	lightDir = glm::normalize(lightDir);
	if (lightDir.y > 0.999)
	{
		lightDir.y = 0.999;
		lightDir.x = 0.001;
	}
	else if (lightDir.y < -0.999)
	{
		lightDir.y = -0.999;
		lightDir.x = 0.001;
	}

	uint32_t newIndex{ static_cast<uint32_t>(m_lightData.size()) };
	LightFormat& lightData{ m_lightData.emplace_back() };
	lightData.position = position;
	lightData.spectrum = color * power;
	lightData.length = length;
	lightData.lightDir = lightDir;
	lightData.falloffCos = std::cos(std::min(std::min(cutoffStartAngle, cutoffEndAngle), static_cast<float>(M_PI_2)));
	lightData.cutoffCos = std::cos(std::min(cutoffEndAngle, static_cast<float>(M_PI_2)));

	glm::vec4 boundingSphere{
		lightData.cutoffCos > glm::one_over_root_two<float>()
			?
			glm::vec4{position + lightDir * (length / 2.0f), length / 2.0f}
		:
			glm::vec4{ position + lightDir * length * lightData.cutoffCos, length * std::sqrt(1 - lightData.cutoffCos * lightData.cutoffCos) } };

	m_boundingSpheres.push_back(boundingSphere);

	m_typeData.push_back(LightFormat::TYPE_SPOT);
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
	m_currentFurthestLight = std::max(20.0f, m_currentFurthestLight);
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
	
	std::lock_guard<std::mutex> lock{m_mutex};
	m_countDataReady = true;
	m_cv.notify_one();
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
void Clusterer::cmdPassConductTileTest(VkCommandBuffer cb, DescriptorManager& descriptorManager)
{
	VkRenderingInfo renderInfo{};
	renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	renderInfo.renderArea = { .offset{0,0}, .extent{.width = m_widthInTiles, .height = m_heightInTiles} };
	renderInfo.layerCount = 1;
	renderInfo.colorAttachmentCount = 0;

	vkCmdFillBuffer(cb, m_tileData.getBufferHandle(), m_tileData.getOffset(), VK_WHOLE_SIZE/*We can use it here, because this buffer is not suballocated.*/, 0);

	BarrierOperations::cmdExecuteBarrier(cb, std::span<const VkMemoryBarrier2>{
		{BarrierOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_SHADER_WRITE_BIT)}
	});

	vkCmdBeginRendering(cb, &renderInfo);
		
		//Waiting untill light counts are ready
		std::unique_lock<std::mutex> lock{m_mutex};
		m_cv.wait(lock, [this] { return m_countDataReady; });
		m_countDataReady = false;
		descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
			m_pointLightTileTestPipeline.getResourceSets(), m_pointLightTileTestPipeline.getResourceSetsInUse(), m_pointLightTileTestPipeline.getPipelineLayoutHandle());
		VkBuffer vertexBinding[1]{ m_lightBoundingVolumeVertexData.getBufferHandle() };
		VkDeviceSize vertexOffsets[1]{ m_lightBoundingVolumeVertexData.getOffset() };
		vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
		m_pointLightTileTestPipeline.cmdBind(cb);
		vkCmdDraw(cb, POINT_LIGHT_BV_VERTEX_COUNT, m_nonculledPointLightCount, 0, 0);

		descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
			m_spotLightTileTestPipeline.getResourceSets(), m_spotLightTileTestPipeline.getResourceSetsInUse(), m_spotLightTileTestPipeline.getPipelineLayoutHandle());
		m_spotLightTileTestPipeline.cmdBind(cb);
		vkCmdDraw(cb, SPOT_LIGHT_BV_VERTEX_COUNT, m_nonculledSpotLightCount, 0, 0);
		
	vkCmdEndRendering(cb);
}
void Clusterer::cmdDrawBVs(VkCommandBuffer cb, DescriptorManager& descriptorManager, Pipeline& pointLPipeline, Pipeline& spotLPipeline, VkRenderingInfo& renderInfo)
{
	descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
		pointLPipeline.getResourceSets(), pointLPipeline.getResourceSetsInUse(), pointLPipeline.getPipelineLayoutHandle());
	VkBuffer vertexBinding[1]{ m_lightBoundingVolumeVertexData.getBufferHandle() };
	VkDeviceSize vertexOffsets[1]{ m_lightBoundingVolumeVertexData.getOffset() };
	vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
	pointLPipeline.cmdBind(cb);
	vkCmdDraw(cb, POINT_LIGHT_BV_VERTEX_COUNT, m_nonculledPointLightCount, 0, 0);

	descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
		spotLPipeline.getResourceSets(), spotLPipeline.getResourceSetsInUse(), spotLPipeline.getPipelineLayoutHandle());
	spotLPipeline.cmdBind(cb);
	vkCmdDraw(cb, SPOT_LIGHT_BV_VERTEX_COUNT, m_nonculledSpotLightCount, 0, 0);

	/*LightFormat::Types* sortedTypeDataPtr{ reinterpret_cast<LightFormat::Types*>(m_sortedTypeData.getData()) };
	for (int i{ 0 }; i < m_nonculledLightsCount; ++i)
	{
		std::cout << uint32_t(sortedTypeDataPtr[m_nonculledLightsData[i].index]) << ' ';
	}
	std::cout << '\n';*/
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

void Clusterer::createTileTestObjects(const BufferMapped& viewprojDataUB)
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

	std::vector<ResourceSet> resourceSets[2]{};

	//Binding 0
	//viewproj matrices
	VkDescriptorSetLayoutBinding viewprojBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT viewprojAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = viewprojDataUB.getSize() };

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
	//Binding 4
	//Tiling constants : tile width, tile height, max light num
	VkDescriptorSetLayoutBinding constDataBinding{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorAddressInfoEXT constDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_constData.getDeviceAddress(), .range = m_constData.getSize() };

	resourceSets[0].push_back({ m_device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
	{viewprojBinding, tilesLightsDataBinding, pointLightIndicesBinding, lightDataBinding, constDataBinding},  {},
		{{{.pUniformBuffer = &viewprojAddressInfo}}, 
		{{.pStorageBuffer = &tilesLightsDataAddressInfo}}, 
		{{.pStorageBuffer = &pointLightIndicesAddressInfo}}, 
		{{.pStorageBuffer = &lightDataAddressInfo}}, 
		{{.pUniformBuffer = &constDataAddressInfo}}} });

	resourceSets[1].push_back({ m_device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
	{viewprojBinding, tilesLightsDataBinding, spotLightIndicesBinding, lightDataBinding, constDataBinding},  {},
		{{{.pUniformBuffer = &viewprojAddressInfo}},
		{{.pStorageBuffer = &tilesLightsDataAddressInfo}},
		{{.pStorageBuffer = &spotLightIndicesAddressInfo}},
		{{.pStorageBuffer = &lightDataAddressInfo}},
		{{.pUniformBuffer = &constDataAddressInfo}}} });

	m_pointLightTileTestPipeline.initializeGraphics(assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/raster_tile_point_light_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/raster_tile_frag.spv"} } },
		resourceSets[0],
		{ {PosOnlyVertex::getBindingDescription()} },
		{ PosOnlyVertex::getAttributeDescriptions() });
	m_spotLightTileTestPipeline.initializeGraphics(assembler,
		{ { ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/raster_tile_spot_light_vert.spv"}, ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/raster_tile_frag.spv"} } },
		resourceSets[1],
		{},
		{});
}
void Clusterer::uploadBuffersData(FrameCommandBufferSet& cmdBufferSet, VkQueue queue)
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

}
void Clusterer::getNewLight(LightFormat* lightData, glm::vec4* boundingSphere, LightFormat::Types type)
{
	uint32_t newIndex = m_lightData.size();
	EASSERT(newIndex < MAX_LIGHTS, "App", "Number of lights exceeds the maximum.");
	lightData = &m_lightData.emplace_back();
	boundingSphere = &m_boundingSpheres[newIndex];

	m_typeData.push_back(type);
}