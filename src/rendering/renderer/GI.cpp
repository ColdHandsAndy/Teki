#include "src/rendering/renderer/GI.h"

GI::GI(VkDevice device, uint32_t windowWidth, uint32_t windowHeight, BufferBaseHostAccessible& baseHostBuffer, BufferBaseHostInaccessible& baseDeviceBuffer, Clusterer& clusterer) :
	m_baseOccupancyMap{ device, VK_FORMAT_R32_UINT, BOM_PACKED_WIDTH, BOM_PACKED_HEIGHT, BOM_PACKED_DEPTH, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, Image::GRAPHICS_AND_COMPUTE_BIT },
	m_emissionMetRoughVoxelmap{ device, VK_FORMAT_R16G16B16A16_UINT, VOXELMAP_RESOLUTION, VOXELMAP_RESOLUTION, VOXELMAP_RESOLUTION, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_dynamicEmissionVoxelmap{ device, VK_FORMAT_R16G16B16A16_UINT, VOXELMAP_RESOLUTION, VOXELMAP_RESOLUTION, VOXELMAP_RESOLUTION, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_albedoNormalVoxelmap{ device, VK_FORMAT_R8G8B8A8_UINT, VOXELMAP_RESOLUTION, VOXELMAP_RESOLUTION, VOXELMAP_RESOLUTION, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_rayAlignedOccupancyMapArray{
		CompileTimeArray::uniform_array_from_args<Image, ROM_NUMBER>
			(device, VK_FORMAT_R32_UINT, ROM_PACKED_WIDTH, ROM_PACKED_HEIGHT, ROM_PACKED_DEPTH, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false)
	},
	m_rayAlignedOccupancyMapArrayStable{
		CompileTimeArray::uniform_array_from_args<Image, STABLE_ROM_NUMBER>
			(device, VK_FORMAT_R32_UINT, ROM_PACKED_WIDTH, ROM_PACKED_HEIGHT, ROM_PACKED_DEPTH, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_ASPECT_COLOR_BIT, false) 
	},
	m_ddgiRadianceProbes{ device, VK_FORMAT_R16G16B16A16_SFLOAT,
		DDGI_PROBE_LIGHT_SIDE_SIZE * DDGI_PROBE_X_COUNT * DDGI_PROBE_Z_COUNT, DDGI_PROBE_LIGHT_SIDE_SIZE * DDGI_PROBE_Y_COUNT,
		VK_IMAGE_USAGE_STORAGE_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_ddgiDistanceProbes{ device, VK_FORMAT_R16_SFLOAT,
		DDGI_PROBE_LIGHT_SIDE_SIZE * DDGI_PROBE_X_COUNT * DDGI_PROBE_Z_COUNT, DDGI_PROBE_LIGHT_SIDE_SIZE * DDGI_PROBE_Y_COUNT,
		VK_IMAGE_USAGE_STORAGE_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_ddgiIrradianceProbes{
		CompileTimeArray::uniform_array_from_args<Image, 2>
			(device, VK_FORMAT_A2B10G10R10_UNORM_PACK32,
			DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS * DDGI_PROBE_X_COUNT * DDGI_PROBE_Z_COUNT, DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS * DDGI_PROBE_Y_COUNT,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT, false)
		},
	m_ddgiVisibilityProbes{
		CompileTimeArray::uniform_array_from_args<Image, 2>
			(device, VK_FORMAT_R16G16_SFLOAT,
			DDGI_PROBE_VISIBILITY_SIDE_SIZE_WITH_BORDERS * DDGI_PROBE_X_COUNT * DDGI_PROBE_Z_COUNT, DDGI_PROBE_VISIBILITY_SIDE_SIZE_WITH_BORDERS * DDGI_PROBE_Y_COUNT,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
			VK_IMAGE_ASPECT_COLOR_BIT, false)
		},
	m_specularReflectionGlossy{ device, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
		windowWidth, windowHeight / 2,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_specularReflectionRough{ device, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
		windowWidth, windowHeight / 2,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_depthSpec{ device, VK_FORMAT_R32_SFLOAT,
		windowWidth, windowHeight / 2,
		VK_IMAGE_USAGE_STORAGE_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_refdirSpec{ device, VK_FORMAT_R8G8_UNORM,
		windowWidth, windowHeight / 2,
		VK_IMAGE_USAGE_STORAGE_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_distToHit{ device, VK_FORMAT_R16_SFLOAT,
		windowWidth, windowHeight / 2,
		VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		VK_IMAGE_ASPECT_COLOR_BIT, false },
	m_sphereVertexData{ baseDeviceBuffer },
	m_pcDataBOM{ OCCUPANCY_METER_SIZE / 2.0, OCCUPANCY_RESOLUTION, VOXELMAP_RESOLUTION },
	m_ROMAtransformMatrices{ {baseHostBuffer, sizeof(glm::mat4x3) * ROM_NUMBER}, {baseHostBuffer, sizeof(glm::mat4x3) * ROM_NUMBER} },
	m_stableROMAtransformMatrices{ {baseHostBuffer, sizeof(glm::mat4x3) * 8}, {baseHostBuffer, sizeof(glm::mat4x3) * 8} },
	m_mappedDirections{ {baseHostBuffer, sizeof(glm::vec4) * DDGI_PROBE_LIGHT_SIDE_SIZE * DDGI_PROBE_LIGHT_SIDE_SIZE}, {baseHostBuffer, sizeof(glm::vec4) * DDGI_PROBE_LIGHT_SIDE_SIZE * DDGI_PROBE_LIGHT_SIDE_SIZE} },
	m_giMetadata{ baseHostBuffer, sizeof(GIMetaData) },
	m_events{ device },
	m_clusterer{ &clusterer }
{
}
GI::~GI()
{
}

void GI::initialize(VkDevice device,
	const ResourceSet& drawDataRS, const ResourceSet& transformMatricesRS, const ResourceSet& materialsTexturesRS, const ResourceSet& distantProbeRS, const ResourceSet& BRDFLUTRS, const ResourceSet& shadowMapsRS, VkSampler generalSampler)
{
	VkDescriptorSetLayoutBinding bindingBOM{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo imageInfoBOM{ .imageView = m_baseOccupancyMap.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	m_resSetWriteBOM.initializeSet(device, 1, {},
		std::array{ bindingBOM },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &imageInfoBOM} }},
		false);
	imageInfoBOM.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	m_resSetReadBOM.initializeSet(device, 1, {},
		std::array{ bindingBOM },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &imageInfoBOM} }},
		false);

	VkDescriptorSetLayoutBinding bindingROMA{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = ROM_NUMBER, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT };
	std::array<VkDescriptorImageInfo, ROM_NUMBER> imageInfosROMA{};
	std::vector<VkDescriptorDataEXT> descDataROMA(ROM_NUMBER);
	for (int i{ 0 }; i < imageInfosROMA.size(); ++i)
	{
		imageInfosROMA[i].imageView = m_rayAlignedOccupancyMapArray[i].getImageView();
		imageInfosROMA[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		descDataROMA[i].pStorageImage = &imageInfosROMA[i];
	}
	VkDescriptorSetLayoutBinding bindingStableROMA{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 8, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT };
	std::array<VkDescriptorImageInfo, STABLE_ROM_NUMBER> imageInfosStableROMA{};
	std::vector<VkDescriptorDataEXT> descDataStableROMA(STABLE_ROM_NUMBER);
	for (int i{ 0 }; i < imageInfosStableROMA.size(); ++i)
	{
		imageInfosStableROMA[i].imageView = m_rayAlignedOccupancyMapArrayStable[i].getImageView();
		imageInfosStableROMA[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		descDataStableROMA[i].pStorageImage = &imageInfosStableROMA[i];
	}
	m_resSetWriteROMA.initializeSet(device, 1, {},
		std::array{ bindingROMA, bindingStableROMA },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		descDataROMA, descDataStableROMA},
		false);
	for (int i{ 0 }; i < imageInfosROMA.size(); ++i)
	{
		imageInfosROMA[i].imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	}
	VkDescriptorSetLayoutBinding bindingViewmatsROMA{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT viewmatsROMAAddressInfo0{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_ROMAtransformMatrices[0].getDeviceAddress(), .range = m_ROMAtransformMatrices[0].getSize() };
	VkDescriptorAddressInfoEXT viewmatsROMAAddressInfo1{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_ROMAtransformMatrices[1].getDeviceAddress(), .range = m_ROMAtransformMatrices[1].getSize() };
	descDataROMA.resize(ROM_NUMBER * 2);
	for (int i{ 0 }; i < imageInfosROMA.size(); ++i)
	{
		descDataROMA[ROM_NUMBER + i].pStorageImage = &imageInfosROMA[i];
	}
	m_resSetReadROMA.initializeSet(device, 2, {},
		std::array{ bindingROMA, bindingViewmatsROMA },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		descDataROMA,
			std::vector<VkDescriptorDataEXT>{ {.pUniformBuffer = &viewmatsROMAAddressInfo0}, { .pUniformBuffer = &viewmatsROMAAddressInfo1 } }},
		false);
	for (int i{ 0 }; i < imageInfosStableROMA.size(); ++i)
	{
		imageInfosStableROMA[i].imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	}
	VkDescriptorSetLayoutBinding bindingViewmatsStableROMA{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT viewmatsStableROMAAddressInfo0{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_stableROMAtransformMatrices[0].getDeviceAddress(), .range = m_stableROMAtransformMatrices[0].getSize() };
	VkDescriptorAddressInfoEXT viewmatsStableROMAAddressInfo1{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_stableROMAtransformMatrices[1].getDeviceAddress(), .range = m_stableROMAtransformMatrices[1].getSize() };
	descDataStableROMA.resize(STABLE_ROM_NUMBER * 2);
	for (int i{ 0 }; i < imageInfosStableROMA.size(); ++i)
	{
		descDataStableROMA[STABLE_ROM_NUMBER + i].pStorageImage = &imageInfosStableROMA[i];
	}
	bindingStableROMA.binding = 0;
	m_resSetReadStableROMA.initializeSet(device, 2, {},
		std::array{ bindingStableROMA, bindingViewmatsStableROMA },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		descDataStableROMA,
			std::vector<VkDescriptorDataEXT>{ {.pUniformBuffer = &viewmatsStableROMAAddressInfo0}, { .pUniformBuffer = &viewmatsStableROMAAddressInfo1 } }},
		false);

	VkDescriptorSetLayoutBinding bindingEmissionMetRoughVM{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo imageInfoEmissionMetRoughVM{ .imageView = m_emissionMetRoughVoxelmap.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	m_resSetEmissionMetRoughWrite.initializeSet(device, 1, {},
		std::array{ bindingEmissionMetRoughVM },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &imageInfoEmissionMetRoughVM} } },
		false);
	imageInfoEmissionMetRoughVM.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	m_resSetEmissionMetRoughRead.initializeSet(device, 1, {},
		std::array{ bindingEmissionMetRoughVM },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &imageInfoEmissionMetRoughVM} } },
		false);
	VkDescriptorSetLayoutBinding bindingAlbedoNormalVM{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo imageInfoAlbedoNormalVM{ .imageView = m_albedoNormalVoxelmap.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	m_resSetAlbedoNormalWrite.initializeSet(device, 1, {},
		std::array{ bindingAlbedoNormalVM },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &imageInfoAlbedoNormalVM} } },
		false);
	imageInfoAlbedoNormalVM.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	m_resSetAlbedoNormalRead.initializeSet(device, 1, {},
		std::array{ bindingAlbedoNormalVM },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &imageInfoAlbedoNormalVM} } },
		false);
	VkDescriptorSetLayoutBinding bindingDynamicEmissionVM{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo imageInfoDynamicEmissionVM{ .imageView = m_dynamicEmissionVoxelmap.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	m_resSetDynamicEmissionWrite.initializeSet(device, 1, {},
		std::array{ bindingDynamicEmissionVM },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &imageInfoDynamicEmissionVM} } },
		false);
	imageInfoDynamicEmissionVM.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	m_resSetDynamicEmissionRead.initializeSet(device, 1, {},
		std::array{ bindingDynamicEmissionVM },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &imageInfoDynamicEmissionVM} } },
		false);

	VkDescriptorSetLayoutBinding bindingRadianceProbes{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo radianceProbesAddressInfo{ .imageView = m_ddgiRadianceProbes.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorSetLayoutBinding bindingDistanceProbes{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo distanceProbesAddressInfo{ .imageView = m_ddgiDistanceProbes.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	m_resSetProbesWrite.initializeSet(device, 1, {},
		std::array{ bindingRadianceProbes, bindingDistanceProbes },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &radianceProbesAddressInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &distanceProbesAddressInfo} }},
		false);
	radianceProbesAddressInfo.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	m_resSetRadianceProbesRead.initializeSet(device, 1, {},
		std::array{ bindingRadianceProbes },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &radianceProbesAddressInfo} }},
		false);
	bindingDistanceProbes.binding = 0;
	distanceProbesAddressInfo.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	m_resSetDistanceProbesRead.initializeSet(device, 1, {},
		std::array{ bindingDistanceProbes },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &distanceProbesAddressInfo} }},
		false);

	VkDescriptorSetLayoutBinding bindingIrradianceProbeHistory{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorSetLayoutBinding bindingIrradianceProbeNew{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo irradianceProbesAddressInfo0{ .imageView = m_ddgiIrradianceProbes[0].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo irradianceProbesAddressInfo1{ .imageView = m_ddgiIrradianceProbes[1].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	m_resSetIrradProbesWrite.initializeSet(device, 2, {},
		std::array{ bindingIrradianceProbeHistory, bindingIrradianceProbeNew },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &irradianceProbesAddressInfo1}, { .pStorageImage = &irradianceProbesAddressInfo0 } },
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &irradianceProbesAddressInfo0}, { .pStorageImage = &irradianceProbesAddressInfo1 } }, },
		false);
	VkDescriptorSetLayoutBinding bindingVisibilityProbeHistory{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorSetLayoutBinding bindingVisibilityProbeNew{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo visibilityProbesAddressInfo0{ .imageView = m_ddgiVisibilityProbes[0].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorImageInfo visibilityProbesAddressInfo1{ .imageView = m_ddgiVisibilityProbes[1].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	m_resSetVisibProbesWrite.initializeSet(device, 2, {},
		std::array{ bindingVisibilityProbeHistory, bindingVisibilityProbeNew },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &visibilityProbesAddressInfo1}, { .pStorageImage = &visibilityProbesAddressInfo0 } },
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &visibilityProbesAddressInfo0}, { .pStorageImage = &visibilityProbesAddressInfo1 } }, },
		false);

	VkDescriptorSetLayoutBinding bindingIrradianceProbes{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorSetLayoutBinding bindingVisibilityProbes{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	irradianceProbesAddressInfo0.sampler = generalSampler;
	irradianceProbesAddressInfo1.sampler = generalSampler;
	irradianceProbesAddressInfo0.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	irradianceProbesAddressInfo1.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	visibilityProbesAddressInfo0.sampler = generalSampler;
	visibilityProbesAddressInfo1.sampler = generalSampler;
	visibilityProbesAddressInfo0.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	visibilityProbesAddressInfo1.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
	m_resSetIndirectDiffuseLighting.initializeSet(device, 2, {},
		std::array{ bindingIrradianceProbes, bindingVisibilityProbes },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
		std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &irradianceProbesAddressInfo0}, { .pCombinedImageSampler = &irradianceProbesAddressInfo1 } },
		std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &visibilityProbesAddressInfo0}, { .pCombinedImageSampler = &visibilityProbesAddressInfo1 } }},
		true);
	VkDescriptorSetLayoutBinding bindingMappedDirections{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT mappedDirectionsAddressInfo0{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_mappedDirections[0].getDeviceAddress(), .range = m_mappedDirections[0].getSize() };
	VkDescriptorAddressInfoEXT mappedDirectionsAddressInfo1{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_mappedDirections[1].getDeviceAddress(), .range = m_mappedDirections[1].getSize() };
	m_resSetMappedDirections.initializeSet(device, 2, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ bindingMappedDirections },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{{
				std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pUniformBuffer = &mappedDirectionsAddressInfo0 }, VkDescriptorDataEXT{ .pUniformBuffer = &mappedDirectionsAddressInfo1 } }}},
		false);

	VkDescriptorSetLayoutBinding bindingReflectionImage{ .binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo reflectionImageInfo{ .imageView = m_specularReflectionGlossy.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorSetLayoutBinding bindingHitDistImage{ .binding = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo hitDistImageInfo{ .imageView = m_distToHit.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	VkDescriptorSetLayoutBinding depthImage{ .binding = 2,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo depthImageInfo{ .imageView = m_depthSpec.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding refdirImage{ .binding = 3,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo refdirImageInfo{ .imageView = m_refdirSpec.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
	m_resSetSpecularWrite.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ bindingReflectionImage, bindingHitDistImage, depthImage, refdirImage },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{{
				std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &reflectionImageInfo } },
					std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &hitDistImageInfo } },
					std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &depthImageInfo } },
					std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &refdirImageInfo } },
			}},
		false);
	{
		bindingReflectionImage.descriptorCount = 2;
		bindingReflectionImage.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		VkDescriptorImageInfo reflecGlossyImageInfo{ .sampler = generalSampler, .imageView = m_specularReflectionGlossy.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
		VkDescriptorImageInfo reflecRoughImageInfo{ .sampler = generalSampler, .imageView = m_specularReflectionRough.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
		bindingHitDistImage.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		hitDistImageInfo.sampler = generalSampler;
		hitDistImageInfo.imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL;
		depthImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		refdirImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		m_resSetIndirectSpecularLighting.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
			std::array{ bindingReflectionImage, bindingHitDistImage, depthImage, refdirImage },
			std::array<VkDescriptorBindingFlags, 0>{},
			std::vector<std::vector<VkDescriptorDataEXT>>{{
					std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pSampledImage = &reflecGlossyImageInfo }, VkDescriptorDataEXT{ .pSampledImage = &reflecRoughImageInfo } },
						std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pSampledImage = &hitDistImageInfo } },
						std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &depthImageInfo } },
						std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &refdirImageInfo } },
				}},
			false);
	}
	VkDescriptorSetLayoutBinding bindingSrcDownscaleImages{ .binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorSetLayoutBinding bindingDstDownscaleImages{ .binding = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo srcImageInfo{ .imageView = m_specularReflectionGlossy.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
	VkDescriptorImageInfo dstImageInfo{ .imageView = m_specularReflectionRough.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
	m_resSetBilateral.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ bindingSrcDownscaleImages, bindingDstDownscaleImages },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{{
				std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &srcImageInfo } },
					std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &dstImageInfo } },
			}},
		false);

	VkDescriptorSetLayoutBinding bindingProbesParameters{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT };
	VkDescriptorAddressInfoEXT probesParametersAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_giMetadata.getDeviceAddress(), .range = m_giMetadata.getSize() };
	m_resSetGIMetadata.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ bindingProbesParameters },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{{
			std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pUniformBuffer = &probesParametersAddressInfo } }}},
		false);



	PipelineAssembler assembler{ device };
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0, VK_CULL_MODE_NONE);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, OCCUPANCY_RESOLUTION, OCCUPANCY_RESOLUTION);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DISABLED);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_ENABLED, VK_SAMPLE_COUNT_4_BIT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_NO_ATTACHMENT);

	std::array<std::reference_wrapper<const ResourceSet>, 6> resourceSetsVoxelize{ transformMatricesRS, drawDataRS, materialsTexturesRS, m_resSetWriteBOM, m_resSetEmissionMetRoughWrite, m_resSetAlbedoNormalWrite };
	m_voxelize.initializeGraphics(assembler,
		{ { ShaderStage{.stage = VK_SHADER_STAGE_VERTEX_BIT, .filepath = "shaders/cmpld/gi_voxelization_vert.spv"},
		ShaderStage{.stage = VK_SHADER_STAGE_GEOMETRY_BIT, .filepath = "shaders/cmpld/gi_voxelization_geom.spv"},
		ShaderStage{.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .filepath = "shaders/cmpld/gi_voxelization_frag.spv"} } },
		resourceSetsVoxelize,
		{ {StaticVertex::getBindingDescription()} },
		{ StaticVertex::getAttributeDescriptions() },
		{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(m_pcDataBOM)}} });


	std::array<std::reference_wrapper<const ResourceSet>, 2> resourceSetsCreateROMA{ m_resSetReadBOM, m_resSetWriteROMA };
	m_createROMA.initializaCompute(device, "shaders/cmpld/gi_create_ROMA_comp.spv", resourceSetsCreateROMA,
		{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(m_pcDataROM)}} });


	std::array<std::reference_wrapper<const ResourceSet>, 8> resourceSetsTraceProbes{
		m_resSetProbesWrite, m_resSetGIMetadata, 
		m_resSetReadROMA, 
		m_resSetDynamicEmissionRead, m_resSetAlbedoNormalRead, 
		m_resSetIndirectDiffuseLighting, distantProbeRS, BRDFLUTRS };
	m_traceProbes.initializaCompute(device, "shaders/cmpld/gi_probe_tracing_comp.spv", resourceSetsTraceProbes,
	{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(m_pcDataTraceProbes)}} });


	std::array<std::reference_wrapper<const ResourceSet>, 3> resourceSetsComputeIrradiance{ m_resSetRadianceProbesRead, m_resSetIrradProbesWrite, m_resSetMappedDirections };
	m_computeIrradiance.initializaCompute(device, "shaders/cmpld/gi_compute_irradiance_comp.spv", resourceSetsComputeIrradiance);

	std::array<std::reference_wrapper<const ResourceSet>, 3> resourceSetsComputeVisibility{ m_resSetDistanceProbesRead, m_resSetVisibProbesWrite, m_resSetMappedDirections };
	m_computeVisibility.initializaCompute(device, "shaders/cmpld/gi_compute_visibility_comp.spv", resourceSetsComputeVisibility);


	std::array<std::reference_wrapper<const ResourceSet>, 3> resourceSetsInjectLight{ shadowMapsRS, m_resSetAlbedoNormalRead, m_resSetDynamicEmissionWrite };
	m_injectLight.initializaCompute(device, "shaders/cmpld/gi_inject_light_comp.spv", resourceSetsInjectLight,
		{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(m_pcDataLightInjection)}} });

	std::array<std::reference_wrapper<const ResourceSet>, 2> resourceSetsMergeEmission{ m_resSetEmissionMetRoughRead, m_resSetDynamicEmissionWrite };
	m_mergeEmission.initializaCompute(device, "shaders/cmpld/gi_emission_merge_comp.spv", resourceSetsMergeEmission);

	std::array<std::reference_wrapper<const ResourceSet>, 8> resourceSetsTraceSpecular{ 
		m_resSetSpecularWrite, m_resSetGIMetadata,
		m_resSetReadStableROMA,
		m_resSetDynamicEmissionRead, m_resSetAlbedoNormalRead,
		m_resSetIndirectDiffuseLighting, distantProbeRS, BRDFLUTRS };
	m_traceSpecular.initializaCompute(device, "shaders/cmpld/gi_trace_specular_comp.spv", resourceSetsTraceSpecular,
	{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(m_pcDataTraceSpecular)}} });

	std::array<std::reference_wrapper<const ResourceSet>, 1> resourceSetsBilateral{ m_resSetBilateral };
	m_bilateral.initializaCompute(device, "shaders/cmpld/bilateral_comp.spv", resourceSetsBilateral,
	{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(m_pcDataBilateral)}} });



	GIMetaData* metaData{ reinterpret_cast<GIMetaData*>(m_giMetadata.getData()) };
	auto& gridData{ metaData->cascades[0].gridData };
	gridData.relOriginProbePos = -glm::vec3(OCCUPANCY_METER_SIZE / 2.0) + glm::vec3(DDGI_PROBE_X_OFFSET, DDGI_PROBE_Y_OFFSET, DDGI_PROBE_Z_OFFSET);
	gridData.relEndProbePos = glm::vec3(OCCUPANCY_METER_SIZE / 2.0) - glm::vec3(DDGI_PROBE_X_OFFSET, DDGI_PROBE_Y_OFFSET, DDGI_PROBE_Z_OFFSET);
	gridData.invProbeTextureResolution = glm::vec2(1.0 / m_ddgiIrradianceProbes[0].getWidth(), 1.0 / m_ddgiIrradianceProbes[0].getHeight());
	gridData.probeFurthestActiveDistance = DDGI_PROBE_MAX_VISIBILITY_RANGE;
	gridData.probeCountX = DDGI_PROBE_X_COUNT;
	gridData.probeCountY = DDGI_PROBE_Y_COUNT;
	gridData.probeCountZ = DDGI_PROBE_Z_COUNT;
	gridData.probeDistX = DDGI_PROBE_X_DISTANCE;
	gridData.probeDistY = DDGI_PROBE_Y_DISTANCE;
	gridData.probeDistZ = DDGI_PROBE_Z_DISTANCE;
	gridData.probeInvDistX = static_cast<float>(1.0 / DDGI_PROBE_X_DISTANCE);
	gridData.probeInvDistY = static_cast<float>(1.0 / DDGI_PROBE_Y_DISTANCE);
	gridData.probeInvDistZ = static_cast<float>(1.0 / DDGI_PROBE_Z_DISTANCE);
	gridData.shadowBias = 0.75 * glm::min(glm::min(DDGI_PROBE_X_DISTANCE, DDGI_PROBE_Y_DISTANCE), DDGI_PROBE_Z_DISTANCE) * TUNABLE_SHADOW_BIAS;
	auto& voxelData{ metaData->cascades[0].voxelData };
	voxelData.resolutionROM = OCCUPANCY_RESOLUTION;
	voxelData.resolutionVM = VOXELMAP_RESOLUTION;
	voxelData.occupationMeterSize = OCCUPANCY_METER_SIZE;
	voxelData.occupationHalfMeterSize = OCCUPANCY_METER_SIZE / 2.0;
	voxelData.invOccupationHalfMeterSize = static_cast<float>(1.0 / voxelData.occupationHalfMeterSize);
	voxelData.offsetNormalScaleROM = 1.7 * BIT_TO_METER_SCALE;
	auto& specData{ metaData->specData };
	specData.specImageRes = glm::ivec2(m_specularReflectionGlossy.getWidth(), m_specularReflectionGlossy.getHeight());
	specData.invSpecImageRes = glm::vec2(1.0 / m_specularReflectionGlossy.getWidth(), 1.0 / m_specularReflectionGlossy.getHeight());
}

void GI::cmdVoxelize(VkCommandBuffer cb, const BufferMapped& indirectDrawCmdData, const Buffer& vertexData, const Buffer& indexData, uint32_t drawCmdCount, uint32_t drawCmdOffset, uint32_t drawCmdStride)
{
	cmdTransferClearVoxelized(cb);

	VkImageMemoryBarrier2 barriers[3]{ SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		m_baseOccupancyMap.getImageHandle(), m_baseOccupancyMap.getSubresourceRange()),
	SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		m_albedoNormalVoxelmap.getImageHandle(), m_albedoNormalVoxelmap.getSubresourceRange()),
	SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		m_emissionMetRoughVoxelmap.getImageHandle(), m_emissionMetRoughVoxelmap.getSubresourceRange()) };
	SyncOperations::cmdExecuteBarrier(cb, barriers);

	cmdPassVoxelize(cb, indirectDrawCmdData, vertexData, indexData, drawCmdCount, drawCmdOffset, drawCmdStride);

	barriers[0] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
		m_baseOccupancyMap.getImageHandle(), m_baseOccupancyMap.getSubresourceRange());
	barriers[1] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
		m_albedoNormalVoxelmap.getImageHandle(), m_albedoNormalVoxelmap.getSubresourceRange());
	barriers[2] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
		m_emissionMetRoughVoxelmap.getImageHandle(), m_emissionMetRoughVoxelmap.getSubresourceRange());
	SyncOperations::cmdExecuteBarrier(cb, barriers);
}


void GI::cmdTransferClearVoxelized(VkCommandBuffer cb)
{
	VkImageMemoryBarrier2 barriers[3]{ SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		m_baseOccupancyMap.getImageHandle(), m_baseOccupancyMap.getSubresourceRange()),
	SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		m_albedoNormalVoxelmap.getImageHandle(), m_albedoNormalVoxelmap.getSubresourceRange()),
	SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		m_emissionMetRoughVoxelmap.getImageHandle(), m_emissionMetRoughVoxelmap.getSubresourceRange()) };

	SyncOperations::cmdExecuteBarrier(cb, barriers);

	VkClearColorValue clearVal{ .uint32 = {0, 0, 0, 0} };
	VkImageSubresourceRange subresourceRange{};
	subresourceRange = m_baseOccupancyMap.getSubresourceRange();
	vkCmdClearColorImage(cb, m_baseOccupancyMap.getImageHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &subresourceRange);
	subresourceRange = m_albedoNormalVoxelmap.getSubresourceRange();
	vkCmdClearColorImage(cb, m_albedoNormalVoxelmap.getImageHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &subresourceRange);
	subresourceRange = m_emissionMetRoughVoxelmap.getSubresourceRange();
	vkCmdClearColorImage(cb, m_emissionMetRoughVoxelmap.getImageHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &subresourceRange);
}
void GI::cmdTransferClearDynamicEmissionVoxelmap(VkCommandBuffer cb)
{
	VkClearColorValue clearVal{ .uint32 = {0, 0, 0, 0} };
	VkImageSubresourceRange subresourceRange{ m_dynamicEmissionVoxelmap.getSubresourceRange() };
	vkCmdClearColorImage(cb, m_dynamicEmissionVoxelmap.getImageHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &subresourceRange);
}
void GI::cmdPassVoxelize(VkCommandBuffer cb, const BufferMapped& indirectDrawCmdData, const Buffer& vertexData, const Buffer& indexData, uint32_t drawCmdCount, uint32_t drawCmdOffset, uint32_t drawCmdStride)
{
	VkRenderingInfo renderInfo{};
	renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	renderInfo.renderArea = { .offset{0,0}, .extent{.width = OCCUPANCY_RESOLUTION, .height = OCCUPANCY_RESOLUTION} };
	renderInfo.layerCount = 1;
	renderInfo.colorAttachmentCount = 0;

	VkBuffer vertexBindings[1]{ vertexData.getBufferHandle() };
	VkDeviceSize vertexBindingOffsets[1]{ vertexData.getOffset() };

	vkCmdBeginRendering(cb, &renderInfo);

	vkCmdBindVertexBuffers(cb, 0, 1, vertexBindings, vertexBindingOffsets);
	vkCmdBindIndexBuffer(cb, indexData.getBufferHandle(), indexData.getOffset(), VK_INDEX_TYPE_UINT32);
	m_voxelize.cmdBindResourceSets(cb);
	m_voxelize.cmdBind(cb);
	vkCmdPushConstants(cb, m_voxelize.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(m_pcDataBOM), &m_pcDataBOM);
	vkCmdDrawIndexedIndirect(cb, indirectDrawCmdData.getBufferHandle(), drawCmdOffset, drawCmdCount, drawCmdStride);

	vkCmdEndRendering(cb);
}
void GI::cmdDispatchCreateROMA(VkCommandBuffer cb)
{
	constexpr uint32_t haltonSeqSize = 128;
	static uint32_t n{ 0 };
	n = (n + 1) % haltonSeqSize;
	static const HaltonSequence<haltonSeqSize, 2> haltonSequence{};

	constexpr int strataCountHor{ DDGI_PROBE_LIGHT_SIDE_SIZE };
	constexpr int strataCountVert{ DDGI_PROBE_LIGHT_SIDE_SIZE };
	constexpr float stratumHorSize{ 1.0f / DDGI_PROBE_LIGHT_SIDE_SIZE };
	constexpr float stratumVertSize{ 1.0f / DDGI_PROBE_LIGHT_SIDE_SIZE };

	m_createROMA.cmdBind(cb);
	m_createROMA.cmdBindResourceSets(cb);
	for (int i{ 0 }; i < ROM_NUMBER; ++i)
	{
		float u{};
		float v{};
		u = ((i % strataCountHor) + haltonSequence.getElement(n, 0)) * stratumHorSize;
		v = ((i / strataCountVert) + haltonSequence.getElement(n, 1)) * stratumVertSize;
		//u = ((i % strataCountHor) + 0.5) * stratumHorSize;
		//v = ((i / strataCountVert) + 0.5) * stratumVertSize;
		m_pcDataROM.stable = 0u;
		m_pcDataROM.directionZ = generateHemisphereDirectionOctohedral(u, v);
		m_pcDataROM.directionX =
			glm::normalize(std::abs(m_pcDataROM.directionZ.y) < 0.9999
				?
				glm::cross(glm::vec3{ 0.0, 1.0, 0.0 }, m_pcDataROM.directionZ)
				:
				glm::cross(glm::vec3{ 0.0, 0.0, glm::sign(-m_pcDataROM.directionZ.y) }, m_pcDataROM.directionZ));
		m_pcDataROM.directionY = glm::cross(m_pcDataROM.directionZ, m_pcDataROM.directionX);
		glm::mat3x4* trMat{ reinterpret_cast<glm::mat3x4*>(m_ROMAtransformMatrices[m_currentBuffers].getData()) + i };
		(*trMat)[0] = glm::vec4{ m_pcDataROM.directionX, 0.0 };
		(*trMat)[1] = glm::vec4{ m_pcDataROM.directionY, 0.0 };
		(*trMat)[2] = glm::vec4{ m_pcDataROM.directionZ, 0.0 };
		m_pcDataROM.indexROM = i;
		m_pcDataROM.resolution = OCCUPANCY_RESOLUTION;
		glm::vec3 originShift{ m_pcDataROM.directionX + m_pcDataROM.directionY + m_pcDataROM.directionZ };
		m_pcDataROM.originROMInLocalBOM = (glm::vec3(1.0f) - originShift) * (static_cast<float>(OCCUPANCY_RESOLUTION / 2));
		m_pcDataROM.originROMInLocalBOM += originShift * 0.5f;
		vkCmdPushConstants(cb, m_createROMA.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_pcDataROM), &m_pcDataROM);
		constexpr uint32_t groupSizeX{ 8 };
		constexpr uint32_t groupSizeY{ 8 };
		constexpr uint32_t groupSizeZ{ 1 };
		vkCmdDispatch(cb, DISPATCH_SIZE(ROM_PACKED_WIDTH, groupSizeX), DISPATCH_SIZE(ROM_PACKED_HEIGHT, groupSizeY), DISPATCH_SIZE(ROM_PACKED_DEPTH, groupSizeZ));
	}

	constexpr int strataCountHorStable{ 2 };
	constexpr int strataCountVertStable{ 4 };
	constexpr float stratumHorSizeStable{ 1.0f / 2 };
	constexpr float stratumVertSizeStable{ 1.0f / 4 };
	for (int i{ 0 }; i < STABLE_ROM_NUMBER; ++i)
	{
		float u{};
		float v{};
		u = ((i % strataCountHorStable) + 0.5) * stratumHorSizeStable;
		v = ((i / strataCountVertStable) + 0.5) * stratumVertSizeStable;
		m_pcDataROM.stable = 1u;
		m_pcDataROM.directionZ = generateHemisphereDirectionOctohedral(u, v);
		m_pcDataROM.directionX =
			glm::normalize(std::abs(m_pcDataROM.directionZ.y) < 0.9999
				?
				glm::cross(glm::vec3{ 0.0, 1.0, 0.0 }, m_pcDataROM.directionZ)
				:
				glm::cross(glm::vec3{ 0.0, 0.0, glm::sign(-m_pcDataROM.directionZ.y) }, m_pcDataROM.directionZ));
		m_pcDataROM.directionY = glm::cross(m_pcDataROM.directionZ, m_pcDataROM.directionX);
		glm::mat3x4* trMat{ reinterpret_cast<glm::mat3x4*>(m_stableROMAtransformMatrices[m_currentBuffers].getData()) + i };
		(*trMat)[0] = glm::vec4{ m_pcDataROM.directionX, 0.0 };
		(*trMat)[1] = glm::vec4{ m_pcDataROM.directionY, 0.0 };
		(*trMat)[2] = glm::vec4{ m_pcDataROM.directionZ, 0.0 };
		m_pcDataROM.indexROM = i;
		m_pcDataROM.resolution = OCCUPANCY_RESOLUTION;
		glm::vec3 originShift{ m_pcDataROM.directionX + m_pcDataROM.directionY + m_pcDataROM.directionZ };
		m_pcDataROM.originROMInLocalBOM = (glm::vec3(1.0f) - originShift) * (static_cast<float>(OCCUPANCY_RESOLUTION / 2));
		m_pcDataROM.originROMInLocalBOM += originShift * 0.5f;
		vkCmdPushConstants(cb, m_createROMA.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_pcDataROM), &m_pcDataROM);
		constexpr uint32_t groupSizeX{ 8 };
		constexpr uint32_t groupSizeY{ 8 };
		constexpr uint32_t groupSizeZ{ 1 };
		vkCmdDispatch(cb, DISPATCH_SIZE(ROM_PACKED_WIDTH, groupSizeX), DISPATCH_SIZE(ROM_PACKED_HEIGHT, groupSizeY), DISPATCH_SIZE(ROM_PACKED_DEPTH, groupSizeZ));
	}

	constexpr uint32_t indexLookUp[32]
	{
		31, 23, 15, 7,   0, 8,  16, 24,
		30, 22, 14, 6,   1, 9,  17, 25,
		29, 21, 13, 5,   2, 10, 18, 26,
		28, 20, 12, 4,   3, 11, 19, 27
	};
	for (int i{ 0 }; i < ROM_NUMBER; ++i)
	{
		glm::vec4* mappedDirection{ reinterpret_cast<glm::vec4*>(m_mappedDirections[m_currentBuffers].getData()) + i };
		*mappedDirection = glm::vec4{ (*(reinterpret_cast<glm::mat3x4*>(m_ROMAtransformMatrices[m_currentBuffers].getData()) + i))[2] };
	}
	for (int i{ 0 }; i < ROM_NUMBER; ++i)
	{
		glm::vec4* mappedDirection{ reinterpret_cast<glm::vec4*>(m_mappedDirections[m_currentBuffers].getData()) + ROM_NUMBER + i };
		*mappedDirection = -glm::vec4{ (*(reinterpret_cast<glm::mat3x4*>(m_ROMAtransformMatrices[m_currentBuffers].getData()) + indexLookUp[i]))[2] };
	}
}
void GI::cmdDispatchInjectLights(VkCommandBuffer cb)
{
	m_injectLight.cmdBind(cb);
	m_injectLight.cmdBindResourceSets(cb);

	m_pcDataLightInjection.voxelmapOriginWorld = SCENE_ORIGIN - glm::vec3(OCCUPANCY_METER_SIZE / 2.0);
	m_pcDataLightInjection.voxelmapScale = 1.0 / VOXEL_TO_METER_SCALE;
	m_pcDataLightInjection.voxelmapResolution = VOXELMAP_RESOLUTION;

	for (int i{ 0 }, lightID{ 1 }; i < m_injectedLightsCount; ++i, ++lightID)
	{
		const Clusterer::LightFormat& lightData{ m_clusterer->m_lightData[m_injectedLightsIndices[i]] };
		Clusterer::LightFormat::Types type{ m_clusterer->m_typeData[m_injectedLightsIndices[i]] };


		m_pcDataLightInjection.spectrum = lightData.spectrum;
		m_pcDataLightInjection.lightLength = lightData.length;
		m_pcDataLightInjection.listIndex = lightData.shadowListIndex;
		m_pcDataLightInjection.lightID = lightID;

		if (lightData.length < VOXEL_TO_METER_SCALE)
			continue;

		constexpr uint32_t groupSize{ 8 };
		if (type == Clusterer::LightFormat::Types::TYPE_POINT)
		{
			for (int j{ 0 }; j < 6; ++j)
			{
				m_pcDataLightInjection.type = type;
				m_pcDataLightInjection.fovScale = 1.0f;
				const uint32_t injectionSize{ std::min(DISPATCH_SIZE(uint32_t((lightData.length * 2 * m_pcDataLightInjection.fovScale) / VOXEL_TO_METER_SCALE), groupSize), static_cast<uint32_t>(VOXELMAP_RESOLUTION / groupSize)) };
				m_pcDataLightInjection.injectionScale = 1.0f / injectionSize;
				m_pcDataLightInjection.layerIndex = j;
				m_pcDataLightInjection.viewmatIndex = lightData.shadowMatrixIndex + j;
				vkCmdPushConstants(cb, m_injectLight.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_pcDataLightInjection), &m_pcDataLightInjection);
				vkCmdDispatch(cb, injectionSize, injectionSize, 1);
			}
		}
		else
		{
			m_pcDataLightInjection.type = type;
			m_pcDataLightInjection.fovScale = std::sqrt(1.0f - lightData.cutoffCos * lightData.cutoffCos) / lightData.cutoffCos;
			const uint32_t injectionSize{ std::min(DISPATCH_SIZE(uint32_t((lightData.length * 2 * m_pcDataLightInjection.fovScale) / VOXEL_TO_METER_SCALE), groupSize), static_cast<uint32_t>(VOXELMAP_RESOLUTION / groupSize)) };
			m_pcDataLightInjection.injectionScale = 1.0f / injectionSize;
			m_pcDataLightInjection.layerIndex = lightData.shadowLayerIndex;
			m_pcDataLightInjection.viewmatIndex = lightData.shadowMatrixIndex;
			m_pcDataLightInjection.lightDir = lightData.lightDir;
			m_pcDataLightInjection.cutoffCos = lightData.cutoffCos;
			m_pcDataLightInjection.falloffCos = lightData.falloffCos;
			vkCmdPushConstants(cb, m_injectLight.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_pcDataLightInjection), &m_pcDataLightInjection);
			vkCmdDispatch(cb, injectionSize, injectionSize, 1);
		}

		SyncOperations::cmdExecuteBarrier(cb, { {SyncOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)} });
	}
}
void GI::cmdDispatchMergeEmission(VkCommandBuffer cb)
{
	m_mergeEmission.cmdBind(cb);
	m_mergeEmission.cmdBindResourceSets(cb);

	constexpr uint32_t groupSize{ 4 };
	vkCmdDispatch(cb, DISPATCH_SIZE(VOXELMAP_RESOLUTION, groupSize), DISPATCH_SIZE(VOXELMAP_RESOLUTION, groupSize), DISPATCH_SIZE(VOXELMAP_RESOLUTION, groupSize));
}
void GI::cmdDispatchTraceProbes(VkCommandBuffer cb, bool skyboxEnabled)
{
	m_traceProbes.setResourceInUse(2, m_currentBuffers);
	m_traceProbes.setResourceInUse(5, m_currentNewProbes);
	m_traceProbes.cmdBind(cb);
	m_traceProbes.cmdBindResourceSets(cb);

	m_pcDataTraceProbes.skyboxEnabled = skyboxEnabled ? 1u : 0u;
	vkCmdPushConstants(cb, m_traceProbes.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &m_pcDataTraceProbes);
	constexpr uint32_t groupSizeX{ DDGI_PROBE_LIGHT_SIDE_SIZE };
	constexpr uint32_t groupSizeY{ DDGI_PROBE_LIGHT_SIDE_SIZE };
	constexpr uint32_t groupSizeZ{ 1 };
	vkCmdDispatch(cb, DISPATCH_SIZE(DDGI_PROBE_X_COUNT * DDGI_PROBE_Z_COUNT * DDGI_PROBE_LIGHT_SIDE_SIZE, groupSizeX), DISPATCH_SIZE(DDGI_PROBE_Y_COUNT * DDGI_PROBE_LIGHT_SIDE_SIZE, groupSizeY), 1);
}
void GI::cmdDispatchTraceSpecular(VkCommandBuffer cb, const glm::mat4& worldFromNDC, const glm::vec3& campos, bool skyboxEnabled)
{
	m_traceSpecular.setResourceInUse(2, m_currentBuffers);
	m_traceSpecular.setResourceInUse(5, m_currentNewProbes);
	m_traceSpecular.cmdBind(cb);
	m_traceSpecular.cmdBindResourceSets(cb);

	m_pcDataTraceSpecular.sceneCenter = SCENE_ORIGIN;
	m_pcDataTraceSpecular.worldFromNDC = worldFromNDC;
	m_pcDataTraceSpecular.campos = campos;
	m_pcDataTraceSpecular.skyboxEnabled = skyboxEnabled ? 1u : 0u;
	vkCmdPushConstants(cb, m_traceSpecular.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_pcDataTraceSpecular), &m_pcDataTraceSpecular);
	constexpr uint32_t groupSizeX{ 8 };
	constexpr uint32_t groupSizeY{ 8 };
	constexpr uint32_t groupSizeZ{ 1 };
	vkCmdDispatch(cb, DISPATCH_SIZE(m_specularReflectionGlossy.getWidth(), groupSizeX), DISPATCH_SIZE(m_specularReflectionGlossy.getHeight(), groupSizeY), 1);
}
void GI::cmdDispatchBlurSpecular(VkCommandBuffer cb)
{
	m_bilateral.cmdBindResourceSets(cb);
	m_bilateral.cmdBind(cb);

	m_pcDataBilateral.imgRes = glm::ivec2(m_specularReflectionGlossy.getWidth(), m_specularReflectionGlossy.getHeight());
	vkCmdPushConstants(cb, m_bilateral.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_pcDataBilateral), &m_pcDataBilateral);
	constexpr uint32_t groupSizeX{ 8 };
	constexpr uint32_t groupSizeY{ 8 };
	constexpr uint32_t groupSizeZ{ 1 };
	vkCmdDispatch(cb, DISPATCH_SIZE(m_pcDataBilateral.imgRes.x, groupSizeX), DISPATCH_SIZE(m_pcDataBilateral.imgRes.y, groupSizeY), groupSizeZ);
}
void GI::cmdDispatchComputeIrradianceAndVisibility(VkCommandBuffer cb)
{
	m_computeIrradiance.setResourceInUse(1, m_currentNewProbes);
	m_computeIrradiance.setResourceInUse(2, m_currentBuffers);
	m_computeIrradiance.cmdBind(cb);
	m_computeIrradiance.cmdBindResourceSets(cb);
	constexpr uint32_t groupSizeXI{ DDGI_PROBE_LIGHT_SIDE_SIZE };
	constexpr uint32_t groupSizeYI{ DDGI_PROBE_LIGHT_SIDE_SIZE };
	constexpr uint32_t groupSizeZI{ 1 };
	vkCmdDispatch(cb, DISPATCH_SIZE(DDGI_PROBE_X_COUNT * DDGI_PROBE_Z_COUNT * DDGI_PROBE_LIGHT_SIDE_SIZE, groupSizeXI), DISPATCH_SIZE(DDGI_PROBE_Y_COUNT * DDGI_PROBE_LIGHT_SIDE_SIZE, groupSizeYI), groupSizeZI);

	m_computeVisibility.setResourceInUse(1, m_currentNewProbes);
	m_computeVisibility.setResourceInUse(2, m_currentBuffers);
	m_computeVisibility.cmdBind(cb);
	m_computeVisibility.cmdBindResourceSets(cb);
	constexpr uint32_t groupSizeXV{ DDGI_PROBE_VISIBILITY_SIDE_SIZE };
	constexpr uint32_t groupSizeYV{ DDGI_PROBE_VISIBILITY_SIDE_SIZE };
	constexpr uint32_t groupSizeZV{ 1 };
	vkCmdDispatch(cb, DISPATCH_SIZE(DDGI_PROBE_X_COUNT * DDGI_PROBE_Z_COUNT * DDGI_PROBE_LIGHT_SIDE_SIZE, groupSizeXV), DISPATCH_SIZE(DDGI_PROBE_Y_COUNT * DDGI_PROBE_LIGHT_SIDE_SIZE, groupSizeYV), groupSizeZV);
}

void GI::initializeDebug(VkDevice device, const ResourceSet& viewprojRS, uint32_t width, uint32_t height, BufferBaseHostAccessible& baseHostBuffer, VkSampler generalSampler, CommandBufferSet& cmdBufferSet, VkQueue queue)
{
	PipelineAssembler assembler{ device };
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0, VK_CULL_MODE_NONE);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, width, height);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_POINT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);
	std::array<std::reference_wrapper<const ResourceSet>, 5> resourceSetsDebugVoxel{ viewprojRS, m_resSetReadBOM, m_resSetReadROMA, m_resSetDynamicEmissionRead, m_resSetAlbedoNormalRead };
	m_debugVoxel.initializeGraphics(assembler,
		{ { ShaderStage{.stage = VK_SHADER_STAGE_VERTEX_BIT, .filepath = "shaders/cmpld/debug_voxel_vert.spv"},
		ShaderStage{.stage = VK_SHADER_STAGE_GEOMETRY_BIT, .filepath = "shaders/cmpld/debug_voxel_geom.spv"},
		ShaderStage{.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .filepath = "shaders/cmpld/color_frag.spv"} } },
		resourceSetsDebugVoxel,
		{},
		{},
		{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, .offset = 0, .size = sizeof(m_pcDataDebugVoxel)}} });

	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, width, height);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);

	VkDescriptorSetLayoutBinding bindingRadianceProbes{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo radianceProbesAddressInfo{ .imageView = m_ddgiRadianceProbes.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding bindingIrradianceProbeHistory{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo irradianceProbesAddressInfo0{ .sampler = generalSampler, .imageView = m_ddgiIrradianceProbes[0].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
	VkDescriptorImageInfo irradianceProbesAddressInfo1{ .sampler = generalSampler, .imageView = m_ddgiIrradianceProbes[1].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding bindingVisibilityProbeHistory{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	VkDescriptorImageInfo visibilityProbesAddressInfo0{ .sampler = generalSampler, .imageView = m_ddgiVisibilityProbes[0].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
	VkDescriptorImageInfo visibilityProbesAddressInfo1{ .sampler = generalSampler, .imageView = m_ddgiVisibilityProbes[1].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL };
	m_resSetProbesDebug.initializeSet(device, 2, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ bindingRadianceProbes, bindingIrradianceProbeHistory, bindingVisibilityProbeHistory },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>
		{{
			std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pStorageImage = &radianceProbesAddressInfo }, VkDescriptorDataEXT{ .pStorageImage = &radianceProbesAddressInfo } },
			std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pSampledImage = &irradianceProbesAddressInfo0 }, VkDescriptorDataEXT{ .pSampledImage = &irradianceProbesAddressInfo1 } },
			std::vector<VkDescriptorDataEXT>{ VkDescriptorDataEXT{ .pSampledImage = &visibilityProbesAddressInfo0 }, VkDescriptorDataEXT{ .pSampledImage = &visibilityProbesAddressInfo1 } },
		}},
		true);
	std::array<std::reference_wrapper<const ResourceSet>, 2> resourceSetsDebugProbe{ viewprojRS, m_resSetProbesDebug };
	m_debugProbes.initializeGraphics(assembler,
		{ { ShaderStage{.stage = VK_SHADER_STAGE_VERTEX_BIT, .filepath = "shaders/cmpld/debug_probe_vert.spv"},
		ShaderStage{.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .filepath = "shaders/cmpld/debug_probe_frag.spv"} } },
		resourceSetsDebugProbe,
		{ {PosOnlyVertex::getBindingDescription()} },
		{ PosOnlyVertex::getAttributeDescriptions() },
		{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(m_pcDataDebugProbes)}} });
	m_sphereVertexCount = LoaderOBJ::loadOBJfile("internal/sphere.obj", m_sphereVertexData, LoaderOBJ::POS_VERT, baseHostBuffer, cmdBufferSet, queue);
}

void GI::cmdDrawBOM(VkCommandBuffer cb, const glm::vec3& camPos)
{
	m_pcDataDebugVoxel.resolution = OCCUPANCY_RESOLUTION;
	m_pcDataDebugVoxel.voxelSize = OCCUPANCY_METER_SIZE / OCCUPANCY_RESOLUTION;
	m_pcDataDebugVoxel.camPos = camPos;
	m_pcDataDebugVoxel.debugType = UiData::VoxelDebugType::BOM_VOXEL_DEBUG;
	m_debugVoxel.cmdBindResourceSets(cb);
	m_debugVoxel.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugVoxel.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0, sizeof(m_pcDataDebugVoxel), &m_pcDataDebugVoxel);
	vkCmdDraw(cb, OCCUPANCY_RESOLUTION * OCCUPANCY_RESOLUTION * OCCUPANCY_RESOLUTION, 1, 0, 0);
}
void GI::cmdDrawROM(VkCommandBuffer cb, const glm::vec3& camPos, uint32_t romaIndex)
{
	m_pcDataDebugVoxel.resolution = OCCUPANCY_RESOLUTION;
	m_pcDataDebugVoxel.voxelSize = OCCUPANCY_METER_SIZE / OCCUPANCY_RESOLUTION;
	m_pcDataDebugVoxel.camPos = camPos;
	m_pcDataDebugVoxel.debugType = UiData::VoxelDebugType::ROM_VOXEL_DEBUG;
	m_pcDataDebugVoxel.indexROMA = romaIndex;
	m_debugVoxel.cmdBindResourceSets(cb);
	m_debugVoxel.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugVoxel.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0, sizeof(m_pcDataDebugVoxel), &m_pcDataDebugVoxel);
	vkCmdDraw(cb, OCCUPANCY_RESOLUTION * OCCUPANCY_RESOLUTION * OCCUPANCY_RESOLUTION, 1, 0, 0);
}
void GI::cmdDrawAlbedo(VkCommandBuffer cb, const glm::vec3& camPos)
{
	m_pcDataDebugVoxel.resolution = VOXELMAP_RESOLUTION;
	m_pcDataDebugVoxel.voxelSize = OCCUPANCY_METER_SIZE / VOXELMAP_RESOLUTION;
	m_pcDataDebugVoxel.camPos = camPos;
	m_pcDataDebugVoxel.debugType = UiData::VoxelDebugType::ALBEDO_VOXEL_DEBUG;
	m_debugVoxel.cmdBindResourceSets(cb);
	m_debugVoxel.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugVoxel.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0, sizeof(m_pcDataDebugVoxel), &m_pcDataDebugVoxel);
	vkCmdDraw(cb, VOXELMAP_RESOLUTION * VOXELMAP_RESOLUTION * VOXELMAP_RESOLUTION, 1, 0, 0);
}
void GI::cmdDrawMetalness(VkCommandBuffer cb, const glm::vec3& camPos)
{
	m_pcDataDebugVoxel.resolution = VOXELMAP_RESOLUTION;
	m_pcDataDebugVoxel.voxelSize = OCCUPANCY_METER_SIZE / VOXELMAP_RESOLUTION;
	m_pcDataDebugVoxel.camPos = camPos;
	m_pcDataDebugVoxel.debugType = UiData::VoxelDebugType::METALNESS_VOXEL_DEBUG;
	m_debugVoxel.cmdBindResourceSets(cb);
	m_debugVoxel.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugVoxel.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0, sizeof(m_pcDataDebugVoxel), &m_pcDataDebugVoxel);
	vkCmdDraw(cb, VOXELMAP_RESOLUTION * VOXELMAP_RESOLUTION * VOXELMAP_RESOLUTION, 1, 0, 0);
}
void GI::cmdDrawRoughness(VkCommandBuffer cb, const glm::vec3& camPos)
{
	m_pcDataDebugVoxel.resolution = VOXELMAP_RESOLUTION;
	m_pcDataDebugVoxel.voxelSize = OCCUPANCY_METER_SIZE / VOXELMAP_RESOLUTION;
	m_pcDataDebugVoxel.camPos = camPos;
	m_pcDataDebugVoxel.debugType = UiData::VoxelDebugType::ROUGHNESS_VOXEL_DEBUG;
	m_debugVoxel.cmdBindResourceSets(cb);
	m_debugVoxel.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugVoxel.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0, sizeof(m_pcDataDebugVoxel), &m_pcDataDebugVoxel);
	vkCmdDraw(cb, VOXELMAP_RESOLUTION * VOXELMAP_RESOLUTION * VOXELMAP_RESOLUTION, 1, 0, 0);
}
void GI::cmdDrawEmission(VkCommandBuffer cb, const glm::vec3& camPos)
{
	m_pcDataDebugVoxel.resolution = VOXELMAP_RESOLUTION;
	m_pcDataDebugVoxel.voxelSize = OCCUPANCY_METER_SIZE / VOXELMAP_RESOLUTION;
	m_pcDataDebugVoxel.camPos = camPos;
	m_pcDataDebugVoxel.debugType = UiData::VoxelDebugType::EMISSION_VOXEL_DEBUG;
	m_debugVoxel.cmdBindResourceSets(cb);
	m_debugVoxel.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugVoxel.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT, 0, sizeof(m_pcDataDebugVoxel), &m_pcDataDebugVoxel);
	vkCmdDraw(cb, VOXELMAP_RESOLUTION * VOXELMAP_RESOLUTION * VOXELMAP_RESOLUTION, 1, 0, 0);
}

void GI::cmdDrawRadianceProbes(VkCommandBuffer cb)
{
	m_pcDataDebugProbes.firstProbePosition = glm::vec3((1.0f / DDGI_PROBE_X_COUNT) - 1.0f, (1.0f / DDGI_PROBE_Y_COUNT) - 1.0f, (1.0f / DDGI_PROBE_Z_COUNT) - 1.0f) * (OCCUPANCY_METER_SIZE / 2);
	m_pcDataDebugProbes.probeCountX = DDGI_PROBE_X_COUNT;
	m_pcDataDebugProbes.probeCountY = DDGI_PROBE_Y_COUNT;
	m_pcDataDebugProbes.probeCountZ = DDGI_PROBE_Z_COUNT;
	m_pcDataDebugProbes.xDist = DDGI_PROBE_X_DISTANCE;
	m_pcDataDebugProbes.yDist = DDGI_PROBE_Y_DISTANCE;
	m_pcDataDebugProbes.debugType = 0;

	VkBuffer vertexBinding[1]{ m_sphereVertexData.getBufferHandle() };
	VkDeviceSize vertexOffsets[1]{ m_sphereVertexData.getOffset() };
	vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
	m_debugProbes.cmdBindResourceSets(cb);
	m_debugProbes.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugProbes.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(m_pcDataDebugProbes), &m_pcDataDebugProbes);
	vkCmdDraw(cb, m_sphereVertexCount, DDGI_PROBE_X_COUNT * DDGI_PROBE_Y_COUNT * DDGI_PROBE_Z_COUNT, 0, 0);
}
void GI::cmdDrawIrradianceProbes(VkCommandBuffer cb)
{
	m_pcDataDebugProbes.firstProbePosition = glm::vec3((1.0f / DDGI_PROBE_X_COUNT) - 1.0f, (1.0f / DDGI_PROBE_Y_COUNT) - 1.0f, (1.0f / DDGI_PROBE_Z_COUNT) - 1.0f) * (OCCUPANCY_METER_SIZE / 2);
	m_pcDataDebugProbes.probeCountX = DDGI_PROBE_X_COUNT;
	m_pcDataDebugProbes.probeCountY = DDGI_PROBE_Y_COUNT;
	m_pcDataDebugProbes.probeCountZ = DDGI_PROBE_Z_COUNT;
	m_pcDataDebugProbes.xDist = DDGI_PROBE_X_DISTANCE;
	m_pcDataDebugProbes.yDist = DDGI_PROBE_Y_DISTANCE;
	m_pcDataDebugProbes.zDist = DDGI_PROBE_Z_DISTANCE;
	m_pcDataDebugProbes.debugType = 1;
	m_pcDataDebugProbes.invIrradianceTextureResolution = glm::vec2(1.0 / m_ddgiIrradianceProbes[0].getWidth(), 1.0 / m_ddgiIrradianceProbes[0].getHeight());

	VkBuffer vertexBinding[1]{ m_sphereVertexData.getBufferHandle() };
	VkDeviceSize vertexOffsets[1]{ m_sphereVertexData.getOffset() };
	vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
	m_debugProbes.setResourceInUse(1, m_currentNewProbes);
	m_debugProbes.cmdBindResourceSets(cb);
	m_debugProbes.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugProbes.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(m_pcDataDebugProbes), &m_pcDataDebugProbes);
	vkCmdDraw(cb, m_sphereVertexCount, DDGI_PROBE_X_COUNT * DDGI_PROBE_Y_COUNT * DDGI_PROBE_Z_COUNT, 0, 0);
}
void GI::cmdDrawVisibilityProbes(VkCommandBuffer cb)
{
	m_pcDataDebugProbes.firstProbePosition = glm::vec3((1.0f / DDGI_PROBE_X_COUNT) - 1.0f, (1.0f / DDGI_PROBE_Y_COUNT) - 1.0f, (1.0f / DDGI_PROBE_Z_COUNT) - 1.0f) * (OCCUPANCY_METER_SIZE / 2);
	m_pcDataDebugProbes.probeCountX = DDGI_PROBE_X_COUNT;
	m_pcDataDebugProbes.probeCountY = DDGI_PROBE_Y_COUNT;
	m_pcDataDebugProbes.probeCountZ = DDGI_PROBE_Z_COUNT;
	m_pcDataDebugProbes.xDist = DDGI_PROBE_X_DISTANCE;
	m_pcDataDebugProbes.yDist = DDGI_PROBE_Y_DISTANCE;
	m_pcDataDebugProbes.zDist = DDGI_PROBE_Z_DISTANCE;
	m_pcDataDebugProbes.debugType = 2;
	m_pcDataDebugProbes.invIrradianceTextureResolution = glm::vec2(1.0 / m_ddgiVisibilityProbes[0].getWidth(), 1.0 / m_ddgiVisibilityProbes[0].getHeight());
	m_pcDataDebugProbes.invProbeMaxActiveDistance = 1.0 / DDGI_PROBE_MAX_VISIBILITY_RANGE;

	VkBuffer vertexBinding[1]{ m_sphereVertexData.getBufferHandle() };
	VkDeviceSize vertexOffsets[1]{ m_sphereVertexData.getOffset() };
	vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
	m_debugProbes.setResourceInUse(1, m_currentNewProbes);
	m_debugProbes.cmdBindResourceSets(cb);
	m_debugProbes.cmdBind(cb);
	vkCmdPushConstants(cb, m_debugProbes.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(m_pcDataDebugProbes), &m_pcDataDebugProbes);
	vkCmdDraw(cb, m_sphereVertexCount, DDGI_PROBE_X_COUNT * DDGI_PROBE_Y_COUNT * DDGI_PROBE_Z_COUNT, 0, 0);
}