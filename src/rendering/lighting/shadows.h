#ifndef SHADOWS_HEADER
#define SHADOWS_HEADER

#include <vector>
#include <list>
#include <algorithm>
#include <intrin.h>

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/vulkan_object_handling/vulkan_object_handler.h"
#include "src/rendering/renderer/clusterer.h"
#include "src/rendering/renderer/culling.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_abstraction/BB.h"

#include "src/tools/time_measurement.h"

#define MAX_POINT_LIGHT_SHADOWS 64
#define MAX_SPOT_LIGHT_SHADOWS 64

class ShadowCaster
{
private:
	VkDevice m_device{};

	uint32_t m_shadowMapsLayerCount{ 4 };
	using DrawsIndex = uint32_t;
	struct ShadowMapInfo
	{
		ImageListContainer::ImageListContainerIndices shadowMapIndices{};
		uint32_t drawsIndex{};
		uint32_t viewMatIndex{};
		float proj00{};
	};
	struct ShadowCubeMapInfo
	{
		ImageListContainer::ImageListContainerIndices shadowMapIndices{};
		uint32_t drawsIndex{};
		uint32_t viewMatIndex{};
	};
	std::vector<ShadowMapInfo> m_indicesForShadowMaps{};
	std::vector<ShadowCubeMapInfo> m_indicesForShadowCubeMaps{};
	std::vector<std::vector<uint32_t>> m_drawCommandIndices{};
	ImageListContainer m_shadowMaps;
	std::vector<ImageList> m_shadowCubeMaps{};
	bool m_newLightsAdded{ false };
	uint32_t m_viewMatCount{ 0 };
	BufferBaseHostAccessible m_shadowMapViewMatrices;
	BufferMapped* const m_indirectDataBuffer{ nullptr };

	OBBs* m_rUnitsBoundingBoxes{};

	Pipeline m_shadowMapPass{};

	Clusterer* const m_clusterer{ nullptr };

	struct { float far{}; float near{}; float cubeProj00{}; float proj22{}; float proj32{}; } m_frustumData{};

public:
	ShadowCaster(VkDevice device, Clusterer& clusterer, 
		BufferMapped& indirectDataBuffer, 
		const BufferMapped& modelTransformDataSSBO,
		const BufferMapped& drawDataSSBO,
		OBBs& boundingBoxes) :
		m_shadowMaps{ device, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, false,
		VkSamplerCreateInfo{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
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
			.unnormalizedCoordinates = VK_FALSE }, m_shadowMapsLayerCount, VK_IMAGE_ASPECT_DEPTH_BIT }, m_device{ device }, m_clusterer{ &clusterer }, m_indirectDataBuffer{ &indirectDataBuffer },
			m_shadowMapViewMatrices{ device, sizeof(glm::mat4) * (MAX_POINT_LIGHT_SHADOWS * 6 + MAX_SPOT_LIGHT_SHADOWS), 
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, BufferBase::NULL_FLAG, false, true }, m_rUnitsBoundingBoxes{ &boundingBoxes }
	{
		PipelineAssembler assembler{ device };
		assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_VIEWPORT);
		assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DYNAMIC);
		assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
		assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
		assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
		assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_SHADOW_MAP);
		assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
		assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT, VK_COMPARE_OP_LESS_OR_EQUAL);
		assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEPTH_ATTACHMENT_ONLY);

		std::vector<ResourceSet> resourceSets{};

		VkDescriptorSetLayoutBinding shadowMapViewMatricesBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
		VkDescriptorAddressInfoEXT shadowMapViewMatricesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_shadowMapViewMatrices.getDeviceAddress(), .range = m_shadowMapViewMatrices.getSize() };

		VkDescriptorSetLayoutBinding modelTransformBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
		VkDescriptorAddressInfoEXT modelTransformAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = modelTransformDataSSBO.getDeviceAddress(), .range = modelTransformDataSSBO.getSize() };

		VkDescriptorSetLayoutBinding drawDataBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
		VkDescriptorAddressInfoEXT drawDataAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = drawDataSSBO.getDeviceAddress(), .range = drawDataSSBO.getSize() };

		resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
		{shadowMapViewMatricesBinding, modelTransformBinding, drawDataBinding},  {},
			{{{.pStorageBuffer = &shadowMapViewMatricesAddressInfo}},
			{{.pStorageBuffer = &modelTransformAddressInfo}},
			{{.pStorageBuffer = &drawDataAddressInfo}}}, false });

		m_shadowMapPass.initializeGraphics(assembler, { {ShaderStage{ VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/shadow_pass_vert.spv"}, 
			ShaderStage{ VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/shadow_pass_frag.spv"}} }, resourceSets,
			{ {StaticVertex::getBindingDescription()} }, { {StaticVertex::getAttributeDescriptions()[0]} }, 
			{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(uint32_t) * 7}}});
	}
	~ShadowCaster()
	{

	}

	void submitFrustum(float near, float far)
	{
		m_frustumData.near = near;
		m_frustumData.far = far;
		float h{ static_cast<float>(1.0 / std::tan((glm::radians(90.0) * 0.5))) };
		m_frustumData.cubeProj00 = h;
		m_frustumData.proj22 = far / (far - near);
		m_frustumData.proj32 = (-near * far) / (far - near);
	}

	void prepareDataForShadowMapRendering()
	{
		for (int i{ 0 }, a{ 0 }; i < m_clusterer->m_nonculledLightsCount; ++i)
		{
			uint32_t index{ m_clusterer->m_nonculledLightsData[i].index };
			Clusterer::LightFormat& light{ m_clusterer->m_lightData[index] };
			auto type{ m_clusterer->m_typeData[index] };

			if (light.shadowListIndex == -1)
				continue;

			if (type == Clusterer::LightFormat::TYPE_SPOT)
			{
				m_indicesForShadowMaps.push_back(
					{.shadowMapIndices = {.listIndex = static_cast<uint16_t>(light.shadowListIndex), .layerIndex = static_cast<uint16_t>(light.shadowLayerIndex)},
					.drawsIndex = static_cast<uint32_t>(a),
					.viewMatIndex = static_cast<uint32_t>(light.shadowMatrixIndex),
					.proj00 = light.cutoffCos / (std::sqrt(1 - light.cutoffCos * light.cutoffCos))});
				glm::vec4 boundingSphere{ m_clusterer->m_boundingSpheres[index] };
				cullMeshes(glm::vec3{boundingSphere}, boundingSphere.w, m_drawCommandIndices[a++]);
			}
			else
			{
				m_indicesForShadowCubeMaps.push_back(
					{.shadowMapIndices = {.listIndex = static_cast<uint16_t>(light.shadowListIndex), .layerIndex = 0}, 
					.drawsIndex = static_cast<uint32_t>(a), 
					.viewMatIndex = static_cast<uint32_t>(light.shadowMatrixIndex) });
				glm::vec4 boundingSphere{ m_clusterer->m_boundingSpheres[index] };
				cullMeshes(glm::vec3{boundingSphere}, boundingSphere.w, m_drawCommandIndices[a++]);
			}
		}
		
		if (!m_indicesForShadowMaps.empty())
			std::sort(m_indicesForShadowMaps.begin(), m_indicesForShadowMaps.end(), [](auto& one, auto& two) -> bool { return one.shadowMapIndices.listIndex < two.shadowMapIndices.listIndex; });
	}

	void cmdRenderShadowMaps(VkCommandBuffer cb, DescriptorManager& descriptorManager)
	{
		if (m_newLightsAdded)
		{
			m_shadowMaps.cmdTransitionLayoutsFromUndefined(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			for (auto& shadowCubeMap : m_shadowCubeMaps)
				shadowCubeMap.cmdTransitionLayoutFromUndefined(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			m_newLightsAdded = false;
		}
		 
		cmdChangeLayouts(cb, 
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, 
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT);

		m_shadowMapPass.cmdBind(cb);
		descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowMapPass.getResourceSets(), m_shadowMapPass.getResourceSetsInUse(), m_shadowMapPass.getPipelineLayoutHandle());
		cmdRenderShadowOnedirMaps(cb);
		cmdRenderShadowCubeMaps(cb);

		cmdChangeLayouts(cb, 
			VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

		m_indicesForShadowMaps.clear();
		m_indicesForShadowCubeMaps.clear();
	}

	ImageListContainer& getShadowMaps()
	{
		return m_shadowMaps;
	}
	std::vector<ImageList>& getShadowCubeMaps()
	{
		return m_shadowCubeMaps;
	}
	BufferBaseHostAccessible& getShadowViewMatrices()
	{
		return m_shadowMapViewMatrices;
	}

private:
	ImageListContainer::ImageListContainerIndices addShadowMap(uint32_t width, uint32_t height)
	{
		m_newLightsAdded = true;
		return m_shadowMaps.getNewImage(width, height, VK_FORMAT_D32_SFLOAT);
	}
	uint32_t addShadowCubeMap(uint32_t sideLength)
	{
		m_newLightsAdded = true;
		uint32_t index{ static_cast<uint32_t>(m_shadowCubeMaps.size()) };
		constexpr uint32_t cubemapLayerCount{ 6 };
		m_shadowCubeMaps.emplace_back(m_device, sideLength, sideLength, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, false, cubemapLayerCount, VK_IMAGE_ASPECT_DEPTH_BIT);
		return index;
	}
	void cullMeshes(const glm::vec3& pos, float rad, std::vector<uint32_t>& drawCommandIndices)
	{
		drawCommandIndices.clear();

		float *xsOfXaxii, *ysOfXaxii, *zsOfXaxii, 
			*xsOfYaxii, *ysOfYaxii, *zsOfYaxii, 
			*xsOfZaxii, *ysOfZaxii, *zsOfZaxii, 
			*centersX, *centersY, *centersZ, 
			*extentsX, *extentsY, *extentsZ;
		uint32_t meshCount{ m_rUnitsBoundingBoxes->getAxiiOBBs(
			&xsOfXaxii, &ysOfXaxii, &zsOfXaxii, 
			&xsOfYaxii, &ysOfYaxii, &zsOfYaxii, 
			&xsOfZaxii, &ysOfZaxii, &zsOfZaxii,
			&centersX, &centersY, &centersZ,
			&extentsX, &extentsY, &extentsZ) };
		__m128 xP{_mm_load1_ps(&pos.x)};
		__m128 yP{_mm_load1_ps(&pos.y)};
		__m128 zP{_mm_load1_ps(&pos.z)};
		__m128 r{_mm_load1_ps(&rad)};
		__m128 r2{_mm_mul_ps(r, r)};
		for (int i{ 0 }; i < meshCount; i += 4)
		{
			__m128 xC{_mm_load_ps(centersX + i)};
			__m128 yC{_mm_load_ps(centersY + i)};
			__m128 zC{_mm_load_ps(centersZ + i)};

			__m128 xCP{_mm_sub_ps(xP, xC)};
			__m128 yCP{_mm_sub_ps(yP, yC)};
			__m128 zCP{_mm_sub_ps(zP, zC)};

			__m128 xU{_mm_load_ps(xsOfXaxii + i)};
			__m128 yU{_mm_load_ps(ysOfXaxii + i)};
			__m128 zU{_mm_load_ps(zsOfXaxii + i)};
			__m128 xV{_mm_load_ps(xsOfYaxii + i)};
			__m128 yV{_mm_load_ps(ysOfYaxii + i)};
			__m128 zV{_mm_load_ps(zsOfYaxii + i)};
			__m128 xW{_mm_load_ps(xsOfZaxii + i)};
			__m128 yW{_mm_load_ps(ysOfZaxii + i)};
			__m128 zW{_mm_load_ps(zsOfZaxii + i)};

			__m128 dpU{_mm_add_ps(_mm_add_ps(_mm_mul_ps(xCP, xU), _mm_mul_ps(yCP, yU)), _mm_mul_ps(zCP, zU))};
			__m128 dpV{_mm_add_ps(_mm_add_ps(_mm_mul_ps(xCP, xV), _mm_mul_ps(yCP, yV)), _mm_mul_ps(zCP, zV))};
			__m128 dpW{_mm_add_ps(_mm_add_ps(_mm_mul_ps(xCP, xW), _mm_mul_ps(yCP, yW)), _mm_mul_ps(zCP, zW))};

			for (int j{ 0 }; j < 4; ++j)
			{
				dpU.m128_f32[j] = glm::clamp(dpU.m128_f32[j], -*(extentsX + i + j), *(extentsX + i + j));

				dpV.m128_f32[j] = glm::clamp(dpV.m128_f32[j], -*(extentsY + i + j), *(extentsY + i + j));

				dpW.m128_f32[j] = glm::clamp(dpW.m128_f32[j], -*(extentsZ + i + j), *(extentsZ + i + j));
			}

			xC = _mm_add_ps(xC, _mm_add_ps(_mm_add_ps(_mm_mul_ps(xU, dpU), _mm_mul_ps(yU, dpU)), _mm_mul_ps(zU, dpU)));
			yC = _mm_add_ps(yC, _mm_add_ps(_mm_add_ps(_mm_mul_ps(xV, dpV), _mm_mul_ps(yV, dpV)), _mm_mul_ps(zV, dpV)));
			zC = _mm_add_ps(zC, _mm_add_ps(_mm_add_ps(_mm_mul_ps(xW, dpW), _mm_mul_ps(yW, dpW)), _mm_mul_ps(zW, dpW)));

			__m128 xD{_mm_sub_ps(xP, xC)};
			__m128 yD{_mm_sub_ps(yP, yC)};
			__m128 zD{_mm_sub_ps(zP, zC)};

			__m128 d2{_mm_add_ps(_mm_add_ps(_mm_mul_ps(xD, xD), _mm_mul_ps(yD, yD)), _mm_mul_ps(zD, zD))};

			__m128 res = _mm_cmp_ps(d2, r2, _CMP_LE_OS);

			for (int j{ 0 }; j < 4; ++j)
			{
				if (res.m128_u32[j] && (i + j < meshCount))
					drawCommandIndices.push_back(i + j);
			}
		}
	}


	void calcViewMatrix(uint32_t index, const glm::vec3& pos, const glm::vec3& dir)
	{
		glm::mat4* mat{ reinterpret_cast<glm::mat4*>(m_shadowMapViewMatrices.getData()) + index };
		*mat = glm::lookAt(pos, pos + dir, (dir.y < 0.999 && dir.y > -0.999) ? glm::vec3{0.0, 1.0, 0.0} : glm::vec3{ 0.0, 0.0, 1.0 });
	}
	void calcCubeViewMatrices(uint32_t index, const glm::vec3& pos)
	{
		glm::mat4* mat{ reinterpret_cast<glm::mat4*>(m_shadowMapViewMatrices.getData()) + index };
		*(mat++) = glm::lookAt(pos, pos + glm::vec3{1.0, 0.0, 0.0}, glm::vec3{0.0, 1.0, 0.0});
		*(mat++) = glm::lookAt(pos, pos + glm::vec3{-1.0, 0.0, 0.0}, glm::vec3{0.0, 1.0, 0.0});
		*(mat++) = glm::lookAt(pos, pos + glm::vec3{0.0, 1.0, 0.0}, glm::vec3{0.0, 0.0, -1.0});
		*(mat++) = glm::lookAt(pos, pos + glm::vec3{0.0, -1.0, 0.0}, glm::vec3{0.0, 0.0, 1.0});
		*(mat++) = glm::lookAt(pos, pos + glm::vec3{0.0, 0.0, 1.0}, glm::vec3{0.0, 1.0, 0.0});
		*mat = glm::lookAt(pos, pos + glm::vec3{0.0, 0.0, -1.0}, glm::vec3{0.0, 1.0, 0.0});
	}
	uint32_t addSpotViewMatrix(const glm::vec3& pos, const glm::vec3& dir)
	{
		uint32_t index{ m_viewMatCount };
		++m_viewMatCount;
		EASSERT(m_viewMatCount <= MAX_SPOT_LIGHT_SHADOWS, "App", "Too many shadow maps");
		calcViewMatrix(index, pos, dir);
		return index;
	}
	uint32_t addPointViewMatrices(const glm::vec3& pos)
	{
		uint32_t index{ m_viewMatCount };
		m_viewMatCount += 6;
		EASSERT(m_viewMatCount <= MAX_POINT_LIGHT_SHADOWS, "App", "Too many shadow maps");
		calcCubeViewMatrices(index, pos);
		return index;
	}

	void cmdChangeLayouts(VkCommandBuffer cb, VkImageLayout srcLayout, VkImageLayout dstLayout, VkPipelineStageFlags srcStages, VkPipelineStageFlags dstStages)
	{
		static std::vector<VkImageMemoryBarrier2> barriers{};

		for (int i{ 0 }; i < m_indicesForShadowMaps.size(); ++i)
		{
			VkImageSubresourceRange range{ m_shadowMaps.getImageListSubresourceRange(m_indicesForShadowMaps[i].shadowMapIndices.listIndex) };
			range.baseArrayLayer = m_indicesForShadowMaps[i].shadowMapIndices.layerIndex;
			range.layerCount = 1;
			barriers.push_back(BarrierOperations::constructImageBarrier(
				srcStages, dstStages,
				0, 0,
				srcLayout, dstLayout,
				m_shadowMaps.getImageHandle(m_indicesForShadowMaps[i].shadowMapIndices.listIndex), range));
		}
		for (int i{ 0 }; i < m_indicesForShadowCubeMaps.size(); ++i)
		{
			barriers.push_back(BarrierOperations::constructImageBarrier(
				srcStages, dstStages, 
				0, 0,
				srcLayout, dstLayout, 
				m_shadowCubeMaps[m_indicesForShadowCubeMaps[i].shadowMapIndices.listIndex].getImageHandle(), m_shadowCubeMaps[m_indicesForShadowCubeMaps[i].shadowMapIndices.listIndex].getSubresourceRange()));
		}

		BarrierOperations::cmdExecuteBarrier(cb, barriers);
		barriers.clear();
	}

	void cmdRenderShadowOnedirMaps(VkCommandBuffer cb)
	{
		for (int i{ 0 }; i < m_indicesForShadowMaps.size();)
		{
			if (m_drawCommandIndices[m_indicesForShadowMaps[i].drawsIndex].size() == 0)
			{
				++i;
				continue;
			}

			uint32_t list{ m_indicesForShadowMaps[i].shadowMapIndices.listIndex };

			VkRenderingAttachmentInfo attachment{};
			attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			attachment.imageView = m_shadowMaps.getImageViewHandle(list);
			attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			attachment.clearValue = { .depthStencil = {.depth = m_frustumData.far, .stencil = 0} }; //It clears the entire list, if want to cache shadow maps, need to clear appropriate layers manually
			attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = { .offset{0,0} };
			m_shadowMaps.getImageListResolution(list, renderInfo.renderArea.extent.width, renderInfo.renderArea.extent.height);
			renderInfo.layerCount = m_shadowMapsLayerCount;
			renderInfo.pDepthAttachment = &attachment;
			renderInfo.colorAttachmentCount = 0;
		 
			int j{ 0 };
			VkViewport viewports[1]{ {.x = 0, .y = 0,
				.width = static_cast<float>(renderInfo.renderArea.extent.width), .height = static_cast<float>(renderInfo.renderArea.extent.height), .minDepth = 0.0, .maxDepth = m_frustumData.far } };
			vkCmdSetViewport(cb, 0, 1, viewports);
			vkCmdBeginRendering(cb, &renderInfo);
			
			struct { int32_t layer; uint32_t drawDataIndex; float proj00; float proj11; float proj22; float proj32; uint32_t viewMatrixIndex; } pcData;
			pcData.proj22 = m_frustumData.proj22;
			pcData.proj32 = m_frustumData.proj32;

			while ((i + j) < m_indicesForShadowMaps.size())
			{
				if (m_indicesForShadowMaps[i + j].shadowMapIndices.listIndex != m_indicesForShadowMaps[i + (j == 0 ? 0 : (j - 1))].shadowMapIndices.listIndex)
					goto next;
				pcData.layer = static_cast<int32_t>(m_indicesForShadowMaps[i + j].shadowMapIndices.layerIndex);
				pcData.viewMatrixIndex = m_indicesForShadowMaps[i + j].viewMatIndex;
				pcData.proj00 = m_indicesForShadowMaps[i + j].proj00;
				pcData.proj11 = -pcData.proj00;
				auto& drawIndices{ m_drawCommandIndices[m_indicesForShadowMaps[i].drawsIndex] };
				IndirectData* drawCommands{ reinterpret_cast<IndirectData*>(m_indirectDataBuffer->getData()) };
				for (int k{ 0 }; k < drawIndices.size(); ++k)
				{
					pcData.drawDataIndex = drawIndices[k];
					vkCmdPushConstants(cb, m_shadowMapPass.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pcData), &pcData);
					auto& dc{ drawCommands[pcData.drawDataIndex].cmd };
					vkCmdDrawIndexed(cb, dc.indexCount, dc.instanceCount, dc.firstIndex, dc.vertexOffset, dc.firstInstance);
				}

				++j;
			}
			next:
			vkCmdEndRendering(cb);
			i += j;
		}
	}
	void cmdRenderShadowCubeMaps(VkCommandBuffer cb)
	{
		for (int i{ 0 }; i < m_indicesForShadowCubeMaps.size(); ++i)
		{
			uint32_t list{ m_indicesForShadowCubeMaps[i].shadowMapIndices.listIndex };

			VkRenderingAttachmentInfo attachment{};
			attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
			attachment.imageView = m_shadowCubeMaps[list].getImageView();
			attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
			attachment.clearValue = { .depthStencil = {.depth = m_frustumData.far, .stencil = 0} };
			attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			VkRenderingInfo renderInfo{};
			renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
			renderInfo.renderArea = { .offset{0,0} };
			renderInfo.renderArea.extent.width = m_shadowCubeMaps[list].getWidth();
			renderInfo.renderArea.extent.height = renderInfo.renderArea.extent.width;
			renderInfo.layerCount = 6;
			renderInfo.pDepthAttachment = &attachment;
			renderInfo.colorAttachmentCount = 0;

			int j{ 0 };
			VkViewport viewports[1]{ {.x = 0, .y = 0,
				.width = static_cast<float>(renderInfo.renderArea.extent.width), .height = static_cast<float>(renderInfo.renderArea.extent.height), .minDepth = 0.0, .maxDepth = m_frustumData.far } };
			vkCmdSetViewport(cb, 0, 1, viewports);
			vkCmdBeginRendering(cb, &renderInfo);

			struct { int32_t layer; uint32_t drawDataIndex; float proj00; float proj11; float proj22; float proj32; uint32_t viewMatrixIndex; } pcData;
			pcData.proj00 = m_frustumData.cubeProj00;
			pcData.proj11 = -m_frustumData.cubeProj00;
			pcData.proj22 = m_frustumData.proj22;
			pcData.proj32 = m_frustumData.proj32;

			auto& drawIndices{ m_drawCommandIndices[m_indicesForShadowCubeMaps[i].drawsIndex] };
			
			for (int j{ 0 }; j < 6; ++j)
			{
				pcData.layer = j;
				pcData.viewMatrixIndex = m_indicesForShadowCubeMaps[i].viewMatIndex + j;
				IndirectData* drawCommands{ reinterpret_cast<IndirectData*>(m_indirectDataBuffer->getData()) };
				for (int k{ 0 }; k < drawIndices.size(); ++k)
				{
					pcData.drawDataIndex = drawIndices[k];
					vkCmdPushConstants(cb, m_shadowMapPass.getPipelineLayoutHandle(), VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pcData), &pcData);
					auto& dc{ drawCommands[pcData.drawDataIndex].cmd };
					vkCmdDrawIndexed(cb, dc.indexCount, dc.instanceCount, dc.firstIndex, dc.vertexOffset, dc.firstInstance);
				}
			}

			vkCmdEndRendering(cb);
			i += j;
		}
	}

	friend class LightTypes::LightBase;
	friend class LightTypes::PointLight;
	friend class LightTypes::SpotLight;
};

#endif