#include "src/rendering/renderer/deferred_lighting.h"

DeferredLighting::DeferredLighting(VkDevice device, uint32_t width, uint32_t height,
	const DepthBuffer& depthBuffer,
	const HBAO& hbao,
	const ResourceSet& viewprojRS,
	const ResourceSet& transformMatricesRS,
	const ResourceSet& materialsTexturesRS,
	const ResourceSet& shadowMapsRS,
	const ResourceSet& indirectDiffiseLightingRS,
	const ResourceSet& indirectSpecularLightingRS,
	const ResourceSet& indirectLightingMetadataRS,
	const ResourceSet& distantProbeRS,
	const ResourceSet& drawDataRS,
	const ResourceSet& BRDFLUTRS,
	const ResourceSet& directLightingRS,
	VkSampler generalSampler)
	: m_UV{ device, VK_FORMAT_R32_UINT, width, height, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
	m_tangentFrame{ device, VK_FORMAT_A2B10G10R10_UNORM_PACK32, width, height, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
	m_drawID{ device, VK_FORMAT_R16_UINT, width, height, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
	m_outputFramebuffer{ device, VK_FORMAT_R16G16B16A16_SFLOAT, width, height, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
	m_depthBuffer{ depthBuffer }
{
	PipelineAssembler assembler{ device };

	assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
	assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, width, height);
	assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
	assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
	assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
	assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT);
	assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
	assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED, 3);
	VkFormat formats[3]{ VK_FORMAT_R32_UINT, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_R16_UINT };
	assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT, formats, ARRAYSIZE(formats));

	std::array<std::reference_wrapper<const ResourceSet>, 3> resourceSets0{ viewprojRS, transformMatricesRS, drawDataRS };

	m_uvBufferPipeline.initializeGraphics(assembler,
		{ { ShaderStage{.stage = VK_SHADER_STAGE_VERTEX_BIT, .filepath = "shaders/cmpld/uv_buffer_vert.spv"},  ShaderStage{.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .filepath = "shaders/cmpld/uv_buffer_frag.spv"} } },
		resourceSets0,
		{ {StaticVertex::getBindingDescription()} },
		{ StaticVertex::getAttributeDescriptions() });


	VkDescriptorSetLayoutBinding uvBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo uvImageInfo{ .imageView = m_UV.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding tangFrameBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo tangFrameImageInfo{ .imageView = m_tangentFrame.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding drawIDBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo drawIDImageInfo{ .imageView = m_drawID.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding depthBinding{ .binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo depthImageInfo{ .sampler = generalSampler, .imageView = m_depthBuffer.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL };
	VkDescriptorSetLayoutBinding aoBinding{ .binding = 4, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo aoImageInfo{ .sampler = generalSampler, .imageView = hbao.getAO().getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
	VkDescriptorSetLayoutBinding framebufferBinding{ .binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo framebufferImageInfo{ .imageView = m_outputFramebuffer.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
	m_resSet.initializeSet(device, 1, VkDescriptorSetLayoutCreateFlagBits{},
		std::array{ uvBinding, tangFrameBinding, drawIDBinding, depthBinding, aoBinding, framebufferBinding },
		std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &uvImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &tangFrameImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &drawIDImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &depthImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &aoImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &framebufferImageInfo} } },
		true);


	std::array<std::reference_wrapper<const ResourceSet>, 11> resourceSets1{
		viewprojRS, m_resSet, 
		materialsTexturesRS, shadowMapsRS, 
		indirectDiffiseLightingRS, indirectSpecularLightingRS, indirectLightingMetadataRS,
		distantProbeRS,
		drawDataRS, 
		BRDFLUTRS, 
		directLightingRS };
	m_lightingComputePipeline.initializaCompute(device, "shaders/cmpld/lighting_pass_comp.spv", resourceSets1,
		{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(m_pcData) }} });

	m_pcData.invResolution = { 1.0 / width, 1.0 / height };

	m_imageBarriers = {
		SyncOperations::constructImageBarrier(
			VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
			m_outputFramebuffer.getImageHandle(), m_outputFramebuffer.getSubresourceRange()),
		SyncOperations::constructImageBarrier(
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			m_UV.getImageHandle(), m_UV.getSubresourceRange()),
		SyncOperations::constructImageBarrier(
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			m_tangentFrame.getImageHandle(), m_tangentFrame.getSubresourceRange()),
		SyncOperations::constructImageBarrier(
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			m_drawID.getImageHandle(), m_drawID.getSubresourceRange()),
		SyncOperations::constructImageBarrier(
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
			VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
			m_depthBuffer.getImageHandle(), m_depthBuffer.getDepthBufferSubresourceRange()) };

	m_dependencyInfo = SyncOperations::createDependencyInfo(m_imageBarriers);
}

void DeferredLighting::cmdPassDrawToUVBuffer(VkCommandBuffer cb, const Culling& culling, const Buffer& vertexData, const Buffer& indexData)
	{
		SyncOperations::cmdExecuteBarrier(cb, 
			{{
			SyncOperations::constructImageBarrier(
				VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_ACCESS_NONE, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				m_UV.getImageHandle(), m_UV.getSubresourceRange()),
			SyncOperations::constructImageBarrier(
				VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_ACCESS_NONE, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				m_tangentFrame.getImageHandle(), m_tangentFrame.getSubresourceRange()),
			SyncOperations::constructImageBarrier(
				VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_ACCESS_NONE, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				m_drawID.getImageHandle(), m_drawID.getSubresourceRange()),
			SyncOperations::constructImageBarrier(
				VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				VK_ACCESS_NONE, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
				m_depthBuffer.getImageHandle(), m_depthBuffer.getDepthBufferSubresourceRange())
			}});

		VkRenderingAttachmentInfo colorAttachmentInfos[3]{ 
			{
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.imageView = m_UV.getImageView(),
				.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = VkClearValue{.color{.float32{0.0f, 0.0f}} }
			}, 
			{
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.imageView = m_tangentFrame.getImageView(),
				.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = VkClearValue{.color{.float32{0.0f, 0.0f, 0.0f, 0.0f}} }
			}, 
			{
				.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
				.imageView = m_drawID.getImageView(),
				.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.clearValue = VkClearValue{.color{.uint32 = 0} }
			}};
		VkRenderingAttachmentInfo depthAttachmentInfo{
			.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
			.imageView = m_depthBuffer.getImageView(),
			.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.clearValue = {.depthStencil = {.depth = 0.0f, .stencil = 0} }
		};

		uint32_t width{ m_UV.getWidth() };
		uint32_t height{ m_UV.getHeight() };

		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = { .offset{0,0}, .extent{.width = width, .height = height} };
		renderInfo.layerCount = 1;
		renderInfo.colorAttachmentCount = ARRAYSIZE(colorAttachmentInfos);
		renderInfo.pColorAttachments = colorAttachmentInfos;
		renderInfo.pDepthAttachment = &depthAttachmentInfo;

		VkBuffer vertexBindings[1]{ vertexData.getBufferHandle() };
		VkDeviceSize vertexBindingOffsets[1]{ vertexData.getOffset() };

		vkCmdBeginRendering(cb, &renderInfo);
			
			vkCmdBindVertexBuffers(cb, 0, 1, vertexBindings, vertexBindingOffsets);
			vkCmdBindIndexBuffer(cb, indexData.getBufferHandle(), indexData.getOffset(), VK_INDEX_TYPE_UINT32);
			m_uvBufferPipeline.cmdBindResourceSets(cb);
			m_uvBufferPipeline.cmdBind(cb);
			vkCmdDrawIndexedIndirectCount(cb,
				culling.getDrawCommandBufferHandle(), culling.getDrawCommandBufferOffset(),
				culling.getDrawCountBufferHandle(), culling.getDrawCountBufferOffset(),
				culling.getMaxDrawCount(), culling.getDrawCommandBufferStride());

		vkCmdEndRendering(cb);
	}

void DeferredLighting::cmdDispatchLightingCompute(VkCommandBuffer cb, uint32_t indirectCurrentSet)
{
	constexpr uint32_t indirectResourceSetIndex{ 4 };
	m_lightingComputePipeline.setResourceInUse(indirectResourceSetIndex, indirectCurrentSet);
	m_lightingComputePipeline.cmdBindResourceSets(cb);
	m_lightingComputePipeline.cmdBind(cb);
	vkCmdPushConstants(cb, m_lightingComputePipeline.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_pcData), &m_pcData);
	constexpr uint32_t groupSize{ 8 };
	vkCmdDispatch(cb, DISPATCH_SIZE(m_UV.getWidth(), groupSize), DISPATCH_SIZE(m_UV.getHeight(), groupSize), 1);
}