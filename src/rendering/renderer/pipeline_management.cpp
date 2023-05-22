#include "pipeline_management.h"

#include "src/rendering/shader_management/shader_operations.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"



PipelineAssembler::PipelineAssembler(VkDevice device)
{
	m_device = device;
}
PipelineAssembler::~PipelineAssembler()
{
}


void PipelineAssembler::setDynamicState(StatePresets preset)
{
	switch (preset)
	{
	case DYNAMIC_STATE_DEFAULT:
		m_dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Invalid state preset.")
			break;
	}
}
void PipelineAssembler::setViewportState(StatePresets preset, uint32_t viewportWidth, uint32_t viewportHeight)
{
	switch (preset)
	{
	case VIEWPORT_STATE_DEFAULT:
		m_viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		m_viewportState.viewportCount = 1;
		m_viewport = { .x = 0, .y = 0, .width = static_cast<float>(viewportWidth), .height = static_cast<float>(viewportHeight), .minDepth = 0.0f, .maxDepth = 1.0f };
		m_viewportState.pViewports = &m_viewport;
		m_viewportState.scissorCount = 1;
		m_rect = { .offset{0, 0}, .extent{.width = viewportWidth, .height = viewportHeight} };
		m_viewportState.pScissors = &m_rect;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Invalid state preset.")
			break;
	}
}
void PipelineAssembler::setInputAssemblyState(StatePresets preset)
{
	switch (preset)
	{
	case INPUT_ASSEMBLY_STATE_DEFAULT:
		m_inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		m_inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		m_inputAssemblyState.primitiveRestartEnable = VK_FALSE;
		break;
	case INPUT_ASSEMBLY_STATE_LINE_DRAWING:
		m_inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		m_inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		m_inputAssemblyState.primitiveRestartEnable = VK_FALSE;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Invalid state preset.")
			break;
	}
}
void PipelineAssembler::setTesselationState(StatePresets preset)
{
	switch (preset)
	{
	case TESSELATION_STATE_DEFAULT:
		m_tesselationState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Invalid state preset.")
			break;
	}
}
void  PipelineAssembler::setMultisamplingState(StatePresets preset)
{
	switch (preset)
	{
	case MULTISAMPLING_STATE_DISABLED:
		m_multisamplingState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		m_multisamplingState.sampleShadingEnable = VK_FALSE;
		m_multisamplingState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Invalid state preset.")
			break;
	}
}
void PipelineAssembler::setRasterizationState(StatePresets preset, float lineWidth, VkCullModeFlags cullMode)
{
	switch (preset)
	{
	case RASTERIZATION_STATE_DEFAULT:
		m_rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		m_rasterizationState.depthClampEnable = VK_FALSE;
		m_rasterizationState.rasterizerDiscardEnable = VK_FALSE;
		m_rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
		m_rasterizationState.lineWidth = lineWidth;
		m_rasterizationState.cullMode = cullMode;
		m_rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		m_rasterizationState.depthBiasEnable = VK_FALSE;
		break;
	case RASTERIZATION_STATE_LINE_POLYGONS:
		m_rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		m_rasterizationState.depthClampEnable = VK_FALSE;
		m_rasterizationState.rasterizerDiscardEnable = VK_FALSE;
		m_rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
		m_rasterizationState.lineWidth = lineWidth;
		m_rasterizationState.cullMode = cullMode;
		m_rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		m_rasterizationState.depthBiasEnable = VK_FALSE;
		break;
	case RASTERIZATION_STATE_POINT_VERTICES:
		m_rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		m_rasterizationState.depthClampEnable = VK_FALSE;
		m_rasterizationState.rasterizerDiscardEnable = VK_FALSE;
		m_rasterizationState.polygonMode = VK_POLYGON_MODE_POINT;
		m_rasterizationState.lineWidth = lineWidth;
		m_rasterizationState.cullMode = cullMode;
		m_rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		m_rasterizationState.depthBiasEnable = VK_FALSE;
		break;
	case RASTERIZATION_STATE_DISABLED:
		m_rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		m_rasterizationState.depthClampEnable = VK_FALSE;
		m_rasterizationState.rasterizerDiscardEnable = VK_TRUE;
		m_rasterizationState.polygonMode = VK_POLYGON_MODE_POINT;
		m_rasterizationState.lineWidth = lineWidth;
		m_rasterizationState.cullMode = cullMode;
		m_rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		m_rasterizationState.depthBiasEnable = VK_FALSE;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Invalid state preset.")
			break;
	}
}
void PipelineAssembler::setDepthStencilState(StatePresets preset)
{
	switch (preset)
	{
	case DEPTH_STENCIL_STATE_DEFAULT:
		m_depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		m_depthStencilState.depthTestEnable = VK_TRUE;
		m_depthStencilState.depthWriteEnable = VK_TRUE;
		m_depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
		m_depthStencilState.depthBoundsTestEnable = VK_FALSE;
		m_depthStencilState.stencilTestEnable = VK_FALSE;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Invalid state preset.")
			break;
	}
}
void PipelineAssembler::setColorBlendState(StatePresets preset)
{
	switch (preset)
	{
	case COLOR_BLEND_STATE_DEFAULT:
		m_colorBlendingState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		m_colorBlendingState.logicOpEnable = VK_FALSE;
		m_colorBlendingState.logicOp = VK_LOGIC_OP_COPY;
		m_colorBlendingState.attachmentCount = 1;
		m_colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		m_colorBlendAttachment.blendEnable = VK_FALSE;
		m_colorBlendingState.pAttachments = &m_colorBlendAttachment;
		m_colorBlendingState.blendConstants[0] = 0.0f;
		m_colorBlendingState.blendConstants[1] = 0.0f;
		m_colorBlendingState.blendConstants[2] = 0.0f;
		m_colorBlendingState.blendConstants[3] = 0.0f;
		break;
	default:
		ASSERT_ALWAYS(false, "App", "Invalid state preset.")
			break;
	}
}

const VkPipelineDynamicStateCreateInfo& PipelineAssembler::getDynamicState() const
{
	return m_dynamicState;
}

const VkPipelineViewportStateCreateInfo& PipelineAssembler::getViewportState() const
{
	return m_viewportState;
}

const VkPipelineInputAssemblyStateCreateInfo& PipelineAssembler::getInputAssemblyState() const
{
	return m_inputAssemblyState;
}

const VkPipelineTessellationStateCreateInfo& PipelineAssembler::getTesselationState() const
{
	return m_tesselationState;
}

const VkPipelineMultisampleStateCreateInfo& PipelineAssembler::getMultisamplingState() const
{
	return m_multisamplingState;
}

const VkPipelineRasterizationStateCreateInfo& PipelineAssembler::getRasterizationState() const
{
	return m_rasterizationState;
}

const VkPipelineDepthStencilStateCreateInfo& PipelineAssembler::getDepthStencilState() const
{
	return m_depthStencilState;
}

const VkPipelineColorBlendStateCreateInfo& PipelineAssembler::getColorBlendState() const
{
	return m_colorBlendingState;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Pipeline::Pipeline(const PipelineAssembler& assembler, std::span<const ShaderStage> shaders, std::vector<ResourceSet>& resourceSets, std::span<const VkVertexInputBindingDescription> bindings, std::span<const VkVertexInputAttributeDescription> attributes)
{
	m_device = assembler.getDevice();
	m_bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

	VkGraphicsPipelineCreateInfo pipelineCI{};
	pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCI.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
	pipelineCI.pDynamicState = &assembler.getDynamicState();
	pipelineCI.pViewportState = &assembler.getViewportState();
	pipelineCI.stageCount = shaders.size();
	pipelineCI.pInputAssemblyState = &assembler.getInputAssemblyState();
	pipelineCI.pTessellationState = &assembler.getTesselationState();
	pipelineCI.pMultisampleState = &assembler.getMultisamplingState();
	pipelineCI.pRasterizationState = &assembler.getRasterizationState();
	pipelineCI.pDepthStencilState = &assembler.getDepthStencilState();
	pipelineCI.pColorBlendState = &assembler.getColorBlendState();
	VkPipelineVertexInputStateCreateInfo vertInputState{};
	vertInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertInputState.vertexBindingDescriptionCount = bindings.size();
	vertInputState.pVertexBindingDescriptions = bindings.data();
	vertInputState.vertexAttributeDescriptionCount = attributes.size();
	vertInputState.pVertexAttributeDescriptions = attributes.data();
	pipelineCI.pVertexInputState = &vertInputState;

	VkPipelineRenderingCreateInfo attachmentsFormats{};
	attachmentsFormats.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	attachmentsFormats.colorAttachmentCount = 1;
	VkFormat format{ VK_FORMAT_B8G8R8A8_SRGB };
	attachmentsFormats.pColorAttachmentFormats = &format;
	attachmentsFormats.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

	pipelineCI.pNext = &attachmentsFormats;

	VkPipelineShaderStageCreateInfo* shaderStages{ new VkPipelineShaderStageCreateInfo[shaders.size()] };
	for (uint32_t i{ 0 }; i < shaders.size(); ++i)
	{
		shaderStages[i] = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = shaders[i].stage,
			.module = ShaderOperations::createModule(m_device, ShaderOperations::getShaderCode(shaders[i].filepath)),
			.pName = "main"
		};
	}
	pipelineCI.pStages = shaderStages;

	

	VkPipelineLayoutCreateInfo pipelineLayoutCI{};
	pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	std::vector<VkDescriptorSetLayout> setLayouts(resourceSets.size());
	for (uint32_t i{ 0 }; i < resourceSets.size(); ++i)
	{
		setLayouts[i] = resourceSets[i].getSetLayout();
	}
	pipelineLayoutCI.setLayoutCount = setLayouts.size();
	pipelineLayoutCI.pSetLayouts = setLayouts.data();
	ASSERT_ALWAYS(vkCreatePipelineLayout(m_device, &pipelineLayoutCI, nullptr, &m_pipelineLayoutHandle) == VK_SUCCESS, "Vulkan", "Pipeline layout creation failed.");
	pipelineCI.layout = m_pipelineLayoutHandle;
	m_resourceSets = std::move(resourceSets);
	m_setsInUse = std::vector<uint32_t>(m_resourceSets.size(), 0u);

	ASSERT_ALWAYS(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_pipelineHandle) == VK_SUCCESS, "Vulkan", "Pipeline creation failed.");
	


	for (uint32_t i{ 0 }; i < shaders.size(); ++i)
	{
		vkDestroyShaderModule(m_device, (shaderStages + i)->module, nullptr);
	}
	delete[] shaderStages;
}

Pipeline::Pipeline(VkDevice device, fs::path computeShaderFilepath, std::vector<ResourceSet>& resourceSets)
{
	m_device = device;
	m_bindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;

	VkPipelineLayoutCreateInfo pipelineLayoutCI{};
	pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	std::vector<VkDescriptorSetLayout> setLayouts(resourceSets.size());
	for (uint32_t i{ 0 }; i < resourceSets.size(); ++i)
	{
		setLayouts[i] = resourceSets[i].getSetLayout();
	}
	pipelineLayoutCI.setLayoutCount = setLayouts.size();
	pipelineLayoutCI.pSetLayouts = setLayouts.data();
	ASSERT_ALWAYS(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &m_pipelineLayoutHandle) == VK_SUCCESS, "Vulkan", "Pipeline layout creation failed.");
	m_resourceSets = std::move(resourceSets);
	m_setsInUse = std::vector<uint32_t>(m_resourceSets.size(), 0u);
	
	VkPipelineShaderStageCreateInfo shaderStage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	shaderStage.module = ShaderOperations::createModule(device, ShaderOperations::getShaderCode(computeShaderFilepath));
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.pName = "main";

	VkComputePipelineCreateInfo compPipelineCI{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	compPipelineCI.layout = m_pipelineLayoutHandle;
	compPipelineCI.stage = shaderStage;
	compPipelineCI.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
	
	ASSERT_ALWAYS(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compPipelineCI, nullptr, &m_pipelineHandle) == VK_SUCCESS, "Vulkan", "Compute pipeline creation failed.");

	vkDestroyShaderModule(device, shaderStage.module, nullptr);
}

Pipeline::Pipeline(Pipeline&& srcPipeline) noexcept
{
	if (this == &srcPipeline)
		return;

	m_device = srcPipeline.m_device;

	m_pipelineHandle = srcPipeline.m_pipelineHandle;
	m_pipelineLayoutHandle = srcPipeline.m_pipelineLayoutHandle;
	m_bindPoint = srcPipeline.m_bindPoint;

	m_resourceSets = std::move(srcPipeline.m_resourceSets);
	m_setsInUse = std::move(srcPipeline.m_setsInUse);

	srcPipeline.m_invalid = true;
}

Pipeline::~Pipeline()
{
	if (!m_invalid)
	{
		vkDestroyPipeline(m_device, m_pipelineHandle, nullptr);
		vkDestroyPipelineLayout(m_device, m_pipelineLayoutHandle, nullptr);
	}
}

VkPipeline Pipeline::getPipelineHandle() const
{
	return m_pipelineHandle;
}

VkPipelineLayout Pipeline::getPipelineLayoutHandle() const
{
	return m_pipelineLayoutHandle;
}

std::vector<ResourceSet>& Pipeline::getResourceSets()
{
	return m_resourceSets;
}

const std::vector<uint32_t>& Pipeline::getResourceSetsInUse()
{
	return m_setsInUse;
}

void Pipeline::setResourceInUse(uint32_t resSetIndex, uint32_t value)
{
	m_setsInUse[resSetIndex] = value;
}

void Pipeline::cmdBind(VkCommandBuffer cb)
{
	vkCmdBindPipeline(cb, m_bindPoint, m_pipelineHandle);
}