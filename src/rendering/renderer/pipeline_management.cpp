#include "pipeline_management.h"

#include "src/rendering/data_abstraction/vertex_layouts.h"

Pipeline::Pipeline(VkDevice device, std::vector<ShaderStage>& shaders, std::vector<ResourceSet>& resourceSets, std::span<const VkVertexInputBindingDescription> bindings, std::span<const VkVertexInputAttributeDescription> attributes)
{
	m_device = device;

	VkPipelineLayoutCreateInfo					 pipelineLayoutCI{};
	pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	std::vector<VkDescriptorSetLayout> setLayouts(resourceSets.size());
	for (uint32_t i{0}; i < resourceSets.size(); ++i)
	{
		setLayouts[i] = resourceSets[i].getSetLayout();
	}
	pipelineLayoutCI.setLayoutCount = setLayouts.size();
	pipelineLayoutCI.pSetLayouts = setLayouts.data();
	ASSERT_ALWAYS(vkCreatePipelineLayout(device, &pipelineLayoutCI, nullptr, &m_pipelineLayoutHandle) == VK_SUCCESS, "Vulkan", "Pipeline layout creation failed.");
	m_resourceSets = std::move(resourceSets);
	m_resourceSetsLayout = std::vector<uint32_t>(m_resourceSets.size(), 0u);

	VkPipelineDynamicStateCreateInfo			 dynamicState{};
	dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;

	VkPipelineViewportStateCreateInfo			 viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewportState.viewportCount = 1;
	VkViewport viewport{ .x = 0, .y = WINDOW_HEIGHT_DEFAULT, .width = WINDOW_WIDTH_DEFAULT, .height = -WINDOW_HEIGHT_DEFAULT, .minDepth = 0.0f, .maxDepth = 1.0f };
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	VkRect2D rect{ .offset{0, 0}, .extent{.width = WINDOW_WIDTH_DEFAULT, .height = WINDOW_HEIGHT_DEFAULT} };
	viewportState.pScissors = &rect;

	VkPipelineShaderStageCreateInfo*			 shaderStages{ new VkPipelineShaderStageCreateInfo[shaders.size()] };
	for (uint32_t i{0}; i < shaders.size(); ++i)
	{
		shaderStages[i] = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 
			.stage = shaders[i].stage, 
			.module = shaders[i].module, 
			.pName = shaders[i].entryPointName.c_str() 
		};
	}

	VkPipelineInputAssemblyStateCreateInfo		 inputAssemblyState{};
	inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	inputAssemblyState.primitiveRestartEnable = VK_FALSE;

	VkPipelineVertexInputStateCreateInfo		 vertexInputState{};
	vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertexInputState.vertexBindingDescriptionCount = bindings.size();
	vertexInputState.pVertexBindingDescriptions = bindings.data();
	vertexInputState.vertexAttributeDescriptionCount = attributes.size();
	vertexInputState.pVertexAttributeDescriptions = attributes.data();

	VkPipelineTessellationStateCreateInfo	     tesselationState{};
	tesselationState.sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;

	VkPipelineMultisampleStateCreateInfo		 multisamplingState{};
	multisamplingState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisamplingState.sampleShadingEnable = VK_FALSE;
	multisamplingState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineRasterizationStateCreateInfo		 rasterizationState{};
	rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterizationState.depthClampEnable = VK_FALSE;
	rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizationState.lineWidth = 1.0f;
	rasterizationState.cullMode = VK_CULL_MODE_NONE;
	rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizationState.depthBiasEnable = VK_FALSE;

	VkPipelineDepthStencilStateCreateInfo		 depthStencilState{};
	depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depthStencilState.depthTestEnable = VK_TRUE;
	depthStencilState.depthWriteEnable = VK_TRUE;
	depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
	depthStencilState.depthBoundsTestEnable = VK_FALSE;
	depthStencilState.stencilTestEnable = VK_FALSE;

	VkPipelineColorBlendStateCreateInfo			 colorBlendingState{};
	colorBlendingState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	colorBlendingState.logicOpEnable = VK_FALSE;
	colorBlendingState.logicOp = VK_LOGIC_OP_COPY;
	colorBlendingState.attachmentCount = 1;
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;
	colorBlendingState.pAttachments = &colorBlendAttachment;
	colorBlendingState.blendConstants[0] = 0.0f;
	colorBlendingState.blendConstants[1] = 0.0f;
	colorBlendingState.blendConstants[2] = 0.0f;
	colorBlendingState.blendConstants[3] = 0.0f;

	VkGraphicsPipelineCreateInfo pipelineCI{};
	pipelineCI.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipelineCI.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
	pipelineCI.layout = m_pipelineLayoutHandle;
	pipelineCI.pDynamicState = nullptr;
	pipelineCI.pViewportState = &viewportState;
	pipelineCI.stageCount = shaders.size();
	pipelineCI.pStages = shaderStages;
	pipelineCI.pInputAssemblyState = &inputAssemblyState;
	pipelineCI.pVertexInputState = &vertexInputState;
	pipelineCI.pTessellationState = nullptr;
	pipelineCI.pMultisampleState = &multisamplingState;
	pipelineCI.pRasterizationState = &rasterizationState;
	pipelineCI.pDepthStencilState = &depthStencilState;
	pipelineCI.pColorBlendState = &colorBlendingState;

	VkPipelineRenderingCreateInfo attachmentsFormats{};
	attachmentsFormats.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	attachmentsFormats.colorAttachmentCount = 1;
	VkFormat format{ VK_FORMAT_B8G8R8A8_SRGB };
	attachmentsFormats.pColorAttachmentFormats = &format;
	attachmentsFormats.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
	pipelineCI.pNext = &attachmentsFormats;

	ASSERT_ALWAYS(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &m_pipelineHandle) == VK_SUCCESS, "Vulkan", "Pipeline creation failed.");

	for (auto& shaderData : shaders)
	{
		vkDestroyShaderModule(device, shaderData.module, nullptr);
	}
	delete[] shaderStages;
}

Pipeline::Pipeline(Pipeline&& srcPipeline) noexcept
{
	if (this == &srcPipeline)
		return;

	m_device = srcPipeline.m_device;

	m_pipelineHandle = srcPipeline.m_pipelineHandle;
	m_pipelineLayoutHandle = srcPipeline.m_pipelineLayoutHandle;

	m_resourceSets = std::move(srcPipeline.m_resourceSets);
	m_resourceSetsLayout = std::move(srcPipeline.m_resourceSetsLayout);

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

const std::vector<uint32_t>& Pipeline::getCurrentResourceLayout()
{
	return m_resourceSetsLayout;
}

void Pipeline::setResourceIndex(uint32_t resSetIndex, uint32_t value)
{
	m_resourceSetsLayout[resSetIndex] = value;
}

void Pipeline::cmdBind(VkCommandBuffer cb)
{
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineHandle);
}



PipelineCompute::PipelineCompute(VkDevice device, const VkShaderModule sModule, std::vector<ResourceSet>& resourceSets)
{
	m_device = device;

	VkPipelineLayoutCreateInfo					 pipelineLayoutCI{};
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
	m_resourceSetsLayout = std::vector<uint32_t>(m_resourceSets.size(), 0u);
	
	VkPipelineShaderStageCreateInfo shaderStage{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	shaderStage.module = sModule;
	shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStage.pName = "main";

	VkComputePipelineCreateInfo compPipelineCI{ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
	compPipelineCI.layout = m_pipelineLayoutHandle;
	compPipelineCI.stage = shaderStage;
	compPipelineCI.flags = VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;

	ASSERT_ALWAYS(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compPipelineCI, nullptr, &m_pipelineHandle) == VK_SUCCESS, "Vulkan", "Compute pipeline creation failed.");

	vkDestroyShaderModule(device, sModule, nullptr);
}

PipelineCompute::PipelineCompute(PipelineCompute&& srcPipeline) noexcept
{
	if (this == &srcPipeline)
		return;

	m_device = srcPipeline.m_device;

	m_pipelineHandle = srcPipeline.m_pipelineHandle;
	m_pipelineLayoutHandle = srcPipeline.m_pipelineLayoutHandle;

	m_resourceSets = std::move(srcPipeline.m_resourceSets);
	m_resourceSetsLayout = std::move(srcPipeline.m_resourceSetsLayout);

	srcPipeline.m_invalid = true;
}
PipelineCompute::~PipelineCompute()
{
	if (!m_invalid)
	{
		vkDestroyPipeline(m_device, m_pipelineHandle, nullptr);
		vkDestroyPipelineLayout(m_device, m_pipelineLayoutHandle, nullptr);
	}
}

VkPipeline PipelineCompute::getPipelineHandle() const
{
	return m_pipelineHandle;
}
VkPipelineLayout PipelineCompute::getPipelineLayoutHandle() const
{
	return m_pipelineLayoutHandle;
}
std::vector<ResourceSet>& PipelineCompute::getResourceSets()
{
	return m_resourceSets;
}
const std::vector<uint32_t>& PipelineCompute::getCurrentResourceLayout()
{
	return m_resourceSetsLayout;
}
void PipelineCompute::setResourceIndex(uint32_t resSetIndex, uint32_t value)
{
	m_resourceSetsLayout[resSetIndex] = value;
}


void PipelineCompute::cmdBind(VkCommandBuffer cb)
{
	vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineHandle);
}