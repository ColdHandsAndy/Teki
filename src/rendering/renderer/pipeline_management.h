#ifndef PIPELINE_MANAGEMENT_HEADER
#define PIPELINE_MANAGEMENT_HEADER

#include <string>
#include <filesystem>
#include <functional>
#include <span>

#include <vulkan/vulkan.h>

#include "src/rendering/renderer/descriptor_management.h"

struct ShaderStage;
class Pipeline;
class ResourceSet;
namespace fs = std::filesystem;

class PipelineAssembler
{
public:
	PipelineAssembler(VkDevice device);
	~PipelineAssembler();

	enum AssemblerPreset
	{
		MAX_ASSEMBLER_PRESETS
	};

	enum StatePresets
	{
		DYNAMIC_STATE_DEFAULT,
		DYNAMIC_STATE_VIEWPORT,
		VIEWPORT_STATE_DEFAULT,
		VIEWPORT_STATE_DYNAMIC,
		INPUT_ASSEMBLY_STATE_DEFAULT,
		INPUT_ASSEMBLY_STATE_POINT,
		INPUT_ASSEMBLY_STATE_TRIANGLE_FAN_DRAWING,
		INPUT_ASSEMBLY_STATE_TRIANGLE_STRIP_DRAWING,
		INPUT_ASSEMBLY_STATE_LINE_DRAWING,
		INPUT_ASSEMBLY_STATE_LINE_STRIP_DRAWING,
		TESSELATION_STATE_DEFAULT,
		MULTISAMPLING_STATE_DISABLED,
		MULTISAMPLING_STATE_ENABLED,
		RASTERIZATION_STATE_DEFAULT,
		RASTERIZATION_STATE_LINE_POLYGONS,
		RASTERIZATION_STATE_POINT_VERTICES,
		RASTERIZATION_STATE_SHADOW_MAP,
		RASTERIZATION_STATE_DISABLED,
		DEPTH_STENCIL_STATE_DEFAULT,
		DEPTH_STENCIL_STATE_TEST_ONLY,
		DEPTH_STENCIL_STATE_WRITE_ONLY,
		DEPTH_STENCIL_STATE_SKYBOX,
		DEPTH_STENCIL_STATE_DISABLED,
		COLOR_BLEND_STATE_DISABLED,
		COLOR_BLEND_STATE_DEFAULT,
		PIPELINE_RENDERING_STATE_DEFAULT,
		PIPELINE_RENDERING_STATE_DEPTH_ATTACHMENT_ONLY,
		PIPELINE_RENDERING_STATE_NO_ATTACHMENT,
		MAX_STATE_PRESETS
	};

	VkDevice getDevice() const { return m_device; };

	void setDynamicState(StatePresets preset);
	void setViewportState(StatePresets preset, uint32_t viewportWidth = 1, uint32_t viewportHeight = 1, uint32_t viewportCount = 1);
	void setInputAssemblyState(StatePresets preset);
	void setTesselationState(StatePresets preset);
	void setMultisamplingState(StatePresets preset, VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT);
	void setRasterizationState(StatePresets preset, float lineWidth = 1.0f, VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT);
	void setDepthStencilState(StatePresets preset, VkCompareOp cmpOp = VK_COMPARE_OP_GREATER_OR_EQUAL);
	void setColorBlendState(StatePresets preset, uint32_t attachmentCount = 1);
	void setPipelineRenderingState(StatePresets preset, VkFormat* colorAttachmentFormats, uint32_t attachmentsCount);
	void setPipelineRenderingState(StatePresets preset, VkFormat colorAttachmentFormat = VK_FORMAT_B8G8R8A8_UNORM);

	const VkPipelineDynamicStateCreateInfo& getDynamicState() const;
	const VkPipelineViewportStateCreateInfo& getViewportState() const;
	const VkPipelineInputAssemblyStateCreateInfo& getInputAssemblyState() const;
	const VkPipelineTessellationStateCreateInfo& getTesselationState() const;
	const VkPipelineMultisampleStateCreateInfo& getMultisamplingState() const;
	const VkPipelineRasterizationStateCreateInfo& getRasterizationState() const;
	const VkPipelineDepthStencilStateCreateInfo& getDepthStencilState() const;
	const VkPipelineColorBlendStateCreateInfo& getColorBlendState() const;
	const VkPipelineRenderingCreateInfo& getPipelineRenderingState() const;

private:
	VkDevice m_device{};

	VkPipelineDynamicStateCreateInfo			 m_dynamicState{};
	VkDynamicState*								 m_dynamicStateValues{ nullptr };
	VkPipelineViewportStateCreateInfo			 m_viewportState{};
	VkViewport									 m_viewport{};
	VkRect2D									 m_rect{};
	VkPipelineInputAssemblyStateCreateInfo		 m_inputAssemblyState{};
	VkPipelineTessellationStateCreateInfo	     m_tesselationState{};
	VkPipelineMultisampleStateCreateInfo		 m_multisamplingState{};
	VkPipelineRasterizationStateCreateInfo		 m_rasterizationState{};
	VkPipelineDepthStencilStateCreateInfo		 m_depthStencilState{};
	VkPipelineColorBlendStateCreateInfo			 m_colorBlendingState{};
	VkPipelineColorBlendAttachmentState			 m_colorBlendAttachments[8]{};
	VkFormat									 m_colorAttachmentFormats[8]{};
	VkPipelineRenderingCreateInfo				 m_pipelineRenderingState{};
};



class Pipeline
{
private:
	VkDevice m_device{};

	VkPipeline m_pipelineHandle{};
	VkPipelineLayout m_pipelineLayoutHandle{};
	VkPipelineBindPoint m_bindPoint{};

	std::vector<std::reference_wrapper<const ResourceSet>> m_resourceSets{};
	std::vector<uint32_t> m_setsInUse{};

	bool m_invalid{ true };

public:
	Pipeline();
	Pipeline(const PipelineAssembler& assembler, 
		std::span<const ShaderStage> shaders,
		std::span<std::reference_wrapper<const ResourceSet>> resourceSets,
		std::span<const VkVertexInputBindingDescription> bindings = {}, 
		std::span<const VkVertexInputAttributeDescription> attributes = {},
		std::span<const VkPushConstantRange> pushConstantsRanges = {});
	Pipeline(VkDevice device, fs::path computeShaderFilepath, std::span<std::reference_wrapper<const ResourceSet>> resourceSets, std::span<const VkPushConstantRange> pushConstantsRanges = {});
	Pipeline(Pipeline&& srcPipeline) noexcept;
	~Pipeline();

	VkPipeline getPipelineHandle() const;
	VkPipelineLayout getPipelineLayoutHandle() const;
	const std::vector<std::reference_wrapper<const ResourceSet>>& getResourceSets() const;
	const std::vector<uint32_t>& getResourceSetsInUse() const;
	void setResourceInUse(uint32_t resSetIndex, uint32_t value);
	void rewriteDescriptor(uint32_t setIndex, uint32_t bindingIndex, uint32_t copyIndex, uint32_t arrayIndex, const VkDescriptorDataEXT& descriptorData)
	{
		EASSERT(setIndex < m_resourceSets.size(), "App", "Incorrect set index.");

		m_resourceSets[setIndex].get().rewriteDescriptor(bindingIndex, copyIndex, arrayIndex, descriptorData);
	}

	void cmdBind(VkCommandBuffer cb);
	void cmdBindResourceSets(VkCommandBuffer cb);

	void initializeGraphics(const PipelineAssembler& assembler,
		std::span<const ShaderStage> shaders,
		std::span<std::reference_wrapper<const ResourceSet>> resourceSets,
		std::span<const VkVertexInputBindingDescription> bindings = {},
		std::span<const VkVertexInputAttributeDescription> attributes = {},
		std::span<const VkPushConstantRange> pushConstantsRanges = {});

	void initializaCompute(VkDevice device, const fs::path& computeShaderFilepath, std::span<std::reference_wrapper<const ResourceSet>> resourceSets, std::span<const VkPushConstantRange> pushConstantsRanges = {});

private:
	Pipeline(Pipeline&) = delete;
	void operator=(Pipeline&) = delete;
};

struct ShaderStage
{
	VkShaderStageFlagBits stage{};
	std::string filepath{};
};

#endif