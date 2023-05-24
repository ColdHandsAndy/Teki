#ifndef PIPELINE_MANAGEMENT_HEADER
#define PIPELINE_MANAGEMENT_HEADER

#include <string>
#include <filesystem>

#include "vulkan/vulkan.h"

#include "src/rendering/renderer/descriptor_management.h"

#define WINDOW_WIDTH_DEFAULT  1280
#define WINDOW_HEIGHT_DEFAULT 720

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
		VIEWPORT_STATE_DEFAULT,
		INPUT_ASSEMBLY_STATE_DEFAULT,
		INPUT_ASSEMBLY_STATE_LINE_DRAWING,
		TESSELATION_STATE_DEFAULT,
		MULTISAMPLING_STATE_DISABLED,
		RASTERIZATION_STATE_DEFAULT,
		RASTERIZATION_STATE_LINE_POLYGONS,
		RASTERIZATION_STATE_POINT_VERTICES,
		RASTERIZATION_STATE_DISABLED,
		DEPTH_STENCIL_STATE_DEFAULT,
		DEPTH_STENCIL_STATE_SKYBOX,
		DEPTH_STENCIL_STATE_DISABLED,
		COLOR_BLEND_STATE_DEFAULT,
		MAX_STATE_PRESETS
	};

	VkDevice getDevice() const { return m_device; };

	void setDynamicState(StatePresets preset);
	void setViewportState(StatePresets preset, uint32_t viewportWidth, uint32_t viewportHeight);
	void setInputAssemblyState(StatePresets preset);
	void setTesselationState(StatePresets preset);
	void setMultisamplingState(StatePresets preset);
	void setRasterizationState(StatePresets preset, float lineWidth = 1.0f, VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT);
	void setDepthStencilState(StatePresets preset);
	void setColorBlendState(StatePresets preset);

	const VkPipelineDynamicStateCreateInfo& getDynamicState() const;
	const VkPipelineViewportStateCreateInfo& getViewportState() const;
	const VkPipelineInputAssemblyStateCreateInfo& getInputAssemblyState() const;
	const VkPipelineTessellationStateCreateInfo& getTesselationState() const;
	const VkPipelineMultisampleStateCreateInfo& getMultisamplingState() const;
	const VkPipelineRasterizationStateCreateInfo& getRasterizationState() const;
	const VkPipelineDepthStencilStateCreateInfo& getDepthStencilState() const;
	const VkPipelineColorBlendStateCreateInfo& getColorBlendState() const;

private:
	VkDevice m_device{};

	VkPipelineDynamicStateCreateInfo			 m_dynamicState{};
	VkPipelineViewportStateCreateInfo			 m_viewportState{};
	VkViewport									 m_viewport{};
	VkRect2D									 m_rect{};
	VkPipelineInputAssemblyStateCreateInfo		 m_inputAssemblyState{};
	VkPipelineTessellationStateCreateInfo	     m_tesselationState{};
	VkPipelineMultisampleStateCreateInfo		 m_multisamplingState{};
	VkPipelineRasterizationStateCreateInfo		 m_rasterizationState{};
	VkPipelineDepthStencilStateCreateInfo		 m_depthStencilState{};
	VkPipelineColorBlendStateCreateInfo			 m_colorBlendingState{};
	VkPipelineColorBlendAttachmentState			 m_colorBlendAttachment{};
};



class Pipeline
{
private:
	VkDevice m_device{};

	VkPipeline m_pipelineHandle{};
	VkPipelineLayout m_pipelineLayoutHandle{};
	VkPipelineBindPoint m_bindPoint{};

	std::vector<ResourceSet> m_resourceSets{};
	std::vector<uint32_t> m_setsInUse{};

	bool m_invalid{ false };

public:
	//Pipeline(VkDevice device, std::vector<ShaderStage>& shaders, std::vector<ResourceSet>& resourceSets, std::span<const VkVertexInputBindingDescription> bindings, std::span<const VkVertexInputAttributeDescription> attributes);
	Pipeline(const PipelineAssembler& assembler, std::span<const ShaderStage> shaders, std::vector<ResourceSet>& resourceSets, std::span<const VkVertexInputBindingDescription> bindings = {}, std::span<const VkVertexInputAttributeDescription> attributes = {});
	Pipeline(VkDevice device, fs::path computeShaderFilepath, std::vector<ResourceSet>& resourceSets);
	Pipeline(Pipeline&& srcPipeline) noexcept;
	~Pipeline();

	VkPipeline getPipelineHandle() const;
	VkPipelineLayout getPipelineLayoutHandle() const;
	std::vector<ResourceSet>& getResourceSets();
	const std::vector<uint32_t>& getResourceSetsInUse();
	void setResourceInUse(uint32_t resSetIndex, uint32_t value);

	void cmdBind(VkCommandBuffer cb);


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