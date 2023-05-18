#ifndef PIPELINE_MANAGEMENT_HEADER
#define PIPELINE_MANAGEMENT_HEADER

#include <string>

#include "vulkan/vulkan.h"

#include "src/rendering/renderer/descriptor_management.h"

#define WINDOW_WIDTH_DEFAULT  1280
#define WINDOW_HEIGHT_DEFAULT 720

struct ShaderStage;

class Pipeline
{
private:
	VkDevice m_device{};

	VkPipeline m_pipelineHandle{};
	VkPipelineLayout m_pipelineLayoutHandle{};

	std::vector<ResourceSet> m_resourceSets{};
	std::vector<uint32_t> m_resourceSetsLayout{};

	bool m_invalid{ false };

public:
	Pipeline(VkDevice device, std::vector<ShaderStage>& shaders, std::vector<ResourceSet>& resourceSets, std::span<const VkVertexInputBindingDescription> bindings, std::span<const VkVertexInputAttributeDescription> attributes);
	Pipeline(Pipeline&& srcPipeline) noexcept;
	~Pipeline();

	VkPipeline getPipelineHandle() const;
	VkPipelineLayout getPipelineLayoutHandle() const;
	std::vector<ResourceSet>& getResourceSets();
	const std::vector<uint32_t>& getCurrentResourceLayout();
	void setResourceIndex(uint32_t resSetIndex, uint32_t value);

	void cmdBind(VkCommandBuffer cb);


private:
	Pipeline(Pipeline&) = delete;
	void operator=(Pipeline&) = delete;
};



class PipelineCompute
{
private:
	VkDevice m_device{};

	VkPipeline m_pipelineHandle{};
	VkPipelineLayout m_pipelineLayoutHandle{};

	std::vector<ResourceSet> m_resourceSets{};
	std::vector<uint32_t> m_resourceSetsLayout{};

	bool m_invalid{ false };

public:
	PipelineCompute(VkDevice device, const VkShaderModule sModule, std::vector<ResourceSet>& resourceSets);
	PipelineCompute(PipelineCompute&& srcPipeline) noexcept;
	~PipelineCompute();

	VkPipeline getPipelineHandle() const;
	VkPipelineLayout getPipelineLayoutHandle() const;
	std::vector<ResourceSet>& getResourceSets();
	const std::vector<uint32_t>& getCurrentResourceLayout();
	void setResourceIndex(uint32_t resSetIndex, uint32_t value);

	void cmdBind(VkCommandBuffer cb);


private:
	PipelineCompute(PipelineCompute&) = delete;
	void operator=(PipelineCompute&) = delete;

};

struct ShaderStage
{
	VkShaderModule module{};
	VkShaderStageFlagBits stage{};
	std::string entryPointName{};
};

#endif