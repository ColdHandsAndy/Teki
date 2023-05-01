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

	bool m_invalid{ false };

public:
	Pipeline(VkDevice device, std::vector<ShaderStage>& shaders, std::vector<ResourceSet>& resourceSets, std::span<const VkVertexInputBindingDescription> bindings, std::span<const VkVertexInputAttributeDescription> attributes);
	Pipeline(Pipeline&& srcPipeline) noexcept;
	~Pipeline();

	VkPipeline getPipelineHandle() const;
	VkPipelineLayout getPipelineLayoutHandle() const;
	std::vector<ResourceSet>& getResourceSets();

	void cmdBind(VkCommandBuffer cb);


private:
	Pipeline(Pipeline&) = delete;
	void operator=(Pipeline&) = delete;
};

struct ShaderStage
{
	VkShaderModule module{};
	VkShaderStageFlagBits stage{};
	std::string entryPointName{};
};

#endif