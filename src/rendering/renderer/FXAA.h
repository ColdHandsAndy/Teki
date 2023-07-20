#ifndef FXAA_CLASS_HEADER
#define FXAA_CLASS_HEADER

#include <cstdint>

#include <vulkan/vulkan.h>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/data_management/image_classes.h"

class FXAA
{
private:
	VkDevice m_device{};

	uint32_t m_windowWidth{};
	uint32_t m_windowHeight{};
	float m_invWindowWidth{};
	float m_invWindowHeight{};

	const Image* m_inputImage{};

	Pipeline m_FXAApass{};

	VkSampler m_sampler{};

public:
	FXAA(VkDevice device, uint32_t windowWidth, uint32_t windowHeight, const Image& framebuffer) : m_device{ device }, m_windowWidth{ windowWidth }, m_windowHeight{ windowHeight }, m_inputImage{ &framebuffer }
	{
		m_invWindowWidth = 1.0 / windowWidth;
		m_invWindowHeight = 1.0 / windowHeight;

		VkSamplerCreateInfo samplerCI{
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		.mipLodBias = 0.0f,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0,
		.compareEnable = VK_FALSE,
		.compareOp = VK_COMPARE_OP_ALWAYS,
		.minLod = 0.0f,
		.maxLod = 128.0f,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE };
		vkCreateSampler(device, &samplerCI, nullptr, &m_sampler);

		VkDescriptorSetLayoutBinding inputImageBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
		VkDescriptorImageInfo inputImageInfo{ .sampler = m_sampler, .imageView = m_inputImage->getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

		std::vector<ResourceSet> resourceSets{};
		resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
			{{inputImageBinding}},  {},
				{ {{.pCombinedImageSampler = &inputImageInfo}} } });

		PipelineAssembler assembler{ device };
		assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
		assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, windowWidth, windowHeight);
		assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
		assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
		assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
		assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0f, VK_CULL_MODE_NONE);
		assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
		assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DISABLED);
		assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);
		m_FXAApass.initializeGraphics(assembler,
			{ {ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/fullscreen_vert.spv"},
			ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/fxaa_frag.spv"}} },
			resourceSets, {}, {},
			{ { VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(float) * 2}} });
	}
	~FXAA()
	{
		vkDestroySampler(m_device, m_sampler, nullptr);
	}


	void cmdPassFXAA(VkCommandBuffer cb, DescriptorManager& descriptorManager, VkImageView outputAttachment)
	{
		VkRenderingAttachmentInfo attachmentInfo{};
		attachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		attachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		attachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		attachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachmentInfo.clearValue = VkClearValue{ .color{.float32{0.0f}} };
		attachmentInfo.imageView = outputAttachment;

		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = { .offset{0,0}, .extent{.width = m_windowWidth, .height = m_windowHeight} };
		renderInfo.layerCount = 1;
		renderInfo.colorAttachmentCount = 1;
		renderInfo.pColorAttachments = &attachmentInfo;

		vkCmdBeginRendering(cb, &renderInfo);
			
			descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
				m_FXAApass.getResourceSets(), m_FXAApass.getResourceSetsInUse(), m_FXAApass.getPipelineLayoutHandle());
			m_FXAApass.cmdBind(cb);
			float pcData[2]{ m_invWindowWidth, m_invWindowHeight };
			vkCmdPushConstants(cb, m_FXAApass.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float) * 2, pcData);
			vkCmdDraw(cb, 3, 1, 0, 0);

		vkCmdEndRendering(cb);
	}

	void changeFramebuffer(const Image& framebuffer)
	{
		m_inputImage = &framebuffer;
	}
};

#endif