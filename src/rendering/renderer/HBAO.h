#ifndef HBAO_CLASS_HEADER
#define HBAO_CLASS_HEADER

#define RANDOM_TEXTURE_SIZE 4

#include <cstdint>
#include <random>

#include <glm/gtc/constants.hpp>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"

class HBAO
{
private:
	VkDevice m_device;

	uint32_t m_aoRenderWidth{};
	uint32_t m_aoRenderHeight{};

	Image m_linearDepthImage;
	Image m_AOImage;
	Image m_blurredAOImage;
	Image m_randTex;
	Image m_depthBuffer;

	Pipeline m_linearExpandedFOVDepthPass;
	Pipeline m_HBAOpass;
	Pipeline m_blurHBAOpass;

	static constexpr float expansionScale{ 1.1 };
	struct
	{
		glm::vec4 uvTransformData;

		glm::vec2 invResolution;
		glm::vec2 resolution;

		float radius;
		float aoExponent;
		float angleBias;
		float negInvR2;
	} m_hbaoInfo;
	float m_blurSharpness{};

	BufferBaseHostAccessible m_viewprojexpDataUB{ m_device, sizeof(glm::mat4) * 2, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT };

	VkSampler m_hbaoSampler{};

public:
	HBAO(VkDevice device, uint32_t aoRenderWidth, uint32_t aoRenderHeight)
		: m_linearDepthImage{ device, VK_FORMAT_R32_SFLOAT, aoRenderWidth, aoRenderHeight, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
			m_AOImage{ device, VK_FORMAT_R16_UNORM, aoRenderWidth, aoRenderHeight, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
				m_blurredAOImage{ device, VK_FORMAT_R16_UNORM, aoRenderWidth, aoRenderHeight, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
					m_randTex{ device, VK_FORMAT_R16G16B16A16_SNORM, RANDOM_TEXTURE_SIZE, RANDOM_TEXTURE_SIZE, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
						m_depthBuffer{ device, VK_FORMAT_D32_SFLOAT, aoRenderWidth, aoRenderHeight, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT },
							m_device{ device }
	{
		m_aoRenderWidth = aoRenderWidth;
		m_aoRenderHeight = aoRenderHeight;
		PipelineAssembler assembler{ device };
		assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
		assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, aoRenderWidth, aoRenderHeight);
		assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
		assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
		assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
		assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT, 1.0f, VK_CULL_MODE_NONE);
		assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
		assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DISABLED);
		assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT, VK_FORMAT_R16_UNORM);

		VkSamplerCreateInfo samplerCI{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_TRUE,
			.maxAnisotropy = 1.0,
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS,
			.minLod = 0.0f,
			.maxLod = 128.0f,
			.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
			.unnormalizedCoordinates = VK_FALSE };
		vkCreateSampler(device, &samplerCI, nullptr, &m_hbaoSampler);

		VkDescriptorSetLayoutBinding depthImageBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
		VkDescriptorImageInfo depthImageInfo{ .sampler = m_hbaoSampler, .imageView = m_linearDepthImage.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

		VkDescriptorSetLayoutBinding randBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
		VkDescriptorImageInfo randImageInfo{ .sampler = m_hbaoSampler, .imageView = m_randTex.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

		std::vector<ResourceSet> resourceSets[2]{};
		resourceSets[0].push_back({device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
			{ depthImageBinding, randBinding },  {},
				{{{ .pCombinedImageSampler = &depthImageInfo }}, {{.pCombinedImageSampler = &randImageInfo }}} });

		m_HBAOpass.initializeGraphics(assembler,
			{ {ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/fullscreen_vert.spv"},
			ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/hbao_frag.spv"}} },
			resourceSets[0], {}, {},
			{{ VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(m_hbaoInfo)}}});

		VkDescriptorSetLayoutBinding aoImageBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
		VkDescriptorImageInfo aoImageInfo{ .sampler = m_hbaoSampler, .imageView = m_AOImage.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

		resourceSets[1].push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
			{ depthImageBinding, aoImageBinding },  {},
				{{{.pCombinedImageSampler = &depthImageInfo }}, {{.pCombinedImageSampler = &aoImageInfo }}} });

		m_blurHBAOpass.initializeGraphics(assembler,
			{ {ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/fullscreen_vert.spv"},
			ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/hbao_blur_frag.spv"}} },
			resourceSets[1], {}, {},
			{ { VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, .offset = 0, .size = sizeof(glm::vec3)}} });



		m_hbaoInfo.invResolution = glm::vec2{ 1.0 / m_aoRenderWidth, 1.0 / m_aoRenderHeight };
		m_hbaoInfo.resolution = glm::vec2{ m_aoRenderWidth, m_aoRenderHeight };

		m_hbaoInfo.radius = 2.3;
		m_hbaoInfo.aoExponent = 1.2;
		m_hbaoInfo.angleBias = 0.18;
		m_hbaoInfo.negInvR2 = -1.0 / (m_hbaoInfo.radius * m_hbaoInfo.radius);

		m_blurSharpness = 3.0;
	}
	~HBAO()
	{
		vkDestroySampler(m_device, m_hbaoSampler, nullptr);
	}

	uint32_t getAOImageWidth()
	{
		return m_aoRenderWidth;
	}
	uint32_t getAOImageHeight()
	{
		return m_aoRenderHeight;
	}
	const Image& getAO()
	{
		return m_blurredAOImage;
	}
	const Image& getLinearDepthImage()
	{
		return m_linearDepthImage;
	}
	const VkImageView getLinearDepthImageView()
	{
		return m_linearDepthImage.getImageView();
	}

	void setHBAOsettings(float radius, float aoExponent, float angleBias, float sharpness)
	{
		m_hbaoInfo.radius = radius;
		m_hbaoInfo.aoExponent = aoExponent;
		m_hbaoInfo.angleBias = angleBias;
		m_hbaoInfo.negInvR2 = -1.0 / (m_hbaoInfo.radius * m_hbaoInfo.radius);

		m_blurSharpness = sharpness;
	}

	void acquireDepthPassData(const BufferMapped& modelTransformDataUB, const BufferMapped& perDrawDataIndicesSSBO)
	{
		VkDescriptorSetLayoutBinding viewprojBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
		VkDescriptorAddressInfoEXT viewprojAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = m_viewprojexpDataUB.getDeviceAddress(), .range = m_viewprojexpDataUB.getSize() };

		VkDescriptorSetLayoutBinding modelTransformBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
		VkDescriptorAddressInfoEXT modelTransformAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = modelTransformDataUB.getDeviceAddress(), .range = modelTransformDataUB.getSize() };

		VkDescriptorSetLayoutBinding uniformDrawIndicesBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
		VkDescriptorAddressInfoEXT uniformDrawIndicesAddressInfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = perDrawDataIndicesSSBO.getDeviceAddress(), .range = perDrawDataIndicesSSBO.getSize() };

		PipelineAssembler assembler{ m_device };
		assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
		assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, m_aoRenderWidth, m_aoRenderHeight);
		assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_DEFAULT);
		assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
		assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
		assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_DEFAULT);
		assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
		assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
		assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT, VK_FORMAT_R32_SFLOAT);

		std::vector<ResourceSet> resourceSets{};
		VkDevice device{ assembler.getDevice() };
		resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{}, 1,
			{ viewprojBinding, modelTransformBinding, uniformDrawIndicesBinding },  {0, { VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT }, 0},
				{{{.pUniformBuffer = &viewprojAddressInfo}}, {{.pUniformBuffer = &modelTransformAddressInfo}}, {{.pStorageBuffer = &uniformDrawIndicesAddressInfo}}} });

		m_linearExpandedFOVDepthPass.initializeGraphics(assembler,
			{ {ShaderStage{VK_SHADER_STAGE_VERTEX_BIT, "shaders/cmpld/lin_depth_vert.spv"},
			ShaderStage{VK_SHADER_STAGE_FRAGMENT_BIT, "shaders/cmpld/lin_depth_frag.spv"}} },
			resourceSets,
			{ {StaticVertex::getBindingDescription()} },
			{ StaticVertex::getAttributeDescriptions() });
	}
	void fiilRandomRotationImage(FrameCommandBufferSet& cmdBufferSet, VkQueue queue)
	{
		VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };

			BufferBaseHostAccessible staging(m_device, RANDOM_TEXTURE_SIZE * RANDOM_TEXTURE_SIZE * sizeof(short) * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

			//Check if dividing by numDir is useful
			constexpr int numDir{ 8 };
			short* texdata{ reinterpret_cast<short*>(staging.getData()) };
			constexpr uint32_t shortScale{ 1 << 15 };

			std::random_device dev;
			std::mt19937 rng(dev());
			std::uniform_real_distribution<> dist(0.0, 1.0);

			for (int i{ 0 }; i < RANDOM_TEXTURE_SIZE * RANDOM_TEXTURE_SIZE; ++i)
			{
				float angle{ static_cast<float>(2.0 * M_PI * dist(rng) / numDir) };

				float cosA = std::cos(angle);
				float sinA = std::sin(angle);
				float jitter = static_cast<float>(dist(rng));
				float empty{ 0.0 };

				texdata[i * 4 + 0] = (signed short)(shortScale * cosA);
				texdata[i * 4 + 1] = (signed short)(shortScale * sinA);
				texdata[i * 4 + 2] = (signed short)(shortScale * jitter);
				texdata[i * 4 + 3] = (signed short)(shortScale * empty);
			}

			BarrierOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
				{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					0, VK_ACCESS_TRANSFER_WRITE_BIT,
					VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
					m_randTex.getImageHandle(),
					{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 })}
			});
			m_randTex.cmdCopyDataFromBuffer(cb, staging.getBufferHandle(), staging.getOffset(), 0, 0, RANDOM_TEXTURE_SIZE, RANDOM_TEXTURE_SIZE);
			BarrierOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
				{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
					0, 0,
					VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					m_randTex.getImageHandle(),
					{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1 })}
			});

		cmdBufferSet.endRecording(cb);

		VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
		vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(queue);
	}

	void submitFrustum(double near, double far, double aspect, double FOV)
	{
		//Modified perspective matrix to avoid border artifacts
		float hDIVf = std::tan((FOV * 0.5));
		float hModif = 1.0 / (hDIVf * expansionScale);
		float w = hModif / aspect;
		float a = -near / (far - near);
		float b = (near * far) / (far - near);

		glm::mat4 projExpanded{
			glm::vec4(w, 0.0, 0.0, 0.0),
				glm::vec4(0.0, -hModif, 0.0, 0.0),
				glm::vec4(0.0, 0.0, a, 1.0),
				glm::vec4(0.0, 0.0, b, 0.0)
		};

		constexpr int projMatOffsetInMat4{ 1 };
		*(reinterpret_cast<glm::mat4*>(m_viewprojexpDataUB.getData()) + projMatOffsetInMat4) = projExpanded;

		m_hbaoInfo.radius = m_hbaoInfo.radius * 0.5f * (float(m_aoRenderHeight) / (std::tan(FOV) * 2.0f));
		m_hbaoInfo.uvTransformData = glm::vec4{ 2.0 / w, -2.0 / hModif, -1.0 / w, 1.0 / hModif };
	}
	void submitViewMatrix(const glm::mat4& viewMat)
	{
		constexpr int viewMatOffsetInMat4{ 0 };
		*(reinterpret_cast<glm::mat4*>(m_viewprojexpDataUB.getData()) + viewMatOffsetInMat4) = viewMat;
	}

	void cmdPassCalcHBAO(VkCommandBuffer cb, DescriptorManager& descriptorManager, const Buffer& vertexData, const Buffer& indexData, const BufferMapped& indirectCmdBuffer, uint32_t drawCount)
	{
		VkRenderingAttachmentInfo linearDepthAttachmentInfo{};
		linearDepthAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		linearDepthAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		linearDepthAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		linearDepthAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		linearDepthAttachmentInfo.clearValue = VkClearValue{ .color{.float32{10000.0f, 10000.0f, 10000.0f}} };
		linearDepthAttachmentInfo.imageView = m_linearDepthImage.getImageView();
		VkRenderingAttachmentInfo hbaoCalcAttachmentInfo{};
		hbaoCalcAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		hbaoCalcAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		hbaoCalcAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		hbaoCalcAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		hbaoCalcAttachmentInfo.clearValue = VkClearValue{ .color{.float32{1.0f}} };
		hbaoCalcAttachmentInfo.imageView = m_AOImage.getImageView();
		VkRenderingAttachmentInfo hbaoBlurAttachmentInfo{};
		hbaoBlurAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		hbaoBlurAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		hbaoBlurAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		hbaoBlurAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		hbaoBlurAttachmentInfo.clearValue = VkClearValue{ .color{.float32{1.0f}} };
		hbaoBlurAttachmentInfo.imageView = m_blurredAOImage.getImageView();
		VkRenderingAttachmentInfo depthBufferAttachmentInfo{};
		depthBufferAttachmentInfo.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
		depthBufferAttachmentInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		depthBufferAttachmentInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthBufferAttachmentInfo.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depthBufferAttachmentInfo.clearValue = VkClearValue{ .depthStencil{.depth = 0.0f} };
		depthBufferAttachmentInfo.imageView = m_depthBuffer.getImageView();

		VkRenderingAttachmentInfo attachments[3]{ linearDepthAttachmentInfo, hbaoCalcAttachmentInfo, hbaoBlurAttachmentInfo };

		VkRenderingInfo renderInfo{};
		renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
		renderInfo.renderArea = { .offset{0,0}, .extent{.width = m_aoRenderWidth, .height = m_aoRenderHeight} };
		renderInfo.layerCount = 1;
		renderInfo.pDepthAttachment = &depthBufferAttachmentInfo;


		renderInfo.colorAttachmentCount = 1;
		renderInfo.pColorAttachments = attachments + 0;

		BarrierOperations::cmdExecuteBarrier(cb,
			{{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				m_linearDepthImage.getImageHandle(), m_linearDepthImage.getSubresourceRange()),
			BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
				VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				0,
				VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
				m_depthBuffer.getImageHandle(),
				{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1})}
			});

		vkCmdBeginRendering(cb, &renderInfo);

			this->cmdCalculateLinearDepth(cb, descriptorManager, vertexData, indexData, indirectCmdBuffer, drawCount);

		vkCmdEndRendering(cb);

		BarrierOperations::cmdExecuteBarrier(cb, 
			{{BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				m_linearDepthImage.getImageHandle(), m_linearDepthImage.getSubresourceRange()),
			BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				0, VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				m_AOImage.getImageHandle(), m_AOImage.getSubresourceRange())}
		});

		renderInfo.colorAttachmentCount = 1;
		renderInfo.pColorAttachments = attachments + 1;

		vkCmdBeginRendering(cb, &renderInfo);

			this->cmdCalculateHBAO(cb, descriptorManager);

		vkCmdEndRendering(cb);

		BarrierOperations::cmdExecuteBarrier(cb,
			{ {BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				m_AOImage.getImageHandle(), m_AOImage.getSubresourceRange()),
			BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				0, VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
				m_blurredAOImage.getImageHandle(), m_blurredAOImage.getSubresourceRange())}
			});

		renderInfo.colorAttachmentCount = 1;
		renderInfo.pColorAttachments = attachments + 2;

		vkCmdBeginRendering(cb, &renderInfo);

			this->cmdBlurHBAO(cb, descriptorManager);

		vkCmdEndRendering(cb);

		BarrierOperations::cmdExecuteBarrier(cb,
			{ {BarrierOperations::constructImageBarrier(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				m_blurredAOImage.getImageHandle(), m_blurredAOImage.getSubresourceRange())}
			});
	}

private:
	void cmdCalculateLinearDepth(VkCommandBuffer cb, DescriptorManager& descriptorManager, const Buffer& vertexData, const Buffer& indexData, const BufferMapped& indirectCmdBuffer, uint32_t drawCount)
	{
		descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
			m_linearExpandedFOVDepthPass.getResourceSets(), m_linearExpandedFOVDepthPass.getResourceSetsInUse(), m_linearExpandedFOVDepthPass.getPipelineLayoutHandle());
		VkBuffer vertexBindings[1]{ vertexData.getBufferHandle() };
		VkDeviceSize vertexBindingOffsets[1]{ vertexData.getOffset() };
		vkCmdBindVertexBuffers(cb, 0, 1, vertexBindings, vertexBindingOffsets);
		vkCmdBindIndexBuffer(cb, indexData.getBufferHandle(), indexData.getOffset(), VK_INDEX_TYPE_UINT32);
		m_linearExpandedFOVDepthPass.cmdBind(cb);
		vkCmdDrawIndexedIndirect(cb, indirectCmdBuffer.getBufferHandle(), indirectCmdBuffer.getOffset(), drawCount, sizeof(VkDrawIndexedIndirectCommand));
	}
	void cmdCalculateHBAO(VkCommandBuffer cb, DescriptorManager& descriptorManager)
	{
		descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
			m_HBAOpass.getResourceSets(), m_HBAOpass.getResourceSetsInUse(), m_HBAOpass.getPipelineLayoutHandle());
		m_HBAOpass.cmdBind(cb);
		vkCmdPushConstants(cb, m_HBAOpass.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(m_hbaoInfo), &m_hbaoInfo);
		vkCmdDraw(cb, 3, 1, 0, 0);
	}
	void cmdBlurHBAO(VkCommandBuffer cb, DescriptorManager& descriptorManager)
	{
		descriptorManager.cmdSubmitPipelineResources(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
			m_blurHBAOpass.getResourceSets(), m_blurHBAOpass.getResourceSetsInUse(), m_blurHBAOpass.getPipelineLayoutHandle());
		m_blurHBAOpass.cmdBind(cb);
		glm::vec3 pcData{ m_hbaoInfo.invResolution, m_blurSharpness };
		vkCmdPushConstants(cb, m_blurHBAOpass.getPipelineLayoutHandle(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pcData), &pcData);
		vkCmdDraw(cb, 3, 1, 0, 0);
	}
};

#endif