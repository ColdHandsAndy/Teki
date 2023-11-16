#include "src/rendering/renderer/HBAO.h"

HBAO::HBAO(VkDevice device, uint32_t aoRenderWidth, uint32_t aoRenderHeight, const DepthBuffer& depthBuffer, CommandBufferSet& cmdBufferSet, VkQueue queue) :
	m_AOImage{ device, VK_FORMAT_R16_UNORM, aoRenderWidth, aoRenderHeight, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
	m_blurredAOImage{ device, VK_FORMAT_R16_UNORM, aoRenderWidth, aoRenderHeight, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
	m_randTex{ device, VK_FORMAT_R16G16B16A16_SNORM, RANDOM_TEXTURE_SIZE, RANDOM_TEXTURE_SIZE, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT },
	m_device{ device }
{
	m_aoRenderWidth = aoRenderWidth;
	m_aoRenderHeight = aoRenderHeight;

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
	samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	vkCreateSampler(device, &samplerCI, nullptr, &m_randSampler);


	VkDescriptorSetLayoutBinding depthImageBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo depthImageInfo{ .sampler = m_hbaoSampler, .imageView = depthBuffer.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL};

	VkDescriptorSetLayoutBinding randBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo randImageInfo{ .sampler = m_randSampler, .imageView = m_randTex.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	VkDescriptorSetLayoutBinding aoOutImageBinding{ .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo aoOutImageInfo{ .imageView = m_AOImage.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

	m_resSets[0].initializeSet(device, 1, VkDescriptorSetLayoutCreateFlags{},
		std::array{ depthImageBinding, randBinding, aoOutImageBinding }, std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &depthImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &randImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &aoOutImageInfo} }},
		true);

	std::reference_wrapper<const ResourceSet> resSet[1]{ m_resSets[0] };

	m_HBAOpass.initializaCompute(device, "shaders/cmpld/hbao_comp.spv", resSet, 
		{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(m_hbaoInfo)}} });

	VkDescriptorSetLayoutBinding aoInImageBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo aoInImageInfo{ .sampler = m_hbaoSampler, .imageView = m_AOImage.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

	VkDescriptorSetLayoutBinding aoBlurredBinding{ .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorImageInfo aoBlurredInfo{ .imageView = m_blurredAOImage.getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };

	m_resSets[1].initializeSet(device, 1, VkDescriptorSetLayoutCreateFlags{},
		std::array{ aoInImageBinding, aoBlurredBinding }, std::array<VkDescriptorBindingFlags, 0>{},
		std::vector<std::vector<VkDescriptorDataEXT>>{
			std::vector<VkDescriptorDataEXT>{ {.pCombinedImageSampler = &aoInImageInfo} },
			std::vector<VkDescriptorDataEXT>{ {.pStorageImage = &aoBlurredInfo} }},
		true);

	resSet[0] = m_resSets[1];

	m_blurHBAOpass.initializaCompute(device, "shaders/cmpld/hbao_blur_comp.spv", resSet, 
		{ {VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(glm::vec2) + sizeof(glm::uvec2)}}});

	m_imageBarriers[0] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
		m_AOImage.getImageHandle(), m_AOImage.getSubresourceRange());
	m_imageBarriers[1] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		m_blurredAOImage.getImageHandle(), m_blurredAOImage.getSubresourceRange());

	m_dependencyInfo = SyncOperations::createDependencyInfo(std::span(m_imageBarriers + 0, m_imageBarriers + 2));

	m_hbaoInfo.invResolution = glm::vec2{ 1.0 / m_aoRenderWidth, 1.0 / m_aoRenderHeight };
	m_hbaoInfo.resolution = glm::uvec2{ m_aoRenderWidth, m_aoRenderHeight };

	m_hbaoInfo.radius = 4.5;
	m_hbaoInfo.aoExponent = 1.22;
	m_hbaoInfo.angleBias = 0.18;
	m_hbaoInfo.negInvR2 = -1.0 / (m_hbaoInfo.radius * m_hbaoInfo.radius);

	fiilRandomRotationImage(cmdBufferSet, queue);
}
HBAO::~HBAO()
{
	vkDestroySampler(m_device, m_hbaoSampler, nullptr);
	vkDestroySampler(m_device, m_randSampler, nullptr);
}

void HBAO::cmdTransferClearBuffers(VkCommandBuffer cb)
{
	SyncOperations::cmdExecuteBarrier(cb,
		{ {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		m_AOImage.getImageHandle(), m_AOImage.getSubresourceRange()),
		SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		m_blurredAOImage.getImageHandle(), m_blurredAOImage.getSubresourceRange())} });

	VkClearColorValue clearVal{ .float32 = {1.0} };
	VkImageSubresourceRange subrRange{ m_AOImage.getSubresourceRange() };
	vkCmdClearColorImage(cb, m_AOImage.getImageHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &subrRange);
	subrRange = m_blurredAOImage.getSubresourceRange();
	vkCmdClearColorImage(cb, m_blurredAOImage.getImageHandle(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearVal, 1, &subrRange);
}

void HBAO::cmdDispatchHBAO(VkCommandBuffer cb)
{
	SyncOperations::cmdExecuteBarrier(cb,
		{ {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
		m_AOImage.getImageHandle(), m_AOImage.getSubresourceRange())} });

	m_HBAOpass.cmdBindResourceSets(cb);
	m_HBAOpass.cmdBind(cb);
	vkCmdPushConstants(cb, m_HBAOpass.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(m_hbaoInfo), &m_hbaoInfo);
	constexpr uint32_t groupSize{ 8 };
	vkCmdDispatch(cb, DISPATCH_SIZE(m_aoRenderWidth, groupSize), DISPATCH_SIZE(m_aoRenderHeight, groupSize), 1);
}
void HBAO::cmdDispatchHBAOBlur(VkCommandBuffer cb)
{
	m_blurHBAOpass.cmdBindResourceSets(cb);
	m_blurHBAOpass.cmdBind(cb);
	struct { glm::vec2 invResolution; glm::uvec2 resolution; } pcData;
	pcData.invResolution = m_hbaoInfo.invResolution;
	pcData.resolution = m_hbaoInfo.resolution;
	vkCmdPushConstants(cb, m_blurHBAOpass.getPipelineLayoutHandle(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcData), &pcData);
	constexpr uint32_t groupSize{ 8 };
	vkCmdDispatch(cb, DISPATCH_SIZE(m_aoRenderWidth, groupSize), DISPATCH_SIZE(m_aoRenderHeight, groupSize), 1);
}

void HBAO::setHBAOsettings(float radius, float aoExponent, float angleBias)
{
	m_hbaoInfo.radius = radius;
	m_hbaoInfo.aoExponent = aoExponent;
	m_hbaoInfo.angleBias = angleBias;
	m_hbaoInfo.negInvR2 = -1.0 / (m_hbaoInfo.radius * m_hbaoInfo.radius);
}

void HBAO::fiilRandomRotationImage(CommandBufferSet& cmdBufferSet, VkQueue queue)
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

	SyncOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
		{SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
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
	SyncOperations::cmdExecuteBarrier(cb, std::span<const VkImageMemoryBarrier2>{
		{SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
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

void HBAO::submitFrustum(double near, double far, double aspect, double FOV)
{
	float h{ static_cast<float>(1.0 / std::tan((FOV * 0.5))) };
	float w{ static_cast<float>(h / aspect) };

	m_hbaoInfo.farPlane = far;
	m_hbaoInfo.nearPlane = near;
	m_hbaoInfo.radius = m_hbaoInfo.radius * 0.5f * (float(m_aoRenderHeight) / (std::tan(FOV) * 2.0f));
	m_hbaoInfo.projectionData = glm::vec4{ 2.0 / w, -2.0 / h, -1.0 / w, 1.0 / h };
}