#ifndef HBAO_CLASS_HEADER
#define HBAO_CLASS_HEADER

#define RANDOM_TEXTURE_SIZE 4

#include <cstdint>
#include <random>

#include <glm/gtc/constants.hpp>

#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/renderer/barrier_operations.h"

class HBAO
{
private:
	VkDevice m_device{};

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
	float m_frustumFar{};
	float m_frustumFOV{};
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
	VkSampler m_randSampler{};

public:
	HBAO(VkDevice device, uint32_t aoRenderWidth, uint32_t aoRenderHeight);
	~HBAO();

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

	void setRadius(float radius)
	{
		m_hbaoInfo.negInvR2 = -1.0 / (radius * radius);
		m_hbaoInfo.radius = radius * 0.5f * (float(m_aoRenderHeight) / (std::tan(m_frustumFOV) * 2.0f));
	}
	void setExponent(float exponent)
	{
		m_hbaoInfo.aoExponent = exponent;
	}
	void setAngleBias(float bias)
	{
		m_hbaoInfo.angleBias = bias;
	}
	void setBlurSharpness(float sharpness)
	{
		m_blurSharpness = sharpness;
	}

	void setHBAOsettings(float radius, float aoExponent, float angleBias, float sharpness);

	void acquireDepthPassData(const BufferMapped& modelTransformDataUB, const BufferMapped& perDrawDataIndicesSSBO);
	void fiilRandomRotationImage(FrameCommandBufferSet& cmdBufferSet, VkQueue queue);

	void submitFrustum(double near, double far, double aspect, double FOV);
	void submitViewMatrix(const glm::mat4& viewMat);

	void cmdPassCalcHBAO(VkCommandBuffer cb, DescriptorManager& descriptorManager, const Buffer& vertexData, const Buffer& indexData, const BufferMapped& indirectCmdBuffer, uint32_t drawCount);

private:
	void cmdCalculateLinearDepth(VkCommandBuffer cb, DescriptorManager& descriptorManager, const Buffer& vertexData, const Buffer& indexData, const BufferMapped& indirectCmdBuffer, uint32_t drawCount);
	void cmdCalculateHBAO(VkCommandBuffer cb, DescriptorManager& descriptorManager);
	void cmdBlurHBAO(VkCommandBuffer cb, DescriptorManager& descriptorManager);
};

#endif