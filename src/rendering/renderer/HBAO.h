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
#include "src/rendering/renderer/sync_operations.h"
#include "src/rendering/renderer/culling.h"

class HBAO
{
private:
	VkDevice m_device{};

	uint32_t m_aoRenderWidth{};
	uint32_t m_aoRenderHeight{};

	Image m_randTex;
	Image m_AOImage;
	Image m_blurredAOImage;

	std::array<ResourceSet, 2> m_resSets{};

	Pipeline m_HBAOpass;
	Pipeline m_blurHBAOpass;

	VkImageMemoryBarrier2 m_imageBarriers[2]{};
	VkDependencyInfo m_dependencyInfo{};

	float m_frustumFar{};
	float m_frustumFOV{};
	struct
	{
		glm::vec4 projectionData;

		glm::vec2 invResolution;
		glm::uvec2 resolution;

		float radius;
		float aoExponent;
		float angleBias;
		float negInvR2;

		float farPlane;
		float nearPlane;
	} m_hbaoInfo;

	VkSampler m_hbaoSampler{};
	VkSampler m_randSampler{};

public:
	HBAO(VkDevice device, uint32_t aoRenderWidth, uint32_t aoRenderHeight, const DepthBuffer& depthBuffer, CommandBufferSet& cmdBufferSet, VkQueue queue);
	~HBAO();

	uint32_t getAOImageWidth()
	{
		return m_aoRenderWidth;
	}
	uint32_t getAOImageHeight()
	{
		return m_aoRenderHeight;
	}
	const Image& getAO() const
	{
		return m_blurredAOImage;
	}
	const VkDependencyInfo& getDependency()
	{
		return m_dependencyInfo;
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

	void setHBAOsettings(float radius, float aoExponent, float angleBias);


	void submitFrustum(double near, double far, double aspect, double FOV);

	void cmdTransferClearBuffers(VkCommandBuffer cb);

	void cmdDispatchHBAO(VkCommandBuffer cb);
	void cmdDispatchHBAOBlur(VkCommandBuffer cb);

private:
	void fiilRandomRotationImage(CommandBufferSet& cmdBufferSet, VkQueue queue);
};

#endif