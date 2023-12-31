#ifndef BB_HEADER
#define BB_HEADER

#include <vector>

#include <glm/glm.hpp>

#include "src/tools/asserter.h"
#include "src/tools/alignment.h"

class OBBs
{
private:
	uint32_t m_count{ 0 };
	uint32_t m_maxCount{ 0 };
	float* m_data{};

	float* m_axii{};
	float* m_extents{};
	float* m_centers{};

	static constexpr int dimensionsNum{ 3 };
	static constexpr int boxVertexCount{ 8 };

public:
	enum Position
	{
		TFL,
		TFR,
		TBL,
		TBR,
		BFL,
		BFR,
		BBL,
		BBR,
		ALL_POS
	};
	OBBs(int boxCount)
	{
		m_maxCount = ALIGNED_SIZE(boxCount, 4);
		
		m_data = { new float[dimensionsNum * boxVertexCount * m_maxCount] };
		m_axii = { new float[dimensionsNum * dimensionsNum * m_maxCount] };
		m_extents = { new float[dimensionsNum * m_maxCount] };
		m_centers = { new float[dimensionsNum * m_maxCount] };
	}
	~OBBs()
	{
		delete[] m_data;
		delete[] m_axii;
		delete[] m_extents;
		delete[] m_centers;
	}

	uint32_t getBBCount() const
	{
		return m_count;
	}

	void getPointsOBB(int index, float** xs, float** ys, float** zs) const
	{
		EASSERT(index < m_count, "App", "Undefined data accessed.");
		*xs = m_data + 0 * boxVertexCount * m_maxCount + boxVertexCount * index;
		*ys = m_data + 1 * boxVertexCount * m_maxCount + boxVertexCount * index;
		*zs = m_data + 2 * boxVertexCount * m_maxCount + boxVertexCount * index;
	}

	uint32_t getAxiiOBBs(
		float** xsOfXaxii, float** ysOfXaxii, float** zsOfXaxii, 
		float** xsOfYaxii, float** ysOfYaxii, float** zsOfYaxii, 
		float** xsOfZaxii, float** ysOfZaxii, float** zsOfZaxii,
		float** centersX, float** centersY, float** centersZ,
		float** extentsX, float** extentsY, float** extentsZ) const
	{
		*xsOfXaxii = m_axii + 0 * m_maxCount;
		*ysOfXaxii = m_axii + 1 * m_maxCount;
		*zsOfXaxii = m_axii + 2 * m_maxCount;
											
		*xsOfYaxii = m_axii + 3 * m_maxCount;
		*ysOfYaxii = m_axii + 4 * m_maxCount;
		*zsOfYaxii = m_axii + 5 * m_maxCount;
											
		*xsOfZaxii = m_axii + 6 * m_maxCount;
		*ysOfZaxii = m_axii + 7 * m_maxCount;
		*zsOfZaxii = m_axii + 8 * m_maxCount;

		*extentsX = m_extents + 0 * m_maxCount;
		*extentsY = m_extents + 1 * m_maxCount;
		*extentsZ = m_extents + 2 * m_maxCount;

		*centersX = m_centers + 0 * m_maxCount;
		*centersY = m_centers + 1 * m_maxCount;
		*centersZ = m_centers + 2 * m_maxCount;

		return m_count;
	}

	void getBoundingSphere(int index, float* pos, float* rad) const
	{
		EASSERT(index < m_count, "App", "Undefined data accessed.");

		float xMin = *(m_data + 0 * boxVertexCount * m_maxCount + boxVertexCount * index + BBR);
		float yMin = *(m_data + 1 * boxVertexCount * m_maxCount + boxVertexCount * index + BBR);
		float zMin = *(m_data + 2 * boxVertexCount * m_maxCount + boxVertexCount * index + BBR);

		float xMax = *(m_data + 0 * boxVertexCount * m_maxCount + boxVertexCount * index + TFL);
		float yMax = *(m_data + 1 * boxVertexCount * m_maxCount + boxVertexCount * index + TFL);
		float zMax = *(m_data + 2 * boxVertexCount * m_maxCount + boxVertexCount * index + TFL);

		pos[0] = (xMin + xMax) / 2.0;
		pos[1] = (yMin + yMax) / 2.0;
		pos[2] = (zMin + zMax) / 2.0;
		float xDif = xMax - xMin;
		float yDif = yMax - yMin;
		float zDif = zMax - zMin;
		*rad = std::sqrt(xDif * xDif + yDif * yDif + zDif * zDif);
	}

	void transformOBB(int index, const glm::mat4& transformMatrix)
	{
		float* xs = m_data + 0 * boxVertexCount * m_maxCount + boxVertexCount * index;
		float* ys = m_data + 1 * boxVertexCount * m_maxCount + boxVertexCount * index;
		float* zs = m_data + 2 * boxVertexCount * m_maxCount + boxVertexCount * index;
		for (int i{ 0 }; i < ALL_POS; ++i)
		{
			glm::vec3 newPos{ transformMatrix * glm::vec4{xs[i], ys[i], zs[i], 1.0} };
			xs[i] = newPos.x;
			ys[i] = newPos.y;
			zs[i] = newPos.z;
		}

		float* xsOfXaxii{ m_axii + 0 * m_maxCount + index };
		float* ysOfXaxii{ m_axii + 1 * m_maxCount + index };
		float* zsOfXaxii{ m_axii + 2 * m_maxCount + index };

		float* xsOfYaxii{ m_axii + 3 * m_maxCount + index };
		float* ysOfYaxii{ m_axii + 4 * m_maxCount + index };
		float* zsOfYaxii{ m_axii + 5 * m_maxCount + index };

		float* xsOfZaxii{ m_axii + 6 * m_maxCount + index };
		float* ysOfZaxii{ m_axii + 7 * m_maxCount + index };
		float* zsOfZaxii{ m_axii + 8 * m_maxCount + index };

		glm::vec3 x{ xs[7] - xs[6], ys[7] - ys[6], zs[7] - zs[6] };
		glm::vec3 y{ xs[2] - xs[6], ys[2] - ys[6], zs[2] - zs[6] };
		glm::vec3 z{ xs[4] - xs[6], ys[4] - ys[6], zs[4] - zs[6] };

		glm::vec3 xn{ glm::normalize(x) };
		glm::vec3 yn{ glm::normalize(y) };
		glm::vec3 zn{ glm::normalize(z) };

		*xsOfXaxii = xn.x;
		*ysOfXaxii = xn.y;
		*zsOfXaxii = xn.z;

		*xsOfYaxii = yn.x;
		*ysOfYaxii = yn.y;
		*zsOfYaxii = yn.z;

		*xsOfZaxii = zn.x;
		*ysOfZaxii = zn.y;
		*zsOfZaxii = zn.z;

		*(m_extents + 0 * m_maxCount + index) = glm::length(x) / 2.0;
		*(m_extents + 1 * m_maxCount + index) = glm::length(y) / 2.0;
		*(m_extents + 2 * m_maxCount + index) = glm::length(z) / 2.0;

		glm::vec3 c{ (xs[1] + xs[6]) / 2.0, (ys[1] + ys[6]) / 2.0, (zs[1] + zs[6]) / 2.0 };

		*(m_centers + 0 * m_maxCount + index) = c.x;
		*(m_centers + 1 * m_maxCount + index) = c.y;
		*(m_centers + 2 * m_maxCount + index) = c.z;
	}

	//Input data should be ordered like in the enum
	void addOBB(float* obbData)
	{
		EASSERT(m_count < m_maxCount, "App", "More OBBs added than space allocated.");

		float* dataToFillX{ m_data + 0 * boxVertexCount * m_maxCount + boxVertexCount * m_count };
		float* dataToFillY{ m_data + 1 * boxVertexCount * m_maxCount + boxVertexCount * m_count };
		float* dataToFillZ{ m_data + 2 * boxVertexCount * m_maxCount + boxVertexCount * m_count };
		
		for (int i{ 0 }; i < ALL_POS; ++i)
		{
			dataToFillX[i] = obbData[3 * i + 0];
			dataToFillY[i] = obbData[3 * i + 1];
			dataToFillZ[i] = obbData[3 * i + 2];
		}

		float* xsOfXaxii{ m_axii + 0 * m_maxCount + m_count };
		float* ysOfXaxii{ m_axii + 1 * m_maxCount + m_count };
		float* zsOfXaxii{ m_axii + 2 * m_maxCount + m_count };

		float* xsOfYaxii{ m_axii + 3 * m_maxCount + m_count };
		float* ysOfYaxii{ m_axii + 4 * m_maxCount + m_count };
		float* zsOfYaxii{ m_axii + 5 * m_maxCount + m_count };

		float* xsOfZaxii{ m_axii + 6 * m_maxCount + m_count };
		float* ysOfZaxii{ m_axii + 7 * m_maxCount + m_count };
		float* zsOfZaxii{ m_axii + 8 * m_maxCount + m_count };

		glm::vec3 x{ obbData[3 * 7 + 0] - obbData[3 * 6 + 0], obbData[3 * 7 + 1] - obbData[3 * 6 + 1], obbData[3 * 7 + 2] - obbData[3 * 6 + 2] };
		glm::vec3 y{ obbData[3 * 2 + 0] - obbData[3 * 6 + 0], obbData[3 * 2 + 1] - obbData[3 * 6 + 1], obbData[3 * 2 + 2] - obbData[3 * 6 + 2] };
		glm::vec3 z{ obbData[3 * 4 + 0] - obbData[3 * 6 + 0], obbData[3 * 4 + 1] - obbData[3 * 6 + 1], obbData[3 * 4 + 2] - obbData[3 * 6 + 2] };

		glm::vec3 xn{ glm::normalize(x) };
		glm::vec3 yn{ glm::normalize(y) };
		glm::vec3 zn{ glm::normalize(z) };

		*xsOfXaxii = xn.x;
		*ysOfXaxii = xn.y;
		*zsOfXaxii = xn.z;
				 	  
		*xsOfYaxii = yn.x;
		*ysOfYaxii = yn.y;
		*zsOfYaxii = yn.z;
				 	  
		*xsOfZaxii = zn.x;
		*ysOfZaxii = zn.y;
		*zsOfZaxii = zn.z;

		*(m_extents + 0 * m_maxCount + m_count) = glm::length(x) / 2.0;
		*(m_extents + 1 * m_maxCount + m_count) = glm::length(y) / 2.0;
		*(m_extents + 2 * m_maxCount + m_count) = glm::length(z) / 2.0;

		glm::vec3 c{ (obbData[3 * 1 + 0] + obbData[3 * 6 + 0]) / 2.0, (obbData[3 * 1 + 1] + obbData[3 * 6 + 1]) / 2.0, (obbData[3 * 1 + 2] + obbData[3 * 6 + 2]) / 2.0 };

		*(m_centers + 0 * m_maxCount + m_count) = c.x;
		*(m_centers + 1 * m_maxCount + m_count) = c.y;
		*(m_centers + 2 * m_maxCount + m_count) = c.z;

		++m_count;
	}

#ifdef _DEBUG

	Pipeline* visOBBPipeline{ nullptr };
	BufferBaseHostAccessible* vb{ nullptr };
	constexpr static uint32_t vertCount{ 18 };
public:
	void initVisualizationResources(VkDevice device, uint32_t wWidth, uint32_t wHeight, BufferMapped& viewprojDataUB)
	{
		PipelineAssembler assembler{ device };
		assembler.setDynamicState(PipelineAssembler::DYNAMIC_STATE_DEFAULT);
		assembler.setViewportState(PipelineAssembler::VIEWPORT_STATE_DEFAULT, wWidth, wHeight, 1);
		assembler.setInputAssemblyState(PipelineAssembler::INPUT_ASSEMBLY_STATE_TRIANGLE_STRIP_DRAWING);
		assembler.setTesselationState(PipelineAssembler::TESSELATION_STATE_DEFAULT);
		assembler.setMultisamplingState(PipelineAssembler::MULTISAMPLING_STATE_DISABLED);
		assembler.setRasterizationState(PipelineAssembler::RASTERIZATION_STATE_LINE_POLYGONS, 1.6f, VK_CULL_MODE_NONE);
		assembler.setColorBlendState(PipelineAssembler::COLOR_BLEND_STATE_DISABLED);
		assembler.setDepthStencilState(PipelineAssembler::DEPTH_STENCIL_STATE_DEFAULT);
		assembler.setPipelineRenderingState(PipelineAssembler::PIPELINE_RENDERING_STATE_DEFAULT);

		std::vector<ResourceSet> resourceSets{};
		VkDescriptorSetLayoutBinding uniformViewProjBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT };
		VkDescriptorAddressInfoEXT uniformViewProjAddressinfo{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT, .address = viewprojDataUB.getDeviceAddress(), .range = sizeof(glm::mat4) * 2 };
		resourceSets.push_back({ assembler.getDevice(), 0, VkDescriptorSetLayoutCreateFlags{}, 1, {uniformViewProjBinding},  {}, {{{.pUniformBuffer = &uniformViewProjAddressinfo}}}, false });
		const VkVertexInputBindingDescription bindingDescription{
			.binding = 0,
			.stride = sizeof(glm::vec3),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX
		};
		const std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions
		{
			VkVertexInputAttributeDescription
			{
				.location = 0,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = 0
			}
		};
		visOBBPipeline = { new Pipeline };
		visOBBPipeline->initializeGraphics(assembler, 
			{{ShaderStage{.stage = VK_SHADER_STAGE_VERTEX_BIT, .filepath = "shaders/cmpld/simple_proj_vert.spv"}, ShaderStage{.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .filepath = "shaders/cmpld/color_frag.spv"}}}, resourceSets, 
			{ {bindingDescription} },
			attributeDescriptions);

		vb = { new BufferBaseHostAccessible{device, m_count * sizeof(glm::vec3) * vertCount, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, false, false} };
		glm::vec3* vbData{ reinterpret_cast<glm::vec3*>(vb->getData()) };
		float* xs = m_data + 0 * boxVertexCount * m_maxCount;
		float* ys = m_data + 1 * boxVertexCount * m_maxCount;
		float* zs = m_data + 2 * boxVertexCount * m_maxCount;
		for (int i{ 0 }; i < m_count; ++i)
		{
			*(vbData++) = glm::vec3(xs[0], ys[0], zs[0]);
			*(vbData++) = glm::vec3(xs[4], ys[4], zs[4]);
			*(vbData++) = glm::vec3(xs[1], ys[1], zs[1]);
			*(vbData++) = glm::vec3(xs[5], ys[5], zs[5]);
			*(vbData++) = glm::vec3(xs[3], ys[3], zs[3]);
			*(vbData++) = glm::vec3(xs[7], ys[7], zs[7]);
			*(vbData++) = glm::vec3(xs[2], ys[2], zs[2]);
			*(vbData++) = glm::vec3(xs[6], ys[6], zs[6]);
			*(vbData++) = glm::vec3(xs[0], ys[0], zs[0]);
			*(vbData++) = glm::vec3(xs[1], ys[1], zs[1]);
			*(vbData++) = glm::vec3(xs[3], ys[3], zs[3]);
			*(vbData++) = glm::vec3(xs[2], ys[2], zs[2]);
			*(vbData++) = glm::vec3(xs[0], ys[0], zs[0]);
			*(vbData++) = glm::vec3(xs[7], ys[7], zs[7]);
			*(vbData++) = glm::vec3(xs[6], ys[6], zs[6]);
			*(vbData++) = glm::vec3(xs[4], ys[4], zs[4]);
			*(vbData++) = glm::vec3(xs[5], ys[5], zs[5]);
			*(vbData++) = glm::vec3(xs[7], ys[7], zs[7]);
			xs += boxVertexCount;
			ys += boxVertexCount;
			zs += boxVertexCount;
		}
	}

	void cmdVisualizeOBBs(VkCommandBuffer cb)
	{
		VkBuffer vertexBinding[1]{ vb->getBufferHandle() };
		VkDeviceSize vertexOffsets[1]{ vb->getOffset() };
		vkCmdBindVertexBuffers(cb, 0, 1, vertexBinding, vertexOffsets);
		visOBBPipeline->cmdBind(cb);
		for (int i{ 0 }; i < m_count; ++i)
			vkCmdDraw(cb, vertCount, 1, vertCount * i, 0);
	}
#endif
};


#endif