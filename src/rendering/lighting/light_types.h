#ifndef LIGHTING_TYPES_HEADER
#define LIGHTING_TYPES_HEADER


#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

#include "src/rendering/renderer/clusterer.h"
#include "src/rendering/lighting/shadows.h"

namespace LightTypes
{

	class DirectionalLight
	{
	private:
		glm::vec3 m_color{};
		float m_power{};
		glm::vec3 m_direction{};

		struct DirectionalLightData
		{
			alignas(16) glm::vec3 spectrum{};
			alignas(16) glm::vec3 lightDir{};
		};

	public:
		DirectionalLight(const glm::vec3& lightColor, float lightPower, const glm::vec3& lightDir) : m_color{ lightColor}, m_power{ lightPower }, m_direction{ glm::normalize(lightDir)} {}

		static uint32_t getDataByteSize()
		{
			return sizeof(DirectionalLightData);
		}
		void plantData(void* dataPtr) const
		{
			DirectionalLightData dataToPass{ m_color * m_power, m_direction };
			std::memcpy(dataPtr, &dataToPass, sizeof(dataToPass));
		}

		void changeDirection(const glm::vec3& lightDir)
		{
			m_direction = glm::normalize(lightDir);
		}
	};

	class LightBase
	{
	protected:
		glm::vec3 m_color{};
		float m_power{};
		bool m_hasShadow{ false };

		Clusterer::LightFormat* m_data{ nullptr };
		glm::vec4* m_boundingSphere{ nullptr };

		inline static Clusterer* m_clusterer{ nullptr };
		inline static ShadowCaster* m_caster{ nullptr };

	protected:
		LightBase(const glm::vec3& lightColor, float lightPower) : m_color{ lightColor }, m_power{ lightPower } {}

	public:
		void changeColor(const glm::vec3& color)
		{
			m_color = color;
			m_data->spectrum = color * m_power;
		}
		void changePower(float power)
		{
			m_power = std::max(0.0f, power);
			m_data->spectrum = m_color * m_power;
		}
		void changeSize(float size)
		{
			m_data->lightSize = std::max(0.0f, size);
		}

		static void assignGlobalClusterer(Clusterer& clusterer)
		{
			m_clusterer = &clusterer;
		}
		static void assignGlobalShadowCaster(ShadowCaster& caster)
		{
			m_caster = &caster;
		}
	};

	class PointLight : public LightBase
	{
	public:
		PointLight(const glm::vec3& worldPos, const glm::vec3& lightColor, float lightPower, float radius, uint32_t shadowMapSize = 0, float lightSize = 1.0f) : LightBase{ lightColor, lightPower }
		{
			EASSERT(m_clusterer != nullptr, "App", "Global Clusterer has not been assigned.");
			m_clusterer->getNewLight(&m_data, &m_boundingSphere, Clusterer::LightFormat::TYPE_POINT);
			m_data->position = worldPos;
			m_data->length = radius;
			m_data->spectrum = lightColor * lightPower;
			*m_boundingSphere = calculateBoundingSphere();

			if (shadowMapSize)
			{
				m_data->shadowListIndex = m_caster->addShadowCubeMap(shadowMapSize);
				m_data->shadowMatrixIndex = m_caster->addPointViewMatrices(worldPos);
				m_data->lightSize = lightSize;
				m_hasShadow = true;
				m_caster->m_drawCommandIndices.resize(m_caster->m_drawCommandIndices.size() + 6);
			}
			else
			{
				m_data->shadowListIndex = -1;
			}
		}

		void changeAll(const glm::vec3& position, const glm::vec3& lightColor, float lightPower, float radius, float lightSize = 0.0f)
		{
			m_data->position = position;
			m_data->length = radius;
			m_data->spectrum = lightColor * lightPower;
			*m_boundingSphere = calculateBoundingSphere();
			if (m_hasShadow)
			{
				m_data->lightSize = lightSize;
				m_caster->calcCubeViewMatrices(m_data->shadowMatrixIndex, m_data->position);
			}
		}
		void changePosition(const glm::vec3& position)
		{
			m_data->position = position;
			m_boundingSphere->x = position.x;
			m_boundingSphere->y = position.y;
			m_boundingSphere->z = position.z;
			if (m_hasShadow)
				m_caster->calcCubeViewMatrices(m_data->shadowMatrixIndex, position);
		}
		void changeRadius(float radius)
		{
			m_data->length = radius;
			m_boundingSphere->w = radius;
			if (m_hasShadow)
				m_caster->calcCubeViewMatrices(m_data->shadowMatrixIndex, m_data->position);
		}

	private:
		glm::vec4 calculateBoundingSphere()
		{
			return { m_data->position, m_data->length };
		}
	};

	class SpotLight : public LightBase
	{
	public:
		SpotLight(const glm::vec3& worldPos, const glm::vec3& lightColor, float lightPower, float length, const glm::vec3& lightDir, float falloffAngle, float cutoffAngle, uint32_t shadowMapSize = 0, float lightSize = 1.0f)
			: LightBase{ lightColor, lightPower }
		{
			EASSERT(m_clusterer != nullptr, "App", "Global Clusterer has not been assigned.");
			m_clusterer->getNewLight(&m_data, &m_boundingSphere, Clusterer::LightFormat::TYPE_SPOT);
			m_data->position = worldPos;
			m_data->spectrum = lightColor * lightPower;
			m_data->cutoffCos = std::cos(std::min(cutoffAngle, static_cast<float>(M_PI_2)));
			m_data->falloffCos = std::cos(std::min(std::min(falloffAngle, cutoffAngle), static_cast<float>(M_PI_2)));
			m_data->lightDir = glm::normalize(lightDir);
			if (m_data->lightDir.y > 0.999)
			{
				m_data->lightDir.y = 0.999;
				m_data->lightDir.x = 0.001;
			}
			else if (m_data->lightDir.y < -0.999)
			{
				m_data->lightDir.y = -0.999;
				m_data->lightDir.x = 0.001;
			}
			m_data->length = length;
			*m_boundingSphere = calculateBoundingSphere();

			if (shadowMapSize)
			{
				auto listAndLayer{ m_caster->addShadowMap(shadowMapSize, shadowMapSize) };
				m_data->shadowListIndex = listAndLayer.listIndex;
				m_data->shadowLayerIndex = listAndLayer.layerIndex;
				m_data->shadowMatrixIndex = m_caster->addSpotViewMatrix(worldPos, m_data->lightDir);
				m_data->lightSize = lightSize * (m_data->cutoffCos / std::sqrt(1 - m_data->cutoffCos * m_data->cutoffCos));
				m_hasShadow = true;
				m_caster->m_drawCommandIndices.emplace_back();
			}
			else
			{
				m_data->shadowListIndex = -1;
			}
		}

		void changeAll(const glm::vec3& position, const glm::vec3& lightColor, float lightPower, float length, const glm::vec3& lightDir, float lightSize = 0.0f, float falloffAngle = -1.0f, float cutoffAngle = -1.0f)
		{
			m_data->position = position;
			m_data->spectrum = lightColor * lightPower;

			if (falloffAngle < 0.0f && cutoffAngle < 0.0f)
			{
				m_data->cutoffCos = std::cos(std::min(cutoffAngle, static_cast<float>(M_PI_2)));
				m_data->falloffCos = std::cos(std::min(std::min(falloffAngle, cutoffAngle), static_cast<float>(M_PI_2)));
			}

			m_data->lightDir = glm::normalize(lightDir);
			if (m_data->lightDir.y > 0.999)
			{
				m_data->lightDir.y = 0.999;
				m_data->lightDir.x = 0.001;
			}
			else if (m_data->lightDir.y < -0.999)
			{
				m_data->lightDir.y = -0.999;
				m_data->lightDir.x = 0.001;
			}

			m_data->length = length;
			*m_boundingSphere = calculateBoundingSphere();

			if (m_hasShadow)
			{
				m_data->lightSize = lightSize;
				m_caster->calcCubeViewMatrices(m_data->shadowMatrixIndex, m_data->position);
			}
		}
		void changePosition(const glm::vec3& position)
		{
			m_data->position = position;
			*m_boundingSphere = calculateBoundingSphere();

			if (m_hasShadow)
				m_caster->calcViewMatrix(m_data->shadowMatrixIndex, position, m_data->lightDir);
		}
		void changeDirection(const glm::vec3& lightDir)
		{
			m_data->lightDir = glm::normalize(lightDir);
			*m_boundingSphere = calculateBoundingSphere();

			if (m_hasShadow)
				m_caster->calcViewMatrix(m_data->shadowMatrixIndex, m_data->position, m_data->lightDir);
		}
		void changeLength(float length)
		{
			m_data->length = length;
			*m_boundingSphere = calculateBoundingSphere();
		}
		void changeCutoff(float cutoffAngle)
		{
			m_data->cutoffCos = std::cos(std::min(cutoffAngle, static_cast<float>(M_PI_2)));
			m_data->falloffCos = std::max(m_data->cutoffCos, m_data->falloffCos);
			m_data->lightSize = m_data->lightSize * (m_data->cutoffCos / std::sqrt(1 - m_data->cutoffCos * m_data->cutoffCos));
			*m_boundingSphere = calculateBoundingSphere();
			if (m_hasShadow)
				m_caster->calcViewMatrix(m_data->shadowMatrixIndex, m_data->position, m_data->lightDir);
		}
		void changeFalloff(float falloffAngle)
		{
			m_data->falloffCos = std::max(m_data->cutoffCos, std::cos(std::min(falloffAngle, static_cast<float>(M_PI_2))));
		}

	private:
		glm::vec4 calculateBoundingSphere()
		{
			return { m_data->cutoffCos > glm::one_over_root_two<float>()
			?
			glm::vec4{m_data->position + m_data->lightDir * (m_data->length / 2.0f), m_data->length / 2.0f}
			:
			glm::vec4{m_data->position + m_data->lightDir * m_data->length * m_data->cutoffCos, m_data->length * std::sqrt(1 - m_data->cutoffCos * m_data->cutoffCos)} };
		}
	};

}


#endif