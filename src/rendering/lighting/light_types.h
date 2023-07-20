#ifndef LIGHTING_TYPES_HEADER
#define LIGHTING_TYPES_HEADER


#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

#include "src/rendering/renderer/clusterer.h"

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
		DirectionalLight(glm::vec3 lightColor, float lightPower, glm::vec3 lightDir) : m_color{ lightColor}, m_power{ lightPower }, m_direction{ glm::normalize(lightDir)} {}

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

		Clusterer::LightFormat* m_data{ nullptr };
		glm::vec4* m_boundingSphere{ nullptr };

		inline static Clusterer* m_clusterer{ nullptr };

	protected:
		LightBase(glm::vec3 lightColor, float lightPower) : m_color{ lightColor }, m_power{ lightPower } {}

	public:
		void changeColor(const glm::vec3& color)
		{
			m_color = color;
			m_data->spectrum = color * m_power;
		}
		void changePower(float power)
		{
			m_power = power;
			m_data->spectrum = m_color * power;
		}

		static void assignGlobalClusterer(Clusterer& clusterer)
		{
			m_clusterer = &clusterer;
		}
	};

	class PointLight : public LightBase
	{
	public:
		PointLight(glm::vec3 worldPos, glm::vec3 lightColor, float lightPower, float radius) : LightBase{ lightColor, lightPower }
		{
			EASSERT(m_clusterer != nullptr, "App", "Global Clusterer has not been assigned.");
			m_clusterer->getNewLight(&m_data, &m_boundingSphere, Clusterer::LightFormat::TYPE_POINT);
			m_data->position = worldPos;
			m_data->length = radius;
			m_data->spectrum = lightColor * lightPower;
			*m_boundingSphere = calculateBoundingSphere();
		}

		void changePosition(const glm::vec3& position)
		{
			m_data->position = position;
			m_boundingSphere->x = position.x;
			m_boundingSphere->y = position.y;
			m_boundingSphere->z = position.z;
		}
		void changeRadius(float radius)
		{
			m_data->length = radius;
			m_boundingSphere->w = radius;
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
		SpotLight(glm::vec3 worldPos, glm::vec3 lightColor, float lightPower, float length, glm::vec3 lightDir, float cutoffStartAngle, float cutoffEndAngle)
			: LightBase{ lightColor, lightPower }
		{
			EASSERT(m_clusterer != nullptr, "App", "Global Clusterer has not been assigned.");
			m_clusterer->getNewLight(&m_data, &m_boundingSphere, Clusterer::LightFormat::TYPE_SPOT);
			m_data->position = worldPos;
			m_data->spectrum = lightColor * lightPower;
			m_data->cutoffCos = std::cos(std::min(cutoffEndAngle, static_cast<float>(M_PI_2)));
			m_data->falloffCos = std::cos(std::min(std::min(cutoffStartAngle, cutoffEndAngle), static_cast<float>(M_PI_2)));
			m_data->lightDir = glm::normalize(lightDir);
			if (lightDir.y > 0.999)
			{
				m_data->lightDir.y = 0.999;
				m_data->lightDir.x = 0.001;
			}
			else if (lightDir.y < -0.999)
			{
				m_data->lightDir.y = -0.999;
				m_data->lightDir.x = 0.001;
			}
			m_data->length = length;
			*m_boundingSphere = calculateBoundingSphere();
		}

		void changePosition(const glm::vec3& position)
		{
			m_boundingSphere->x = position.x + m_boundingSphere->x - m_data->position.x;
			m_boundingSphere->y = position.y + m_boundingSphere->y - m_data->position.y;
			m_boundingSphere->z = position.z + m_boundingSphere->z - m_data->position.z;
			m_data->position = position;
		}
		void changeDirection(const glm::vec3& lightDir)
		{
			m_data->lightDir = glm::normalize(lightDir);
			calculateBoundingSphere();
		}
		void changeLength(float length)
		{
			m_data->length = length;
			calculateBoundingSphere();
		}
		void changeCutoff(float cutoffAngle)
		{
			m_data->cutoffCos = std::cos(std::min(cutoffAngle, static_cast<float>(M_PI_2)));
			calculateBoundingSphere();
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