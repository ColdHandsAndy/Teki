#ifndef LIGHTING_TYPES_HEADER
#define LIGHTING_TYPES_HEADER

#define NEGLIGIBLE_LIGHT_CONTRIBUTION 0.02f

#include <cmath>
#include <cstdint>

#include "glm/glm.hpp"

namespace LightTypes
{
	static class LightBase
	{
	protected:
		glm::vec3 m_lightColor{};
		float m_lightPower{};

		LightBase() = default;
		LightBase(glm::vec3 lightColor, float lightPower) : m_lightColor{ lightColor }, m_lightPower{ lightPower } {}

	public:
		const glm::vec3& getColor()
		{
			return m_lightColor;
		}
		float getLightPower()
		{
			return m_lightPower;
		}

		virtual void changeColor(const glm::vec3& lightColor)
		{
			m_lightColor = lightColor;
		}
		virtual void changePower(float lightPower)
		{
			m_lightPower = lightPower;
		}

	};

	class DirectionalLight : public LightBase
	{
	private:
		glm::vec3 m_lightDir{};

		struct DirectionalLightData
		{
			alignas(16) glm::vec3 spectrum{};
			alignas(16) glm::vec3 lightDir{};
		};

	public:
		DirectionalLight(glm::vec3 lightColor, float lightPower, glm::vec3 lightDir) : LightBase{ lightColor, lightPower }, m_lightDir{ glm::normalize(lightDir)} {}

		static uint32_t getDataByteSize()
		{
			return sizeof(DirectionalLightData);
		}
		void plantData(void* dataPtr)
		{
			DirectionalLightData dataToPass{ m_lightColor * m_lightPower, m_lightDir };
			std::memcpy(dataPtr, &dataToPass, sizeof(dataToPass));
		}

		void changeDirection(const glm::vec3& lightDir)
		{
			m_lightDir = lightDir;
		}

	};

	class PointLight : public LightBase
	{
	private:
		glm::vec3 m_worldPos{};
		float m_radius{};
		
		struct PointLightData
		{
			alignas(16) glm::vec3 position{};
			alignas(16) glm::vec3 spectrum{};
		};
	public:
		PointLight(glm::vec3 worldPos, glm::vec3 lightColor, float lightPower) : LightBase{ lightColor, lightPower }, m_worldPos{ worldPos }
		{
			evaluateLightRadius(lightColor, lightPower);
		}

		const glm::vec3& getPosition()
		{
			return m_worldPos;
		}
		float getRadius()
		{
			return m_radius;
		}
		static uint32_t getDataByteSize()
		{
			return sizeof(PointLightData);
		}
		void plantData(void* dataPtr)
		{
			PointLightData dataToPass{ m_worldPos, m_lightColor * m_lightPower };
			std::memcpy(dataPtr, &dataToPass, sizeof(dataToPass));
		}

		void changePos(const glm::vec3& newPos)
		{
			m_worldPos = newPos;
		}
		void changeColor(const glm::vec3& lightColor) override
		{
			m_lightColor = lightColor;
			evaluateLightRadius(m_lightColor, m_lightPower);
		}
		void changePower(float lightPower) override
		{
			m_lightPower = lightPower;
			evaluateLightRadius(m_lightColor, m_lightPower);
		}

	private:
		void evaluateLightRadius(const glm::vec3& lightColor, float lightPower)
		{
			m_radius = std::sqrt(std::max({ lightColor.x, lightColor.y, lightColor.z }) * lightPower / NEGLIGIBLE_LIGHT_CONTRIBUTION);
		}

	};

	class SpotLight : public LightBase
	{
	private:
		glm::vec3 m_worldPos{};
		glm::vec3 m_lightDirection{};
		float m_lightLength{};
		float m_startCutoffCos{};
		float m_endCutoffCos{};

		struct SpotLightData
		{
			alignas(16) glm::vec3 position{};
			alignas(16) glm::vec4 spectrum_startCutoffCos{};
			alignas(16) glm::vec4 lightDir_endCutoffCos{};
		};

	public:
		SpotLight(glm::vec3 worldPos, glm::vec3 lightColor, float lightPower, glm::vec3 lightDir, float cutoffStartAngle, float cutoffEndAngle) : LightBase{ lightColor, lightPower }, m_worldPos{ worldPos }, m_lightDirection{ glm::normalize(lightDir) }
		{
			m_startCutoffCos = std::cos(std::min(cutoffStartAngle, static_cast<float>(M_PI_2)));
			m_endCutoffCos = std::cos(std::min(cutoffEndAngle, static_cast<float>(M_PI_2)));
			evaluateLightLength(m_lightColor, m_lightPower);
		}

		const glm::vec3& getDirection()
		{
			return m_lightDirection;
		}
		float getLength()
		{
			return m_lightLength;
		}
		const glm::vec3& getPosition()
		{
			return m_worldPos;
		}
		float getAngle()
		{
			return std::acos(m_endCutoffCos);
		}
		static uint32_t getDataByteSize()
		{
			return sizeof(SpotLightData);
		}
		void plantData(void* dataPtr)
		{
			SpotLightData dataToPass{ m_worldPos, glm::vec4{ m_lightColor * m_lightPower, m_startCutoffCos }, glm::vec4{ m_lightDirection, m_endCutoffCos } };
			std::memcpy(dataPtr, &dataToPass, sizeof(dataToPass));
		}
		void changeDirection(const glm::vec3& lightDir)
		{
			m_lightDirection = lightDir;
		}

		void changePos(const glm::vec3& newPos)
		{
			m_worldPos = newPos;
		}
		void changeColor(const glm::vec3& lightColor) override
		{
			m_lightColor = lightColor;
			evaluateLightLength(m_lightColor, m_lightPower);
		}
		void changePower(float lightPower) override
		{
			m_lightPower = lightPower;
			evaluateLightLength(m_lightColor, m_lightPower);
		}

	private:
		void evaluateLightLength(const glm::vec3& lightColor, float lightPower)
		{
			m_lightLength = std::sqrt(std::max({ lightColor.x, lightColor.y, lightColor.z }) * lightPower / NEGLIGIBLE_LIGHT_CONTRIBUTION);
		}

	};
}

#endif