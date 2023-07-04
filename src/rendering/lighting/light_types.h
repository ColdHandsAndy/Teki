#ifndef LIGHTING_TYPES_HEADER
#define LIGHTING_TYPES_HEADER


#include <cmath>
#include <cstdint>

#include <glm/glm.hpp>

#include "src/rendering/renderer/clusterer.h"

namespace LightTypes
{
//	//General light types. Could be used in the future
//	static class LightBase
//	{
//	protected:
//		glm::vec3 m_lightColor{};
//		float m_lightPower{};
//
//		LightBase() = default;
//		LightBase(glm::vec3 lightColor, float lightPower) : m_lightColor{ lightColor }, m_lightPower{ lightPower } {}
//
//	public:
//		const glm::vec3& getColor() const
//		{
//			return m_lightColor;
//		}
//		float getLightPower() const
//		{
//			return m_lightPower;
//		}
//
//		virtual void changeColor(const glm::vec3& lightColor)
//		{
//			m_lightColor = lightColor;
//		}
//		virtual void changePower(float lightPower)
//		{
//			m_lightPower = lightPower;
//		}
//
//	};
//
//	class DirectionalLight : public LightBase
//	{
//	private:
//		glm::vec3 m_lightDir{};
//
//		struct DirectionalLightData
//		{
//			alignas(16) glm::vec3 spectrum{};
//			alignas(16) glm::vec3 lightDir{};
//		};
//
//	public:
//		DirectionalLight(glm::vec3 lightColor, float lightPower, glm::vec3 lightDir) : LightBase{ lightColor, lightPower }, m_lightDir{ glm::normalize(lightDir)} {}
//
//		static uint32_t getDataByteSize()
//		{
//			return sizeof(DirectionalLightData);
//		}
//		void plantData(void* dataPtr) const
//		{
//			DirectionalLightData dataToPass{ m_lightColor * m_lightPower, m_lightDir };
//			std::memcpy(dataPtr, &dataToPass, sizeof(dataToPass));
//		}
//
//		void changeDirection(const glm::vec3& lightDir)
//		{
//			m_lightDir = lightDir;
//		}
//
//	};
//
//	class PointLight : public LightBase
//	{
//	private:
//		glm::vec3 m_worldPos{};
//		float m_radius{};
//		
//		/*struct PointLightData
//		{
//			alignas(16) glm::vec4 position_length{};
//			alignas(16) glm::vec3 spectrum{};
//		};*/
//	public:
//		PointLight(glm::vec3 worldPos, glm::vec3 lightColor, float lightPower, float radius) : LightBase{ lightColor, lightPower }, m_worldPos{ worldPos }, m_radius{ radius } {}
//
//		const glm::vec3& getPosition() const
//		{
//			return m_worldPos;
//		}
//		float getRadius() const
//		{
//			return m_radius;
//		}
//		static uint32_t getDataByteSize()
//		{
//			return sizeof(PointLightData);
//		}
//		void plantData(void* dataPtr) const
//		{
//			PointLightData dataToPass{ glm::vec4(m_worldPos, m_radius), m_lightColor * m_lightPower };
//			std::memcpy(dataPtr, &dataToPass, sizeof(dataToPass));
//		}
//
//		void changePos(const glm::vec3& newPos)
//		{
//			m_worldPos = newPos;
//		}
//		void changeColor(const glm::vec3& lightColor) override
//		{
//			m_lightColor = lightColor;
//		}
//		void changePower(float lightPower) override
//		{
//			m_lightPower = lightPower;
//		}
//
//	};
//
//	class SpotLight : public LightBase
//	{
//	private:
//		glm::vec3 m_worldPos{};
//		glm::vec3 m_lightDirection{};
//		float m_lightLength{};
//		float m_startCutoffCos{};
//		float m_endCutoffCos{};
//		float m_angle{};
//
//		/*struct SpotLightData
//		{
//			alignas(16) glm::vec4 position_radius{};
//			alignas(16) glm::vec4 spectrum_startCutoffCos{};
//			alignas(16) glm::vec4 lightDir_endCutoffCos{};
//		};*/
//
//	public:
//		SpotLight(glm::vec3 worldPos, glm::vec3 lightColor, float lightPower, float length, glm::vec3 lightDir, float cutoffStartAngle, float cutoffEndAngle) 
//			: LightBase{ lightColor, lightPower }, m_worldPos{ worldPos }, m_lightDirection{ glm::normalize(lightDir) }, m_lightLength{ length }, m_angle{ cutoffEndAngle }
//		{
//			m_endCutoffCos = std::cos(std::min(cutoffEndAngle, static_cast<float>(M_PI_2)));
//			m_startCutoffCos = std::cos(std::min(std::min(cutoffStartAngle, cutoffEndAngle), static_cast<float>(M_PI_2)));
//		}
//
//		const glm::vec3& getDirection() const
//		{
//			return m_lightDirection;
//		}
//		float getLength() const
//		{
//			return m_lightLength;
//		}
//		const glm::vec3& getPosition() const
//		{
//			return m_worldPos;
//		}
//		float getAngle() const
//		{
//			return m_angle;
//		}
//		float getCos() const
//		{
//			return m_endCutoffCos;
//		}
//		static uint32_t getDataByteSize()
//		{
//			return sizeof(SpotLightData);
//		}
//		void plantData(void* dataPtr) const
//		{
//			SpotLightData dataToPass{ glm::vec4(m_worldPos, m_lightLength), glm::vec4{ m_lightColor * m_lightPower, m_startCutoffCos }, glm::vec4{ m_lightDirection, m_endCutoffCos } };
//			std::memcpy(dataPtr, &dataToPass, sizeof(dataToPass));
//		}
//		void changeDirection(const glm::vec3& lightDir)
//		{
//			m_lightDirection = lightDir;
//		}
//
//		void changePos(const glm::vec3& newPos)
//		{
//			m_worldPos = newPos;
//		}
//		void changeColor(const glm::vec3& lightColor) override
//		{
//			m_lightColor = lightColor;
//		}
//		void changePower(float lightPower) override
//		{
//			m_lightPower = lightPower;
//		}
//
//	};

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
			ASSERT_ALWAYS(m_clusterer != nullptr, "App", "Global Clusterer has not been assigned.");
			m_clusterer->getNewLight(m_data, m_boundingSphere, Clusterer::LightFormat::TYPE_POINT);
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
			ASSERT_ALWAYS(m_clusterer != nullptr, "App", "Global Clusterer has not been assigned.");
			m_clusterer->getNewLight(m_data, m_boundingSphere, Clusterer::LightFormat::TYPE_SPOT);
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