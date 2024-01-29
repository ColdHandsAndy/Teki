#ifndef CAMERA_HEADER
#define CAMERA_HEADER

#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "src/tools/projection.h"
#include "src/tools/asserter.h"

class Camera
{
private:
	glm::vec3 m_cameraPosition{ 0.0, 0.0, -10.0 };
	glm::vec3 m_cameraForwardDirection{ 0.0, 0.0, 1.0 };
	glm::vec3 m_cameraUpDirection{ 0.0, 1.0, 0.0 };
	glm::vec3 m_cameraSideDirection{ 1.0, 0.0, 0.0 };

	float m_speed{ 10.0 };
	float m_sensitivity{ 3.4 };

	glm::vec2 m_last2DInput{};
	bool m_cameraPositionChanged{ false };

	float m_near{};
	float m_far{};
	float m_FOV{};
	float m_aspect{};

public:
	
	enum Direction
	{
		RIGHT,
		LEFT,
		UP,
		DOWN,
		FORWARD,
		BACK,
	};


public:
	Camera() = default;
	Camera(float near, float far, float FOV, float aspect)
		: m_near{ near }, m_far{ far }, m_FOV{ FOV }, m_aspect{ aspect }
	{

	}

	const glm::vec3& getPosition() const
	{
		return m_cameraPosition;
	}
	const glm::vec3& getForwardDirection() const
	{
		return m_cameraForwardDirection;
	}
	const glm::vec3& getUpDirection() const
	{
		return m_cameraUpDirection;
	}

	const float getSpeed() const { return m_speed; };
	const float getSensitivity() const { return m_sensitivity; };
	const float getNear() const { return m_near; };
	const float getFar() const { return m_far; };
	const float getFOV() const { return m_FOV; };
	const float getAspect() const { return m_aspect; };
	const bool cameraPositionChanged() 
	{ 
		return m_cameraPositionChanged;
	};

	void setPosition(const glm::vec3& vec) { m_cameraPosition = vec; };
	void setForwardDirection(const glm::vec3& vec) { m_cameraForwardDirection = vec; };
	void setSpeed(float a) { m_speed = a; };
	void setSensetivity(float a) { m_sensitivity = a; };
	void setNear(float a) { m_near = a; };
	void setFar(float a) { m_far = a; };
	void setFOV(float a) { m_FOV = a; };
	void setAspect(float a) { m_aspect = a; };
	void setCameraPositionLeftUnchanged()
	{
		m_cameraPositionChanged = false;
	}

	void move(Direction dir, float deltaTime)
	{
		switch (dir)
		{
		case Direction::RIGHT:
			m_cameraPosition += m_cameraSideDirection * m_speed * deltaTime;
			break;
		case Direction::LEFT:
			m_cameraPosition -= m_cameraSideDirection * m_speed * deltaTime;
			break;
		case Direction::UP:
			m_cameraPosition += m_cameraUpDirection * m_speed * deltaTime;
			break;
		case Direction::DOWN:
			m_cameraPosition -= m_cameraUpDirection * m_speed * deltaTime;
			break;
		case Direction::FORWARD:
			m_cameraPosition += m_cameraForwardDirection * m_speed * deltaTime;
			break;
		case Direction::BACK:
			m_cameraPosition -= m_cameraForwardDirection * m_speed * deltaTime;
			break;
		default:
			EASSERT(false, "App", "Direction enum is unknown.");
			break;
		}

		m_cameraPositionChanged = true;
	}

	void moveFrom2DInput(const glm::vec2& normalizedInput, bool invalidateLastPos)
	{
		glm::vec2 inputDelta{ invalidateLastPos ? glm::vec2(0.0) : normalizedInput - m_last2DInput };

		float pitch{ inputDelta.y * m_sensitivity };
		float yaw{ inputDelta.x * m_sensitivity };

		glm::quat qp{ glm::angleAxis(pitch, m_cameraSideDirection) };
		glm::quat qy{ glm::angleAxis(yaw, m_cameraUpDirection) };

		glm::quat q{ qp * qy };

		glm::vec3 newDir{ glm::mat3_cast(q) * m_cameraForwardDirection };

		if (glm::abs(glm::dot(newDir, m_cameraUpDirection)) < 0.999)
		{
			m_cameraForwardDirection = newDir;
			m_cameraSideDirection = glm::normalize(glm::cross(m_cameraUpDirection, m_cameraForwardDirection));
		}

		m_last2DInput = normalizedInput;
	}
};

#endif