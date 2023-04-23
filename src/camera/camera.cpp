#include "src/camera/camera.h"
#include "src/world_state_class/world_state.h"

#include <iomanip>

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch )
{
	m_cameraPosition = position;
	m_worldUp = up;
	m_yaw = yaw;
	m_pitch = pitch;
	updateCamVectors();
}

glm::vec3 Camera::currentPos() const
{
	return m_cameraPosition;
}

glm::vec3 Camera::getCamFront() const
{
	return m_cameraFront;
}

float Camera::curMovementSpeed()
{
	return m_movementSpeed;
}
float Camera::curTurningSpeed()
{
	return m_turningSpeed;
}
void Camera::changeMovementSpeed(float newSpeed)
{
	m_movementSpeed = newSpeed;
}
void Camera::changeTurningSpeed(float newSpeed)
{
	m_turningSpeed = newSpeed;
}

void Camera::changePositiom(const glm::vec3& newPos)
{
	m_cameraPosition = newPos;
}

void Camera::processKeyboard(Camera::Direction direction)
{
	float velocity = m_movementSpeed * WorldState::getDeltaTime();
	
	if (direction == Direction::FORWARD)
	{
		m_cameraPosition += m_cameraFront * velocity;
	}
	if (direction == Direction::BACKWARD)
	{
		m_cameraPosition -= m_cameraFront * velocity;
	}
	if (direction == Direction::LEFT)
	{
		m_cameraPosition -= m_cameraRight * velocity;
	}
	if (direction == Direction::RIGHT)
	{
		m_cameraPosition += m_cameraRight * velocity;
	}
}

void Camera::processMouseMovement(float xoffset, float yoffset)
{
	xoffset *= m_turningSpeed;
	yoffset *= m_turningSpeed;

	m_yaw += xoffset;
	m_pitch += yoffset;

	if (m_pitch > glm::radians(89.0f))
		m_pitch = glm::radians(89.0f);
	if (m_pitch < -glm::radians(89.0f))
		m_pitch = -glm::radians(89.0f);

	updateCamVectors();
}

glm::mat4 Camera::getViewMatrix()
{
	return glm::lookAt(m_cameraPosition, m_cameraPosition + m_cameraFront, m_cameraUp);
}

void Camera::updateCamVectors()
{
	glm::vec3 newFront{};
	newFront.x = glm::cos(m_pitch) * glm::cos(m_yaw);
	newFront.y = glm::sin(m_pitch);
	newFront.z = glm::sin(m_yaw) * glm::cos(m_pitch);
	m_cameraFront = glm::normalize(newFront);

	m_cameraRight = glm::normalize(glm::cross(m_cameraFront, m_worldUp));
	m_cameraUp = glm::cross(m_cameraRight, m_cameraFront);
}