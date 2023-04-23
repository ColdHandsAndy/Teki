#ifndef CAMERA_HEADER
#define CAMERA_HEADER

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Camera
{
public:
	enum Direction
	{
		FORWARD,
		BACKWARD,
		LEFT,
		RIGHT,
		all_directions
	};
private:
	glm::vec3 m_cameraPosition{ 0.0f, 0.0f, 1.0f };
	glm::vec3 m_worldUp{ 0.0f, 1.0f, 0.0f };
	glm::vec3 m_cameraUp{ 0.0f, 1.0f, 0.0f };
	glm::vec3 m_cameraFront{ 0.0f, 0.0f, -1.0f };
	glm::vec3 m_cameraRight{ 1.0f, 0.0f, 0.0f };

	float m_yaw{-glm::half_pi<float>()};
	float m_pitch{ 0.0f };

	float m_movementSpeed{ 1.5f };
	float m_turningSpeed{ 0.002f };

public:
	Camera() = default;
	Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch);

	glm::vec3 currentPos() const;
	glm::vec3 getCamFront() const;

	float curMovementSpeed();
	float curTurningSpeed();
	void changeMovementSpeed(float newSpeed);
	void changeTurningSpeed(float newSpeed);

	void changePositiom(const glm::vec3& newPos);

	void processKeyboard(Camera::Direction direction);

	void processMouseMovement(float xoffset, float yoffset);

	glm::mat4 getViewMatrix();

private:
	void updateCamVectors();

};

#endif