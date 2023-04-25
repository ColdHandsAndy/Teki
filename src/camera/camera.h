#ifndef CAMERA_HEADER
#define CAMERA_HEADER

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/ext/quaternion_transform.hpp>

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

	float m_movementSpeed{ 1.5f };
	float m_turningSpeed{ 0.002f };

public:
	Camera() = default;
	Camera(glm::vec3 position, glm::vec3 front);

	glm::vec3 currentPos() const;
	glm::vec3 getCamFront() const;

	glm::mat4 getViewMatrix();

private:
	void updateCamVectors();

};

#endif