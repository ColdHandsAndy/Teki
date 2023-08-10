#ifndef PROJECTION_HEADER
#define PROJECTION_HEADER

#include <cmath>

#include <glm/glm.hpp>

glm::mat4 getProjectionRZ(float FOV, float aspect, float zNear, float zFar)
{
	float h{ static_cast<float>(1.0 / std::tan((FOV * 0.5))) };
	float w{ h / aspect };
	float a{ -zNear / (zFar - zNear) };
	float b{ (zNear * zFar) / (zFar - zNear) };

	glm::mat4 mat{
		glm::vec4(w, 0.0, 0.0, 0.0),
		glm::vec4(0.0, -h, 0.0, 0.0),
		glm::vec4(0.0, 0.0, a, 1.0),
		glm::vec4(0.0, 0.0, b, 0.0)
	};
	return mat;
}

glm::mat4 getProjection(float FOV, float aspect, float zNear, float zFar)
{
	float h{ static_cast<float>(1.0 / std::tan((FOV * 0.5))) };
	float w{ h / aspect };
	float a{ zFar / (zFar - zNear) };
	float b{ (-zNear * zFar) / (zFar - zNear) };

	glm::mat4 mat{
		glm::vec4(w, 0.0, 0.0, 0.0),
		glm::vec4(0.0, -h, 0.0, 0.0),
		glm::vec4(0.0, 0.0, a, 1.0),
		glm::vec4(0.0, 0.0, b, 0.0)
	};
	return mat;
}

#endif