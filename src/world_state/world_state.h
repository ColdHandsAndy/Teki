#ifndef WORLD_STATE_HEADER
#define WORLD_STATE_HEADER

#include <GLFW/glfw3.h>

namespace WorldState
{
	inline double deltaTime;
	inline double lastFrame;

	inline double getDeltaTime()
	{
		return deltaTime;
	}

	inline void initialize()
	{
		deltaTime = 0.0;
		lastFrame = glfwGetTime();
	}

	inline void refreshFrameTime()
	{
		double currentTime{ glfwGetTime() };
		deltaTime = currentTime - lastFrame;
		lastFrame = currentTime;
	}
}

#endif