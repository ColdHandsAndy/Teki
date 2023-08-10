#include "src/window/window.h"

Window::Window(uint32_t width, uint32_t height, std::string windowName, bool borderless) : m_width{ width }, m_height{ height }
{
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	glfwWindowHint(GLFW_DECORATED, !borderless);

	m_window = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
}

Window::Window(std::string windowName)
{
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	GLFWmonitor* monitor{ glfwGetPrimaryMonitor() };
	const GLFWvidmode* mode{ glfwGetVideoMode(monitor) };
	glfwWindowHint(GLFW_RED_BITS, mode->redBits);
	glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
	glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
	glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
	m_window = glfwCreateWindow(mode->width, mode->height, windowName.c_str(), monitor, NULL);
	m_width = mode->width;
	m_height = mode->height;
}

uint32_t Window::getWidth() const
{
	return m_width;
}

uint32_t Window::getHeight() const
{
	return m_height;
}

Window::operator GLFWwindow* () const
{
	return m_window;
}