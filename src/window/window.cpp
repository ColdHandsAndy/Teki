#include "src/window/window.h"

Window::Window(uint32_t width, uint32_t height, std::string windowName) : m_width{ width }, m_height{ height }
{
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	m_window = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
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