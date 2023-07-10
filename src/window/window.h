#ifndef WINDOW_CLASS_HEADER
#define WINDOW_CLASS_HEADER

#include <string>
#include <GLFW/glfw3.h>

class Window
{
private:
	GLFWwindow* m_window{ nullptr };

	uint32_t m_width{};
	uint32_t m_height{};

private:

public:
	Window() = default;
	Window(uint32_t width, uint32_t height, std::string windowName);
	Window(std::string windowName);

	~Window() 
	{

	};

	uint32_t getWidth() const;
	uint32_t getHeight() const;

	operator GLFWwindow* () const;
};

#endif