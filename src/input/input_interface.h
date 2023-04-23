#ifndef INPUT_INTERFACE_CLASS_HEADER
#define INPUT_INTERFACE_CLASS_HEADER

#include <GLFW/glfw3.h>

#include "src/window/window.h"

class InputInterface
{
private:
	Window* m_window{ nullptr };

public:
	InputInterface() = delete;
	InputInterface(Window& window);
	~InputInterface();



};

#endif
