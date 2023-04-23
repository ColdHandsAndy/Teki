#ifndef RENDERER_CLASS_HEADER
#define RENDERER_CLASS_HEADER

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "src/window/window.h"

class [[nodiscard]] Renderer
{
private:
    Window m_window;

public:
	Renderer();
	~Renderer();
};

#endif 
