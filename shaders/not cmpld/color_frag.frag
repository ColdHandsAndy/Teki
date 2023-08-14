#version 460

layout(location = 0) in vec3 inpColor;
layout(location = 0) out vec4 finalColor;

void main()
{
	finalColor = vec4(inpColor, 1.0);
}