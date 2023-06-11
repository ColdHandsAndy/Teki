#version 460

layout (set = 0, binding = 1) uniform samplerCube samplerCubeMap;


layout (location = 0) in vec3 inUVW;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec4 color = texture(samplerCubeMap, inUVW);
	color.w = 1.0;
	outFragColor = color;
}