#version 460

layout (set = 1, binding = 0) uniform samplerCube samplerCubeMap;


layout (location = 0) in vec3 inUVW;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	vec4 color = texture(samplerCubeMap, inUVW);
	gl_FragDepth = 0.0;
	outFragColor = color;
}