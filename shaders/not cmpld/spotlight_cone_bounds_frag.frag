#version 460

layout(location = 0) out vec4 outFragColor;

void main() 
{
	vec3 lineColor = vec3(0.705, 0.298, 0.941);
	outFragColor = vec4(lineColor, 1.0);
}