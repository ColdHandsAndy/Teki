#version 460

layout(location = 0) out vec4 outFragColor;

void main() 
{
	vec3 lineColor = vec3(0.913, 0.447, 0.054) * 0.5;
	outFragColor = vec4(lineColor, 1.0);
}