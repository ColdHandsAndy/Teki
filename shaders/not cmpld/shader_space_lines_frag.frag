#version 460

layout(push_constant) uniform PushConsts 
{
	vec3 camPos;
} pushConstants;

layout(location = 0) in vec3 lineColor;
layout(location = 1) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

void main() 
{
	float distanceToPoint = distance(fragPos.xz, pushConstants.camPos.xz);
	vec4 color = vec4(lineColor, 1.0);
	if (distanceToPoint > 30.0)
		color.w = -(1.0/15.0) * distanceToPoint + 3.0;
    outColor = color;
}