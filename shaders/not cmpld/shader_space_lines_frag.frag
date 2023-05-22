#version 460

layout(location = 0) in vec3 lineColor;

layout(location = 0) out vec4 outColor;

void main() 
{
    outColor = vec4(lineColor, 1.0);
}