#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 lineColor;
layout(location = 1) out vec3 fragPos;

layout(set = 0, binding = 0) uniform Viewproj 
{
    mat4 view;
    mat4 proj;
} viewproj;

void main() 
{
    fragPos = position;
    lineColor = color * 0.8;
    gl_Position = viewproj.proj * viewproj.view * vec4(position, 1.0);
}