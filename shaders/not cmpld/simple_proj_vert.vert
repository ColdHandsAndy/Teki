#version 460

layout(set = 0, binding = 0) uniform ViewProjMatrices
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(location = 0) in vec3 position;
layout(location = 0) out vec3 outColor;

void main() 
{
    vec4 vertPos = viewproj.proj * viewproj.view * vec4(position, 1.0);
    gl_Position = vertPos;
	outColor = vec3(0.03, 0.98, 0.1);
}