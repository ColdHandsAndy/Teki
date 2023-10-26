#version 460

layout(set = 0, binding = 0) uniform Viewproj
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 outUVW;

void main() 
{
    outUVW = position;
    vec4 vertPos = vec4(position, 1.0);
    mat4 skyboxView = viewproj.view;
    skyboxView[3] = vec4(0.0, 0.0, 0.0, 1.0);
    vertPos = viewproj.proj * skyboxView * vertPos;
    gl_Position = vertPos;
}