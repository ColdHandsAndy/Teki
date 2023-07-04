#version 460

layout(set = 0, binding = 0) uniform UBO
{
    mat4 view;
    mat4 proj;
} skyboxTransform;

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 outUVW;

void main() 
{
    outUVW = position;
    vec4 vertPos = vec4(position, 1.0);
    vertPos = skyboxTransform.proj * skyboxTransform.view * vertPos;
    gl_Position = vertPos;
}