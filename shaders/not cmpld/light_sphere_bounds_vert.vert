#version 460

layout(set = 0, binding = 0) uniform ViewprojTransform
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(location = 0) in vec3 vertPosition;
layout(location = 1) in vec4 lightPos_lightRadius;

void main() 
{
    vec3 lightVertPos = (vertPosition * lightPos_lightRadius.w) + lightPos_lightRadius.xyz;
    gl_Position = viewproj.proj * viewproj.view * vec4(lightVertPos, 1.0);
}