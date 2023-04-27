#version 460

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() 
{
    vec3 colors[4] = { vec3(0, 0.4, 0.85), vec3(0.67, 0, 0), vec3(0.6, 1.0, 0.6), vec3(0, 0, 0) };
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);

    fragColor = colors[gl_VertexIndex % 4];
}