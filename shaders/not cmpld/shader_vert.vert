#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in uint packedNormals4x8;
layout(location = 2) in uint packedTangents4x8;
layout(location = 3) in uint packedTexCoords2x16;

layout(location = 0) out vec3 fragColor;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() 
{
    vec3 norm = vec3(unpackSnorm4x8(packedNormals4x8));

    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position, 1.0);

    fragColor = norm;
}