#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable

layout(location = 0) in vec3 position;
layout(location = 1) in uint packedNormals4x8;
layout(location = 2) in uint packedTangents4x8;
layout(location = 3) in uint packedTexCoords2x16;

layout(location = 0) out vec2 texCoords;
layout(location = 1) out flat uint listIndex;
layout(location = 2) out flat uint layerIndex;

layout(set = 0, binding = 0) uniform UBO1 
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(set = 1, binding = 0) uniform UBO2 
{
    mat4 transMatrices[8];
} translations;

struct DrawDataIndices
{
    uint8_t index0;
    uint8_t index1;
    uint8_t index2;
    uint8_t index3;
};
layout(set = 3, binding = 0) buffer SSBO1 
{
    DrawDataIndices drawData[];
} drawDataIndices;


void main() 
{
    vec3 norm = vec3(unpackSnorm4x8(packedNormals4x8));

    gl_Position = viewproj.proj * viewproj.view * translations.transMatrices[drawDataIndices.drawData[gl_DrawID].index0] * vec4(position, 1.0);

    listIndex = drawDataIndices.drawData[gl_DrawID].index1;
    layerIndex = drawDataIndices.drawData[gl_DrawID].index2;
    texCoords = unpackHalf2x16(packedTexCoords2x16);
}