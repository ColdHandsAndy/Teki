#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable

layout(location = 0) in vec3 position;
layout(location = 1) in uint packedNormals4x8;
layout(location = 2) in uint packedTangents4x8;
layout(location = 3) in uint packedTexCoords2x16;

layout(location = 0) out flat uint drawID;
layout(location = 1) out vec3 normal;
layout(location = 2) out vec3 tangent;
layout(location = 3) out vec2 texCoords;
layout(location = 4) out vec3 fragmentPos;



layout(set = 0, binding = 0) uniform ViewProjMatrices
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(set = 1, binding = 1) uniform ModelMatrices 
{
    mat4 modelMatrices[8];
} modelMatrices;

struct DrawDataIndices
{
    uint8_t modelIndex;
    uint8_t index1;
    uint8_t index2;
    uint8_t index3;
    uint8_t bcIndexList;
    uint8_t bcIndexLayer;
    uint8_t nmIndexList;
    uint8_t nmIndexLayer;
    uint8_t mrIndexList;
    uint8_t mrIndexLayer;
    uint8_t emIndexList;
    uint8_t emIndexLayer;
};
layout(set = 2, binding = 0) buffer DrawDataIndicesBuffer 
{
    DrawDataIndices drawData[];
} drawDataIndices;


void main() 
{
    mat4 modelmat = modelMatrices.modelMatrices[drawDataIndices.drawData[gl_DrawID].modelIndex];
    vec4 worldPos = modelmat * vec4(position, 1.0);
    fragmentPos = vec3(worldPos);
    vec4 viewPos = viewproj.view * worldPos;
    gl_Position = viewproj.proj * viewPos;

    vec3 norm = vec3(unpackSnorm4x8(packedNormals4x8));
    vec3 tang = vec3(unpackSnorm4x8(packedTangents4x8));

    drawID = gl_DrawID;
    normal = mat3(modelmat) * vec3(norm.r, norm.b, -norm.g);
    //tangent = vec3(tang.r, tang.b, -tang.g);
    tangent = tang;
    texCoords = unpackHalf2x16(packedTexCoords2x16);
}