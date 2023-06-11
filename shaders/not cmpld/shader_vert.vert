#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable

layout(location = 0) in vec3 position;
layout(location = 1) in uint packedNormals4x8;
layout(location = 2) in uint packedTangents4x8;
layout(location = 3) in uint packedTexCoords2x16;

layout(location = 0) out flat uint drawID;
layout(location = 1) out vec3 outPos;
layout(location = 2) out vec3 outNorm;
layout(location = 3) out vec4 outTang;
layout(location = 4) out vec2 outTexC;



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
    DrawDataIndices indices[];
} drawDataIndices;


void main() 
{
    drawID = gl_DrawID;
	
    mat4 modelmat = modelMatrices.modelMatrices[drawDataIndices.indices[gl_DrawID].modelIndex];
    vec4 worldPos = modelmat * vec4(position, 1.0);
    gl_Position = viewproj.proj * viewproj.view * worldPos;

    vec3 norm = vec3(unpackSnorm4x8(packedNormals4x8));
    vec4 tang = vec4(unpackSnorm4x8(packedTangents4x8));
	
	outPos = vec3(worldPos);
    outNorm = normalize(mat3(modelmat) * vec3(norm.r, norm.g, norm.b));
    outTang = vec4(normalize(mat3(modelmat) * vec3(tang.r, tang.g, tang.b)), tang.w);
	outTexC = unpackHalf2x16(packedTexCoords2x16);
}