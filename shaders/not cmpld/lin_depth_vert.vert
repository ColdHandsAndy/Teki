#version 460

#extension GL_GOOGLE_include_directive						:  enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable

#include "misc.h"

layout(location = 0) in vec3 position;
layout(location = 1) in uint packedNormals4x8;
layout(location = 2) in uint packedTangents4x8;
layout(location = 3) in uint packedTexCoords2x16;

layout(location = 0) out float outDepth;


layout(set = 0, binding = 0) uniform ViewProjMatrices
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(set = 0, binding = 1) uniform ModelMatrices 
{
    mat4 modelMatrices[8];
} modelMatrices;

layout(set = 0, binding = 2) buffer DrawDataIndicesBuffer 
{
    DrawData indices[];
} drawDataIndices;


void main() 
{
    mat4 modelmat = modelMatrices.modelMatrices[drawDataIndices.indices[gl_DrawID].modelIndex];
    vec4 worldPos = modelmat * vec4(position, 1.0);
    gl_Position = viewproj.proj * viewproj.view * worldPos;

	outDepth = gl_Position.w;
}