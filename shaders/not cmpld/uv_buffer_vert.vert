#version 460

#extension GL_GOOGLE_include_directive						:  enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8     :  enable

#include "misc.h"

layout(location = 0) in vec3 position;
layout(location = 1) in uint packedNormals4x8;
layout(location = 2) in uint packedTangents4x8;
layout(location = 3) in uint packedTexCoords2x16;

layout(location = 0) out flat uint drawID;
layout(location = 1) out vec3 outNorm;
layout(location = 2) out vec3 outTang;
layout(location = 3) out flat float outTangSign;
layout(location = 4) out vec2 outTexC;

#define COORDINATE_TRANSFORMATION_SET_INDEX 0
#include "coordinate_transformation_set.h"

layout(set = 1, binding = 0) buffer ModelMatrices 
{
    mat4 modelMatrices[];
} modelMatrices;

layout(set = 2, binding = 0) buffer DrawDataBuffer 
{
    DrawData data[];
} drawData;
layout(set = 2, binding = 1) buffer DrawDataIndexBuffer 
{
    uint data[];
} drawDataIndices;

void main() 
{
	drawID = drawDataIndices.data[gl_DrawID];
	
    mat4 modelmat = modelMatrices.modelMatrices[drawData.data[drawID].modelIndex];
    gl_Position = coordTransformData.ndcFromWorld * modelmat * vec4(position, 1.0);

    vec3 norm = vec3(unpackSnorm4x8(packedNormals4x8));
    vec4 tang = vec4(unpackSnorm4x8(packedTangents4x8));
	
	outNorm = normalize(mat3(modelmat) * norm);
	outTang = normalize(mat3(modelmat) * tang.xyz);
	outTangSign = tang.w;
	outTexC = unpackHalf2x16(packedTexCoords2x16);
}