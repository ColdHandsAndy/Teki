#version 460

#extension GL_GOOGLE_include_directive						:  enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8     :  enable

#include "misc.h"

layout(location = 0) in vec3 position;
layout(location = 1) in uint packedNormals4x8;
layout(location = 2) in uint packedTangents4x8;
layout(location = 3) in uint packedTexCoords2x16;

layout(push_constant) uniform PushConsts 
{
	float halfSide;
	uint resolutionBOM;
	uint resolutionVM;
} pushConstants;

layout(set = 0, binding = 0) buffer ModelMatrices 
{
    mat4 modelMatrices[];
} modelMatrices;

layout(set = 1, binding = 0) buffer DrawDataBuffer 
{
    DrawData data[];
} drawData;


vec3 transformOrthographicallyAxisAlignedCube(vec3 vertPos, vec3 center, float halfSide)
{
	return (vertPos - center) / halfSide;
}

layout(location = 0) out vec2 outTexCoords;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out flat uint out_bcList_bcLayer_emList_emLayer;
layout(location = 3) out flat uint out_mrList_mrLayer;

void main() 
{
    DrawData drawdata = drawData.data[gl_DrawID];

    out_bcList_bcLayer_emList_emLayer = (uint(drawdata.bcIndexList) << (8 * 3)) | (uint(drawdata.bcIndexLayer) << (8 * 2)) | (uint(drawdata.emIndexList) << (8 * 1)) | (uint(drawdata.emIndexLayer));
    out_mrList_mrLayer = (uint(drawdata.mrIndexList) << (8 * 1)) | (uint(drawdata.mrIndexLayer));

    mat4 modelmat = modelMatrices.modelMatrices[drawdata.modelIndex];
    gl_Position = vec4(transformOrthographicallyAxisAlignedCube(vec3(modelmat * vec4(position, 1.0)), vec3(0.0), pushConstants.halfSide), 1.0);
    outTexCoords = unpackHalf2x16(packedTexCoords2x16);
    outNormal = normalize(mat3(modelmat) * vec3(unpackSnorm4x8(packedNormals4x8)));
}