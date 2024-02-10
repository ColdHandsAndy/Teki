#version 460

#extension GL_GOOGLE_include_directive						:  enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8     :  enable
#extension GL_ARB_shader_viewport_layer_array               :  enable

#include "bindless.h"
#include "misc.h"

layout(location = 0) in vec3 position;

layout(push_constant) uniform PushConsts 
{
	int layer;
    uint drawDataIndex;
    float proj00;
    float proj11;
    float proj22;
    float proj32;
    uint viewMatrixIndex;
} pushConstants;

layout(set = 0, binding = 0) buffer ViewMatrices
{
    mat4 mats[];
} viewmatrices;

layout(set = 0, binding = 1) buffer ModelMatrices 
{
    mat4 modelMatrices[];
} modelMatrices;

layout(set = 0, binding = 2) buffer DrawDataBuffer 
{
    DrawData data[];
} drawData;

void main()
{
    mat4 modelmat = modelMatrices.modelMatrices[drawData.data[pushConstants.drawDataIndex].modelIndex];
    vec4 worldPos = modelmat * vec4(position, 1.0);
    vec4 pos = viewmatrices.mats[pushConstants.viewMatrixIndex] * worldPos;
    pos.x = pos.x * pushConstants.proj00;
    pos.y = pos.y * pushConstants.proj11;
    pos.w = pos.z;
	pos.z = pos.z * pushConstants.proj22 + pushConstants.proj32;
    gl_Position = pos;
    gl_Layer = pushConstants.layer;
}