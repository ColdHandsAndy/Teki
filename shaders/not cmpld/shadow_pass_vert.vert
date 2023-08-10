#version 460

#extension GL_GOOGLE_include_directive						:  enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8     :  enable
#extension GL_ARB_shader_viewport_layer_array               :  enable

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

layout(location = 0) out float outLinDepth;

void main()
{
    mat4 modelmat = modelMatrices.modelMatrices[drawData.data[pushConstants.drawDataIndex].modelIndex];
    vec4 worldPos = modelmat * vec4(position, 1.0);
    vec4 viewPos = viewmatrices.mats[pushConstants.viewMatrixIndex] * worldPos;
	vec4 projPos;
    projPos.x = viewPos.x * pushConstants.proj00;
    projPos.y = viewPos.y * pushConstants.proj11;
    projPos.w = viewPos.z;
	projPos.z = viewPos.z * pushConstants.proj22 + pushConstants.proj32;
    outLinDepth = viewPos.z;
    gl_Position = projPos;
    gl_Layer = pushConstants.layer;
}