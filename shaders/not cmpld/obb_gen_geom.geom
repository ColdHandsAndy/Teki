#version 460 core

#extension GL_GOOGLE_include_directive						:  enable

#define COORDINATE_TRANSFORMATION_SET_INDEX 0
#include "coordinate_transformation_set.h"

layout(push_constant) uniform PushConsts 
{
	vec4 xAxis_extent;
	vec4 yAxis_extent;
	vec4 zAxis_extent;
	vec3 center;
} pushConstants;

layout (points) in;
layout (line_strip, max_vertices = 18) out;

layout(location = 0) out vec3 outColor;

void main() 
{    
	vec3 center = pushConstants.center;
	vec3 xMove = pushConstants.xAxis_extent.xyz * pushConstants.xAxis_extent.w;
	vec3 yMove = pushConstants.yAxis_extent.xyz * pushConstants.yAxis_extent.w;
	vec3 zMove = pushConstants.zAxis_extent.xyz * pushConstants.zAxis_extent.w;

    gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove + xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove - xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove - xMove - zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove + xMove - zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove + xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
    EndPrimitive();
	
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove + xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove - xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove - xMove - zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove + xMove - zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove + xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
    EndPrimitive();
	
	gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove + xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove + xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
    EndPrimitive();

	gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove - xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove - xMove + zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
    EndPrimitive();

	gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove - xMove - zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove - xMove - zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
    EndPrimitive();

	gl_Position = coordTransformData.ndcFromWorld * vec4(center + yMove + xMove - zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(center - yMove + xMove - zMove, 1.0);
	outColor = vec3(0.03, 0.98, 0.1);
    EmitVertex();
    EndPrimitive();
} 