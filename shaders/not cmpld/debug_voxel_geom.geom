#version 460 core

#extension GL_GOOGLE_include_directive						:  enable

#define COORDINATE_TRANSFORMATION_SET_INDEX 0
#include "coordinate_transformation_set.h"

layout(push_constant) uniform PushConsts 
{
	vec3 camPos;
	uint debugType;
	uint resolution;
	float voxelSize;
	uint indexROMA;
} pushConstants;

layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

layout(location = 0) in vec3[1] inColor;
layout(location = 1) in flat uint[1] renderVoxel;

layout(location = 0) out vec3 outColor;

void main() 
{    
	if (!bool(renderVoxel[0]))
		return;
		
	vec3 voxpos = gl_in[0].gl_Position.xyz;
	vec3 viewDir = pushConstants.camPos - voxpos;
	vec3 xMove = vec3(1.0, 0.0, 0.0) * pushConstants.voxelSize * 0.5;
	vec3 yMove = vec3(0.0, 1.0, 0.0) * pushConstants.voxelSize * 0.5;
	vec3 zMove = vec3(0.0, 0.0, 1.0) * pushConstants.voxelSize * 0.5;
	
    gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + xMove * sign(viewDir.x) - yMove - zMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + xMove * sign(viewDir.x) + yMove - zMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + xMove * sign(viewDir.x) - yMove + zMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + xMove * sign(viewDir.x) + yMove + zMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	EndPrimitive();
	
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + yMove * sign(viewDir.y) - xMove - zMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + yMove * sign(viewDir.y) - xMove + zMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + yMove * sign(viewDir.y) + xMove - zMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + yMove * sign(viewDir.y) + xMove + zMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	EndPrimitive();
	
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + zMove * sign(viewDir.z) - xMove - yMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + zMove * sign(viewDir.z) - xMove + yMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + zMove * sign(viewDir.z) + xMove - yMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	gl_Position = coordTransformData.ndcFromWorld * vec4(voxpos + zMove * sign(viewDir.z) + xMove + yMove, 1.0);
	outColor = inColor[0];
    EmitVertex();
	EndPrimitive();
} 