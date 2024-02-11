#version 460

#extension GL_GOOGLE_include_directive						:  enable


layout(push_constant) uniform PushConsts 
{
	vec3 firstProbePosition;
	uint probeCountX;
	uint probeCountY;
	uint probeCountZ;
	float xDist;
	float yDist;
	float zDist;
	uint debugType;
	vec2 invIrradianceTextureResolution;
} pushConstants;

layout(location = 0) in vec3 positionLocal;

layout(location = 0) out vec3 outNorm;
layout(location = 1) out flat ivec3 outProbeID;

#define COORDINATE_TRANSFORMATION_SET_INDEX 0
#include "coordinate_transformation_set.h"

layout(set = 2, binding = 0, rgba8_snorm) uniform readonly image3D ProbeOffsetImage;

void main()
{
    outNorm = normalize(positionLocal);
	outProbeID = ivec3(
		 gl_InstanceIndex % pushConstants.probeCountX, 
		(gl_InstanceIndex % (pushConstants.probeCountX * pushConstants.probeCountY)) / pushConstants.probeCountX, 
		 gl_InstanceIndex / (pushConstants.probeCountX * pushConstants.probeCountY));
	vec3 probeOffset = imageLoad(ProbeOffsetImage, outProbeID).xyz;
	vec3 positionWorld = pushConstants.firstProbePosition + 
		vec3(pushConstants.xDist * (outProbeID.x + probeOffset.x), 
			 pushConstants.yDist * (outProbeID.y + probeOffset.y), 
			 pushConstants.zDist * (outProbeID.z + probeOffset.z));
	float probeSizeModif = 4.0 / float(max(max(pushConstants.probeCountX, pushConstants.probeCountY), pushConstants.probeCountZ));
    vec4 vertPos = coordTransformData.ndcFromWorld * vec4(positionLocal * probeSizeModif + positionWorld, 1.0);
    gl_Position = vertPos;
}