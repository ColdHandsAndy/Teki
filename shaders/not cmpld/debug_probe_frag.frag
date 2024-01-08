#version 460

#extension GL_GOOGLE_include_directive						: enable

#include "gi_data.h"

#define DEBUG_TYPE_RADIANCE 0
#define DEBUG_TYPE_IRRADIANCE 1
#define DEBUG_TYPE_VISIBILITY 2

layout(location = 0) in vec3 inNorm;
layout(location = 1) in flat ivec3 inProbeID;

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
	float invProbeMaxActiveDistance;
} pushConstants;

layout(set = 1, binding = 0, r11f_g11f_b10f) uniform readonly image2D RadianceProbes;
layout(set = 1, binding = 1) uniform sampler2D IrradianceProbes;
layout(set = 1, binding = 2) uniform sampler2D VisibilityProbes;

layout(location = 0) out vec4 finalColor;

vec3 TimothyTonemapper(vec3 x) 
{
    const float a = 1.6;
    const float d = 0.977;
    const float hdrMax = 8.0;
    const float midIn = 0.18;
    const float midOut = 0.267;

    // Can be precomputed
    const float b =
        (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    const float c =
        (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return vec3(pow(x.x, a), pow(x.y, a), pow(x.z, a)) / (vec3(pow(x.x, a * d), pow(x.y, a * d), pow(x.z, a * d)) * b + c);
}

vec2 signNotZero(vec2 v) 
{
	return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
}
ivec2 getProbeCoordFetch(vec3 dir)
{
	vec2 p = dir.xz * (1.0 / (abs(dir.x) + abs(dir.y) + abs(dir.z)));
	vec2 res = (dir.y <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
	return ivec2((res * 0.5 + 0.5) * DDGI_PROBE_LIGHT_SIDE_SIZE);
}
vec2 getProbeCoordSampling(vec3 dir)
{
	vec2 p = dir.xz * (1.0 / (abs(dir.x) + abs(dir.y) + abs(dir.z)));
	vec2 res = (dir.y <= 0.0) ? ((1.0 - abs(p.yx)) * signNotZero(p)) : p;
	return vec2((res * 0.5 + 0.5) * float(DDGI_PROBE_LIGHT_SIDE_SIZE));
}

void main()
{
	vec3 normal = normalize(inNorm);
	
	vec3 res;

	if (pushConstants.debugType == DEBUG_TYPE_RADIANCE)
	{
		ivec2 probeOuterCoord = ivec2(DDGI_PROBE_LIGHT_SIDE_SIZE * inProbeID.x + inProbeID.z * (DDGI_PROBE_LIGHT_SIDE_SIZE * pushConstants.probeCountX), DDGI_PROBE_LIGHT_SIDE_SIZE * inProbeID.y);
		res = TimothyTonemapper(imageLoad(RadianceProbes, probeOuterCoord + getProbeCoordFetch(normal)).xyz);
	}
	else if (pushConstants.debugType == DEBUG_TYPE_IRRADIANCE)
	{
		ivec2 probeOuterCoord = ivec2(DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS * inProbeID.x + inProbeID.z * (DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS * pushConstants.probeCountX), DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS * inProbeID.y);
		vec3 irradiance = texture(IrradianceProbes, (probeOuterCoord + ivec2(1, 1) + getProbeCoordSampling(normal)) * pushConstants.invIrradianceTextureResolution).xyz;
		irradiance = pow(irradiance, vec3(DDGI_IRRADIANCE_GAMMA * 0.5));
		res = TimothyTonemapper(irradiance);
	}
	else if (pushConstants.debugType == DEBUG_TYPE_VISIBILITY)
	{
		ivec2 probeOuterCoord = ivec2(DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS * inProbeID.x + inProbeID.z * (DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS * pushConstants.probeCountX), DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS * inProbeID.y);
		res = texture(VisibilityProbes, (probeOuterCoord + ivec2(1, 1) + getProbeCoordSampling(normal)) * pushConstants.invIrradianceTextureResolution).xxx * pushConstants.invProbeMaxActiveDistance;
	}
	else
	{
		res = vec3(0.0);
	}

	finalColor = vec4(vec3(pow(res.x, 1.0 / 2.2), pow(res.y, 1.0 / 2.2), pow(res.z, 1.0 / 2.2)), 0.0);
}