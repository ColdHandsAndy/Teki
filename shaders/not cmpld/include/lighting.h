#ifndef LIGHTING_HEADER
#define LIGHTING_HEADER

#include "rand.h"
#include "misc.h"
#include "math.h"

#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

//Light data layout
struct UnifiedLightData
{
	vec3 position;
	float lightLength;
	vec3 spectrum;
	float cutoffCos;
	vec3 direction;
	float falloffCos;
	int shadowListIndex;
	uint shadowLayerIndex;
	uint shadowMatrixIndex;
	float lightSize;
};

//Shadow mapping functions
#define hash(p)  fract(sin(dot(p, vec2(11.9898, 78.233))) * 43758.5453)
float blueNoise(vec2 U)
{
	float v = hash(U + vec2(-1.0, 0.0))
		+ hash(U + vec2(1.0, 0.0))
		+ hash(U + vec2(0.0, 1.0))
		+ hash(U + vec2(0.0, -1.0));
	return  hash(U) - v / 4.0 + 0.5;
}
#undef hash
//PCF
float PCF(float bias, float receiverDepth, vec3 uv, samplerShadow samplerSM, texture2DArray shadowMap, ivec2 screenCoord)
{
	float invRes = 1.0 / textureSize(shadowMap, 0).x;

	float biasedReceiverDepth = (-SHADOW_NEAR_DEPTH / (SHADOW_FAR_DEPTH - SHADOW_NEAR_DEPTH)) + (SHADOW_NEAR_DEPTH * SHADOW_FAR_DEPTH) / ((receiverDepth - bias) * (SHADOW_FAR_DEPTH - SHADOW_NEAR_DEPTH));

	const float radius = 0.7;

	float rot = TWO_PI * blueNoise(vec2(screenCoord));
	float rcos = sin(rot);
	float rsin = cos(rot);

	float pcfValue = 0.0;

	vec2 offsetsFar[4] = {
		vec2(2.3, 2.3),
		vec2(-2.3, -2.3),
		vec2(2.3, -2.3),
		vec2(-2.3, 2.3)
	};

	for (int i = 0; i < 4; ++i)
	{
		vec2 offs = offsetsFar[i];
		offs.x = offs.x * rcos - offs.y * rsin;
		offs.y = offs.x * rsin + offs.y * rcos;
		pcfValue += texture(sampler2DArrayShadow(shadowMap, samplerSM), vec4(uv.xy + offs * invRes * radius, uv.z, biasedReceiverDepth)).x;
	}

	if (subgroupMin(pcfValue) > 3.9)
		return 1.0;
	else if (subgroupMax(pcfValue) < 0.1)
		return 0.0;

	vec2 offsetsClose[5] = {
		vec2(1.3, 0.0),
		vec2(-1.3, 0.0),
		vec2(0.0, 1.3),
		vec2(0.0, -1.3),
		vec2(0.0, 0.0)
	};

	for (int i = 0; i < 5; ++i)
	{
		vec2 offs = offsetsClose[i];
		offs.x = offs.x * rcos - offs.y * rsin;
		offs.y = offs.x * rsin + offs.y * rcos;
		pcfValue += texture(sampler2DArrayShadow(shadowMap, samplerSM), vec4(uv.xy + offs * invRes * radius, uv.z, biasedReceiverDepth)).x;
	}

	return pcfValue / 9.0;
}

#endif