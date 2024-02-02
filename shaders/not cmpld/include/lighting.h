#ifndef LIGHTING_HEADER
#define LIGHTING_HEADER

#include "rand.h"

#extension GL_EXT_samplerless_texture_functions : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define PCF_NUM_SAMPLES 12
#define BLOCKER_SEARCH_NUM_SAMPLES 8 
vec2 poissonDisc[16] = {
 vec2( 0.14383161, -0.14100790 ),
 vec2( 0.19984126, 0.78641367 ),
 vec2( 0.34495938, 0.29387760 ),
 vec2( -0.38277543, 0.27676845 ),
 vec2( -0.26496911, -0.41893023 ),
 vec2( 0.53742981, -0.47373420 ),
 vec2( 0.79197514, 0.19090188 ),
 vec2( -0.094184101, -0.92938870 ),
 vec2( -0.94201624, -0.39906216 ),
 vec2( -0.91588581, 0.45771432 ),
 vec2( -0.81544232, -0.87912464 ),
 vec2( 0.97484398, 0.75648379 ),
 vec2( 0.44323325, -0.97511554 ),
 vec2( -0.24188840, 0.99706507 ),
 vec2( -0.81409955, 0.91437590 ),
 vec2( 0.94558609, -0.76890725 )
 };

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
//Plane biasing 
vec2 computeReceiverPlaneDepthBias(vec3 texCoordDX, vec3 texCoordDY)
{
	vec2 biasUV;
	biasUV.x = texCoordDY.y * texCoordDX.z - texCoordDX.y * texCoordDY.z;
	biasUV.y = texCoordDX.x * texCoordDY.z - texCoordDY.x * texCoordDX.z;
	biasUV *= 1.0f / ((texCoordDX.x * texCoordDY.y) - (texCoordDX.y * texCoordDY.x));
	return biasUV;
}
//PCF
float PCF(float bias, float receiverDepth, vec3 uv, samplerShadow samplerSM, texture2DArray shadowMap, sampler2D rotationImage, ivec2 screenCoord)
{
	float invRes = 1.0 / textureSize(shadowMap, 0).x;

	float biasedReceiverDepth = receiverDepth - bias;

	const float radius = 0.7;

	int rotTexSize = textureSize(rotationImage, 0).x;
	ivec2 offsetsFirstCoord = ivec2(screenCoord.x * 0.5, screenCoord.y) % rotTexSize;

	vec2 rRot = bool(screenCoord.x % 2) ? texelFetch(rotationImage, offsetsFirstCoord, 0).xy : texelFetch(rotationImage, offsetsFirstCoord, 0).zw;
	float rcos = rRot.x;
	float rsin = rRot.y;

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

//PCSS
float penumbraSize(float receiverDepth, float blockerDepth)
{
	return (receiverDepth - blockerDepth) / blockerDepth;
}
void findBlocker(out float avgBlockerDepth, out float numBlockers, vec3 uv, float receiverDepth, float lightSize, float nearPlane, sampler samplerSM, texture2DArray shadowMap, float jitter)
{
	float searchWidth = lightSize * 0.05 * (receiverDepth - nearPlane) / (receiverDepth);
	float blockerSum = 0;
	numBlockers = 0;
	
	float rcos = random(jitter * 2000.0);
	float rsin = sqrt(1 - rcos * rcos);

	for (int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++i)
	{
		vec2 offUV = poissonDisc[i] * searchWidth;
		offUV += offUV * jitter;
		offUV.x = offUV.x * rcos - offUV.y * rsin;
		offUV.y = offUV.x * rsin + offUV.y * rcos;
		vec2 newUV = clamp(vec2(uv.xy + offUV), 0.0, 1.0);
		
		float shadowMapDepth = texture(sampler2DArray(shadowMap, samplerSM), vec3(newUV, uv.z)).x;
		if (shadowMapDepth < receiverDepth)
		{
			blockerSum += shadowMapDepth;
			++numBlockers;
		}
	}
	avgBlockerDepth = blockerSum / numBlockers;
}
float filterPCF(vec3 uv, float receiverDepth, float filterRadiusUV, sampler samplerSM, texture2DArray shadowMap, float jitter, float bias)
{
	float sum = 0.0f;
	
	float rcos = random(jitter * 2000.0);
	float rsin = sqrt(1 - rcos * rcos);

	for (int i = 0; i < PCF_NUM_SAMPLES; ++i)
	{
		vec2 offUV = poissonDisc[i] * filterRadiusUV;
		offUV += offUV * jitter;
		offUV.x = offUV.x * rcos - offUV.y * rsin;
		offUV.y = offUV.x * rsin + offUV.y * rcos;
		vec2 newUV = clamp(vec2(uv.xy + offUV), 0.0, 1.0);
		
		sum += texture(sampler2DArray(shadowMap, samplerSM), vec3(newUV, uv.z)).x > receiverDepth - bias ? 1.0 : 0.0;
	}
	return sum / PCF_NUM_SAMPLES;
} 
float PCSS(float receiverDepth, vec3 uv, float lightSize, float bias, float nearPlane, sampler samplerSM, texture2DArray shadowMap)
{
	float jitter = random(vec2(uv * 2000.0));
	
	float avgBlockerDepth = 0;
	float numBlockers = 0;
	findBlocker(avgBlockerDepth, numBlockers, uv, receiverDepth, lightSize, nearPlane, samplerSM, shadowMap, jitter);
	if (numBlockers < 1)
		return 1.0f;
	float penumbraRatio = penumbraSize(receiverDepth, avgBlockerDepth);
	float filterRadiusUV = penumbraRatio * lightSize * nearPlane / receiverDepth;
	
	return filterPCF(uv, receiverDepth, filterRadiusUV, samplerSM, shadowMap, jitter, bias); 
}

#endif