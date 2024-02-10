#ifndef PBR_HEADER
#define PBR_HEADER

#include "math.h"

vec3 F_Schlick(vec3 F0, float F90, float NdotX)
{
	return F0 + (F90 - F0) * pow(1.0 - NdotX, 5.0);
}
float V_SmithGGXCorrelated(float NdotL, float NdotV, float alpha2)
{ 
	float Lambda_GGXV = NdotL * sqrt((-NdotV * alpha2 + NdotV) * NdotV + alpha2);
	float Lambda_GGXL = NdotV * sqrt((-NdotL * alpha2 + NdotL) * NdotL + alpha2);
	
	return 0.5 / (Lambda_GGXV + Lambda_GGXL);
}
float D_GGX(float NdotH, float alpha2)
{
	float d = (NdotH * alpha2 - NdotH) * NdotH + 1;
	return alpha2 / (d * d);
}

float Fr_DisneyDiffuse(float NdotV, float NdotL, float LdotH, float roughness)
{
	float energyBias = mix(0, 0.5, roughness);
	float energyFactor = mix(1.0, 1.0 / 1.51 , roughness);
	float fd90 = energyBias + 2.0 * LdotH * LdotH * roughness;
	vec3 f0 = vec3(1.0, 1.0, 1.0);
	float lightScatter = F_Schlick(f0, fd90, NdotL).r;
	float viewScatter = F_Schlick(f0, fd90, NdotV).r;
	
	return lightScatter * viewScatter * energyFactor;
}

float Fr_LambertDiffuse()
{
	return ONE_OVER_PI;
}


float computeSpecOcclusion(float NdotV, float diffAO, float alpha)
{
	return clamp(pow(NdotV + diffAO, exp2(-16.0 * alpha - 1.0)) - 1.0 + diffAO, 0.0, 1.0);
}

#endif