#version 460

layout(location = 0) in flat int instID;
layout(location = 1) in vec3 posInp;
layout(location = 2) in vec2 texCInp;
layout(location = 3) in vec3 normInp;
layout(location = 4) in vec3 colorInp;
layout(location = 5) in flat float metalnessInp;
layout(location = 6) in flat float roughnessInp;

layout(location = 0) out vec4 finalColor;

layout(set = 0, binding = 2) uniform samplerCube samplerCubeMap;
layout(set = 0, binding = 3) uniform samplerCube samplerCubeMapRad;
layout(set = 0, binding = 4) uniform samplerCube samplerCubeMapIrrad;
layout(set = 0, binding = 5) uniform sampler2D brdfLUT;

layout(push_constant) uniform PushConsts 
{
	vec3 camPos;
} pushConstants;

#define PI 3.141592653589
#define TWO_PI (2.0 * PI)
#define ONE_OVER_PI (1.0 / PI)
#define ONE_OVER_TWO_PI (1.0 / TWO_PI)

#define BRDF_LUT_TEXTURE_SIZE 512
#define RAD_MIP_COUNT 7
#define SAMPLE_COUNT 256


vec3 F_Schlick(vec3 f0, float f90, float u);


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


vec3 F_Schlick(vec3 f0, float f90, float u)
{
	return f0 + (f90 - f0) * pow(1.0 - u , 5.0);
}

float V_SmithGGXCorrelated(float NdotL, float NdotV, float alpha)
{
	float alpha2 = alpha * alpha;
	
	float Lambda_GGXV = NdotL * sqrt((-NdotV * alpha2 + NdotV) * NdotV + alpha2);
	float Lambda_GGXL = NdotV * sqrt((-NdotL * alpha2 + NdotL) * NdotL + alpha2);
	
	return 0.5 / (Lambda_GGXV + Lambda_GGXL);
}

float D_GGX(float NdotH, float alpha)
{
	float alpha2 = alpha * alpha;
	float d = (NdotH * alpha2 - NdotH) * NdotH + 1;
	return alpha2 / (d * d);
}



vec3 getSpecularDominantDir(vec3 N, vec3 R, float alpha, float NdotV)
{
	float lerpFactor = pow(1 - NdotV , 10.8649) * (1 - 0.298475 * log(39.4115 - 39.0029 * alpha)) + 0.298475 * log(39.4115 - 39.0029 * alpha);
	return mix(N, R, lerpFactor);
}
vec3 evalIBLspecular(vec3 N, vec3 R, float NdotV, float alpha, float roughness, vec3 F0)
{
	vec3 dominantR = getSpecularDominantDir(N, R, alpha, NdotV);
	NdotV = max(NdotV, 0.5 / BRDF_LUT_TEXTURE_SIZE);
	float mipLevel = sqrt(roughness) * RAD_MIP_COUNT;
	vec3 specLD = texture(samplerCubeMapRad, dominantR, mipLevel).rgb;
	vec2 specDFG = texture(brdfLUT, vec2(NdotV, alpha)).xy;
	return specLD * (F0 * specDFG.x + specDFG.y);
}

vec3 getDiffuseDominantDir(vec3 N, vec3 V, float NdotV, float alpha)
{
	float a = 1.02341 * alpha - 1.51174;
	float b = -0.511705 * alpha + 0.755868;
	float lerpFactor = clamp((NdotV * a + b) * alpha, 0.0, 1.0);
	return mix(N, V, lerpFactor);
}
vec3 evalIBLdiffuse(vec3 N, vec3 V, float NdotV, float alpha)
{
	vec3 dominantN = getDiffuseDominantDir(N, V, NdotV, alpha);
	vec3 diffLD = texture(samplerCubeMapIrrad, dominantN).rgb;
	float diffDFG = texture(brdfLUT, vec2(NdotV, alpha)).z;
	return diffLD * diffDFG;
}



void main()
{
	vec3 N = normalize(normInp);
	vec3 V = normalize(pushConstants.camPos - posInp);
	vec3 R = reflect(-V, N);
	
	float NdotV = clamp(dot(N, V), 0.0, 1.0);
	
	vec3 F0 = mix(vec3(0.04), vec3(1.0), metalnessInp);
	float roughness = roughnessInp;
	float alpha = roughnessInp * roughnessInp + 0.001;
	float alpha2 = alpha * alpha;
	
	vec3 specIBL = evalIBLspecular(N, R, NdotV, alpha, roughness, F0);
	vec3 diffIBL = evalIBLdiffuse(N, V, NdotV, alpha);
	
	vec3 diffuseWeight = vec3(1.0) - F_Schlick(F0, 1.0, NdotV);
	
	vec3 result = specIBL + diffIBL * diffuseWeight;
	finalColor = vec4(result, 1.0);
}