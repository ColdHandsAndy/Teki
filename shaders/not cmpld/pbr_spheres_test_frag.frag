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


vec3 F_Schlick(vec3 f0, float f90, float u)
{
	return f0 + (f90 - f0) * pow(1.0 - u , 5.0);
}

vec3 getSpecularDominantDir(vec3 N, vec3 R, float alpha, float NdotV)
{
	float lerpFactor = pow(1 - NdotV , 10.8649) * (1 - 0.298475 * log(39.4115 - 39.0029 * alpha)) + 0.298475 * log(39.4115 - 39.0029 * alpha);
	return mix(N, R, lerpFactor);
}
vec3 getDiffuseDominantDir(vec3 N, vec3 V, float NdotV, float alpha)
{
	float a = 1.02341 * alpha - 1.51174;
	float b = -0.511705 * alpha + 0.755868;
	float lerpFactor = clamp((NdotV * a + b) * alpha, 0.0, 1.0);
	return mix(N, V, lerpFactor);
}
vec3 evalIBLspecular(vec3 N, vec3 R, float NdotV, float alpha, float roughness, vec3 F0)
{
	vec3 dominantR = getSpecularDominantDir(N, R, alpha, NdotV);
	NdotV = max(NdotV, 0.5 / BRDF_LUT_TEXTURE_SIZE);
	float mipLevel = sqrt(roughness) * RAD_MIP_COUNT;
	//vec3 specLD = texture(samplerCubeMapRad, dominantR, mipLevel).rgb;
	//vec2 specDFG = texture(brdfLUT, vec2(NdotV, alpha)).xy;
	//return specLD * (F0 * specDFG.x + specDFG.y);

	//
	//vec2 specDFG = texture(brdfLUT, vec2(NdotV, alpha)).xy;
	//return (F0 * specDFG.x + specDFG.y) * vec3(0.5);
	//
	//
	vec2 specDFG = texture(brdfLUT, vec2(NdotV, alpha)).xy;
	return mix(specDFG.xxx, specDFG.yyy, F0) * vec3(0.5);
	//
}
vec3 evalIBLdiffuse(vec3 N, vec3 V, float NdotV, float alpha)
{
	vec3 dominantN = getDiffuseDominantDir(N, V, NdotV, alpha);
	vec3 diffLD = texture(samplerCubeMapIrrad, dominantN).rgb;
	//float diffDFG = texture(brdfLUT, vec2(NdotV, alpha)).z;
	//return diffLD * diffDFG;

	//
	//float diffDFG = texture(brdfLUT, vec2(NdotV, alpha)).z;
	//return vec3(diffDFG) * vec3(0.5);
	//
	//
	float diffDFG = texture(brdfLUT, vec2(NdotV, alpha)).z;
	return vec3(diffDFG) * vec3(0.5);
	//
}

vec3 evalIBL(vec3 N, vec3 R, vec3 V, float NdotV, float alpha, float roughness, vec3 F0)
{
	vec3 Fr = max(vec3(1.0 - alpha), F0) - F0;

    vec3 k_S = F0 + Fr * pow(1.0 - NdotV, 5.0);

	vec3 DFG = texture(brdfLUT, vec2(NdotV, roughness)).xyz;

	vec3 dominantR = getSpecularDominantDir(N, R, alpha, NdotV);
	float mipLevel = sqrt(roughness) * RAD_MIP_COUNT;
	vec3 specLD = texture(samplerCubeMapRad, R, mipLevel).rgb;
	vec3 dominantN = getDiffuseDominantDir(N, V, NdotV, alpha);
	vec3 diffLD = texture(samplerCubeMapIrrad, N).rgb;
    vec3 FssEss = k_S * DFG.x + DFG.y;

    float Ems = (1.0 - (DFG.x + DFG.y));
    vec3 F_avg = F0 + (1.0 - F0) / 21.0;
    vec3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
    vec3 k_D = (1.0 - FssEss - FmsEms);
    vec3 color = FssEss * specLD + (FmsEms + k_D) * diffLD;

	return color;
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
	finalColor = vec4(evalIBL(N, R, V, NdotV, alpha, roughness, F0), 1.0);
	//finalColor = vec4(result, 1.0);
}