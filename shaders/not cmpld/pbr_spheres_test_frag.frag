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

vec3 evalIBL(vec3 N, vec3 R, vec3 V, float NdotV, float alpha, float roughness, vec3 F0)
{
	vec3 Fr = max(vec3(1.0 - alpha), F0) - F0;

    vec3 k_S = F0 + Fr * pow(1.0 - NdotV, 5.0);

	vec3 DFG = texture(brdfLUT, vec2(NdotV, roughness)).xyz;

	float mipLevel = sqrt(roughness) * RAD_MIP_COUNT;
	vec3 specLD = texture(samplerCubeMapRad, R, mipLevel).rgb;
	vec3 diffLD = texture(samplerCubeMapIrrad, N).rgb;
    vec3 FssEss = k_S * DFG.x + DFG.y;

    float Ems = (1.0 - (DFG.x + DFG.y));
    vec3 F_avg = F0 + (1.0 - F0) / 21.0;
    vec3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
    vec3 k_D = (1.0 - FssEss - FmsEms) * colorInp;
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

	finalColor = vec4(evalIBL(N, R, V, NdotV, alpha, roughness, F0), 1.0);
}