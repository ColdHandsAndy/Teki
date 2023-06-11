#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable

layout(depth_unchanged) out float gl_FragDepth;

#define PI 3.141592653589
#define TWO_PI (2.0 * PI)
#define ONE_OVER_PI (1.0 / PI)
#define ONE_OVER_TWO_PI (1.0 / TWO_PI)

#define BRDF_LUT_TEXTURE_SIZE 512
#define RAD_MIP_COUNT 7
#define SAMPLE_COUNT 256



layout(push_constant) uniform PushConsts 
{
	vec3 camPos;
} pushConstants;

layout(set = 0, binding = 0) uniform ViewProjMatrices 
{
    mat4 view;
    mat4 proj;
} viewproj;

layout (set = 1, binding = 0) uniform sampler2DArray imageListArray[64];

struct DrawDataIndicesLayout
{
    uint8_t modelIndex;
    uint8_t index1;
    uint8_t index2;
    uint8_t index3;
    uint8_t bcIndexList;
    uint8_t bcIndexLayer;
    uint8_t nmIndexList;
    uint8_t nmIndexLayer;
    uint8_t mrIndexList;
    uint8_t mrIndexLayer;
    uint8_t emIndexList;
    uint8_t emIndexLayer;
};
layout(set = 2, binding = 0) buffer DrawDataIndices 
{
    DrawDataIndicesLayout indices[];
} drawDataIndices;

struct DirectionalLightLayout
{
    vec3 spectrum;
    vec3 direction;
};
layout(std140, set = 3, binding = 0) uniform DirectionalLight
{
    DirectionalLightLayout light;
} dirLight;
struct PointLightLayout
{
    vec3 position;
    vec3 spectrum;
};
layout(std140, set = 3, binding = 1) uniform PointLights
{
    uint lightNumber;
    PointLightLayout lights[64];
} pointLights;
struct SpotLightLayout
{
    vec3 position;
    vec4 spectrum_startCutoffCos;
    vec4 direction_endCutoffCos;
};
layout(std140, set = 3, binding = 2) uniform SpotLights
{
    uint lightNumber;
    SpotLightLayout lights[64];
} spotLights;

layout(set = 4, binding = 0) uniform samplerCube samplerCubeMap;
layout(set = 4, binding = 1) uniform samplerCube samplerCubeMapRad;
layout(set = 4, binding = 2) uniform samplerCube samplerCubeMapIrrad;
layout(set = 4, binding = 3) uniform sampler2D brdfLUT;



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

vec3 evalIBLspecular(vec3 N, vec3 R, float NdotV, float alpha, float roughness, vec3 F0, vec3 DFG)
{
	vec3 dominantR = getSpecularDominantDir(N, R, alpha, NdotV);
	NdotV = max(NdotV, 0.5 / BRDF_LUT_TEXTURE_SIZE);
	float mipLevel = sqrt(roughness) * RAD_MIP_COUNT;
	vec3 specLD = texture(samplerCubeMapRad, dominantR, mipLevel).rgb;
	return specLD * mix(DFG.xxx, DFG.yyy, F0);
}
vec3 evalIBLdiffuse(vec3 N, vec3 V, float NdotV, float alpha, vec3 DFG)
{
	vec3 dominantN = getDiffuseDominantDir(N, V, NdotV, alpha);
	vec3 diffLD = texture(samplerCubeMapIrrad, dominantN).rgb;
	return diffLD * DFG.z;
}




struct MaterialData
{
	vec3  F0;
	float roughness;
	vec3  albedo;
	float alpha;
	vec3 mscatterComp;
	float alpha2;
	float diffAO;
	float specAO;
};
void evalLightTerms(out vec3 F, out vec3 Fr, out float Fd, float NdotV, float NdotL, float NdotH, float LdotH, MaterialData data)
{
	float D     = D_GGX(NdotH, data.alpha2);
	float Vis   = V_SmithGGXCorrelated(NdotV, NdotL, data.alpha2);
	F			= F_Schlick(data.F0, 1.0, LdotH);
	Fr			= D * F * Vis / PI;
	Fd			= Fr_DisneyDiffuse(NdotV, NdotL, LdotH, data.roughness) / PI;
}
vec3 directionalLightEval(vec3 V, vec3 N, float NdotV, MaterialData data)
{
	vec3 	L = -dirLight.light.direction;
	vec3 	H = normalize(L + V);
	float 	NdotL = clamp(dot(N, L), 0.0, 1.0);
	float 	NdotH = clamp(dot(N, H), 0.0, 1.0);
	float	LdotH = clamp(dot(L, H), 0.0, 1.0);
	
	vec3 	F;
	vec3 	Fr;
	float 	Fd;
	evalLightTerms(F, Fr, Fd, NdotV, NdotL, NdotH, LdotH, data);

	return (Fr * data.mscatterComp * data.specAO + Fd * (vec3(1.0) - F) * data.albedo) * dirLight.light.spectrum * NdotL;
}
vec3 pointLightEval(vec3 spectrum, vec3 unnormL, vec3 V, vec3 N, float NdotV, MaterialData data)
{
	float	dist = length(unnormL);
	vec3 	L = normalize(unnormL);
	vec3 	H = normalize(L + V);
	float 	NdotL = dot(N, L);
	float 	NdotH = dot(N, H);
	float	LdotH = dot(L, H);
	
	vec3 	F;
	vec3 	Fr;
	float 	Fd;
	evalLightTerms(F, Fr, Fd, NdotV, NdotL, NdotH, LdotH, data);

    return (((Fr * data.mscatterComp * data.specAO + Fd * (vec3(1.0) - F)) * data.albedo) * spectrum * NdotL) / (dist * dist);
}
vec3 spotLightEval(vec3 spectrum, vec3 direction, vec3 unnormL, float cutoffStartCos, float cutoffEndCos, vec3 V, vec3 N, float NdotV, MaterialData data)
{
	float	dist = length(unnormL);
	vec3 	L = normalize(unnormL);
	vec3 	H = normalize(L + V);
	float 	NdotL = dot(N, L);
	float 	NdotH = dot(N, H);
	float	LdotH = dot(L, H);
	
	vec3 	F;
	vec3 	Fr;
	float 	Fd;
	evalLightTerms(F, Fr, Fd, NdotV, NdotL, NdotH, LdotH, data);
	
    float cutoffIntensity = clamp((-dot(direction, L) - cutoffEndCos) / (cutoffStartCos - cutoffEndCos), 0.0, 1.0);

    return (((Fr * data.mscatterComp * data.specAO + Fd * (vec3(1.0) - F)) * data.albedo) * spectrum * cutoffIntensity * NdotL) / (dist * dist);
}
vec3 calculateLightContribution(vec3 V, vec3 N, float NdotV, MaterialData data)
{
	vec3 result = vec3(0.0);
	vec3 viewPos = pushConstants.camPos;
    result += directionalLightEval(V, N, NdotV, data);
    uint pointLightCount = pointLights.lightNumber;
    for (uint i = 0; i < pointLightCount; ++i)
    {	
		vec3 spectrum = pointLights.lights[i].spectrum;
		vec3 unnormL = viewPos - pointLights.lights[i].position;
        result += pointLightEval(spectrum, unnormL, V, N, NdotV, data);
    }
    uint spotLightCount = spotLights.lightNumber;
    for (uint i = 0; i < spotLightCount; ++i)
    {
		vec3 spectrum = spotLights.lights[i].spectrum_startCutoffCos.xyz;
		vec3 unnormL = viewPos - spotLights.lights[i].position;
		vec3 direction = spotLights.lights[i].direction_endCutoffCos.xyz;
		float cutoffStartCos = spotLights.lights[i].spectrum_startCutoffCos.w;
		float cutoffEndCos = spotLights.lights[i].direction_endCutoffCos.w;
        result += spotLightEval(spectrum, direction, unnormL, cutoffStartCos, cutoffEndCos, V, N, NdotV, data);
    }
    return result;
}

float computeSpecOcclusion(float NdotV, float diffAO, float alpha)
{
	return clamp(pow(NdotV + diffAO, exp2(-16.0 * alpha - 1.0)) - 1.0 + diffAO, 0.0, 1.0);
}

layout(location = 0) in flat uint drawID;
layout(location = 1) in vec3 inPos;
layout(location = 2) in vec3 inNorm;
layout(location = 3) in vec4 inTang;
layout(location = 4) in vec2 inTexC;

layout(location = 0) out vec4 outputColor;

void main() 
{
	DrawDataIndicesLayout dataIndices = drawDataIndices.indices[drawID];
	
	vec3 N = normalize(inNorm);
	vec3 T = normalize(inTang.xyz);
	vec3 B = cross(inTang.xyz, inNorm) * inTang.w;
	mat3 TBN = mat3(T, B, N);
	N = TBN * normalize((texture(imageListArray[dataIndices.nmIndexList], vec3(inTexC, dataIndices.nmIndexLayer + 0.1)).xyz) * 2.0 - 1.0);
	vec3 V = normalize(pushConstants.camPos - inPos);
	vec3 R = reflect(-V, N);
	
	float NdotV = abs(dot(N, V)) + 0.0001;
	
	vec3 mrData = texture(imageListArray[dataIndices.mrIndexList], vec3(inTexC, dataIndices.mrIndexLayer + 0.1)).xyz;
	
	MaterialData data;
	data.albedo = texture(imageListArray[dataIndices.bcIndexList], vec3(inTexC, dataIndices.bcIndexLayer + 0.1)).xyz;
	data.F0 = mix(vec3(0.04), data.albedo, mrData.b);
	data.roughness = mrData.g;
	data.alpha = mrData.g * mrData.g + 0.001;
	data.alpha2 = data.alpha * data.alpha;
	data.diffAO = mrData.r;
	data.specAO = computeSpecOcclusion(NdotV, mrData.r, data.alpha);

	vec3 DFG = texture(brdfLUT, vec2(NdotV, data.alpha)).xyz;
	data.mscatterComp = 1.0 + data.F0 * (1.0 / DFG.y - 1.0);

	vec3 specIBL = evalIBLspecular(N, R, NdotV, data.alpha, data.roughness, data.F0, DFG);
	vec3 diffuseWeight = vec3(1.0) - F_Schlick(data.F0, 1.0, NdotV);
	vec3 diffIBL = evalIBLdiffuse(N, V, NdotV, data.alpha, DFG) * diffuseWeight * data.albedo;
	
	vec3 lightsContrib = calculateLightContribution(V, N, NdotV, data);

	vec3 result = lightsContrib + specIBL * data.specAO + diffIBL * data.diffAO;

    outputColor = vec4(result, 1.0);
}