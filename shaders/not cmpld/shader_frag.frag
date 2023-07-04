#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int8     : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16    : enable
#extension GL_KHR_shader_subgroup_ballot					: enable
#extension GL_KHR_shader_subgroup_arithmetic				: enable


layout(depth_unchanged) out float gl_FragDepth;


//Math
#define PI 3.141592653589
#define TWO_PI (2.0 * PI)
#define ONE_OVER_PI (1.0 / PI)
#define ONE_OVER_TWO_PI (1.0 / TWO_PI)

//PBR
#define BRDF_LUT_TEXTURE_SIZE 512
#define RAD_MIP_COUNT 7
#define SAMPLE_COUNT 256

//Clustering
#define MAX_LIGHTS 1024
#define MAX_WORDS 32
#define Z_BIN_COUNT 8096
#define UINT16_MAX 65535
#define TILE_PIXEL_WIDTH 8
#define TILE_PIXEL_HEIGHT 8 



layout(location = 0) in flat uint drawID;
layout(location = 1) in vec3 inPos;
layout(location = 2) in vec3 inNorm;
layout(location = 3) in vec4 inTang;
layout(location = 4) in vec2 inTexC;
layout(location = 5) in float inViewDepth;

layout(push_constant) uniform PushConsts 
{
	vec3 camPos;
	float binWidth;
	layout(offset = 16)uint windowTileWidth;
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

struct UnifiedLightData
{
	vec3 position;
	float lightLength;
	vec3 spectrum;
	float cutoffCos;
	vec3 direction;
	float falloffCos;
};
layout(set = 3, binding = 0) uniform LightsData
{
	UnifiedLightData lights[MAX_LIGHTS];
} lightsData;
const uint TYPE_POINT = 0;
const uint TYPE_SPOT = 1;
layout(std430, set = 3, binding = 1) buffer TypeData
{
	uint8_t types[];
} typeData;
layout(set = 3, binding = 2) buffer TilesData
{
	uint tilesWords[];
} tilesData;
struct ZBin
{
	uint16_t minI;
	uint16_t maxI;
};
layout(set = 3, binding = 3) buffer ZBinData
{
	ZBin data[];
} zBinData;

layout(set = 4, binding = 0) uniform samplerCube samplerCubeMap;
layout(set = 4, binding = 1) uniform samplerCube samplerCubeMapRad;
layout(set = 4, binding = 2) uniform samplerCube samplerCubeMapIrrad;
layout(set = 4, binding = 3) uniform sampler2D brdfLUT;

struct DirectionalLightLayout
{
    vec3 spectrum;
    vec3 direction;
};
layout(std140, set = 5, binding = 0) uniform DirectionalLight
{
    DirectionalLightLayout light;
} dirLight;


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



vec3 evaluateIBL(vec3 N, vec3 V, vec3 R, float NdotV, float alpha, float roughness, vec3 F0, vec3 DFG, vec3 albedo, float specAO, float diffAO)
{
    vec3 Fr = max(vec3(1.0 - alpha), F0) - F0;
    vec3 kS = F0 + Fr * pow(1.0 - NdotV, 5.0);

    vec3 FssEss = kS * DFG.x + DFG.y;

    float Ems = (1.0 - (DFG.x + DFG.y));
    vec3 Favg = F0 + (1.0 - F0) / 21.0;
    vec3 FmsEms = Ems * FssEss * Favg / (1.0 - Favg * Ems);
    vec3 kD = albedo * (1.0 - FssEss - FmsEms) * diffAO;
	float mipLevel = sqrt(roughness) * RAD_MIP_COUNT;
	vec3 specLD = texture(samplerCubeMapRad, R, mipLevel).rgb;
	vec3 diffLD = texture(samplerCubeMapIrrad, N).rgb;

    return FssEss * specLD * specAO + (FmsEms + kD) * diffLD;
}
float computeSpecOcclusion(float NdotV, float diffAO, float alpha)
{
	return clamp(pow(NdotV + diffAO, exp2(-16.0 * alpha - 1.0)) - 1.0 + diffAO, 0.0, 1.0);
}



struct MaterialData
{
	vec3  F0;
	float roughness;
	vec3  albedo;
	float alpha;
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

	return (Fr * data.specAO + Fd * (vec3(1.0) - F) * data.albedo) * dirLight.light.spectrum * NdotL;
}
vec3 pointLightEval(vec3 spectrum, vec3 unnormL, float radius, vec3 V, vec3 N, float NdotV, MaterialData data)
{
	float	dist = length(unnormL);
	vec3 	L = normalize(unnormL);
	vec3 	H = normalize(L + V);
	float 	NdotL = clamp(dot(N, L), 0.0, 1.0);
	float 	NdotH = clamp(dot(N, H), 0.0, 1.0);
	float	LdotH = clamp(dot(L, H), 0.0, 1.0);
	
	vec3 	F;
	vec3 	Fr;
	float 	Fd;
	evalLightTerms(F, Fr, Fd, NdotV, NdotL, NdotH, LdotH, data);

	float sqrtNom = clamp(1 - pow(dist / radius, 4), 0.0, 1.0);
	float attenuationTerm = (sqrtNom * sqrtNom) / (dist * dist + 1.0);

    return ((Fr * data.specAO + Fd * (vec3(1.0) - F)) * data.albedo) * spectrum * NdotL * attenuationTerm;
}
vec3 spotLightEval(vec3 spectrum, vec3 direction, vec3 unnormL, float lengthL, float falloffCos, float cutoffCos, vec3 V, vec3 N, float NdotV, MaterialData data)
{
	float	dist = length(unnormL);
	vec3 	L = normalize(unnormL);
	vec3 	H = normalize(L + V);
	float 	NdotL = clamp(dot(N, L), 0.0, 1.0);
	float 	NdotH = clamp(dot(N, H), 0.0, 1.0);
	float	LdotH = clamp(dot(L, H), 0.0, 1.0);
	
	vec3 	F;
	vec3 	Fr;
	float 	Fd;
	evalLightTerms(F, Fr, Fd, NdotV, NdotL, NdotH, LdotH, data);
	
	float sqrtNom = clamp(1 - pow(dist / lengthL, 4), 0.0, 1.0);
	float attenuationTerm = (sqrtNom * sqrtNom) / (dist * dist + 1.0);
    float falloffIntensity = clamp((-dot(direction, L) - cutoffCos) / (falloffCos - cutoffCos), 0.0, 1.0);

    return ((Fr * data.specAO + Fd * (vec3(1.0) - F)) * data.albedo) * spectrum * falloffIntensity * NdotL * attenuationTerm;
}

vec3 processLight(uint index, vec3 V, vec3 N, float NdotV, MaterialData data)
{
	UnifiedLightData light = lightsData.lights[index];
	uint type = typeData.types[index];
	
	vec3 pos = inPos;
	vec3 result = vec3(0.0);
	switch (type)
	{
		case TYPE_POINT:
			result = pointLightEval(light.spectrum, light.position - pos, light.lightLength, V, N, NdotV, data);
			break;
		case TYPE_SPOT:
			result = spotLightEval(light.spectrum, light.direction, light.position - pos, light.lightLength, light.falloffCos, light.cutoffCos, V, N, NdotV, data);
			break;
		default:
			result = vec3(1.0, 0.0, 0.0);
			break;
	}
	return result;
}
uint getTileFirstWordFromScreenPosition()
{
	uint xTile = uint(gl_FragCoord.x / TILE_PIXEL_WIDTH);
	uint yTile = uint(gl_FragCoord.y / TILE_PIXEL_HEIGHT);
	
	return (yTile * pushConstants.windowTileWidth + xTile) * MAX_WORDS;
}
void getZBinMinMaxData(out uint minInd, out uint maxInd)
{
	uint zBinIndex = uint(inViewDepth / pushConstants.binWidth);
	
	minInd = zBinIndex > Z_BIN_COUNT ? UINT16_MAX : zBinData.data[zBinIndex].minI;
	maxInd = zBinIndex > Z_BIN_COUNT ? 0 : zBinData.data[zBinIndex].maxI;
}

vec3 calculateLightContribution(vec3 V, vec3 N, float NdotV, MaterialData data)
{
	//Directional light contrib
	vec3 result = vec3(0.0);
    result += directionalLightEval(V, N, NdotV, data);
	
	
	//Lights and bins merged between workgroups to achieve uniformity
	uint wordMin = 0;
	uint wordMax = max(MAX_WORDS - 1, 0);
	
	uint tileWordsStart = getTileFirstWordFromScreenPosition();
	
	uint minIndexZ;
	uint maxIndexZ;
	getZBinMinMaxData(minIndexZ, maxIndexZ);
	
	uint mergedMin = subgroupMin(minIndexZ);
	uint mergedMax = subgroupMax(maxIndexZ);
	wordMin = max(mergedMin / 32, wordMin);
	wordMax = min(mergedMax / 32, wordMax);
	
	//
	//float modif = 1.0 / 50;
	//result = vec3(0.0, 1.0, 0.0);
	//
	for (uint wordIndex = wordMin; wordIndex <= wordMax; ++wordIndex)
	{
		uint mask = tilesData.tilesWords[tileWordsStart + wordIndex];
		
		//Try to get this out of the loop
		int localMin = clamp(int(minIndexZ) - int(wordIndex) * 32, 0, 31);
		int maskWidth = clamp(int(maxIndexZ) - int(minIndexZ) + 1, 0, 32);
		uint zBinMask = maskWidth == 32 ? uint(0xFFFFFFFF) : bitfieldInsert(0, uint(0xFFFFFFFF), localMin, maskWidth);
		mask &= zBinMask;

		uint mergedMask = subgroupOr(mask);
		while (mergedMask != 0)
		{
			uint bitIndex = findLSB(mergedMask);
			uint lightIndex = wordIndex * 32 + bitIndex;
			mergedMask ^= (1 << bitIndex);
			result += processLight(lightIndex, V, N, NdotV, data);
			//result += vec3(modif, -modif, 0.0);
		}
	}
	
    return result;
}


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

	vec3 DFG = texture(brdfLUT, vec2(NdotV, data.roughness)).xyz;
	
	vec3 IBLcontrib = evaluateIBL(N, V, R, NdotV, data.alpha, data.roughness, data.F0, DFG, data.albedo, data.specAO, data.diffAO);
	vec3 lightsContrib = calculateLightContribution(V, N, NdotV, data);

	vec3 emission = texture(imageListArray[dataIndices.emIndexList], vec3(inTexC, dataIndices.emIndexLayer + 0.1)).xyz;

	vec3 result = lightsContrib + IBLcontrib + emission;

    outputColor = vec4(result, 1.0);
}