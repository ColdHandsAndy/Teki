#version 460

#extension GL_GOOGLE_include_directive						: enable

#include "gi_data.h"

layout(set = 2, binding = 0) uniform sampler2DArray imageListArray[64];

layout(set = 3, binding = 0, r32ui) uniform uimage3D BOM;
layout(set = 4, binding = 0, rgba16ui) uniform writeonly uimage3D EmissionMetRoughVoxelMap;
layout(set = 5, binding = 0, rgba8ui) uniform writeonly uimage3D AlbedoNormalVoxelMap;

layout(location = 0) in vec3 inVoxTexCoords;
layout(location = 1) in vec2 inTexCoords;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in flat uint in_bcList_bcLayer_emList_emLayer;
layout(location = 4) in flat uint in_mrList_mrLayer;

layout(push_constant) uniform PushConsts 
{
	float halfSide;
	uint resolutionBOM;
	uint resolutionVM;
} pushConstants;

void main() 
{
	vec3 coordsBOM = inVoxTexCoords * pushConstants.resolutionBOM;

	const uint packingX = 4;
	const uint packingY = 2;
	const uint packingZ = 4;
	ivec3 coordsPackedBOM = ivec3(coordsBOM.x / packingX, coordsBOM.y / packingY, coordsBOM.z / packingZ);
	
	uint pack = 1 << ((uint(coordsBOM.x) % packingX) * 4) << ((uint(coordsBOM.y) % packingY) * 16) << ((uint(coordsBOM.z) % packingZ));
	
	imageAtomicOr(BOM, coordsPackedBOM, pack);

	vec3 albedo = texture(imageListArray[in_bcList_bcLayer_emList_emLayer >> 24], vec3(inTexCoords, ((in_bcList_bcLayer_emList_emLayer >> 16) & 0xFF) + 0.1), 1.0).xyz;
	vec2 metrough = texture(imageListArray[(in_mrList_mrLayer >> 8) & 0xFF], vec3(inTexCoords, ((in_mrList_mrLayer) & 0xFF) + 0.1), 1.0).zy;
	vec3 emission = texture(imageListArray[(in_bcList_bcLayer_emList_emLayer >> 8) & 0xFF], vec3(inTexCoords, ((in_bcList_bcLayer_emList_emLayer) & 0xFF) + 0.1), 1.0).xyz;
	

	uint emrRG = packHalf2x16(emission.xy);
	uint emrB = packHalf2x16(vec2(emission.z, 0.0));
	uint emrA = (uint(round(metrough.x * 255.0)) << 8) | uint(round(metrough.y * 255.0));

	uvec4 emrData = uvec4(emrRG & 0xFFFF, (emrRG >> 16) & 0xFFFF, emrB & 0xFFFF, emrA);

	uint anR = uint(round(albedo.r * 255.0));
	uint anG = uint(round(albedo.g * 255.0));
	uint anB = uint(round(albedo.b * 255.0));
	vec2 encodedNorm = encodeOctohedralZeroToOne(inNormal);
	uint anA = (uint(round(encodedNorm.y * 15.0)) << 4) | uint(round(encodedNorm.x * 15.0));

	uvec4 anData = uvec4(anR, anG, anB, anA);

	imageStore(EmissionMetRoughVoxelMap, ivec3(inVoxTexCoords * (pushConstants.resolutionVM)), emrData);
	imageStore(AlbedoNormalVoxelMap, ivec3(inVoxTexCoords * (pushConstants.resolutionVM)), anData);
}