#version 460

#extension GL_GOOGLE_include_directive						: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8     : enable

#include "gi_data.h"

layout(push_constant) uniform PushConsts 
{
	vec3 camPos;
	uint debugType;
	uint resolution;
	float voxelSize;
	uint indexROMA;
} pushConstants;

layout(set = 1, binding = 0, r32ui) uniform readonly uimage3D BOM;
layout(set = 2, binding = 0, r32ui) uniform readonly uimage3D ROMA[32];

layout(set = 2, binding = 1) uniform ViewMatricesROMA
{
	mat3x4 viewmats[32];
} viewmatsROMA;

layout(set = 3, binding = 0, rgba16ui) uniform readonly uimage3D EmissionMetRoughVoxelMap;

layout(set = 4, binding = 0, rgba8ui) uniform readonly uimage3D AlbedoNormalVoxelMap;

layout(location = 0) out vec3 outColor;
layout(location = 1) out flat uint renderVoxel;

#define DEBUG_TYPE_BOM 0
#define DEBUG_TYPE_ROM 1
#define DEBUG_TYPE_ALBEDO 2
#define DEBUG_TYPE_METALNESS 3
#define DEBUG_TYPE_ROUGHNESS 4
#define DEBUG_TYPE_EMISSION 5

void main() 
{
	ivec3 voxelCoord = ivec3(gl_VertexIndex % pushConstants.resolution, gl_VertexIndex / (pushConstants.resolution * pushConstants.resolution), (gl_VertexIndex % (pushConstants.resolution * pushConstants.resolution)) / pushConstants.resolution);
	
	if (pushConstants.debugType == DEBUG_TYPE_BOM)
	{
		const uint packingX = 4;
		const uint packingY = 2;
		const uint packingZ = 4;
		renderVoxel = imageLoad(BOM, ivec3(voxelCoord.x / packingX, voxelCoord.y / packingY, voxelCoord.z / packingZ)).x;
		renderVoxel &= 1 << ((voxelCoord.x % packingX) * 4) << ((voxelCoord.y % packingY) * 16) << ((voxelCoord.z % packingZ));
		outColor = vec3(voxelCoord.x % 2, voxelCoord.y % 2, voxelCoord.z % 2);
		vec3 origin = vec3(0.5) * pushConstants.voxelSize * (1.0 - pushConstants.resolution);
		gl_Position = vec4(origin + pushConstants.voxelSize * voxelCoord, 1.0);
	}
	else if (pushConstants.debugType == DEBUG_TYPE_ROM)
	{
		const uint packingX = 1;
		const uint packingY = 1;
		const uint packingZ = 32;
		renderVoxel = imageLoad(ROMA[pushConstants.indexROMA], ivec3(voxelCoord.x / packingX, voxelCoord.y / packingY, voxelCoord.z / packingZ)).x;
		renderVoxel &= 1 << (voxelCoord.z % packingZ);
		outColor = vec3(float(voxelCoord.z) / pushConstants.resolution, float(voxelCoord.z) / pushConstants.resolution, float(voxelCoord.z) / pushConstants.resolution);
		vec3 pos = 
			(0.5 * (1.0 - pushConstants.resolution) + voxelCoord.x) * viewmatsROMA.viewmats[pushConstants.indexROMA][0].xyz * pushConstants.voxelSize + 
			(0.5 * (1.0 - pushConstants.resolution) + voxelCoord.y) * viewmatsROMA.viewmats[pushConstants.indexROMA][1].xyz * pushConstants.voxelSize + 
			(0.5 * (1.0 - pushConstants.resolution) + voxelCoord.z) * viewmatsROMA.viewmats[pushConstants.indexROMA][2].xyz * pushConstants.voxelSize;
		gl_Position = vec4(pos, 1.0);
	}
	else if (pushConstants.debugType == DEBUG_TYPE_ALBEDO)
	{
		vec3 albedo;
		vec3 normal;
		unpackMaterialVM(albedo, normal, imageLoad(AlbedoNormalVoxelMap, ivec3(voxelCoord.x, voxelCoord.y, voxelCoord.z)));

		renderVoxel = uint(any(greaterThan(albedo, vec3(0.0001))));
		outColor = albedo;
		vec3 origin = vec3(0.5) * pushConstants.voxelSize * (1.0 - pushConstants.resolution);
		gl_Position = vec4(origin + pushConstants.voxelSize * voxelCoord, 1.0);
	}
	else if (pushConstants.debugType == DEBUG_TYPE_METALNESS)
	{
		vec3 emission;
		float roughness;
		float metalness;
		unpackEmissionVM(emission, metalness, roughness, imageLoad(EmissionMetRoughVoxelMap, ivec3(voxelCoord.x, voxelCoord.y, voxelCoord.z)));

		renderVoxel = uint(metalness > 0.0001);
		outColor = vec3(metalness);
		vec3 origin = vec3(0.5) * pushConstants.voxelSize * (1.0 - pushConstants.resolution);
		gl_Position = vec4(origin + pushConstants.voxelSize * voxelCoord, 1.0);
	}
	else if (pushConstants.debugType == DEBUG_TYPE_ROUGHNESS)
	{
		vec3 emission;
		float roughness;
		float metalness;
		unpackEmissionVM(emission, metalness, roughness, imageLoad(EmissionMetRoughVoxelMap, ivec3(voxelCoord.x, voxelCoord.y, voxelCoord.z)));

		renderVoxel = uint(roughness > 0.0001);
		outColor = vec3(roughness);
		vec3 origin = vec3(0.5) * pushConstants.voxelSize * (1.0 - pushConstants.resolution);
		gl_Position = vec4(origin + pushConstants.voxelSize * voxelCoord, 1.0);
	}
	else if (pushConstants.debugType == DEBUG_TYPE_EMISSION)
	{
		vec3 emission;
		float roughness;
		float metalness;
		unpackEmissionVM(emission, metalness, roughness, imageLoad(EmissionMetRoughVoxelMap, ivec3(voxelCoord.x, voxelCoord.y, voxelCoord.z)));

		renderVoxel = uint(any(greaterThan(emission, vec3(0.0001))));
		outColor = emission;
		vec3 origin = vec3(0.5) * pushConstants.voxelSize * (1.0 - pushConstants.resolution);
		gl_Position = vec4(origin + pushConstants.voxelSize * voxelCoord, 1.0);
	}
	else
	{
		gl_Position = vec4(0.0);
	}
}