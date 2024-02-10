#ifndef GI_DATA_HEADER
#define GI_DATA_HEADER

#include "math.h"
#include "octohedral.h"

#define ROM_NUMBER 32
#define STABLE_ROM_NUMBER 4
#define GI_ROM_INDEX_MULTIPLIER 8
#define GI_STABLE_ROM_INDEX_MULTIPLIER 4
#define DDGI_PROBE_LIGHT_SIDE_SIZE 8
#define DDGI_PROBE_VISIBILITY_SIDE_SIZE 8
#define DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS (DDGI_PROBE_LIGHT_SIDE_SIZE + 2)
#define DDGI_PROBE_VISIBILITY_SIDE_SIZE_WITH_BORDERS (DDGI_PROBE_VISIBILITY_SIDE_SIZE + 2)
#define DDGI_HYSTERESIS_IRRADIANCE 0.035
#define DDGI_HYSTERESIS_VISIBILITY 0.06
#define DDGI_IRRADIANCE_GAMMA 5.0
#define DDGI_INVERSE_IRRADIANCE_GAMMA (1.0 / DDGI_IRRADIANCE_GAMMA)
#define DDGI_IRRADIANCE_SCALE 4.3379
#define DDGI_IRRADIANCE_INVERSE_SCALE (1.0 / DDGI_IRRADIANCE_SCALE)
#define DDGI_VISIBILIYY_SHARPNESS 32

struct ProbeGridData
{
	vec3 relOriginProbePos;
	//pad
	vec3 relEndProbePos;
	//pad
	vec2 invProbeTextureResolution;
	float probeFurthestActiveDistance;
	uint probeCountX;
	uint probeCountY;
	uint probeCountZ;
	float probeDistX;
	float probeDistY;
	float probeDistZ;
	float probeInvDistX;
	float probeInvDistY;
	float probeInvDistZ;
	float shadowBias;
	//pad3
};

struct SpecularData
{
	ivec2 specImageRes;
	vec2 invSpecImageRes;
};

struct VoxelizationData
{
	uint resolutionROM;
	uint resolutionVM;
	float occupationMeterSize;
	float occupationHalfMeterSize;
	float invOccupationHalfMeterSize;
	float offsetNormalScaleROM;
	//pad2
};

struct Cascade
{
	ProbeGridData gridData;
	VoxelizationData voxelData;
};

struct GIMetaData
{
	Cascade cascades[1];
	SpecularData specData;
};



vec2 getInnerProbeCoordSampling(vec3 dir)
{
	return encodeOctohedralZeroToOne(dir) * DDGI_PROBE_LIGHT_SIDE_SIZE;
}
vec2 getOuterProbeCoordSampling(vec3 gridCoord, float probeCountX)
{
	const int border = 1;
	return vec2(gridCoord.x * DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS + gridCoord.z * probeCountX * DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS + border, gridCoord.y * DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS + border);
}
void getProbeData(out vec3 irradiance, out vec2 visibility, vec3 samplingDirection, uint probeCountX, vec3 gridCoord, vec2 invProbeTextureResolution, sampler2D IProbes, sampler2D VProbes)
{
	vec2 sampleCoords = (getOuterProbeCoordSampling(gridCoord, float(probeCountX)) + getInnerProbeCoordSampling(samplingDirection)) * invProbeTextureResolution;

	irradiance = texture(IProbes, sampleCoords).xyz * DDGI_IRRADIANCE_SCALE;
	visibility = texture(VProbes, sampleCoords).xy;
}
vec3 sampleProbeVolume(ProbeGridData gridData, vec3 sampleDir, vec3 gridPos, vec3 gridCoord, vec3 baseProbeCoord, vec3 trilValues, sampler2D IrradianceProbes, sampler2D VisibilityProbes)
{
	vec3 result;

	vec3 sumIrradiance = vec3(0.0);
	float sumWeight = 0.0;
	for (int i = 0; i < 8; ++i)
	{
		ivec3 offset = ivec3(i, i >> 1, i >> 2) & ivec3(1);

		//Trillinear filtering
		vec3 trilinearComponents = max(vec3(0.001), vec3(bool(offset.x) ? trilValues.x : 1.0 - trilValues.x, bool(offset.y) ? trilValues.y : 1.0 - trilValues.y, bool(offset.z) ? trilValues.z : 1.0 - trilValues.z));
		float weight = 1.0;
		//

		vec3 curProbeCoord = baseProbeCoord + vec3(offset);

		vec3 probeDir = normalize(curProbeCoord - gridCoord); //Try without normalize

		//Normal irradiance wrapping
		float wrappedDP = (dot(probeDir, sampleDir) + 1.0) * 0.5;
		weight *= wrappedDP * wrappedDP + 0.2;
		//

		vec3 irradiance;
		vec2 visibility;
		getProbeData(irradiance, visibility, sampleDir, gridData.probeCountX, curProbeCoord, gridData.invProbeTextureResolution, IrradianceProbes, VisibilityProbes);

		//Visibility weighing
		float distToProbe = distance(gridPos, curProbeCoord * vec3(gridData.probeDistX, gridData.probeDistY, gridData.probeDistZ));

		float variance = abs(visibility.x * visibility.x - visibility.y);
		float chebyshevWeight = 1.0;

		if (distToProbe > visibility.x)
		{
			float t = distToProbe - visibility.x;
			chebyshevWeight = variance / (variance + t * t);
			chebyshevWeight = max(chebyshevWeight * chebyshevWeight * chebyshevWeight, 0.0);
		}

		chebyshevWeight = max(chebyshevWeight, 0.05);
		weight *= chebyshevWeight;
		//

		weight = max(0.000001, weight);

		//Log perception
		const float crushThreshold = 0.2;
		if (weight < crushThreshold)
		{
			weight *= (weight * weight) * (1.0 / (crushThreshold * crushThreshold));
		}
		//
		
		weight *= trilinearComponents.x * trilinearComponents.y * trilinearComponents.z;

		//Decode perceptual encoding
		irradiance = pow(irradiance, vec3(DDGI_IRRADIANCE_GAMMA * 0.5));
		//

		sumIrradiance += vec3(irradiance * weight);
		sumWeight += weight;
	}

	result = sumIrradiance / sumWeight;

	//Go back to linear irradiance
	result = result * result;
	//

	result *= 0.5 * PI;

	return result;
}


void unpackInjectedEmissionVM(out vec3 emission, out uint lightID, uvec4 data)
{
	vec2 rg = unpackHalf2x16(data.x | (data.y << 16));
	float b = unpackHalf2x16(data.z).r;

	emission = vec3(rg, b);

	lightID = data.w;
}
void unpackEmissionVM(out vec3 emission, out float metalness, out float roughness, uvec4 data)
{
	vec2 rg = unpackHalf2x16(data.x | (data.y << 16));
	float b = unpackHalf2x16(data.z).r;

	emission = vec3(rg, b);
	metalness = float(data.w >> 8) * (1.0 / 255.0);
	roughness = float(data.w & 0xFF) * (1.0 / 255.0);
}
void unpackMaterialVM(out vec3 albedo, out vec3 normal, uvec4 data)
{
	albedo = vec3(data.xyz) * (1.0 / 255.0);
	normal = decodeOctohedralZeroToOne(vec2(float(data.w & 0xF) * (1.0 / 15.0), float(data.w >> 4) * (1.0 / 15.0)));
}

#endif