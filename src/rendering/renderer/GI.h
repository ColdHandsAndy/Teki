#ifndef GI_CLASS_HEADER
#define GI_CLASS_HEADER

#include <cmath>
#include <numbers>
#include <random>

#include <glm/glm.hpp>
#include <glm/geometric.hpp>

#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/sync_operations.h"
#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/renderer/clusterer.h"
#include "src/rendering/UI/UIData.h"

#include "src/tools/compile_time_array.h"
#include "src/tools/comp_s.h"
#include "src/tools/obj_loader.h"
#include "src/tools/ld_seq.h"
#include "src/tools/timestamp_queries.h"

#define SCENE_ORIGIN glm::vec3(0.0)

#define OCCUPANCY_RESOLUTION 128
#define OCCUPANCY_METER_SIZE 32.0f
#define BIT_TO_METER_SCALE (OCCUPANCY_METER_SIZE / OCCUPANCY_RESOLUTION)

#define BOM_PACKED_WIDTH (OCCUPANCY_RESOLUTION / 4)
#define BOM_PACKED_HEIGHT (OCCUPANCY_RESOLUTION / 2)
#define BOM_PACKED_DEPTH (OCCUPANCY_RESOLUTION / 4)

#define VOXELMAP_RESOLUTION (OCCUPANCY_RESOLUTION)
#define VOXEL_TO_METER_SCALE (BIT_TO_METER_SCALE * (OCCUPANCY_RESOLUTION / VOXELMAP_RESOLUTION))

#define ROM_NUMBER 32
#define STABLE_ROM_NUMBER 4
#define ROM_PACKED_WIDTH OCCUPANCY_RESOLUTION
#define ROM_PACKED_HEIGHT OCCUPANCY_RESOLUTION
#define ROM_PACKED_DEPTH (OCCUPANCY_RESOLUTION / 32)

#define DDGI_PROBE_LIGHT_SIDE_SIZE 8
#define DDGI_PROBE_VISIBILITY_SIDE_SIZE 8
#define DDGI_PROBE_LIGHT_SIDE_SIZE_WITH_BORDERS (DDGI_PROBE_LIGHT_SIDE_SIZE + 2)
#define DDGI_PROBE_VISIBILITY_SIDE_SIZE_WITH_BORDERS (DDGI_PROBE_VISIBILITY_SIDE_SIZE + 2)
#define DDGI_PROBE_X_COUNT 24
#define DDGI_PROBE_Y_COUNT 24
#define DDGI_PROBE_Z_COUNT 24
#define DDGI_PROBE_X_DISTANCE float(OCCUPANCY_METER_SIZE / DDGI_PROBE_X_COUNT)
#define DDGI_PROBE_Y_DISTANCE float(OCCUPANCY_METER_SIZE / DDGI_PROBE_Y_COUNT)
#define DDGI_PROBE_Z_DISTANCE float(OCCUPANCY_METER_SIZE / DDGI_PROBE_Z_COUNT)
#define DDGI_PROBE_X_OFFSET float(DDGI_PROBE_X_DISTANCE / 2.0)
#define DDGI_PROBE_Y_OFFSET float(DDGI_PROBE_Y_DISTANCE / 2.0)
#define DDGI_PROBE_Z_OFFSET float(DDGI_PROBE_Z_DISTANCE / 2.0)
#define DDGI_PROBE_MAX_VISIBILITY_RANGE (std::sqrt(DDGI_PROBE_X_DISTANCE * DDGI_PROBE_X_DISTANCE + DDGI_PROBE_Y_DISTANCE * DDGI_PROBE_Y_DISTANCE + DDGI_PROBE_Z_DISTANCE * DDGI_PROBE_Z_DISTANCE))

#define TUNABLE_SHADOW_BIAS 0.7

class GI
{
private:
	Image m_baseOccupancyMap;
	Image m_albedoNormalVoxelmap;
	Image m_emissionMetRoughVoxelmap;
	Image m_dynamicEmissionVoxelmap;
	Image m_specularReflectionGlossy;
	Image m_specularReflectionRough;
	Image m_depthSpec;
	Image m_refdirSpec;
	Image m_distToHit;
	Image m_ddgiRadianceProbes;
	Image m_ddgiDistanceProbes;
	std::array<Image, 2> m_ddgiIrradianceProbes;
	std::array<Image, 2> m_ddgiVisibilityProbes;
	std::array<Image, ROM_NUMBER> m_rayAlignedOccupancyMapArray;
	std::array<Image, STABLE_ROM_NUMBER> m_rayAlignedOccupancyMapArrayStable;
	BufferMapped m_ROMAtransformMatrices[2]{};
	BufferMapped m_stableROMAtransformMatrices[2]{};
	BufferMapped m_mappedDirections[2]{};

	uint16_t m_injectedLightsCount{ 0 };
	uint16_t m_injectedLightsIndices[MAX_LIGHTS]{};

	ResourceSet m_resSetWriteBOM{};
	ResourceSet m_resSetReadBOM{};
	ResourceSet m_resSetWriteROMA{};
	ResourceSet m_resSetReadROMA{};
	ResourceSet m_resSetReadStableROMA{};
	ResourceSet m_resSetAlbedoNormalWrite{};
	ResourceSet m_resSetAlbedoNormalRead{};
	ResourceSet m_resSetEmissionMetRoughWrite{};
	ResourceSet m_resSetEmissionMetRoughRead{};
	ResourceSet m_resSetDynamicEmissionWrite{};
	ResourceSet m_resSetDynamicEmissionRead{};
	ResourceSet m_resSetProbesWrite{};
	ResourceSet m_resSetSpecularWrite{};
	ResourceSet m_resSetRadianceProbesRead{};
	ResourceSet m_resSetDistanceProbesRead{};
	ResourceSet m_resSetIrradProbesWrite{};
	ResourceSet m_resSetVisibProbesWrite{};
	ResourceSet m_resSetIndirectDiffuseLighting{};
	ResourceSet m_resSetIndirectSpecularLighting{};
	ResourceSet m_resSetMappedDirections{};
	ResourceSet m_resSetBilateral{};

	Pipeline m_voxelize{};
	Pipeline m_createROMA{};
	Pipeline m_traceProbes{};
	Pipeline m_traceSpecular;
	Pipeline m_computeIrradiance{};
	Pipeline m_computeVisibility{};
	Pipeline m_injectLight{};
	Pipeline m_mergeEmission{};
	Pipeline m_bilateral{};

	uint32_t m_currentNewProbes{ 0 };
	uint32_t m_currentBuffers{ 0 };

	SyncOperations::EventHolder<1> m_events;

	Clusterer* const m_clusterer{ nullptr };

	struct
	{
		float halfSide{};
		uint32_t resolutionBOM{};
		uint32_t resolutionVM{};
	} m_pcDataBOM{};

	struct
	{
		alignas(16) glm::vec3 directionZ{};
		int resolution{};
		alignas(16) glm::vec3 directionX{};
		uint32_t indexROM{};
		alignas(16) glm::vec3 directionY{};
		alignas(16) glm::vec3 originROMInLocalBOM{};
		uint32_t stable;
	} m_pcDataROM{};

	struct
	{
		glm::vec3 voxelmapOriginWorld;
		float injectionScale;
		glm::vec3 spectrum;
		float voxelmapScale;
		glm::vec3 lightDir;
		float cutoffCos;
		float falloffCos;
		float lightLength;
		float fovScale;
		uint32_t lightID;
		uint32_t voxelmapResolution;
		uint32_t listIndex;
		uint32_t layerIndex;
		uint32_t viewmatIndex;
		uint32_t type;
	} m_pcDataLightInjection{};

	struct
	{
		uint32_t skyboxEnabled;
	} m_pcDataTraceProbes{};

	struct 
	{ 
		glm::mat4 worldFromNDC; 
		glm::vec3 sceneCenter; 
		float pad; 
		glm::vec3 campos; 
		uint32_t skyboxEnabled;
	} m_pcDataTraceSpecular{};

	struct 
	{ 
		glm::ivec2 imgRes;
	} m_pcDataBilateral{};
	


	struct alignas(16) ProbeGridData
	{
		alignas(16) glm::vec3 relOriginProbePos;
		//pad
		alignas(16) glm::vec3 relEndProbePos;
		//pad
		alignas(16) glm::vec2 invProbeTextureResolution;
		float probeFurthestActiveDistance;
		uint32_t probeCountX;
		uint32_t probeCountY;
		uint32_t probeCountZ;
		float probeDistX;
		float probeDistY;
		float probeDistZ;
		float probeInvDistX;
		float probeInvDistY;
		float probeInvDistZ;
		float shadowBias;
		//pad3
	};
	static_assert(sizeof(ProbeGridData) == 96);
	struct alignas(16) SpecularData
	{
		glm::ivec2 specImageRes;
		glm::vec2 invSpecImageRes;
	};
	static_assert(sizeof(SpecularData) == 16);
	struct alignas(16) VoxelizationData
	{
		uint32_t resolutionROM;
		uint32_t resolutionVM;
		float occupationMeterSize;
		float occupationHalfMeterSize;
		float invOccupationHalfMeterSize;
		float offsetNormalScaleROM;
		//pad2
	};
	static_assert(sizeof(VoxelizationData) == 32);
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
	BufferMapped m_giMetadata;
	ResourceSet m_resSetGIMetadata;

	struct
	{
		glm::vec3 camPos{};
		uint32_t debugType{};
		uint32_t resolution{};
		float voxelSize{};
		uint32_t indexROMA{};
	} m_pcDataDebugVoxel{};
	Pipeline m_debugVoxel{};

	struct
	{
		glm::vec3 firstProbePosition;
		uint32_t probeCountX;
		uint32_t probeCountY;
		uint32_t probeCountZ;
		float xDist;
		float yDist;
		float zDist;
		uint32_t debugType;
		glm::vec2 invIrradianceTextureResolution;
		float invProbeMaxActiveDistance;
	} m_pcDataDebugProbes{};
	ResourceSet m_resSetProbesDebug{};
	Pipeline m_debugProbes{};
	Buffer m_sphereVertexData{};
	uint32_t m_sphereVertexCount{};

public:
	GI(VkDevice device, uint32_t windowWidth, uint32_t windowHeight, BufferBaseHostAccessible& baseHostBuffer, BufferBaseHostInaccessible& baseDeviceBuffer, Clusterer& clusterer);
	~GI();

	const ResourceSet& getIndirectDiffuseLightingResourceSet()
	{
		return m_resSetIndirectDiffuseLighting;
	}
	const ResourceSet& getIndirectSpecularLightingResourceSet()
	{
		return m_resSetIndirectSpecularLighting;
	}
	const ResourceSet& getIndirectLightingMetadataResourceSet()
	{
		return m_resSetGIMetadata;
	}

	uint32_t getIndirectLightingCurrentSet()
	{
		return m_currentNewProbes;
	}

	uint32_t getCountROM() const
	{
		return ROM_NUMBER;
	}

	const glm::vec3& getScenePosition() const
	{
		return SCENE_ORIGIN;
	}

	void initialize(VkDevice device,
		const ResourceSet& drawDataRS, const ResourceSet& transformMatricesRS, const ResourceSet& materialsTexturesRS, const ResourceSet& distantProbeRS, const ResourceSet& BRDFLUTRS, const ResourceSet& shadowMapsRS, VkSampler generalSampler);

	void cmdVoxelize(VkCommandBuffer cb, const BufferMapped& indirectDrawCmdData, const Buffer& vertexData, const Buffer& indexData, uint32_t drawCmdCount, uint32_t drawCmdOffset, uint32_t drawCmdStride);
	
	template<uint32_t QueryNum>
	void cmdComputeIndirect(VkCommandBuffer cb,
		const glm::mat4& inverseViewProjectionMatrix, 
		const glm::vec3 camPos,
		TimestampQueries<QueryNum>& queries,
		const uint32_t queryIndexGICreateROMA, 
		const uint32_t queryIndexGITraceSpecular,
		const uint32_t queryIndexGITraceProbes,
		const uint32_t queryIndexGIComputeIrradianceAndVisibility, 
		bool skyboxEnabled,
		bool profile)
	{
		changeCurrentBuffers();

		{
			constexpr int romCount{ ROM_NUMBER };
			constexpr int sromCount{ STABLE_ROM_NUMBER };
			VkImageMemoryBarrier2 barriers[romCount + sromCount]{};

			for (int i{ 0 }; i < romCount; ++i)
			{
				barriers[i] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
					VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
					m_rayAlignedOccupancyMapArray[i].getImageHandle(), m_rayAlignedOccupancyMapArray[i].getSubresourceRange());
			}
			for (int i{ romCount }, j{ 0 }; i < romCount + sromCount; ++i, ++j)
			{
				barriers[i] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
					VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
					m_rayAlignedOccupancyMapArrayStable[j].getImageHandle(), m_rayAlignedOccupancyMapArrayStable[j].getSubresourceRange());
			}

			SyncOperations::cmdExecuteBarrier(cb, barriers);
		}

		if (profile) queries.cmdWriteStart(cb, queryIndexGICreateROMA);
		cmdDispatchCreateROMA(cb);
		if (profile) queries.cmdWriteEnd(cb, queryIndexGICreateROMA);

		{
			constexpr int romCount{ ROM_NUMBER };
			constexpr int sromCount{ STABLE_ROM_NUMBER };
			VkImageMemoryBarrier2 barriers[romCount + sromCount + 6]{};
			for (int i{ 0 }; i < romCount; ++i)
			{
				barriers[i] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
					m_rayAlignedOccupancyMapArray[i].getImageHandle(), m_rayAlignedOccupancyMapArray[i].getSubresourceRange());
			}
			for (int i{ romCount }, j{ 0 }; i < romCount + sromCount; ++i, ++j)
			{
				barriers[i] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
					VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
					m_rayAlignedOccupancyMapArrayStable[j].getImageHandle(), m_rayAlignedOccupancyMapArrayStable[j].getSubresourceRange());
			}

			barriers[romCount + sromCount + 0] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				m_ddgiRadianceProbes.getImageHandle(), m_ddgiRadianceProbes.getSubresourceRange());

			barriers[romCount + sromCount + 1] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				m_ddgiDistanceProbes.getImageHandle(), m_ddgiDistanceProbes.getSubresourceRange());

			barriers[romCount + sromCount + 2] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_depthSpec.getImageHandle(), m_depthSpec.getSubresourceRange());
			barriers[romCount + sromCount + 3] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_refdirSpec.getImageHandle(), m_refdirSpec.getSubresourceRange());
			barriers[romCount + sromCount + 4] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				m_specularReflectionGlossy.getImageHandle(), m_specularReflectionGlossy.getSubresourceRange());
			barriers[romCount + sromCount + 5] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				m_distToHit.getImageHandle(), m_distToHit.getSubresourceRange());

			SyncOperations::cmdExecuteBarrier(cb, barriers);
		}

		if (profile) queries.cmdWriteStart(cb, queryIndexGITraceProbes);
		cmdDispatchTraceProbes(cb, skyboxEnabled);
		if (profile) queries.cmdWriteEnd(cb, queryIndexGITraceProbes);

		changeHistoryAndNewProbes();

		{
			VkImageMemoryBarrier2 barriers[4]{};

			barriers[0] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_ddgiRadianceProbes.getImageHandle(), m_ddgiRadianceProbes.getSubresourceRange());

			barriers[1] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_ddgiDistanceProbes.getImageHandle(), m_ddgiDistanceProbes.getSubresourceRange());


			barriers[2] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				m_ddgiIrradianceProbes[m_currentNewProbes].getImageHandle(), m_ddgiIrradianceProbes[m_currentNewProbes].getSubresourceRange());

			barriers[3] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				m_ddgiVisibilityProbes[m_currentNewProbes].getImageHandle(), m_ddgiVisibilityProbes[m_currentNewProbes].getSubresourceRange());

			VkDependencyInfo probeDependency{ SyncOperations::createDependencyInfo(barriers) };

			m_events.cmdSet(cb, 0, probeDependency);

			if (profile) queries.cmdWriteStart(cb, queryIndexGITraceSpecular);
			cmdDispatchTraceSpecular(cb, inverseViewProjectionMatrix, camPos, skyboxEnabled);
			if (profile) queries.cmdWriteEnd(cb, queryIndexGITraceSpecular);


			uint32_t indices[]{ 0 };
			m_events.cmdWait(cb, 1, indices, &probeDependency);
		}

		if (profile) queries.cmdWriteStart(cb, queryIndexGIComputeIrradianceAndVisibility);
		cmdDispatchComputeIrradianceAndVisibility(cb);
		if (profile) queries.cmdWriteEnd(cb, queryIndexGIComputeIrradianceAndVisibility);

		{
			VkImageMemoryBarrier2 barriers[5]{};

			barriers[0] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
				m_depthSpec.getImageHandle(), m_depthSpec.getSubresourceRange());

			barriers[1] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
				m_refdirSpec.getImageHandle(), m_refdirSpec.getSubresourceRange());

			barriers[2] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_specularReflectionGlossy.getImageHandle(), m_specularReflectionGlossy.getSubresourceRange());

			barriers[3] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_distToHit.getImageHandle(), m_distToHit.getSubresourceRange());

			barriers[4] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_NONE, VK_ACCESS_SHADER_READ_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
				m_specularReflectionRough.getImageHandle(), m_specularReflectionRough.getSubresourceRange());

			SyncOperations::cmdExecuteBarrier(cb, barriers);
		}

		cmdDispatchBlurSpecular(cb);

		{
			VkImageMemoryBarrier2 barriers[3]{};

			barriers[0] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_ddgiIrradianceProbes[m_currentNewProbes].getImageHandle(), m_ddgiIrradianceProbes[m_currentNewProbes].getSubresourceRange());

			barriers[1] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_ddgiVisibilityProbes[m_currentNewProbes].getImageHandle(), m_ddgiVisibilityProbes[m_currentNewProbes].getSubresourceRange());

			barriers[2] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
				VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
				m_specularReflectionRough.getImageHandle(), m_specularReflectionRough.getSubresourceRange());

			SyncOperations::cmdExecuteBarrier(cb, barriers);
		}
	}

	template<uint32_t QueryNum>
	void cmdInjectLights(VkCommandBuffer cb,
		TimestampQueries<QueryNum>& queries,
		const uint32_t queryIndexGIInjectLights,
		bool profile)
	{
		SyncOperations::cmdExecuteBarrier(cb, { {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_NONE, VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_ACCESS_NONE, VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			m_dynamicEmissionVoxelmap.getImageHandle(), m_dynamicEmissionVoxelmap.getSubresourceRange())} });

		cmdTransferClearDynamicEmissionVoxelmap(cb);

		{
			VkImageMemoryBarrier2 barriers[1]{};

			barriers[0] = SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
				VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
				m_dynamicEmissionVoxelmap.getImageHandle(), m_dynamicEmissionVoxelmap.getSubresourceRange());

			SyncOperations::cmdExecuteBarrier(cb, barriers);
		}

		if (profile) queries.cmdWriteStart(cb, queryIndexGIInjectLights);
		cmdDispatchInjectLights(cb);
		if (profile) queries.cmdWriteEnd(cb, queryIndexGIInjectLights);

		SyncOperations::cmdExecuteBarrier(cb, { {SyncOperations::constructMemoryBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)} });

		cmdDispatchMergeEmission(cb);

		SyncOperations::cmdExecuteBarrier(cb, { {SyncOperations::constructImageBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_NONE,
			VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_NONE,
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
			m_dynamicEmissionVoxelmap.getImageHandle(), m_dynamicEmissionVoxelmap.getSubresourceRange())} });
	}

private:
	void cmdTransferClearVoxelized(VkCommandBuffer cb);
	void cmdTransferClearDynamicEmissionVoxelmap(VkCommandBuffer cb);
	void cmdPassVoxelize(VkCommandBuffer cb, const BufferMapped& indirectDrawCmdData, const Buffer& vertexData, const Buffer& indexData, uint32_t drawCmdCount, uint32_t drawCmdOffset, uint32_t drawCmdStride);
	void cmdDispatchCreateROMA(VkCommandBuffer cb);
	void cmdDispatchInjectLights(VkCommandBuffer cb);
	void cmdDispatchMergeEmission(VkCommandBuffer cb);
	void cmdDispatchTraceProbes(VkCommandBuffer cb, bool skyboxEnabled);
	void cmdDispatchTraceSpecular(VkCommandBuffer cb, const glm::mat4& worldFromNDC, const glm::vec3& campos, bool skyboxEnabled);
	void cmdDispatchComputeIrradianceAndVisibility(VkCommandBuffer cb);
	void cmdDispatchBlurSpecular(VkCommandBuffer cb);

	void changeHistoryAndNewProbes()
	{
		m_currentNewProbes = m_currentNewProbes ? 0 : 1;
	}

	void changeCurrentBuffers()
	{
		m_currentBuffers = m_currentBuffers ? 0 : 1;
	}

public:
	void initializeDebug(VkDevice device, const ResourceSet& viewprojRS, uint32_t width, uint32_t height, BufferBaseHostAccessible& baseHostBuffer, VkSampler generalSampler, CommandBufferSet& cmdBufferSet, VkQueue queue);


	void cmdDrawBOM(VkCommandBuffer cb, const glm::vec3& camPos);
	void cmdDrawROM(VkCommandBuffer cb, const glm::vec3& camPos, uint32_t romaIndex);
	void cmdDrawAlbedo(VkCommandBuffer cb, const glm::vec3& camPos);
	void cmdDrawMetalness(VkCommandBuffer cb, const glm::vec3& camPos);
	void cmdDrawRoughness(VkCommandBuffer cb, const glm::vec3& camPos);
	void cmdDrawEmission(VkCommandBuffer cb, const glm::vec3& camPos);

	void cmdDrawRadianceProbes(VkCommandBuffer cb);
	void cmdDrawIrradianceProbes(VkCommandBuffer cb);
	void cmdDrawVisibilityProbes(VkCommandBuffer cb);

private:
	void addLightToInject(uint32_t index)
	{
		m_injectedLightsIndices[m_injectedLightsCount++] = index;
	}

	glm::vec3 generateHemisphereDirectionOctohedral(float u, float v)
	{
		u = u * 2.0f - 1.0f;
		v = v * 2.0f - 1.0f;

		glm::vec3 vec{};
		vec.y = 1.0 - std::abs(u) - std::abs(v);
		vec.x = u;
		vec.z = v;

		float t = std::max(-vec.y, 0.0f);

		vec.x += vec.x >= 0.0f ? -t : t;
		vec.z += vec.z >= 0.0f ? -t : t;

		return glm::normalize(vec);
	}


	friend class LightTypes::LightBase;
	friend class LightTypes::PointLight;
	friend class LightTypes::SpotLight;
};

#endif