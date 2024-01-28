#ifndef UI_DATA_HEADER
#define UI_DATA_HEADER

#include <cstdint>

#include <LegitProfiler/ImGuiProfilerRenderer.h>
#include "src/rendering/data_management/buffer_class.h"

struct UiData
{
    enum VoxelDebugType
    {
        BOM_VOXEL_DEBUG,
        ROM_VOXEL_DEBUG,
        ALBEDO_VOXEL_DEBUG,
        METALNESS_VOXEL_DEBUG,
        ROUGHNESS_VOXEL_DEBUG,
        EMISSION_VOXEL_DEBUG,
        NONE_VOXEL_DEBUG
    };
    enum ProbeDebugType
    {
        RADIANCE_PROBE_DEBUG,
        IRRADIANCE_PROBE_DEBUG,
        VISIBILITY_PROBE_DEBUG,
        NONE_PROBE_DEBUG
    };



    bool drawBVs{ false };
    bool drawLightProxies{ true };
    bool drawSpaceGrid{ false };
    bool skyboxEnabled{ true };
    bool hiZvis{ false };
    int hiZmipLevel{ 0 };
    int voxelDebug{ NONE_VOXEL_DEBUG };
    int probeDebug{ NONE_PROBE_DEBUG };
    bool showOBBs{ false };
    int indexROM{ 0 };
    uint32_t countROM{ 1 };
    uint32_t frustumCulledCount{ 0 };
    BufferMapped finalDrawCount;
    std::vector<legit::ProfilerTask> gpuTasks{};
    //std::vector<legit::ProfilerTask> cpuTasks{};
};


#endif