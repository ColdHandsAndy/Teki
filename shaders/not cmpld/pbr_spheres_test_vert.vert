#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

layout(location = 0) out flat int instID;
layout(location = 1) out vec3 posInp;
layout(location = 2) out vec2 texCInp;
layout(location = 3) out vec3 normInp;
layout(location = 4) out vec3 colorInp;
layout(location = 5) out flat float metalnessInp;
layout(location = 6) out flat float roughnessInp;

layout(set = 0, binding = 0) uniform ViewProjMatrices
{
    mat4 view;
    mat4 proj;
} viewproj;

struct InstanceData
{
    vec4 color_met;
    vec4 position_roughness;
};
layout(set = 0, binding = 1) buffer PerInstanceData
{
    InstanceData data[];
} instData;

void main() 
{
    instID = gl_InstanceIndex;
    texCInp = texCoord;
	posInp = position * 1.5 + instData.data[gl_InstanceIndex].position_roughness.xyz;
    normInp = normalize(position);
    colorInp = instData.data[gl_InstanceIndex].color_met.xyz;
    metalnessInp = instData.data[gl_InstanceIndex].color_met.w;
    roughnessInp = instData.data[gl_InstanceIndex].position_roughness.w;
    vec4 vertPos = viewproj.proj * viewproj.view * vec4(posInp, 1.0);
    gl_Position = vertPos;
}