#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int16    :  enable

layout(set = 0, binding = 0) uniform ViewProjMatrices
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(set = 0, binding = 2) buffer IndexBuffer
{
    uint16_t indices[];
} indexBuffer;

struct UnifiedLightData
{
	vec3 position;
	float lightLength;
	vec3 spectrum;
	float cutoffCos;
	vec3 direction;
	float falloffCos;
};
layout(set = 0, binding = 3) buffer LightData
{
    UnifiedLightData lights[];
} lightData;

layout(location = 0) in vec3 inpPosition;
layout(location = 0) out flat uint outLightIndex;

void main()
{
	uint lightIndex = indexBuffer.indices[gl_InstanceIndex];
	outLightIndex = lightIndex;
	UnifiedLightData light = lightData.lights[lightIndex];
	gl_Position = viewproj.proj * viewproj.view * vec4(inpPosition * light.lightLength + light.position, 1.0);
}