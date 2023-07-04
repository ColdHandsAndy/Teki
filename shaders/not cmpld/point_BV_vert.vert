#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int16    :  enable

layout(set = 0, binding = 0) uniform ViewProjMatrices
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(set = 0, binding = 1) buffer IndexBuffer
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
layout(set = 0, binding = 2) buffer LightData
{
    UnifiedLightData lights[];
} lightData;

layout(location = 0) in vec3 position;
layout(location = 0) out vec3 outColor;

void main() 
{
	uint lightIndex = indexBuffer.indices[gl_InstanceIndex];
	UnifiedLightData light = lightData.lights[lightIndex];
    vec4 vertPos = viewproj.proj * viewproj.view * vec4((position * light.lightLength) + light.position, 1.0);
    gl_Position = vertPos;
	outColor = vec3(0.93, 0.48, 0.23);
}