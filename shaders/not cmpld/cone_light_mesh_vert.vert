#version 460

#extension GL_GOOGLE_include_directive						:  enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16    :  enable

#include "lighting.h"

layout(push_constant) uniform PushConstants
{
	float sizemod;
} pushConstants;

layout(set = 0, binding = 0) uniform ViewProjMatrices
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(set = 0, binding = 1) buffer IndexBuffer
{
    uint16_t indices[];
} indexBuffer;

layout(set = 0, binding = 2) buffer LightData
{
    UnifiedLightData lights[];
} lightData;

const uint NO_TRANSFORM = 0;
const uint MID_TRANSFORM = 1;
const uint CAP_TRANSFORM = 2;

vec3 spotVertexTransform(UnifiedLightData light, uint transformType, vec3 transformDirection)
{
	vec3 resPos = vec3(0.0);
	float cosTheta = light.cutoffCos;
	float sinTheta = sqrt(1 - cosTheta * cosTheta);
	float tanHalfTheta = sinTheta / (1.0 + cosTheta);
	
	//vec3 up = vec3(0.0, 1.0, 0.0);
	
	if (transformType != NO_TRANSFORM)
	{
		if (transformType == CAP_TRANSFORM)
		{
			//resPos = up + transformDirection * tanHalfTheta;
			resPos = transformDirection * tanHalfTheta;
			resPos.y += 1.0;
		}
		else
		{
			//resPos = up * cosTheta + transformDirection * sinTheta;
			resPos = transformDirection * sinTheta /*  * 2.0  */; //Multiplication by 2.0 is omitted so the volumes are more usable
			resPos.y += cosTheta;
		}
		//float cosPhi = dot(light.direction, up);
		float cosPhi = light.direction.y;
		float sinPhi = sqrt(1 - cosPhi * cosPhi);
		//vec3 rotAxis = cross(up, light.direction);
		
		vec3 rotAxis = normalize(vec3(light.direction.z, 0.0, -light.direction.x));
		
		resPos = resPos * cosPhi + cross(rotAxis, resPos) * sinPhi + rotAxis * dot(rotAxis, resPos) * (1.0 - cosPhi);
	}
	
	return resPos;
}

//const vec3 spotDirections[3] = (vec3(0.0, 0.0, 1.0), vec3(-0.86603, 0, -0.5), vec3(0.86603, 0, -0.5));
const vec3 spotDirections[6] = vec3[](vec3(0.0, 0.0, 1.0), vec3(-0.86603, 0, -0.5), vec3(-0.86603, 0, -0.5), vec3(0.86603, 0, -0.5), vec3(0.86603, 0, -0.5), vec3(0.0, 0.0, 1.0));

const uint vertexTransformTypes[9] = uint[](NO_TRANSFORM, MID_TRANSFORM, MID_TRANSFORM, MID_TRANSFORM, MID_TRANSFORM, CAP_TRANSFORM, MID_TRANSFORM, CAP_TRANSFORM, CAP_TRANSFORM);
const bool RIGHT = true;
const bool LEFT = false;
const bool directionChoice[9] = bool[](RIGHT, RIGHT, LEFT, LEFT, RIGHT, LEFT, RIGHT, RIGHT, LEFT);

layout(location = 0) out vec3 outColor;

void main()
{
	uint vert = gl_VertexIndex % 9;
	uint part = gl_VertexIndex / 9;

	bool capPart = part > 2;
	uint lightIndex = indexBuffer.indices[gl_InstanceIndex];
	UnifiedLightData light = lightData.lights[lightIndex];
	gl_Position = 
	viewproj.proj * viewproj.view 
	* vec4(spotVertexTransform(light, capPart ? CAP_TRANSFORM : vertexTransformTypes[vert], spotDirections[capPart ? vert * 2 : part * 2 + uint(directionChoice[vert])])
	* light.lightLength * pushConstants.sizemod + light.position, 1.0);
	outColor = vec3(0.35, 0.56, 0.94);
}