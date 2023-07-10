#version 460

#extension GL_KHR_shader_subgroup_arithmetic				: enable


#define PI 3.141592653589

#define FOV_EXPANSION_SCALE 0.9090909
#define FOV_EXPANSION_BIAS  0.0454545

#define RAND_TEXTURE_SIZE 4

#define NUM_DIRECTIONS 8
#define NUM_STEPS 6


layout (location = 0) in vec2 inUV;

layout(push_constant) uniform PushConstants
{
	vec4 uvTransformData;
	
	vec2 invResolution;
	vec2 resolution;
	
	float radius;
	float aoExponent;
	float angleBias;
	float negInvR2;
} pushConstants;


layout(set = 0, binding = 0) uniform sampler2D linDepthTexture;
layout(set = 0, binding = 1) uniform sampler2D randTexture;


vec3 getPos(vec2 uv)
{
	float linDepth = texture(linDepthTexture, uv).x;
	vec4 transformData = pushConstants.uvTransformData;
	return vec3((uv * transformData.xy + transformData.zw) * linDepth, linDepth);
}
vec3 getPos(vec2 uv, float linDepth)
{
	vec4 transformData = pushConstants.uvTransformData;
	return vec3((uv * transformData.xy + transformData.zw) * linDepth, linDepth);
}
vec3 getNorm(vec2 uv, vec3 viewPos)
{
	float linDepth = viewPos.z;
	
	float resX = pushConstants.resolution.x; 
    vec4 H;
	H.x = texture(linDepthTexture, uv - vec2(1.0 / resX, 0.0)).x;
    H.y = texture(linDepthTexture, uv + vec2(1.0 / resX, 0.0)).x;
    H.z = texture(linDepthTexture, uv - vec2(2.0 / resX, 0.0)).x;
    H.w = texture(linDepthTexture, uv + vec2(2.0 / resX, 0.0)).x;
	
	vec2 he = abs(2.0 * H.xy - H.zw - linDepth);
    vec3 hDeriv;
    if (he.x > he.y)
        hDeriv = getPos(uv + vec2(1.0 / resX, 0.0), H.y) - viewPos;
    else
        hDeriv = -getPos(uv - vec2(1.0 / resX, 0.0), H.x) + viewPos;

	float resY = pushConstants.resolution.y; 
    vec4 V;
	V.x = texture(linDepthTexture, uv - vec2(0.0, 1.0 / resY)).x;
    V.y = texture(linDepthTexture, uv + vec2(0.0, 1.0 / resY)).x;
    V.z = texture(linDepthTexture, uv - vec2(0.0, 2.0 / resY)).x;
    V.w = texture(linDepthTexture, uv + vec2(0.0, 2.0 / resY)).x;
	
	vec2 ve = abs(2.0 * V.xy - V.zw - linDepth);
    vec3 vDeriv;
    if (ve.x > ve.y)
        vDeriv = getPos(uv + vec2(0.0, 1.0 / resY), V.y) - viewPos;
    else
        vDeriv = -getPos(uv - vec2(0.0, 1.0 / resY), V.x) + viewPos;

    return normalize(cross(hDeriv, vDeriv));
}
vec2 rotateDirection(vec2 dir, vec2 cosSin)
{
	return vec2(dir.x * cosSin.x - dir.y * cosSin.y, dir.x * cosSin.y + dir.y * cosSin.x);
}


float computeAO(vec3 P, vec3 N, vec3 S)
{
  vec3 V = S - P;
  float VdotV = dot(V, V);
  float NdotV = dot(N, V) * 1.0 / sqrt(VdotV);

  return clamp(NdotV - pushConstants.angleBias, 0.0, 1.0) * clamp(VdotV * pushConstants.negInvR2 + 1.0, 0.0, 1.0);
}
float computeAmbientOcclusion(vec2 uv, float pixelRad, vec4 randInp, vec3 viewPos, vec3 viewNorm)
{
	float stepSizePixels = pixelRad / (NUM_STEPS + 1);

	const float alpha = 2.0 * PI / NUM_DIRECTIONS;
	float occlusion = 0.0;
	
	for (float directionIndex = 0; directionIndex < NUM_DIRECTIONS; ++directionIndex)
    {
		float angle = alpha * directionIndex;
	
		vec2 direction = rotateDirection(vec2(cos(angle), sin(angle)), randInp.xy);
		float rayPixels = (randInp.z * stepSizePixels + 1.0);
	
		for (float stepIndex = 0; stepIndex < NUM_STEPS; ++stepIndex)
		{
			vec2 snappedUV = round(rayPixels * direction) * pushConstants.invResolution + uv;
			vec3 s = getPos(snappedUV);
		
			rayPixels += stepSizePixels;
			
			if (viewPos.z - s.z < 0.5)
				occlusion += computeAO(viewPos, viewNorm, s);
		}
    }

	occlusion *= pushConstants.aoExponent / (NUM_DIRECTIONS * NUM_STEPS);
	
	return clamp(1.0 - occlusion * 2.0,0,1);
}

layout (location = 0) out float AO;

void main()
{
	vec2 fetchUV = inUV * FOV_EXPANSION_SCALE + FOV_EXPANSION_BIAS; //expansion corrected uv
	vec3 viewPos = getPos(fetchUV);
	vec3 viewNorm = getNorm(fetchUV, viewPos);
	
	float pixelRad = pushConstants.radius / viewPos.z;

	if (subgroupMax(pixelRad) < 1.0)
	{
		AO = 1.0;
		return;
	}
		
	vec4 randInp = texture(randTexture, gl_FragCoord.xy / RAND_TEXTURE_SIZE);
	
	float result = computeAmbientOcclusion(fetchUV, pixelRad, randInp, viewPos, viewNorm);
	
	AO = pow(result, pushConstants.aoExponent);
}