#version 460

#define FOV_EXPANSION_SCALE 0.9090909
#define FOV_EXPANSION_BIAS  0.0454545
#define KERNEL_RADIUS 3

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform sampler2D linearDepth;
layout(set = 0, binding = 1) uniform sampler2D AO;

layout(push_constant) uniform PushConsts 
{
	vec2  invResolution;
	float sharpness;
} pushConstants;

layout(location = 0) out float blurredAO;

float blurFunction(vec2 uv, vec2 linDepthUV, float r, float centerD, inout float totalW)
{
	float c = texture(AO, uv).x;
	float d = texture(linearDepth, linDepthUV).x;
	
	const float blurSigma = float(KERNEL_RADIUS) * 0.5;
	const float blurFalloff = 1.0 / (2.0 * blurSigma * blurSigma);
	
	float ddiff = (d - centerD) * pushConstants.sharpness;
	float w = exp2(-r * r * blurFalloff - ddiff * ddiff);
	totalW += w;
	
	return c*w;
}

void main()
{
	vec2 linDepthUV = inUV * FOV_EXPANSION_SCALE + FOV_EXPANSION_BIAS;
	vec2 uv = inUV;
	
	float centerAO = texture(AO, uv).x;
	float centerD = texture(linearDepth, linDepthUV).x;
	
	float totalAO = centerAO;
	float totalW = 1.0;
	
	for (float r = 1; r <= KERNEL_RADIUS; ++r)
	{
		vec2 movedUV = uv + pushConstants.invResolution * r;
		totalAO += blurFunction(movedUV, linDepthUV, r, centerD, totalW);  
	}
	
	for (float r = 1; r <= KERNEL_RADIUS; ++r)
	{
		vec2 movedUV = uv - pushConstants.invResolution * r;
		totalAO += blurFunction(movedUV, linDepthUV, r, centerD, totalW);  
	}
	
	blurredAO = totalAO/totalW;
}