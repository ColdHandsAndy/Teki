#version 460

#extension GL_GOOGLE_include_directive						:  enable

#include "tang_frame.h"

//Math
#define PI 3.141592653589
#define TWO_PI (2.0 * PI)
#define ONE_OVER_PI (1.0 / PI)
#define ONE_OVER_TWO_PI (1.0 / TWO_PI)
#define SQRT_2 1.41421356237309
#define ONE_OVER_SQRT_2 0.7071067811865475244

layout(location = 0) in flat uint drawID;
layout(location = 1) in vec3 inNorm;
layout(location = 2) in vec3 inTang;
layout(location = 3) in flat float inTangSign;
layout(location = 4) in vec2 inTexC;

layout(location = 0) out uint outputUV;
layout(location = 1) out vec4 outputTangentFramePacked;
layout(location = 2) out uint outputDrawID;

#define COORDINATE_TRANSFORMATION_SET_INDEX 0
#include "coordinate_transformation_set.h"

void main()
{
	vec3 N = normalize(inNorm);
	vec3 B = normalize(cross(inTang, N));
	vec3 T = cross(N, B);
	outputTangentFramePacked = packTangentFrame(T,N,B);
    uint drawID_frameHandednessBit = drawID | (inTangSign < 0.0 ? 0x8000 : 0x0000);
	outputDrawID = drawID_frameHandednessBit;
	outputUV = packHalf2x16(inTexC + dFdxFine(inTexC) * coordTransformData.ndcFromWorld[2][0] - dFdyFine(inTexC) * coordTransformData.ndcFromWorld[2][1]);
}