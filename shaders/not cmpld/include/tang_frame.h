#ifndef TANG_FRAME_HEADER
#define TANG_FRAME_HEADER

#include "math.h"

vec4 quaternionFromRotMat(mat3 m)
{
	vec4 q;
	float tr = m[0][0] + m[1][1] + m[2][2];
	if (tr > 0)
	{
		float S = sqrt(tr + 1.0) * 2; // S=4*qw 
		q.w = 0.25 * S;
		q.x = (m[2][1] - m[1][2]) / S;
		q.y = (m[0][2] - m[2][0]) / S;
		q.z = (m[1][0] - m[0][1]) / S;
	}
	else if ((m[0][0] > m[1][1]) && (m[0][0] > m[2][2]))
	{
		float S = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2; // S=4*qx 
		q.w = (m[2][1] - m[1][2]) / S;
		q.x = 0.25 * S;
		q.y = (m[0][1] + m[1][0]) / S;
		q.z = (m[0][2] + m[2][0]) / S;
	}
	else if (m[1][1] > m[2][2])
	{
		float S = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2; // S=4*qy
		q.w = (m[0][2] - m[2][0]) / S;
		q.x = (m[0][1] + m[1][0]) / S;
		q.y = 0.25 * S;
		q.z = (m[1][2] + m[2][1]) / S;
	}
	else
	{
		float S = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2; // S=4*qz
		q.w = (m[1][0] - m[0][1]) / S;
		q.x = (m[0][2] + m[2][0]) / S;
		q.y = (m[1][2] + m[2][1]) / S;
		q.z = 0.25 * S;
	}
	return q;
}

mat3 rotMatFromQuaternion(vec4 q)
{
	mat3 result;
	float xSq = q.x * q.x;
	float ySq = q.y * q.y;
	float zSq = q.z * q.z;
	float wSq = q.w * q.w;
	float twoX = 2.0f * q.x;
	float twoY = 2.0f * q.y;
	float twoW = 2.0f * q.w;
	float xy = twoX * q.y;
	float xz = twoX * q.z;
	float yz = twoY * q.z;
	float wx = twoW * q.x;
	float wy = twoW * q.y;
	float wz = twoW * q.z;

	result[0][0] = wSq + xSq - ySq - zSq;
	result[1][0] = xy - wz;
	result[2][0] = xz + wy;

	result[0][1] = xy + wz;
	result[1][1] = wSq - xSq + ySq - zSq;
	result[2][1] = yz - wx;

	result[0][2] = xz - wy;
	result[1][2] = yz + wx;
	result[2][2] = wSq - xSq - ySq + zSq;

	return result;
}


vec4 packTangentFrame(vec3 T, vec3 N, vec3 B)
{
	vec4 Q = quaternionFromRotMat(transpose(mat3(T, N, B)));

	vec4 Qabs = abs(Q);

	uint indexOfBiggestComponent = 0;

	float biggestCmp = Qabs[indexOfBiggestComponent];

	for (int i = 1; i < 4; ++i)
	{
		if (Qabs[i] > biggestCmp)
		{
			biggestCmp = Qabs[i];
			indexOfBiggestComponent = i;
		}
	}

	if (Q[indexOfBiggestComponent] < 0.0)
		Q = -Q;

	if (indexOfBiggestComponent != 3)
		Q[indexOfBiggestComponent] = Q[3];

	return vec4(Q.xyz * SQRT_2 * 0.5 + 0.5, float(indexOfBiggestComponent) / 3.0 + 0.1);
}

mat3 unpackTangentFrame(vec4 packedTangentFrame, float handedness)
{
	packedTangentFrame.xyz = (packedTangentFrame.xyz * 2.0 - 1.0) * ONE_OVER_SQRT_2;
	uint biggestComponentIndex = uint(packedTangentFrame.w * 3.0 + 0.1);
	float biggestComponent = sqrt(1.0 - packedTangentFrame.x * packedTangentFrame.x - packedTangentFrame.y * packedTangentFrame.y - packedTangentFrame.z * packedTangentFrame.z);

	float temp = packedTangentFrame[biggestComponentIndex];
	packedTangentFrame[3] = temp;
	packedTangentFrame[biggestComponentIndex] = biggestComponent;

	mat3 rotMat = rotMatFromQuaternion(packedTangentFrame);
	rotMat[2] *= handedness;

	return rotMat;
}

#endif