#version 460

#define M_PI 3.1415926535897932384626433832795

layout(set = 0, binding = 0) uniform ViewprojTransform
{
    mat4 view;
    mat4 proj;
} viewproj;

layout(location = 0) in vec3 vertPosition;
layout(location = 1) in vec4 lightDir_lightLength;
layout(location = 2) in vec4 lightPos_lightCutoff;

vec3 rotateToDir(vec3 vector, vec3 direction)
{
    vec3 xaxis = cross(vec3(0.0, 1.0, 0.0), direction);
    xaxis = normalize(xaxis);

    vec3 yaxis = cross(direction, xaxis);
    yaxis = normalize(yaxis);

    xaxis = -xaxis;

    mat3 rotationMat = mat3(vec3(xaxis.x, xaxis.y, xaxis.z), vec3(yaxis.x, yaxis.y, yaxis.z), vec3(direction.x, direction.y, direction.z));

    return rotationMat * vector;
}

vec3 rotateZ(float angle, vec3 vector)
{
    vec3 result = vector;
    float sinVal = sin(angle);
    float cosVal = cos(angle);
    result.x =  vector.x * cosVal - vector.y * sinVal;
	result.y =  vector.x * sinVal + vector.y * cosVal;
	return result;
}

void main() 
{
    vec3 lightVertPos;

    vec3 direction = lightDir_lightLength.xyz;
    vec3 position = lightPos_lightCutoff.xyz;
    float length = lightDir_lightLength.w;
    float cutoff = lightPos_lightCutoff.w;

    if (gl_VertexIndex == 0)
    {
        gl_Position = viewproj.proj * viewproj.view * vec4(vertPosition + position, 1.0);
        return;
    }

    if (dot(direction, vec3(0.0, 1.0, 0.0)) > 0.999 || dot(direction, vec3(0.0, 1.0, 0.0)) < -0.999)
    {  
        direction.x += 0.001;
    }

    lightVertPos = rotateZ((M_PI / 15) * (gl_VertexIndex),  vec3(0.0, sin(cutoff), cos(cutoff)));
    lightVertPos = rotateToDir(lightVertPos, direction);
    gl_Position = viewproj.proj * viewproj.view * vec4(lightVertPos * length + position, 1.0);
    return;
}