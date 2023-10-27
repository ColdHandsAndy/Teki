#version 460

#extension GL_GOOGLE_include_directive						:  enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 lineColor;
layout(location = 1) out vec3 fragPos;

#define COORDINATE_TRANSFORMATION_SET_INDEX 0
#include "coordinate_transformation_set.h"

void main() 
{
    fragPos = position;
    lineColor = color * 0.8;
    gl_Position = coordTransformData.ndcFromWorld * vec4(position, 1.0);
}