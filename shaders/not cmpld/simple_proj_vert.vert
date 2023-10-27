#version 460

#extension GL_GOOGLE_include_directive						:  enable

#define COORDINATE_TRANSFORMATION_SET_INDEX 0
#include "coordinate_transformation_set.h"

layout(location = 0) in vec3 position;
layout(location = 0) out vec3 outColor;

void main() 
{
    vec4 vertPos = coordTransformData.ndcFromWorld * vec4(position, 1.0);
    gl_Position = vertPos;
	outColor = vec3(0.03, 0.98, 0.1);
}