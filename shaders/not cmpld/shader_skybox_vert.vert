#version 460

#extension GL_GOOGLE_include_directive						:  enable

#define COORDINATE_TRANSFORMATION_SET_INDEX 0
#include "coordinate_transformation_set.h"

layout(location = 0) in vec3 position;

layout(location = 0) out vec3 outUVW;

void main() 
{
    outUVW = position;
    vec4 vertPos = vec4(position, 1.0);
    mat4 skyboxView = coordTransformData.viewFromWorld;
    skyboxView[3] = vec4(0.0, 0.0, 0.0, 1.0);
    vertPos = coordTransformData.ndcFromView * skyboxView * vertPos;
    gl_Position = vertPos;
}