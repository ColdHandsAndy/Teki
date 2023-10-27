#ifndef WORLD_TRANSFORM_SET_HEADER
#define WORLD_TRANSFORM_SET_HEADER

struct CoordinateTransformationData
{
	mat4 ndcFromWorld;
	mat4 viewFromWorld;
	mat4 ndcFromView;
	mat4 worldFromNdc;
	mat4 worldFromView;
	mat4 viewFromNdc;
};

layout(set = COORDINATE_TRANSFORMATION_SET_INDEX, binding = 0) uniform CoordinateTransformation
{
	CoordinateTransformationData coordTransformData;
};

#endif