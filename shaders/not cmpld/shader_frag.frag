#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable

layout (set = 2, binding = 0) uniform sampler2DArray imageListArray[64];

struct DrawDataIndices
{
    uint8_t index0;
    uint8_t index1;
    uint8_t index2;
    uint8_t index3;
};
layout(set = 3, binding = 0) buffer SSBO1 
{
    DrawDataIndices drawData[];
} drawDataIndices;

layout(location = 0) in vec2 texCoords;
layout(location = 1) in flat uint listIndex;
layout(location = 2) in flat uint layerIndex;

layout(location = 0) out vec4 outColor;

void main() 
{
   outColor = texture(imageListArray[listIndex], vec3(texCoords, float(layerIndex) + 0.1));
   //outColor = vec4(1.0, 0.0, 0.0, 1.0);
}