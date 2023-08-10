#version 460

layout(location = 0) in float inLinDepth;

void main()
{
    gl_FragDepth = inLinDepth;
}