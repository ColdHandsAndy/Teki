#version 460

layout(location = 0) in float inDepth;

layout(location = 0) out float linDepth;

void main() 
{
    linDepth = inDepth;
}