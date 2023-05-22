#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable

layout(set = 0, binding = 0) uniform UBO1 
{
    mat4 view;
    mat4 proj;
    vec3 lightPos;
    vec3 camPos;
} viewproj;

layout (set = 2, binding = 0) uniform sampler2DArray imageListArray[64];

struct DrawDataIndices
{
    uint8_t modelIndex;
    uint8_t index1;
    uint8_t index2;
    uint8_t index3;
    uint8_t bcIndexList;
    uint8_t bcIndexLayer;
    uint8_t nmIndexList;
    uint8_t nmIndexLayer;
    uint8_t mrIndexList;
    uint8_t mrIndexLayer;
    uint8_t emIndexList;
    uint8_t emIndexLayer;
};
layout(set = 3, binding = 0) buffer SSBO1 
{
    DrawDataIndices drawData[];
} drawDataIndices;

layout(location = 0) in flat uint drawID;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 texCoords;
layout(location = 4) in vec3 vertPosition;

layout(location = 0) out vec4 outColor;

vec3 computeBlinnPhong(vec3 camPos, vec3 particlePos, vec3 lightPos, vec3 normalVec)
{
    vec3 baseColor = vec3(texture(imageListArray[drawDataIndices.drawData[drawID].bcIndexList], vec3(texCoords, float(drawDataIndices.drawData[drawID].bcIndexLayer) + 0.1)));
    vec3 ambientColor = vec3(0.4, 1.0, 0.8) * 0.02 * baseColor;
    vec3 diffuseColor = baseColor;
    vec3 specColor = vec3(1.0);

    vec3 lightDir = lightPos - particlePos;
    float distance = length(lightDir);
    distance = distance * distance;
    lightDir = normalize(lightDir);
    float lambertian = max(dot(lightDir, normalVec), 0.0);
    float specular = 0.0;

    if (lambertian > 0.0)
    {
        vec3 viewDir = normalize(camPos - particlePos);
        vec3 halfDir = normalize(lightDir + viewDir);
        float specAngle = max(dot(halfDir, normalVec), 0.0);
        specular = pow(specAngle, 64.0);
    }

    const vec3 lightColor = vec3(1.0, 1.0, 1.0);
    const float lightPower = 18.0;

    vec3 result = ambientColor +
                  diffuseColor * lambertian * lightColor * lightPower / distance +
                  specColor * specular * lightColor * lightPower / distance;

    return result;
}

void main() 
{
    vec3 resultColor = computeBlinnPhong(viewproj.camPos, vertPosition, viewproj.lightPos, normal);
   
    outColor = vec4(resultColor, 1.0);
    //uint listIndex = drawDataIndices.drawData[drawID].bcIndexList;
    //float layerIndex = drawDataIndices.drawData[drawID].bcIndexLayer;
    //outColor = texture(imageListArray[listIndex], vec3(texCoords, float(layerIndex) + 0.1));
}