#version 460

#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable

layout(push_constant) uniform PushConsts 
{
	vec3 camPos;
} pushConstants;

layout(set = 0, binding = 0) uniform ViewProjMatrices 
{
    mat4 view;
    mat4 proj;
} viewproj;

layout (set = 1, binding = 0) uniform sampler2DArray imageListArray[64];

struct DrawDataIndicesLayout
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
layout(set = 2, binding = 0) buffer DrawDataIndices 
{
    DrawDataIndicesLayout drawData[];
} drawDataIndices;

struct DirectionalLightLayout
{
    vec3 spectrum;
    vec3 direction;
};
layout(std140, set = 3, binding = 0) uniform DirectionalLight
{
    DirectionalLightLayout light;
} dirLight;
struct PointLightLayout
{
    vec3 position;
    vec3 spectrum;
};
layout(std140, set = 3, binding = 1) uniform PointLights
{
    uint lightNumber;
    PointLightLayout lights[64];
} pointLights;
struct SpotLightLayout
{
    vec3 position;
    vec4 spectrum_startCutoffCos;
    vec4 direction_endCutoffCos;
};
layout(std140, set = 3, binding = 2) uniform SpotLights
{
    uint lightNumber;
    SpotLightLayout lights[64];
} spotLights;


layout(location = 0) in flat uint drawID;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 tangent;
layout(location = 3) in vec2 texCoords;
layout(location = 4) in vec3 fragmentPos;

layout(location = 0) out vec4 finalColor;
layout(depth_unchanged) out float gl_FragDepth;

vec3 directionalBlinnPhong(vec3 spectrum, vec3 diffuseColor, vec3 cameraPos, vec3 particlePos, vec3 normalVec, vec3 lightDir)
{
    vec3 specColor = vec3(1.0);

    vec3 dirToLight = -lightDir;
    dirToLight = normalize(dirToLight);
    float lambertian = max(dot(dirToLight, normalVec), 0.0);
    float specular = 0.0;

    if (lambertian > 0.0)
    {
        vec3 viewDir = normalize(cameraPos - particlePos);
        vec3 halfDir = normalize(dirToLight + viewDir);
        float specAngle = max(dot(halfDir, normalVec), 0.0);
        specular = pow(specAngle, 64.0);
    }

    vec3 result = diffuseColor * lambertian * spectrum +
                  specColor * specular * spectrum;

    return result;
}
vec3 pointBlinnPhong(vec3 spectrum, vec3 diffuseColor, vec3 cameraPos, vec3 particlePos, vec3 normalVec, vec3 lightPos)
{
    vec3 specColor = vec3(1.0);

    vec3 dirToLight = lightPos - particlePos;
    float distance = length(dirToLight);
    distance = distance * distance;
    dirToLight = normalize(dirToLight);
    float lambertian = max(dot(dirToLight, normalVec), 0.0);
    float specular = 0.0;

    if (lambertian > 0.0)
    {
        vec3 viewDir = normalize(cameraPos - particlePos);
        vec3 halfDir = normalize(dirToLight + viewDir);
        float specAngle = max(dot(halfDir, normalVec), 0.0);
        specular = pow(specAngle, 64.0);
    }

    vec3 result = diffuseColor * lambertian * spectrum / distance +
                  specColor * specular * spectrum / distance;

    return result;
}
vec3 spotBlinnPhong(vec3 spectrum, vec3 diffuseColor, vec3 cameraPos, vec3 particlePos, vec3 normalVec, vec3 lightPos, vec3 lightDir, float cutoffStartCos, float cutoffEndCos)
{
    vec3 specColor = vec3(1.0);

    vec3 dirToLight = lightPos - particlePos;
    float distance = length(dirToLight);
    distance = distance * distance;
    dirToLight = normalize(dirToLight);
    float lambertian = max(dot(dirToLight, normalVec), 0.0);
    float specular = 0.0;

    if (lambertian > 0.0)
    {
        vec3 viewDir = normalize(cameraPos - particlePos);
        vec3 halfDir = normalize(dirToLight + viewDir);
        float specAngle = max(dot(halfDir, normalVec), 0.0);
        specular = pow(specAngle, 64.0);
    }

    float cutoffIntensity = clamp((-dot(lightDir, dirToLight) - cutoffEndCos) / (cutoffStartCos - cutoffEndCos), 0.0, 1.0);

    vec3 result = (diffuseColor * lambertian * spectrum / distance +
                  specColor * specular * spectrum / distance) * cutoffIntensity;

    return result;
}

vec3 computeDiffuseAndSpecularComponents(vec3 diffuseColor, vec3 cameraPos, vec3 particlePos, vec3 normalVec)
{
    vec3 result = vec3(0.0);
    result += directionalBlinnPhong(dirLight.light.spectrum, diffuseColor, cameraPos, particlePos, normalVec, dirLight.light.direction);
    uint pointLightCount = pointLights.lightNumber;
    for (uint i = 0; i < pointLightCount; ++i)
    {
        result += pointBlinnPhong(pointLights.lights[i].spectrum, diffuseColor, cameraPos, particlePos, normalVec, pointLights.lights[i].position);
    }
    uint spotLightCount = spotLights.lightNumber;
    for (uint i = 0; i < spotLightCount; ++i)
    {
        result += spotBlinnPhong(vec3(spotLights.lights[i].spectrum_startCutoffCos), diffuseColor, cameraPos, particlePos, normalVec, spotLights.lights[i].position, vec3(spotLights.lights[i].direction_endCutoffCos), spotLights.lights[i].spectrum_startCutoffCos.w, spotLights.lights[i].direction_endCutoffCos.w);
    }
    return result;
}

void main() 
{
    vec3 baseColor = vec3(texture(imageListArray[drawDataIndices.drawData[drawID].bcIndexList], vec3(texCoords, float(drawDataIndices.drawData[drawID].bcIndexLayer) + 0.1)));
    vec3 ambientColor = vec3(0.2, 0.8, 0.0) * 0.02 * baseColor;

    vec3 resultColor = ambientColor + computeDiffuseAndSpecularComponents(baseColor, pushConstants.camPos, fragmentPos, normalize(normal));
   
    finalColor = vec4(resultColor, 1.0);
}