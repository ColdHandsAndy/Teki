#ifndef TONEMAP_HEADER
#define TONEMAP_HEADER

float ColToneB(float hdrMax, float contrast, float shoulder, float midIn, float midOut)
{
    return -((-pow(midIn, contrast) + (midOut * (pow(hdrMax, contrast * shoulder) * pow(midIn, contrast) -
        pow(hdrMax, contrast) * pow(midIn, contrast * shoulder) * midOut)) /
        (pow(hdrMax, contrast * shoulder) * midOut - pow(midIn, contrast * shoulder) * midOut)) /
        (pow(midIn, contrast * shoulder) * midOut));
} 
float ColToneC(float hdrMax, float contrast, float shoulder, float midIn, float midOut)
{
    return (pow(hdrMax, contrast * shoulder) * pow(midIn, contrast) - pow(hdrMax, contrast) * pow(midIn, contrast * shoulder) * midOut) /
        (pow(hdrMax, contrast * shoulder) * midOut - pow(midIn, contrast * shoulder) * midOut);
}
float ColTone(float x, vec4 p)
{
    float z = pow(x, p.r);
    return z / (pow(z, p.g) * p.b + p.a);
}
vec3 TimothyTonemapper(vec3 color)
{
    const float hdrMax = 16.0;
    const float contrast = 2.0;
    const float shoulder = 1.0;
    const float midIn = 0.18;
    const float midOut = 0.18;

    float b = ColToneB(hdrMax, contrast, shoulder, midIn, midOut);
    float c = ColToneC(hdrMax, contrast, shoulder, midIn, midOut);

    float peak = max(color.r, max(color.g, color.b));
    vec3 ratio = color / peak;
    peak = ColTone(peak, vec4(contrast, shoulder, b, c));

    float crosstalk = 4.0;
    float saturation = contrast;
    float crossSaturation = contrast * 16.0;

    float white = 1.0;

    float sDivCs = saturation / crossSaturation;
    ratio.x = pow(abs(ratio.x), sDivCs);
    ratio.y = pow(abs(ratio.y), sDivCs);
    ratio.z = pow(abs(ratio.z), sDivCs);

    float powPC = pow(peak, crosstalk);
    ratio.x = mix(ratio.x, white, powPC);
    ratio.y = mix(ratio.y, white, powPC);
    ratio.z = mix(ratio.z, white, powPC);
    ratio.x = pow(abs(ratio.x), crossSaturation);
    ratio.y = pow(abs(ratio.y), crossSaturation);
    ratio.z = pow(abs(ratio.z), crossSaturation);

    color = peak * ratio;
    return color;
}

#endif