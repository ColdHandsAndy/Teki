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

vec3 TimothyTonemapper(vec3 x) 
{
    const float a = 1.6;
    const float d = 0.977;
    const float hdrMax = 8.0;
    const float midIn = 0.18;
    const float midOut = 0.267;

    // Can be precomputed
    const float b =
        (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    const float c =
        (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
        ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return vec3(pow(x.x, a), pow(x.y, a), pow(x.z, a)) / (vec3(pow(x.x, a * d), pow(x.y, a * d), pow(x.z, a * d)) * b + c);
}

#endif