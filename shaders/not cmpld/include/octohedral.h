#ifndef OCTOHEDRAL_HEADER
#define OCTOHEDRAL_HEADER

vec2 encodeOctohedralNegOneToPosOne(vec3 dir)
{
	vec2 p = dir.xz * (1.0 / (abs(dir.x) + abs(dir.y) + abs(dir.z)));
	vec2 res = (dir.y <= 0.0) ? ((1.0 - abs(p.yx)) * vec2((p.x >= 0.0) ? +1.0 : -1.0, (p.y >= 0.0) ? +1.0 : -1.0)) : p;
	return res;
}
vec2 encodeOctohedralZeroToOne(vec3 dir)
{
	return encodeOctohedralNegOneToPosOne(dir) * 0.5 + 0.5;
}
vec3 decodeOctohedralNegOneToPosOne(vec2 uv)
{
	float u = uv.x;
	float v = uv.y;

	vec3 vec;
	vec.y = 1.0 - abs(u) - abs(v);
	vec.x = u;
	vec.z = v;

	float t = max(-vec.y, 0.0f);

	vec.x += vec.x >= 0.0f ? -t : t;
	vec.z += vec.z >= 0.0f ? -t : t;

	return normalize(vec);
}
vec3 decodeOctohedralZeroToOne(vec2 uv)
{
	return decodeOctohedralNegOneToPosOne(uv * 2.0 - 1.0);
}

#endif