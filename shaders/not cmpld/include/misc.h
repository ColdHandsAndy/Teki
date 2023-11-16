#ifndef MISC_HEADER
#define MISC_HEADER

//Texture array coord from direction vector
vec3 getTexArrayCoordinateFromDirection(vec3 v)
{
	vec3 vAbs = abs(v);
	float ma; 
	vec2 coord; 
	float layer;
	if(vAbs.z >= vAbs.x && vAbs.z >= vAbs.y)
	{
		layer = v.z < 0.0 ? 5.0 : 4.0;
		ma = 0.5 / vAbs.z;
		coord = vec2(v.z < 0.0 ? -v.x : v.x, -v.y);
	}
	else if(vAbs.y >= vAbs.x)
	{
		layer = v.y < 0.0 ? 3.0 : 2.0;
		ma = 0.5 / vAbs.y;
		coord = vec2(v.x, v.y < 0.0 ? -v.z : v.z);
	}
	else
	{
		layer = v.x < 0.0 ? 1.0 : 0.0;
		ma = 0.5 / vAbs.x;
		coord = vec2(v.x < 0.0 ? v.z : -v.z, -v.y);
	}
	
	coord = coord * ma + 0.5;
	
	return vec3(coord, layer + 0.1);
}
//Draw data indices layout
struct DrawData
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
//Project sphere to screen space coords
void projectSphere(vec3 p, float r, float proj00, float proj11, out float bvWidth, out float bvHeight, out vec2 bvCenter)
{
    vec3 cr = p * r;
    float czr2 = p.z * p.z - r * r;

    float vx = sqrt(p.x * p.x + czr2);
    float minX = (vx * p.x - cr.z) / (vx * p.z + cr.x);
    float maxX = (vx * p.x + cr.z) / (vx * p.z - cr.x);

    float vy = sqrt(p.y * p.y + czr2);
    float minY = (vy * p.y - cr.z) / (vy * p.z + cr.y);
    float maxY = (vy * p.y + cr.z) / (vy * p.z - cr.y);

	float projMinX = minX * proj00 * 0.5 + 0.5;
	float projMaxX = maxX * proj00 * 0.5 + 0.5;
	float projMinY = minY * proj11 * 0.5 + 0.5;
	float projMaxY = maxY * proj11 * 0.5 + 0.5;
	
	bvWidth  = abs(projMaxX - projMinX);
	bvHeight = abs(projMaxY - projMinY);
	bvCenter = clamp(vec2(projMinX + bvWidth * 0.5, projMinY + bvHeight * 0.5), 0.0, 1.0);
}

vec3 getWorldPositionFromDepth(mat4 worldFromNdc, vec2 uv, float depth)
{
	vec4 res = worldFromNdc * vec4(uv * 2.0 - 1.0, depth, 1.0);
	return res.xyz / res.w;
}

vec3 RGBtoYCoCg(vec3 c)
{
	return mat3(0.25, 0.5, -0.25, 0.5, 0, 0.5, 0.25, -0.5, -0.25) * c;
}
vec3 YCoCgToRGB(vec3 c)
{
	return mat3(1, 1, 1, 1, 0, -1, -1, 1, -1) * c;
}

#endif