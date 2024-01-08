#version 460 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

layout(location = 0) in vec2[3] inTexCoords;
layout(location = 1) in vec3[3] inNormal;
layout(location = 2) in flat uint[3] in_bcList_bcLayer_emList_emLayer;
layout(location = 3) in flat uint[3] in_mrList_mrLayer;

layout(location = 0) out vec3 voxTexCoords;
layout(location = 1) out vec2 outTexCoords;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out flat uint out_bcList_bcLayer_emList_emLayer;
layout(location = 4) out flat uint out_mrList_mrLayer;

void main() 
{    
	vec3 faceNorm = cross(gl_in[1].gl_Position.xyz - gl_in[0].gl_Position.xyz, gl_in[2].gl_Position.xyz - gl_in[0].gl_Position.xyz);
	uint biggestCompIndex = 0;
	float biggestComp = abs(faceNorm[0]);
	
	for (int i = 1; i < 3; ++i)
	{
		float comp = abs(faceNorm[i]);
		if (comp > biggestComp)
		{
			biggestCompIndex = i;
			biggestComp = comp;
		}
	}
	
	vec3 verts[3];
	if (biggestCompIndex == 0)
	{
		verts[0] = vec3(gl_in[0].gl_Position.yz, gl_in[0].gl_Position.x * 0.5 + 0.5);
		verts[1] = vec3(gl_in[1].gl_Position.yz, gl_in[1].gl_Position.x * 0.5 + 0.5);
		verts[2] = vec3(gl_in[2].gl_Position.yz, gl_in[2].gl_Position.x * 0.5 + 0.5);
	}
	else if (biggestCompIndex == 1)
	{
		verts[0] = vec3(gl_in[0].gl_Position.zx, gl_in[0].gl_Position.y * 0.5 + 0.5);
		verts[1] = vec3(gl_in[1].gl_Position.zx, gl_in[1].gl_Position.y * 0.5 + 0.5);
		verts[2] = vec3(gl_in[2].gl_Position.zx, gl_in[2].gl_Position.y * 0.5 + 0.5);
	}
	else
	{
		verts[0] = vec3(gl_in[0].gl_Position.xy, gl_in[0].gl_Position.z * 0.5 + 0.5);
		verts[1] = vec3(gl_in[1].gl_Position.xy, gl_in[1].gl_Position.z * 0.5 + 0.5);
		verts[2] = vec3(gl_in[2].gl_Position.xy, gl_in[2].gl_Position.z * 0.5 + 0.5);
	}
	
	gl_Position = vec4(verts[0], 1.0);
	voxTexCoords = gl_in[0].gl_Position.xyz * 0.5 + vec3(0.5);
	outTexCoords = inTexCoords[0];
	outNormal = inNormal[0];
	out_bcList_bcLayer_emList_emLayer = in_bcList_bcLayer_emList_emLayer[0];
	out_mrList_mrLayer = in_mrList_mrLayer[0];
    EmitVertex();
	gl_Position = vec4(verts[1], 1.0);
	voxTexCoords = gl_in[1].gl_Position.xyz * 0.5 + vec3(0.5);
	outTexCoords = inTexCoords[1];
	outNormal = inNormal[1];
	out_bcList_bcLayer_emList_emLayer = in_bcList_bcLayer_emList_emLayer[0];
	out_mrList_mrLayer = in_mrList_mrLayer[0];
    EmitVertex();
	gl_Position = vec4(verts[2], 1.0);
	voxTexCoords = gl_in[2].gl_Position.xyz * 0.5 + vec3(0.5);
	outTexCoords = inTexCoords[2];
	outNormal = inNormal[2];
	out_bcList_bcLayer_emList_emLayer = in_bcList_bcLayer_emList_emLayer[0];
	out_mrList_mrLayer = in_mrList_mrLayer[0];
    EmitVertex();
	EndPrimitive();
} 