#version 460

//#extension GL_EXT_shader_explicit_arithmetic_types_int8    :  enable
#extension GL_KHR_shader_subgroup					       :  enable
#extension GL_KHR_shader_subgroup_basic    				   :  enable
#extension GL_KHR_shader_subgroup_vote    				   :  enable
#extension GL_KHR_shader_subgroup_arithmetic    		   :  enable

layout(set = 0, binding = 1) buffer TilesData
{
	uint tilesWords[];
} tilesData;

layout(set = 0, binding = 4) uniform TilingConsts
{
	uint windowTileWidth;
	uint windowTileHeight;
	uint maxWordsNum;
} tilingConsts;

layout(location = 0) in flat uint inpLightIndex;

void main()
{
	uint xTile = uint(gl_FragCoord.x);
	uint yTile = uint(gl_FragCoord.y);
	
	uint lightBit = 1 << ( inpLightIndex % 32 );
	uint word = inpLightIndex / 32;
	
	uint tileIndex = yTile * tilingConsts.windowTileWidth + xTile;
	
	atomicOr(tilesData.tilesWords[tileIndex * tilingConsts.maxWordsNum + word], lightBit);
}