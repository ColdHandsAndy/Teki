#ifndef RUNIT_CLASS_HEADER
#define RUNIT_CLASS_HEADER

#include <cstdint>

class RUnit
{
private:
	uint32_t m_offsetVertex{};
	uint32_t m_offsetIndex{};
	uint32_t m_byteSizeVertices{};
	uint32_t m_byteSizeIndices{};

	uint32_t m_commandBufferOffset{};

	uint16_t m_vertexSize{};

	//Material ref

public:
	RUnit() = default;
	RUnit(uint32_t vertOff, uint32_t indOff, uint32_t vertexNum, uint32_t indexNum, uint16_t vertexSize, uint16_t indexSize = sizeof(uint32_t));
	~RUnit() = default;

	uint32_t getOffsetVertex();
	uint32_t getOffsetIndex();
	uint32_t getByteSizeVertices();
	uint32_t getByteSizeIndices();
	uint32_t getCommandBufferOffset();
	uint16_t getVertexSize();
};

#endif