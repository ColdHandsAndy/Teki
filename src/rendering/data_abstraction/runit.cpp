#include "RUnit.h"

RUnit::RUnit(uint32_t vertOff, uint32_t indOff, uint32_t vertexNum, uint32_t indexNum, uint16_t vertexSize, uint16_t indexSize)
    : m_offsetVertex{ vertOff }, m_offsetIndex{ indOff }, m_vertexSize{ vertexSize }
{
    m_byteSizeVertex = vertexNum * vertexSize;
    m_byteSizeIndex = indexNum * indexSize;
}

uint32_t RUnit::getOffsetVertex()
{
    return m_offsetVertex;
}

uint32_t RUnit::getOffsetIndex()
{
    return m_offsetIndex;
}

uint32_t RUnit::getByteSizeVertex()
{
    return m_byteSizeVertex;
}

uint32_t RUnit::getByteSizeIndex()
{
    return m_byteSizeIndex;
}

uint16_t RUnit::getVertexSize()
{
    return m_vertexSize;
}
