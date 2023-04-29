#include "RUnit.h"

RUnit::RUnit(uint32_t vertOff, uint32_t indOff, uint32_t vertexNum, uint32_t indexNum, uint16_t vertexSize, uint16_t indexSize)
    : m_offsetVertex{ vertOff }, m_offsetIndex{ indOff }, m_vertexSize{ vertexSize }
{
    m_byteSizeVertices = vertexNum * vertexSize;
    m_byteSizeIndices = indexNum * indexSize;
}

uint32_t RUnit::getOffsetVertex()
{
    return m_offsetVertex;
}

uint32_t RUnit::getOffsetIndex()
{
    return m_offsetIndex;
}

uint32_t RUnit::getByteSizeVertices()
{
    return m_byteSizeVertices;
}

uint32_t RUnit::getByteSizeIndices()
{
    return m_byteSizeIndices;
}

uint32_t RUnit::getCommandBufferOffset()
{
    return m_commandBufferOffset;
}

uint16_t RUnit::getVertexSize()
{
    return m_vertexSize;
}
