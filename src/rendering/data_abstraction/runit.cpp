#include "RUnit.h"

RUnit::RUnit(uint64_t vertOff, uint64_t indOff, uint16_t vertexSize, uint16_t indexSize, uint64_t vertBufSize, uint64_t indexBufSize, uint16_t imageListIndex, uint16_t imageListLayerIndex)
    : m_offsetVertex{ vertOff }, m_offsetIndex{ indOff }, m_byteSizeVertices{ vertBufSize }, m_byteSizeIndices{ indexBufSize }, m_vertexSize{ vertexSize }, m_indexSize{ indexSize }
{
}

uint64_t RUnit::getOffsetVertex() const
{
    return m_offsetVertex;
}

uint64_t RUnit::getOffsetIndex() const
{
    return m_offsetIndex;
}

uint64_t RUnit::getVertBufByteSize() const
{
    return m_byteSizeVertices;
}

uint64_t RUnit::getIndexBufByteSize() const
{
    return m_byteSizeIndices;
}

uint16_t RUnit::getVertexSize() const
{
    return m_vertexSize;
}

uint16_t RUnit::getIndexSize() const
{
    return m_indexSize;
}

uint64_t RUnit::getDrawCmdBufferOffset() const
{
    return m_commandBufferOffset;
}

std::array<std::pair<uint16_t, uint16_t>, 4>& RUnit::getMaterialIndices()
{
    return m_materialIndices;
}



void RUnit::setVertBufOffset(uint64_t offset)
{
    m_offsetVertex = offset;
}

void RUnit::setIndexBufOffset(uint64_t offset)
{
    m_offsetIndex = offset;
}

void RUnit::setVertBufByteSize(uint64_t size)
{
    m_byteSizeVertices = size;
}

void RUnit::setIndexBufByteSize(uint64_t size)
{
    m_byteSizeIndices = size;
}

void RUnit::setVertexSize(uint16_t size)
{
    m_vertexSize = size;
}

void RUnit::setIndexSize(uint16_t size)
{
    m_indexSize = size;
}

void RUnit::setDrawCmdBufferOffset(uint64_t offset)
{
    m_commandBufferOffset = offset;
}