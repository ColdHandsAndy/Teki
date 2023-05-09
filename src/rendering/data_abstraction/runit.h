#ifndef RUNIT_CLASS_HEADER
#define RUNIT_CLASS_HEADER

#include <cstdint>

class RUnit
{
private:
	uint64_t m_offsetVertex{};
	uint64_t m_offsetIndex{};
	uint64_t m_byteSizeVertices{};
	uint64_t m_byteSizeIndices{};

	uint64_t m_commandBufferOffset{};

	uint16_t m_vertexSize{};
	uint16_t m_indexSize{};
	//Material ref

public:
	RUnit() = default;
	//RUnit(uint64_t vertOff, uint64_t indOff, uint64_t vertexNum, uint64_t indexNum, uint16_t vertexSize, uint16_t indexSize = sizeof(uint32_t));
	RUnit(uint64_t vertOff, uint64_t indOff, uint16_t vertexSize, uint16_t indexSize, uint64_t vertBufSize, uint64_t indexBufSize);
	~RUnit() = default;

	uint64_t getOffsetVertex() const;
	uint64_t getOffsetIndex() const;
	uint64_t getVertBufByteSize() const;
	uint64_t getIndexBufByteSize() const;
	uint16_t getVertexSize() const;
	uint16_t getIndexSize() const;
	uint64_t getDrawCmdBufferOffset() const;

	void setVertBufOffset(uint64_t offset);
	void setIndexBufOffset(uint64_t offset);
	void setVertBufByteSize(uint64_t size);
	void setIndexBufByteSize(uint64_t size);
	void setVertexSize(uint16_t size);
	void setIndexSize(uint16_t size);
	void setDrawCmdBufferOffset(uint64_t offset);
};

#endif