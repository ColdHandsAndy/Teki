#include "mesh.h"

StaticMesh::StaticMesh()
{
}

StaticMesh::~StaticMesh()
{
}


void StaticMesh::setTransformMatrixIndex(uint32_t matrixIndex)
{
	m_transformMatrixIndex = matrixIndex;
}

void StaticMesh::consumeRenderUnits(std::vector<RUnit>& units)
{
	m_RUnits = units;
}

void StaticMesh::consumeRenderUnits(std::vector<RUnit>&& units)
{
	m_RUnits = std::move(units);
}

std::vector<RUnit>& StaticMesh::getRUnits()
{
	return m_RUnits;
}

const uint32_t StaticMesh::getTransformMatrixIndex() const
{
	return m_transformMatrixIndex;
}
