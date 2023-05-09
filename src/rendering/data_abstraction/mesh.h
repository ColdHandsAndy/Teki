#ifndef MESH_CLASS_HEADER
#define MESH_CLASS_HEADER

#include <vector>

#include "glm/glm.hpp"

#include "RUnit.h"

class StaticMesh
{
private:
	std::vector<RUnit> m_RUnits{};
	uint32_t m_transformMatrixIndex{};

public:
	StaticMesh();
	~StaticMesh();

	void setTransformMatrixIndex(uint32_t matrixIndex);
	void consumeRenderUnits(std::vector<RUnit>& units);
	void consumeRenderUnits(std::vector<RUnit>&& units);

	std::vector<RUnit>& getRUnits();
	const uint32_t getTransformMatrixIndex() const;
};

//class InstancedMesh{};
//class DynamicMesh{};

#endif