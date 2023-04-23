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
	//uint32_t m_AABBindex{};

public:

};

//class InstancedMesh{};
//class DynamicMesh{};

#endif