#ifndef BB_HEADER
#define BB_HEADER

#include <vector>

#include "src/tools/asserter.h"

class OBBs
{
private:
	uint32_t m_count{ 0 };
	uint32_t m_maxCount{ 0 };
	float* m_data{};

	static constexpr int dimensionsNum{ 3 };
	static constexpr int boxVertexCount{ 8 };

public:
	enum Position
	{
		TFL,
		TFR,
		TBL,
		TBR,
		BFL,
		BFR,
		BBL,
		BBR,
		ALL_POS
	};
	OBBs(int boxCount)
	{
		m_data = { new float[dimensionsNum * boxVertexCount * boxCount] };
		m_maxCount = boxCount;
	}
	~OBBs()
	{
		delete[] m_data;
	}

	uint32_t getBBCount() const
	{
		return m_count;
	}

	void getOBB(int index, float** xs, float** ys, float** zs) const
	{
		EASSERT(index < m_count, "App", "Undefined data accessed.");
		*xs = m_data + 0 * boxVertexCount * m_maxCount + boxVertexCount * index;
		*ys = m_data + 1 * boxVertexCount * m_maxCount + boxVertexCount * index;
		*zs = m_data + 2 * boxVertexCount * m_maxCount + boxVertexCount * index;
	}

	void transformOBB(int index, const glm::mat4& transformMatrix)
	{
		float* xs = m_data + 0 * boxVertexCount * m_maxCount + boxVertexCount * index;
		float* ys = m_data + 1 * boxVertexCount * m_maxCount + boxVertexCount * index;
		float* zs = m_data + 2 * boxVertexCount * m_maxCount + boxVertexCount * index;
		for (int i{ 0 }; i < ALL_POS; ++i)
		{
			glm::vec3 newPos{ transformMatrix * glm::vec4{xs[i], ys[i], zs[i], 1.0} };
			xs[i] = newPos.x;
			ys[i] = newPos.y;
			zs[i] = newPos.z;
		}
	}

	//Input data should be ordered like in the enum
	void addOBB(float* obbData)
	{
		EASSERT(m_count < m_maxCount, "App", "More OBBs added than space allocated.");

		float* dataToFillX{ m_data + 0 * boxVertexCount * m_maxCount + boxVertexCount * m_count };
		float* dataToFillY{ m_data + 1 * boxVertexCount * m_maxCount + boxVertexCount * m_count };
		float* dataToFillZ{ m_data + 2 * boxVertexCount * m_maxCount + boxVertexCount * m_count };
		
		for (int i{ 0 }; i < ALL_POS; ++i)
		{
			dataToFillX[i] = obbData[3 * i + 0];
			dataToFillY[i] = obbData[3 * i + 1];
			dataToFillZ[i] = obbData[3 * i + 2];
		}

		++m_count;
	}
};


#endif