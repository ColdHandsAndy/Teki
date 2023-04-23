#include "renderer.h"

#include <iostream>
#include <fstream>

#include "src/window/window.h"
#include "src/tools/thread_pool_wrap.h"
#include "src/MDO_handling/MDO_format.h"


//Renderer::Renderer()
//{
//	if (!glfwInit())
//	{
//		glfwTerminate();
//		std::cerr << "[GLFW] : INITIALIZATION ERROR" << std::endl;
//		assert(false);
//	}
//
//	uint32_t width{ 1280 };
//	uint32_t height{ 720 };
//	std::string windowName{ "Engine" };
//
//	m_window = { width, height, windowName };
//
//	if (m_window == NULL)
//	{
//		glfwTerminate();
//		std::cerr << "[GLFW] : Window creation error";
//		assert(false);
//	}
//}
//
//Renderer::~Renderer()
//{
//	glfwDestroyWindow(m_window);
//	glfwTerminate();
//}

//template <typename Component>
//Component getVertexComponent(std::ifstream& sourceStream, std::streampos& currentFilestreamPos, size_t componentByteSize)
//{
//	sourceStream.seekg(currentFilestreamPos);
//	Component comp{};
//	sourceStream.read(reinterpret_cast<char*>(&comp), componentByteSize);
//	currentFilestreamPos += componentByteSize;
//
//	return comp;
//}
//
//void Renderer::loadStaticModel(Model& model)
//{
//	const MDO::ModelSpec spec{ model.getPath() };
//
//	uint32_t meshNum{ spec.getMeshNum() };
//
//	m_staticMeshNum += meshNum;
//
//	model.m_staticMeshes.resize(meshNum);
//
//	ThreadPool& threadPool{ ThreadPoolWrapper::getInstance().getPool() };
//	const unsigned int threadCount{ ThreadPoolWrapper::getInstance().getThreadCount() };
//
//	std::vector<std::ifstream> ifstreamVec{};
//	ifstreamVec.resize(threadCount);
//	for (unsigned int i{ 0 }; i < threadCount; ++i)
//	{
//		ifstreamVec[i].open(model.getPath(), std::ios::binary);
//	}
//
//	auto bufFill{ [&ifstreamVec, threadCount](char* buf, std::streampos filePos, uint32_t byteSize, uint32_t order) -> void
//		{
//			ifstreamVec[order % threadCount].seekg(filePos);
//			ifstreamVec[order % threadCount].read(buf, byteSize);
//		}
//	};
//
//	unsigned int posByteSize{ sizeof(float) * 3 };
//	unsigned int normByteSize{ sizeof(float) * 3 };
//	unsigned int tangByteSize{ sizeof(float) * 3 };
//	unsigned int texcByteSize{ sizeof(float) * 2 };
//
//
//	for (uint32_t i{ 0 }; i < meshNum; ++i)
//	{
//		StaticMesh& mesh{ model.m_staticMeshes[i] };
//		MDO::ModelSpec::MeshSpec meshspec{ spec.getMeshSpec(i) };
//
//		unsigned int vertexOffset{ static_cast<unsigned int>(m_tempGeneralVertexBuffer.size()) };
//		unsigned int indexOffset{ static_cast<unsigned int>(m_tempGeneralIndexBuffer.size()) };
//
//		mesh.m_vertexCount = meshspec.VertexComponentBufferMap[MDO::dataflags::POSITION_BIT].componentsNum;
//		mesh.m_indexCount = meshspec.VertexComponentBufferMap[MDO::dataflags::INDICES_BIT].componentsNum;
//
//		mesh.m_startingVertexPosInVB = vertexOffset;
//		mesh.m_startingIndexPosInIB = indexOffset;
//
//		//TODO: Optimize threaded mesh data loading
//		
//		//piece under development start
//		
//		char* posBuf{ new char[mesh.m_vertexCount * posByteSize] };
//		char* normBuf{ new char[mesh.m_vertexCount * normByteSize] };
//		char* tangBuf{ new char[mesh.m_vertexCount * tangByteSize] };
//		char* texcBuf{ new char[mesh.m_vertexCount * texcByteSize] };
//
//		std::streampos positionBufferPosition{ meshspec.VertexComponentBufferMap[MDO::dataflags::POSITION_BIT].streamPosition };
//		std::streampos normalBufferPosition{ meshspec.VertexComponentBufferMap[MDO::dataflags::NORMAL_BIT].streamPosition };
//		std::streampos tangentBufferPosition{ meshspec.VertexComponentBufferMap[MDO::dataflags::TANGENT_BIT].streamPosition };
//		std::streampos texCoordsBufferPosition{ meshspec.VertexComponentBufferMap[MDO::dataflags::TEX_COORDS_BIT].streamPosition };
//		
//		threadPool.push_task(bufFill, posBuf, positionBufferPosition, mesh.m_vertexCount * posByteSize, 0);
//		threadPool.push_task(bufFill, normBuf, normalBufferPosition, mesh.m_vertexCount * normByteSize, 1);
//		threadPool.push_task(bufFill, tangBuf, tangentBufferPosition, mesh.m_vertexCount * tangByteSize, 2);
//		threadPool.push_task(bufFill, texcBuf, texCoordsBufferPosition, mesh.m_vertexCount * texcByteSize, 3);
//
//		std::streampos streamIndexPosition{ meshspec.VertexComponentBufferMap[MDO::dataflags::INDICES_BIT].streamPosition };
//		unsigned int indexByteSize{ sizeof(GLuint) };
//		char* indexBuf{ new char[mesh.m_indexCount * indexByteSize] };
//
//		threadPool.wait_for_tasks();
//
//		threadPool.push_task(bufFill, indexBuf, streamIndexPosition, mesh.m_indexCount * indexByteSize, 4);
//
//		for (unsigned int j{ 0 }; j < mesh.m_vertexCount; ++j)
//		{
//			m_tempGeneralVertexBuffer.push_back(
//			  { *reinterpret_cast<glm::vec3*>(&posBuf[j * posByteSize]),
//				*reinterpret_cast<glm::vec3*>(&normBuf[j * normByteSize]),
//				*reinterpret_cast<glm::vec3*>(&tangBuf[j * tangByteSize]),
//				*reinterpret_cast<glm::vec3*>(&texcBuf[j * texcByteSize]) }
//			);
//		}
//
//		delete[] posBuf;
//		delete[] normBuf;
//		delete[] tangBuf;
//		delete[] texcBuf;
//
//		threadPool.wait_for_tasks();
//
//		for (GLuint* iterPtr{ reinterpret_cast<GLuint*>(indexBuf) }; iterPtr != reinterpret_cast<GLuint*>(indexBuf) + mesh.m_indexCount; ++iterPtr)
//		{
//			m_tempGeneralIndexBuffer.push_back(*iterPtr);
//		}
//
//		delete[] indexBuf;
//		
//		//piece under development end
//
//		m_tempGeneralIndirectBuffer.push_back({ mesh.m_indexCount, 1, indexOffset, static_cast<int>(vertexOffset), 0 });
//	}
//
//	for (auto& element : ifstreamVec)
//		element.close();
//
//	m_tempGeneralVertexBuffer.shrink_to_fit();
//	m_tempGeneralIndexBuffer.shrink_to_fit();
//	m_tempGeneralIndirectBuffer.shrink_to_fit();
//
//	model.setInitialized();
//}
//
//void Renderer::flushDataIntoGPU()
//{
//	glBindBuffer(GL_ARRAY_BUFFER, m_staticRenderingOpenGLObjects.VBO);
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_staticRenderingOpenGLObjects.EBO);
//	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, m_staticRenderingOpenGLObjects.indirectBO);
//	glBufferData(GL_ARRAY_BUFFER, m_tempGeneralVertexBuffer.size() * sizeof(StaticVertexPBR::Vertex), m_tempGeneralVertexBuffer.data(), GL_STATIC_DRAW);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_tempGeneralIndexBuffer.size() * sizeof(GLuint), m_tempGeneralIndexBuffer.data(), GL_STATIC_DRAW);
//	glBufferData(GL_DRAW_INDIRECT_BUFFER, m_tempGeneralIndirectBuffer.size() * sizeof(DrawElementsIndirectCommand), m_tempGeneralIndirectBuffer.data(), GL_STATIC_DRAW);
//
//	m_tempGeneralVertexBuffer.clear();
//	m_tempGeneralIndexBuffer.clear();
//	m_tempGeneralIndirectBuffer.clear();
//}
//
//uint32_t Renderer::getStaticMeshNum() const
//{
//	return m_staticMeshNum;
//}
//
//const Window& Renderer::getWindow() const
//{
//	return m_window;
//}