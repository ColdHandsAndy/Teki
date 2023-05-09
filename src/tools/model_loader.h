#ifndef MODEL_LOADER_HEADER
#define MODEL_LOADER_HEADER

#include <iostream>

#include "tiny_gltf.h"
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/data_management/buffer_class.h"

namespace fs = std::filesystem;

inline bool loadGLTFModel(tinygltf::Model& model, const fs::path& filename)
{
	tinygltf::TinyGLTF loader{};
	std::string err{};
	std::string warn{};

	bool res = loader.LoadASCIIFromFile(&model, &err, &warn, filename.generic_string());
	if (!warn.empty())
	{
		std::cout << "WARN: " << warn << std::endl;
	}

	if (!err.empty())
	{
		std::cout << "ERR: " << err << std::endl;
	}

	if (!res)
		std::cout << "Failed to load glTF: " << filename << std::endl;
	else
		std::cout << "Loaded glTF: " << filename << std::endl;

	return res;
}

template<typename T>
inline void formVertexChunk(const std::map<std::string, int>& attributes,
							const std::vector<tinygltf::Accessor>& allAccessors,
							const std::vector<tinygltf::Buffer>& allBuffers,
							const std::vector<tinygltf::BufferView>& allBufferViews,
							uint8_t* stagingDataPtr,
							uint64_t& stagingCurrentSize,
							RUnit& renderUnit,
							const glm::dmat4& transformMatrix)
{

};

template<>
inline void formVertexChunk<StaticVertex>(const std::map<std::string, int>& attributes,
										  const std::vector<tinygltf::Accessor>& allAccessors,
										  const std::vector<tinygltf::Buffer>& allBuffers,
										  const std::vector<tinygltf::BufferView>& allBufferViews,
										  uint8_t* stagingDataPtr,
										  uint64_t& stagingCurrentSize,
										  RUnit& renderUnit,
										  const glm::dmat4& transformMatrix)
{
	auto posAccIter{ attributes.find("POSITION") };
	ASSERT_ALWAYS((posAccIter != attributes.end()), "App", "Required accessor in a model was not found.");
	const tinygltf::Accessor& accessorPos{ allAccessors[(*posAccIter).second] };
	const tinygltf::BufferView& bufferViewPos{ allBufferViews[accessorPos.bufferView] };
	const tinygltf::Buffer& bufferPos{ allBuffers[bufferViewPos.buffer] };
	uint64_t countPos{ accessorPos.count };
	uint64_t offsetPos{ bufferViewPos.byteOffset + accessorPos.byteOffset };
	uint64_t stridePos{ bufferViewPos.byteStride == 0 ? sizeof(float) : bufferViewPos.byteStride };

	auto normAccIter{ attributes.find("NORMAL") };
	ASSERT_ALWAYS((normAccIter != attributes.end()), "App", "Required accessor in a model was not found.");
	const tinygltf::Accessor& accessorNorm{ allAccessors[(*normAccIter).second] };
	const tinygltf::BufferView& bufferViewNorm{ allBufferViews[accessorNorm.bufferView] };
	const tinygltf::Buffer& bufferNorm{ allBuffers[bufferViewNorm.buffer] };
	uint64_t countNorm{ accessorNorm.count };
	uint64_t offsetNorm{ bufferViewNorm.byteOffset + accessorNorm.byteOffset };
	uint64_t strideNorm{ bufferViewNorm.byteStride == 0 ? sizeof(float) : bufferViewNorm.byteStride };

	auto tangAccIter{ attributes.find("TANGENT") };
	ASSERT_ALWAYS((tangAccIter != attributes.end()), "App", "Required accessor in a model was not found.");
	const tinygltf::Accessor& accessorTang{ allAccessors[(*tangAccIter).second] };
	const tinygltf::BufferView& bufferViewTang{ allBufferViews[accessorTang.bufferView] };
	const tinygltf::Buffer& bufferTang{ allBuffers[bufferViewTang.buffer] };
	uint64_t countTang{ accessorTang.count };
	uint64_t offsetTang{ bufferViewTang.byteOffset + accessorTang.byteOffset };
	uint64_t strideTang{ bufferViewTang.byteStride == 0 ? sizeof(float) : bufferViewTang.byteStride };

	auto texCoordAccIter{ attributes.find("TEXCOORD_0") };
	ASSERT_ALWAYS((texCoordAccIter != attributes.end()), "App", "Required accessor in a model was not found.");
	const tinygltf::Accessor& accessorTexC{ allAccessors[(*texCoordAccIter).second] };
	const tinygltf::BufferView& bufferViewTexC{ allBufferViews[accessorTexC.bufferView] };
	const tinygltf::Buffer& bufferTexC{ allBuffers[bufferViewTexC.buffer] };
	uint64_t countTexC{ accessorTexC.count };
	uint64_t offsetTexC{ bufferViewTexC.byteOffset + accessorTexC.byteOffset };
	uint64_t strideTexC{};
	int texComponentType{ accessorTexC.componentType };

	// TODO: Implement other types support
	ASSERT_ALWAYS(texComponentType == TINYGLTF_COMPONENT_TYPE_FLOAT, "App", "TexCoords types other than float are unsupported yet.");
	strideTexC = bufferViewTexC.byteStride == 0 ? sizeof(float) : bufferViewTexC.byteStride;
	//

	//switch (texComponentType)
	//{
	//case TINYGLTF_COMPONENT_TYPE_FLOAT:
	//	strideTexC = bufferViewTexC.byteStride == 0 ? sizeof(float) : bufferViewTexC.byteStride;
	//	break;
	//case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
	//	strideTexC = bufferViewTexC.byteStride == 0 ? sizeof(short) : bufferViewTexC.byteStride;
	//	break;
	//case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
	//	strideTexC = bufferViewTexC.byteStride == 0 ? sizeof(UCHAR) : bufferViewTexC.byteStride;
	//	break;
	//default:
	//	ASSERT_ALWAYS(false, "App", "Unknown texCoord component type.")
	//	break;
	//}


	ASSERT_ALWAYS(countPos == countNorm == countTang == countTexC, "App", "Not every vertex in a mesh has equal amount of attributes.");
	//Locked operations
	uint64_t chunkSize{ sizeof(StaticVertex) * countPos };
	StaticVertex* vertexDataPtr{ reinterpret_cast<StaticVertex*>(stagingDataPtr + stagingCurrentSize) };
	renderUnit.setVertBufByteSize(chunkSize);
	renderUnit.setVertBufOffset(stagingCurrentSize);
	stagingCurrentSize += chunkSize;
	//
	const unsigned char* posData{ bufferPos.data.data() + offsetPos };
	const unsigned char* normData{ bufferNorm.data.data() + offsetNorm };
	const unsigned char* tangData{ bufferTang.data.data() + offsetTang };
	const unsigned char* texCoordData{ bufferTexC.data.data() + offsetTexC };
	for (uint64_t i{ 0 }; i < countPos; ++i)
	{
		const float* posDataTyped{ reinterpret_cast<const float*>(posData) }; //Only float is possible for pos attrib
		vertexDataPtr->position = transformMatrix * glm::vec4{ *(posDataTyped + 0), *(posDataTyped + 1), *(posDataTyped + 2), 1.0f };
		posData += stridePos;

		const float* normDataTyped{ reinterpret_cast<const float*>(normData) }; //Only float is possible for norm attrib
		vertexDataPtr->normal = glm::packSnorm4x8(glm::vec4{ *(normDataTyped + 0), *(normDataTyped + 1), *(normDataTyped + 2), 0.0f });
		normData += strideNorm;

		const float* tangDataTyped{ reinterpret_cast<const float*>(tangData) }; //Only float is possible for tang attrib
		vertexDataPtr->tangent = glm::packSnorm4x8(glm::vec4{ *(tangDataTyped + 0), *(tangDataTyped + 1), *(tangDataTyped + 2), 0.0f } *(-(*(tangDataTyped + 3))));
		tangData += strideTang;

		const float* texCoordDataTyped{ reinterpret_cast<const float*>(texCoordData) };
		vertexDataPtr->texCoords = glm::packHalf2x16(glm::vec2{ *texCoordDataTyped, *(texCoordDataTyped + 1) });
		texCoordData += strideTexC;

		++vertexDataPtr;
	}
}

inline void formIndexChunk(const tinygltf::Accessor& indicesAccessor,
						   const std::vector<tinygltf::Buffer>& allBuffers,
						   const std::vector<tinygltf::BufferView>& allBufferViews,
						   uint8_t* stagingDataPtr,
						   uint64_t& stagingCurrentSize,
						   RUnit& renderUnit)
{
	const tinygltf::BufferView& bufferView{ allBufferViews[indicesAccessor.bufferView] };
	const tinygltf::Buffer& buffer{ allBuffers[bufferView.buffer] };
	uint64_t count{ indicesAccessor.count };
	uint64_t offset{ bufferView.byteOffset + indicesAccessor.byteOffset };

	//Locked operations
	uint64_t chunkSize{ sizeof(uint32_t) * count };
	uint32_t* indexDataPtr{ reinterpret_cast<uint32_t*>(stagingDataPtr + stagingCurrentSize) };
	renderUnit.setIndexBufByteSize(chunkSize);
	renderUnit.setIndexBufOffset(stagingCurrentSize);
	stagingCurrentSize += chunkSize;
	//

	switch (indicesAccessor.componentType)
	{
	case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
	{
		const uint32_t* buf = reinterpret_cast<const uint32_t*>(&buffer.data[offset]);

		for (uint64_t i{ 0 }; i < count; ++i)
		{
			*(indexDataPtr++) = buf[i];
		}
		break;
	}
	case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
	{
		const uint16_t* buf = reinterpret_cast<const uint16_t*>(&buffer.data[offset]);
		for (uint64_t i{ 0 }; i < count; ++i)
		{
			*(indexDataPtr++) = buf[i];
		}
		break;
	}
	case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
	{
		const uint8_t* buf = reinterpret_cast<const uint8_t*>(&buffer.data[offset]);
		for (uint64_t i{ 0 }; i < count; ++i)
		{
			*(indexDataPtr++) = buf[i];
		}
		break;
	}
	default:
		ASSERT_ALWAYS(false, "App", "Unknown index type.");
	}
}

inline void processNode(const std::vector<tinygltf::Node>& allNodes,
						const std::vector<tinygltf::Mesh>& allMeshes,
						const std::vector<tinygltf::Accessor>& allAccessors,
						const std::vector<tinygltf::Buffer>& allBuffers,
						const std::vector<tinygltf::BufferView>& allBufferViews,
						const std::vector<tinygltf::Material>& allMaterials,
						int nodeIndex,
						uint8_t* stagingDataPtr,
						uint64_t& stagingCurrentSize,
						StaticMesh& loadedMesh,
						const glm::dmat4& globalTransformMatrix)
{
	const tinygltf::Node node{ allNodes[nodeIndex] };

	glm::dmat4 localTransformMatrix{};
	if (node.matrix.empty())
	{
		glm::dmat4 translate{ 1.0f };
		if (!node.translation.empty())
		{
			translate = glm::translate(glm::dvec3{ node.translation[0], node.translation[1], node.translation[2] });
		}
		glm::dmat4 rotate{ 1.0f };
		if (!node.rotation.empty())
		{
			rotate = glm::mat4_cast(glm::dquat{ node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3] });
		}
		glm::dmat4 scale{ 1.0f };
		if (!node.scale.empty())
		{
			scale = glm::scale(glm::dvec3{ node.scale[0], node.scale[1], node.scale[2] });
		}
		localTransformMatrix = translate * rotate * scale * globalTransformMatrix;
		//localTransformMatrix = globalTransformMatrix * translate * rotate * scale;
	}
	else
	{
		localTransformMatrix = glm::make_mat4x4(node.matrix.data()) * globalTransformMatrix;
		//localTransformMatrix = globalTransformMatrix * glm::make_mat4x4(node.matrix.data());
	}

	if (node.mesh > -1)
	{
		const tinygltf::Mesh& mesh{ allMeshes[node.mesh] };

		for (auto& primitive : mesh.primitives)
		{
			RUnit& renderUnit{ loadedMesh.getRUnits().emplace_back() };
			//Fill buffer with mesh data and init RUnit for one glTF "primitive"
			//Process RUnit vertex buffer and put in staging
			renderUnit.setVertexSize(sizeof(StaticVertex));
			formVertexChunk<StaticVertex>(primitive.attributes, allAccessors, allBuffers, allBufferViews, stagingDataPtr, stagingCurrentSize, renderUnit, localTransformMatrix);

			int indicesIndex{ primitive.indices };
			ASSERT_ALWAYS((indicesIndex >= 0), "App", "Index to indices accessor is inavlid.");
			renderUnit.setIndexSize(sizeof(uint32_t));
			formIndexChunk(allAccessors[indicesIndex], allBuffers, allBufferViews, stagingDataPtr, stagingCurrentSize, renderUnit);

			int materialIndex{ primitive.material };
			ASSERT_ALWAYS((materialIndex >= 0), "App", "Index to material accessor is inavlid.");
		}
	}


	for (auto childNode : node.children)
	{
		processNode(allNodes, allMeshes, allAccessors, allBuffers, allBufferViews, allMaterials, childNode, stagingDataPtr, stagingCurrentSize, loadedMesh, localTransformMatrix);
	}
}

inline void processModelData(const tinygltf::Model& model, 
							 uint8_t* stagingDataPtr, 
							 uint64_t& stagingCurrentSize, 
							 StaticMesh& loadedMesh)
{
	const std::vector<tinygltf::Scene>& scenes{ model.scenes };
	const std::vector<tinygltf::Node>& nodes{ model.nodes };
	const std::vector<tinygltf::Mesh>& meshes{ model.meshes };
	const std::vector<tinygltf::Accessor>& accessors{ model.accessors };
	const std::vector<tinygltf::Buffer>& buffers{ model.buffers };
	const std::vector<tinygltf::BufferView>& bufferViews{ model.bufferViews };
	const std::vector<tinygltf::Material>& materials{ model.materials };

	int modelRUnitsOffset{ 0 };
	for (int i{ 0 }; i < scenes.size(); ++i)
	{
		for (auto rootNode : scenes[i].nodes)
		{
			processNode(nodes, meshes, accessors, buffers, bufferViews, materials, rootNode, stagingDataPtr, stagingCurrentSize, loadedMesh, glm::dmat4{ 1.0f });
		}
	}
}

inline std::vector<StaticMesh> loadStaticMeshes(std::shared_ptr<VulkanObjectHandler> vulkanObjects, FrameCommandBufferSet& commandBufferSet, Buffer& vertexBuffer, Buffer& indexBuffer, BufferMapped& indirectCmdBuffer, std::vector<fs::path> filepaths)
{
	std::vector<StaticMesh> loadedMeshes(filepaths.size());

	BufferBaseHostAccessible resourceStaging{ vulkanObjects->getLogicalDevice(), 1073741824ll, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, {{vulkanObjects->getTransferFamilyIndex()}}};

	uint8_t* stagingDataPtr{ reinterpret_cast<uint8_t*>(resourceStaging.getData()) };
	uint64_t stagingCurrentSize{ 0 };
	for (uint32_t i{ 0 }; i < filepaths.size(); ++i)
	{
		tinygltf::Model model{};
		ASSERT_ALWAYS(loadGLTFModel(model, filepaths[i]), "App", "Failed to load a glTF model.");
		processModelData(model, stagingDataPtr, stagingCurrentSize, loadedMeshes[i]);
	}

	uint64_t verticesByteSize{ 0 };
	uint64_t indicesByteSize{ 0 };
	for (auto& mesh : loadedMeshes)
	{
		std::vector<RUnit>& renderUnits{ mesh.getRUnits() };
		for (uint32_t i{ 0 }; i < renderUnits.size(); ++i)
		{
			verticesByteSize += renderUnits[i].getVertBufByteSize();
			indicesByteSize += renderUnits[i].getIndexBufByteSize();
		}
	}
	verticesByteSize = ALIGNED_SIZE(verticesByteSize, vertexBuffer.getAlignment());
	indicesByteSize = ALIGNED_SIZE(indicesByteSize, indexBuffer.getAlignment());
	vertexBuffer.initialize(verticesByteSize);
	indexBuffer.initialize(indicesByteSize);
	VkDeviceSize rUnitCount{ 0 }; 
	for (auto& mesh : loadedMeshes) 
	{
		rUnitCount += mesh.getRUnits().size();
	};
	indirectCmdBuffer.initialize(rUnitCount * sizeof(VkDrawIndexedIndirectCommand));

	std::vector<VkBufferCopy> copyRegionsVertexBuf{};
	std::vector<VkBufferCopy> copyRegionsIndexBuf{};
	VkDrawIndexedIndirectCommand* indirectCmdData{ reinterpret_cast<VkDrawIndexedIndirectCommand*>(indirectCmdBuffer.getData()) };

	uint64_t offsetIntoVertexData{ 0 };
	uint64_t offsetIntoIndexData{ indexBuffer.getOffset() };
	uint32_t offsetIntoCmdBuffer{ 0 };

	int32_t firstVertex{ 0 };
	uint32_t firstIndex{ 0 };

	for (auto& mesh : loadedMeshes)
	{
		std::vector<RUnit>& renderUnits{ mesh.getRUnits() };
		for (uint32_t i{ 0 }; i < renderUnits.size(); ++i)
		{
			uint64_t vertBufSize{ renderUnits[i].getVertBufByteSize() };
			uint64_t indexBufSize{ renderUnits[i].getIndexBufByteSize() };
			uint32_t indexCount{ static_cast<uint32_t>(indexBufSize / renderUnits[i].getIndexSize()) };

			*(indirectCmdData++) = VkDrawIndexedIndirectCommand{
				.indexCount = indexCount,
				.instanceCount = 1,
				.firstIndex = firstIndex,
				.vertexOffset = firstVertex,
				.firstInstance = 0 };
			firstIndex += indexCount;
			firstVertex += static_cast<int32_t>(vertBufSize / renderUnits[i].getVertexSize());
			copyRegionsVertexBuf.push_back(VkBufferCopy{ .srcOffset = renderUnits[i].getOffsetVertex(), .dstOffset = offsetIntoVertexData, .size = vertBufSize });
			copyRegionsIndexBuf.push_back(VkBufferCopy{ .srcOffset = renderUnits[i].getOffsetIndex(), .dstOffset = offsetIntoIndexData, .size = indexBufSize });
			renderUnits[i].setVertBufOffset(offsetIntoVertexData);
			renderUnits[i].setIndexBufOffset(offsetIntoIndexData);
			renderUnits[i].setDrawCmdBufferOffset(offsetIntoCmdBuffer++);
			offsetIntoVertexData += vertBufSize;
			offsetIntoIndexData += indexBufSize;
		}
	}

	VkCommandBuffer CB{ commandBufferSet.beginRecording(FrameCommandBufferSet::ASYNC_TRANSFER_CB) };
	{
		BufferTools::cmdBufferCopy(CB, resourceStaging.getBufferHandle(), vertexBuffer.getBufferHandle(), copyRegionsVertexBuf.size(), copyRegionsVertexBuf.data());
		BufferTools::cmdBufferCopy(CB, resourceStaging.getBufferHandle(), indexBuffer.getBufferHandle(), copyRegionsIndexBuf.size(), copyRegionsIndexBuf.data());
		VkBufferMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, 
									   .srcQueueFamilyIndex = vulkanObjects->getTransferFamilyIndex(), 
									   .dstQueueFamilyIndex = vulkanObjects->getGraphicsFamilyIndex(), 
									   .buffer = indirectCmdBuffer.getBufferHandle(), 
									   .offset = indirectCmdBuffer.getOffset(), 
									   .size = indirectCmdBuffer.getSize() };
		vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
	}
	commandBufferSet.endRecording(CB);
	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &CB };
	ASSERT_ALWAYS(vkQueueSubmit(vulkanObjects->getQueue(VulkanObjectHandler::TRANSFER_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS, "Vulkan", "Queue submission failed");
	ASSERT_ALWAYS(vkQueueWaitIdle(vulkanObjects->getQueue(VulkanObjectHandler::TRANSFER_QUEUE_TYPE)) == VK_SUCCESS, "Vulkan", "Wait idle failed");
	commandBufferSet.resetCommandBuffer(CB);

	return loadedMeshes;
}


#endif