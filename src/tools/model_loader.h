#ifndef MODEL_LOADER_HEADER
#define MODEL_LOADER_HEADER

#include <iostream>
#include <set>

#include "tiny_gltf.h"
#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>

#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_abstraction/mesh.h"
#include "src/rendering/data_abstraction/runit.h"
#include "src/tools/logging.h"

namespace fs = std::filesystem;

inline bool loadGLTFModel(tinygltf::Model& model, const fs::path& filename)
{
	tinygltf::TinyGLTF loader{};
	std::string err{};
	std::string warn{};
	loader.SetPreserveImageChannels(true);
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

	return res;
}

template<typename T>
inline void formVertexChunk(const std::map<std::string, int>& attributes,
							const std::vector<tinygltf::Accessor>& allAccessors,
							const std::vector<tinygltf::Buffer>& allBuffers,
							const std::vector<tinygltf::BufferView>& allBufferViews,
							uint8_t* const stagingDataPtr,
							uint64_t& stagingCurrentSize,
							RUnit& renderUnit,
							const glm::dmat4& transformMatrix,
							oneapi::tbb::spin_mutex& mutex)
{

};

template<>
inline void formVertexChunk<StaticVertex>(const std::map<std::string, int>& attributes,
										  const std::vector<tinygltf::Accessor>& allAccessors,
										  const std::vector<tinygltf::Buffer>& allBuffers,
										  const std::vector<tinygltf::BufferView>& allBufferViews,
										  uint8_t* const stagingDataPtr,
										  uint64_t& stagingCurrentSize,
										  RUnit& renderUnit,
										  const glm::dmat4& transformMatrix,
										  oneapi::tbb::spin_mutex& mutex)
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
	
	ASSERT_ALWAYS(countPos == countNorm && countNorm == countTang && countTang == countTexC, "App", "Not every vertex in a mesh has equal amount of attributes.");

	//Locked operations
	oneapi::tbb::spin_mutex::scoped_lock lock{};
	lock.acquire(mutex);
	uint64_t chunkSize{ sizeof(StaticVertex) * countPos };
	StaticVertex* vertexDataPtr{ reinterpret_cast<StaticVertex*>(stagingDataPtr + stagingCurrentSize) };
	renderUnit.setVertBufByteSize(chunkSize);
	renderUnit.setVertBufOffset(stagingCurrentSize);
	stagingCurrentSize += chunkSize;
	lock.release();
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
						   uint8_t* const stagingDataPtr,
						   uint64_t& stagingCurrentSize,
						   RUnit& renderUnit,
						   oneapi::tbb::spin_mutex& mutex)
{
	const tinygltf::BufferView& bufferView{ allBufferViews[indicesAccessor.bufferView] };
	const tinygltf::Buffer& buffer{ allBuffers[bufferView.buffer] };
	uint64_t count{ indicesAccessor.count };
	uint64_t offset{ bufferView.byteOffset + indicesAccessor.byteOffset };

	//Locked operations
	oneapi::tbb::spin_mutex::scoped_lock lock{};
	lock.acquire(mutex);
	uint64_t chunkSize{ sizeof(uint32_t) * count };
	uint32_t* indexDataPtr{ reinterpret_cast<uint32_t*>(stagingDataPtr + stagingCurrentSize) };
	renderUnit.setIndexBufByteSize(chunkSize);
	renderUnit.setIndexBufOffset(stagingCurrentSize);
	stagingCurrentSize += chunkSize;
	lock.release();
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
						std::vector<int>& meshMaterialIndices,
						int nodeIndex,
						uint8_t* const stagingDataPtr,
						uint64_t& stagingCurrentSize,
						StaticMesh& loadedMesh,
						const glm::dmat4& globalTransformMatrix,
						oneapi::tbb::spin_mutex& mutex)
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
			formVertexChunk<StaticVertex>(primitive.attributes, allAccessors, allBuffers, allBufferViews, stagingDataPtr, stagingCurrentSize, renderUnit, localTransformMatrix, mutex);

			int indicesIndex{ primitive.indices };
			ASSERT_ALWAYS((indicesIndex >= 0), "App", "Index to indices accessor is inavlid.");
			renderUnit.setIndexSize(sizeof(uint32_t));
			formIndexChunk(allAccessors[indicesIndex], allBuffers, allBufferViews, stagingDataPtr, stagingCurrentSize, renderUnit, mutex);

			int materialIndex{ primitive.material };
			ASSERT_ALWAYS((materialIndex >= 0), "App", "Index to material accessor is inavlid.");

			meshMaterialIndices.push_back(materialIndex);
		}
	}


	for (auto childNode : node.children)
	{
		processNode(allNodes, allMeshes, allAccessors, allBuffers, allBufferViews, meshMaterialIndices, childNode, stagingDataPtr, stagingCurrentSize, loadedMesh, localTransformMatrix, mutex);
	}
}

inline void processMeshData(const tinygltf::Model& model, 
							 uint8_t* const stagingDataPtr,
							 uint64_t& stagingCurrentSize, 
							 StaticMesh& loadedMesh,
							 std::vector<int>& meshMaterialIndices,
							 oneapi::tbb::spin_mutex& mutex)
{
	const std::vector<tinygltf::Scene>& scenes{ model.scenes };
	const std::vector<tinygltf::Node>& nodes{ model.nodes };
	const std::vector<tinygltf::Mesh>& meshes{ model.meshes };
	const std::vector<tinygltf::Accessor>& accessors{ model.accessors };
	const std::vector<tinygltf::Buffer>& buffers{ model.buffers };
	const std::vector<tinygltf::BufferView>& bufferViews{ model.bufferViews };

	LOG_IF_INFO(scenes.size() > 1, "There are {} scenes in a model.", scenes.size());
	for (int i{ 0 }; i < scenes.size(); ++i)
	{
		for (auto rootNode : scenes[i].nodes)
		{
			processNode(nodes, meshes, accessors, buffers, bufferViews, meshMaterialIndices, rootNode, stagingDataPtr, stagingCurrentSize, loadedMesh, glm::dmat4{ 1.0f }, mutex);
		}
	}
}


std::vector<char> getShaderCode(const fs::path& filepath);
VkShaderModule createModule(VkDevice device, std::vector<char>& code);

struct alignas(uint64_t) ImageComponentTransformData
{
	uint64_t dataAddress{};
	uint32_t srcImageWidth{};
	uint32_t srcImageHeight{};
	uint16_t dstImageListIndex{};
	uint16_t dstImageListLayerIndex{};
};
struct ImageTransferData
{
	VkDeviceSize bufferOffset{};
	uint32_t width{};
	uint32_t height{};
	uint32_t dstImageListIndex{};
	uint32_t dstImageLayerIndex{};
};
inline void loadTexturesIntoStaging(VkDevice device,
							 const std::vector<tinygltf::Model>& models,
							 std::vector<StaticMesh>& loadedMeshes,
							 std::vector<ImageList>& loadedTextures,
							 std::vector<ImageTransferData>& transferOperations,
							 std::vector<ImageComponentTransformData>& componentTransformOperations,
							 std::vector<std::vector<int>>& meshesMaterialIndices,
							 uint8_t* const stagingDataPtr,
							 uint64_t stagingDeviceAddress,
							 uint64_t& stagingCurrentSize);

inline PipelineCompute buildComponentTransformComputePipeline(VkDevice device,
	std::vector<ImageList>& loadedTextures,
	std::vector<ImageComponentTransformData>& componentTransformOperations,
	uint8_t* const stagingDataPtr,
	uint64_t stagingDeviceAddress,
	uint64_t& stagingCurrentSize,
	uint32_t stagingAlignment);


inline std::vector<StaticMesh> loadStaticMeshes(std::shared_ptr<VulkanObjectHandler> vulkanObjects,
												DescriptorManager& descriptorManager,
												FrameCommandBufferSet& commandBufferSet, 
												Buffer& vertexBuffer, 
												Buffer& indexBuffer, 
												BufferMapped& indirectCmdBuffer, 
												std::vector<ImageList>& loadedTextures,
												std::vector<fs::path> filepaths)
{
	BufferBaseHostAccessible resourceStaging{ vulkanObjects->getLogicalDevice(), 2147483648ll, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };

	VkDeviceAddress stagingDeviceAddress{ resourceStaging.getBufferDeviceAddress() };
	uint8_t* const stagingDataPtr{ reinterpret_cast<uint8_t*>(resourceStaging.getData()) };
	uint64_t stagingCurrentSize{ 0 };
	
	uint32_t modelNum{ static_cast<uint32_t>(filepaths.size()) };
	std::vector<tinygltf::Model> models(modelNum);
	std::vector<StaticMesh> loadedMeshes(modelNum);
	std::vector<std::vector<int>> meshesMaterialIndices(modelNum);
	
	oneapi::tbb::spin_mutex mutex{};
	oneapi::tbb::parallel_for(0u, static_cast<uint32_t>(modelNum),
		[&filepaths, &stagingDataPtr, &stagingCurrentSize, &models, &loadedMeshes, &meshesMaterialIndices, &mutex](uint32_t i)
		{
			ASSERT_ALWAYS(loadGLTFModel(models[i], filepaths[i]), "App", "Failed to load a glTF model.");
			processMeshData(models[i], stagingDataPtr, stagingCurrentSize, loadedMeshes[i], meshesMaterialIndices[i], mutex);
		});
	
	stagingCurrentSize = ALIGNED_SIZE(stagingCurrentSize, resourceStaging.getBufferAlignment());
	

	
	std::vector<ImageTransferData> transferOperations{};
	std::vector<ImageComponentTransformData> componentTransformOperations{};

	loadTexturesIntoStaging(vulkanObjects->getLogicalDevice(), models, loadedMeshes, loadedTextures, transferOperations, componentTransformOperations, meshesMaterialIndices, stagingDataPtr, stagingDeviceAddress, stagingCurrentSize);

	PipelineCompute compute{ buildComponentTransformComputePipeline(vulkanObjects->getLogicalDevice(), loadedTextures, componentTransformOperations, stagingDataPtr, stagingDeviceAddress, stagingCurrentSize, resourceStaging.getBufferAlignment()) };
	

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



	std::vector<VkImageMemoryBarrier> memBarriers{};
	for (uint32_t i{ 0 }; i < loadedTextures.size(); ++i)
	{
		memBarriers.push_back(VkImageMemoryBarrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
													.srcAccessMask = 0,
													.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
													.newLayout = VK_IMAGE_LAYOUT_GENERAL,
													.image = loadedTextures[i].getImageHandle(),
													.subresourceRange = loadedTextures[i].getSubresourceRange() });
	}
	VkCommandBuffer CB{ commandBufferSet.beginRecording(FrameCommandBufferSet::ASYNC_COMPUTE_CB) };
	{
		BufferTools::cmdBufferCopy(CB, resourceStaging.getBufferHandle(), vertexBuffer.getBufferHandle(), copyRegionsVertexBuf.size(), copyRegionsVertexBuf.data());
		BufferTools::cmdBufferCopy(CB, resourceStaging.getBufferHandle(), indexBuffer.getBufferHandle(), copyRegionsIndexBuf.size(), copyRegionsIndexBuf.data());
		
		vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, memBarriers.size(), memBarriers.data());
		compute.cmdBind(CB);
		for (uint32_t i{ 0 }; i < componentTransformOperations.size(); ++i)
		{
			compute.setResourceIndex(1, i);
			descriptorManager.cmdSubmitPipelineResources(CB, VK_PIPELINE_BIND_POINT_COMPUTE, compute.getResourceSets(), compute.getCurrentResourceLayout(), compute.getPipelineLayoutHandle());
			vkCmdDispatch(CB, componentTransformOperations[i].srcImageWidth / 16, componentTransformOperations[i].srcImageHeight / 16, 1);
		}
		
		for (auto& barrier : memBarriers)
		{
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		}
		vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, memBarriers.size(), memBarriers.data());

		for (uint32_t i{ 0 }; i < transferOperations.size(); ++i)
		{
			auto& op{ transferOperations[i] };
			loadedTextures[op.dstImageListIndex].cmdCopyDataFromBuffer(CB, resourceStaging.getBufferHandle(), 1, &op.bufferOffset, &op.width, &op.height, &op.dstImageLayerIndex);
		}

		for (auto& barrier : memBarriers)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = 0;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			barrier.srcQueueFamilyIndex = vulkanObjects->getComputeFamilyIndex();
			barrier.dstQueueFamilyIndex = vulkanObjects->getGraphicsFamilyIndex();
		}
		vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, memBarriers.size(), memBarriers.data());
	}
	commandBufferSet.endRecording(CB);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &CB };
	ASSERT_ALWAYS(vkQueueSubmit(vulkanObjects->getQueue(VulkanObjectHandler::COMPUTE_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS, "Vulkan", "Queue submission failed");
	ASSERT_ALWAYS(vkQueueWaitIdle(vulkanObjects->getQueue(VulkanObjectHandler::COMPUTE_QUEUE_TYPE)) == VK_SUCCESS, "Vulkan", "Wait idle failed");
	

	VkCommandBuffer CB2{ commandBufferSet.beginRecording(FrameCommandBufferSet::MAIN_CB) };
	for (auto& barrier : memBarriers)
	{
		barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		barrier.srcQueueFamilyIndex = vulkanObjects->getComputeFamilyIndex();
		barrier.dstQueueFamilyIndex = vulkanObjects->getGraphicsFamilyIndex();
	}
	vkCmdPipelineBarrier(CB2, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, memBarriers.size(), memBarriers.data());
	commandBufferSet.endRecording(CB2);

	VkSubmitInfo submitInfo2{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &CB2 };
	ASSERT_ALWAYS(vkQueueSubmit(vulkanObjects->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfo2, VK_NULL_HANDLE) == VK_SUCCESS, "Vulkan", "Queue submission failed");
	ASSERT_ALWAYS(vkQueueWaitIdle(vulkanObjects->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) == VK_SUCCESS, "Vulkan", "Wait idle failed");


	commandBufferSet.resetBuffers();
	
	return loadedMeshes;
}



inline PipelineCompute buildComponentTransformComputePipeline(VkDevice device,
															  std::vector<ImageList>& loadedTextures,
															  std::vector<ImageComponentTransformData>& componentTransformOperations,
															  uint8_t* const stagingDataPtr,
															  uint64_t stagingDeviceAddress,
															  uint64_t& stagingCurrentSize,
															  uint32_t stagingAlignment)
{
	std::vector<ResourceSet> resourceSets{};

	VkDescriptorSetLayoutBinding imageListsBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 64, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	std::vector<VkDescriptorImageInfo> storageImageData(loadedTextures.size());
	std::vector<VkDescriptorDataEXT> imageListsDescData(loadedTextures.size());
	for (uint32_t i{ 0 }; i < imageListsDescData.size(); ++i)
	{
		storageImageData[i] = { .imageView = loadedTextures[i].getImageView(), .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
		imageListsDescData[i].pStorageImage = &storageImageData[i];
	}
	resourceSets.push_back({ device, 0, VkDescriptorSetLayoutCreateFlags{},
		1,
		{imageListsBinding},
		{{VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT}},
		{imageListsDescData} });


	stagingCurrentSize = ALIGNED_SIZE(stagingCurrentSize, stagingAlignment);


	VkDescriptorSetLayoutBinding transformCmdsBinding{ .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	std::vector<VkDescriptorAddressInfoEXT> transCmdData(componentTransformOperations.size());
	std::vector<VkDescriptorDataEXT> transCmdDescData(componentTransformOperations.size());
	for (uint32_t i{ 0 }; i < transCmdData.size(); ++i)
	{
		ImageComponentTransformData* dataPointer{ reinterpret_cast<ImageComponentTransformData*>(stagingDataPtr + stagingCurrentSize) };
		*dataPointer = componentTransformOperations[i];
		transCmdData[i].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT;
		transCmdData[i].address = stagingDeviceAddress + stagingCurrentSize;
		transCmdData[i].range = sizeof(ImageComponentTransformData);
		stagingCurrentSize += sizeof(ImageComponentTransformData);

		stagingCurrentSize = ALIGNED_SIZE(stagingCurrentSize, stagingAlignment);

		transCmdDescData[i].pUniformBuffer = &transCmdData[i];
	}

	resourceSets.push_back({ device, 1, VkDescriptorSetLayoutCreateFlags{},
		static_cast<uint32_t>(componentTransformOperations.size()),
		{transformCmdsBinding},
		{},
		{transCmdDescData} });

	std::vector<char> compCode{ getShaderCode("shaders/cmpld/shader_comp.spv") };
	return PipelineCompute{ device, createModule(device, compCode), resourceSets };
}

inline void loadTexturesIntoStaging(VkDevice device,
									const std::vector<tinygltf::Model>& models,
									std::vector<StaticMesh>& loadedMeshes,
									std::vector<ImageList>& loadedTextures,
									std::vector<ImageTransferData>& transferOperations,
									std::vector<ImageComponentTransformData>& componentTransformOperations,
									std::vector<std::vector<int>>& meshesMaterialIndices,
									uint8_t* const stagingDataPtr,
									uint64_t stagingDeviceAddress,
									uint64_t& stagingCurrentSize)
{
	auto imageFindingFunc{ [device, &loadedTextures, &transferOperations, &componentTransformOperations, &stagingCurrentSize, stagingDeviceAddress](const tinygltf::Image& texture, std::pair<uint16_t, uint16_t>& matIndices)
		{
			bool listFound{ false };
			uint16_t listIndex{ 0 };
			uint16_t layerIndex{ 0 };
			for (uint32_t k{ 0 }; k < loadedTextures.size(); ++k)
			{
				if (loadedTextures[k].getWidth() == texture.width && loadedTextures[k].getHeight() == texture.height && loadedTextures[k].getFormat() == VK_FORMAT_R8G8B8A8_UNORM)
				{
					if (loadedTextures[k].getLayer(layerIndex) == false)
					{
						continue;
					}
					listIndex = k;
					listFound = true;
					break;
				}
			}
			if (!listFound)
			{
				listIndex = loadedTextures.size();
				loadedTextures.emplace_back(device, texture.width, texture.height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT)
					.getLayer(layerIndex);
			}
			matIndices.first = listIndex;
			matIndices.second = layerIndex;

			if (texture.component == 3)
			{
				componentTransformOperations.emplace_back(stagingDeviceAddress + stagingCurrentSize, texture.width, texture.height, listIndex, layerIndex);
			}
			else if (texture.component == 4)
			{
				transferOperations.emplace_back(stagingCurrentSize, texture.width, texture.height, listIndex, layerIndex);
			}
			else
			{
				ASSERT_ALWAYS(false, "App", "Texture has unsupported number of components");
			}
		}
	};
	for (uint32_t i{ 0 }; i < models.size(); ++i)
	{
		const std::vector<tinygltf::Material>& materials{ models[i].materials };
		const std::vector<tinygltf::Texture>& textures{ models[i].textures };
		const std::vector<tinygltf::Image>& images{ models[i].images };

		std::map<std::string, std::pair<uint16_t, uint16_t>> imageListInfo{};
		for (uint32_t j{ 0 }; j < meshesMaterialIndices[i].size(); ++j)
		{
			auto& matIndices{ loadedMeshes[i].getRUnits()[j].getMaterialIndices() };
			const tinygltf::Material& mat{ materials[meshesMaterialIndices[i][j]] };
			int bcIndex{ mat.pbrMetallicRoughness.baseColorTexture.index > -1 ? textures[mat.pbrMetallicRoughness.baseColorTexture.index].source : -1 };
			int nmIndex{ mat.normalTexture.index > -1 ? textures[mat.normalTexture.index].source : -1 };
			int mrIndex{ mat.pbrMetallicRoughness.metallicRoughnessTexture.index > -1 ? textures[mat.pbrMetallicRoughness.metallicRoughnessTexture.index].source : -1 };
			int emIndex{ mat.emissiveTexture.index > -1 ? textures[mat.emissiveTexture.index].source : -1 };
			if (bcIndex > -1)
			{
				const tinygltf::Image& baseColorTex{ images[bcIndex] };
				std::string imgName{ baseColorTex.uri };
				ASSERT_ALWAYS(!imgName.empty(), "App", "Image uri is not present");
				if (!imageListInfo.contains(imgName))
				{
					const tinygltf::Image& baseColorTex{ images[bcIndex] };
					ASSERT_ALWAYS(baseColorTex.pixel_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, "App", "Unsupported component type");
					std::memcpy(stagingDataPtr + stagingCurrentSize, baseColorTex.image.data(), baseColorTex.image.size());

					imageFindingFunc(baseColorTex, matIndices[0]);
					imageListInfo.insert({ imgName, {matIndices[0].first, matIndices[0].second} });

					stagingCurrentSize += baseColorTex.image.size();
				}
				else
				{
					matIndices[0] = imageListInfo.at(imgName);
				}
			}
			else
			{
				matIndices[0] = { 0, 0 };
			}
			if (nmIndex > -1)
			{
				const tinygltf::Image& normalTex{ images[nmIndex] };
				ASSERT_ALWAYS(normalTex.pixel_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, "App", "Unsupported component type");
				std::string imgName{ normalTex.uri };
				ASSERT_ALWAYS(!imgName.empty(), "App", "Image uri is not present");
				if (!imageListInfo.contains(imgName))
				{
					const tinygltf::Image& normalTex{ images[nmIndex] };
					std::memcpy(stagingDataPtr + stagingCurrentSize, normalTex.image.data(), normalTex.image.size());

					imageFindingFunc(normalTex, matIndices[1]);
					imageListInfo.insert({ imgName, {matIndices[1].first, matIndices[1].second} });

					stagingCurrentSize += normalTex.image.size();
				}
				else
				{
					matIndices[1] = imageListInfo.at(imgName);
				}
			}
			else
			{
				matIndices[1] = { 0, 1 };
			}
			if (mrIndex > -1)
			{
				const tinygltf::Image& metalRoughAOTex{ images[mrIndex] };
				ASSERT_ALWAYS(metalRoughAOTex.pixel_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, "App", "Unsupported component type");
				std::string imgName{ metalRoughAOTex.uri };
				ASSERT_ALWAYS(!imgName.empty(), "App", "Image uri is not present");
				if (!imageListInfo.contains(imgName))
				{
					const tinygltf::Image& metalRoughAOTex{ images[mrIndex] };

					std::memcpy(stagingDataPtr + stagingCurrentSize, metalRoughAOTex.image.data(), metalRoughAOTex.image.size());

					imageFindingFunc(metalRoughAOTex, matIndices[2]);
					imageListInfo.insert({ imgName, {matIndices[2].first, matIndices[2].second} });

					stagingCurrentSize += metalRoughAOTex.image.size();
				}
				else
				{
					matIndices[2] = imageListInfo.at(imgName);
				}
			}
			else
			{
				matIndices[2] = { 0, 2 };
			}
			if (emIndex > -1)
			{
				const tinygltf::Image& emissiveTex{ images[emIndex] };
				ASSERT_ALWAYS(emissiveTex.pixel_type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, "App", "Unsupported component type");
				std::string imgName{ emissiveTex.uri };
				ASSERT_ALWAYS(!imgName.empty(), "App", "Image uri is not present");
				if (!imageListInfo.contains(imgName))
				{
					const tinygltf::Image& emissiveTex{ images[emIndex] };
					std::memcpy(stagingDataPtr + stagingCurrentSize, emissiveTex.image.data(), emissiveTex.image.size());

					imageFindingFunc(emissiveTex, matIndices[3]);
					imageListInfo.insert({ imgName, {matIndices[3].first, matIndices[3].second} });

					stagingCurrentSize += emissiveTex.image.size();
				}
				else
				{
					matIndices[3] = imageListInfo.at(imgName);
				}
			}
			else
			{
				matIndices[3] = { 0, 3 };
			}
		}
	}
}

#endif