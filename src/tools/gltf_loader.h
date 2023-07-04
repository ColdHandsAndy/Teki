#ifndef GLTF_LOADER_HEADER
#define GLTF_LOADER_HEADER

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <iostream>
#include <map>

#include <tbb/task_arena.h>
#include <tbb/spin_mutex.h>

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <OpenImageIO/imageio.h>
#include <ktx.h>

#include "src/rendering/shader_management/shader_operations.h"
#include "src/rendering/renderer/pipeline_management.h"
#include "src/rendering/renderer/command_management.h"
#include "src/rendering/renderer/descriptor_management.h"
#include "src/rendering/data_abstraction/vertex_layouts.h"
#include "src/rendering/data_management/buffer_class.h"
#include "src/rendering/data_management/image_classes.h"
#include "src/rendering/data_abstraction/mesh.h"
#include "src/rendering/data_abstraction/runit.h"

#include "src/tools/logging.h"
#include "src/tools/alignment.h"

namespace fs = std::filesystem;

struct MaterialURIs
{
	std::string bcURI{};
	std::string nmURI{};
	std::string mrURI{};
	std::string emURI{};
};


void loadTexturesIntoStaging(uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	uint32_t stagingBufferAlignment,
	std::vector<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>>& texDataTransfersData,	//staging offset, staging size, list index, layer index
	ImageListContainer& loadedTextures,
	std::vector<StaticMesh>& meshes,
	std::vector<MaterialURIs>& meshesMaterialURIs);
inline void processMeshData(cgltf_data* model,
	cgltf_scene& scene,
	const fs::path& workPath,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	StaticMesh& mesh,
	std::vector<MaterialURIs>& meshesMaterialURIs);
inline void processNode(cgltf_data* model,
	cgltf_node* node,
	const fs::path& workPath,
	std::vector<MaterialURIs>& meshesMaterialURIs,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	StaticMesh& loadedMesh,
	const glm::mat4& nodeTransformL);
template<typename T>
inline void formVertexChunk(cgltf_data* model,
	cgltf_primitive* meshData,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	RUnit& renderUnit,
	const glm::mat4& nodeTransform);
template<>
inline void formVertexChunk<StaticVertex>(cgltf_data* model,
	cgltf_primitive* meshData,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	RUnit& renderUnit,
	const glm::mat4& nodeTransform);
inline void formIndexChunk(cgltf_data* model,
	cgltf_accessor* indexAccessor,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	RUnit& renderUnit);



inline std::vector<StaticMesh> loadStaticMeshes(
								Buffer& vertexBuffer,
								Buffer& indexBuffer,
								BufferMapped& indirectCmdBuffer,
								ImageListContainer& loadedTextures,
								std::vector<fs::path> filepaths,
								std::shared_ptr<VulkanObjectHandler> vulkanObjects,
								DescriptorManager& descriptorManager,
								FrameCommandBufferSet& commandBufferSet)
{
	cgltf_options options{};
	int modelCount = filepaths.size();
	cgltf_data** modelsData{ new cgltf_data*[modelCount]};

	std::vector<StaticMesh> meshes{};
	std::vector<MaterialURIs> meshesMaterialURIs{};

	BufferBaseHostAccessible resourceStaging{ vulkanObjects->getLogicalDevice(),
		2147483648ll, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT };
	uint32_t stagingBufferAlignment{ static_cast<uint32_t>(resourceStaging.getAlignment()) };
	uint64_t stagingCurrentSize{ 0 };
	uint8_t* const stagingDataPtr{ reinterpret_cast<uint8_t*>(resourceStaging.getData()) };
	for (int i{ 0 }; i < modelCount; ++i)
	{
		cgltf_result result1 = cgltf_parse_file(&options, filepaths[i].generic_string().c_str(), &(modelsData[i]));
		if (result1 != cgltf_result_success)
		{
			ASSERT_ALWAYS(false, "cgltf", "Parsing failed.");
		}
		cgltf_data* currentModel{ modelsData[i] };
		cgltf_result result2 = cgltf_load_buffers(&options, currentModel, filepaths[i].generic_string().c_str());
		if (result2 == cgltf_result_success)
		{
			meshes.emplace_back();
			for (int j{0}; j < currentModel->scenes_count; ++j)
			{
				processMeshData(currentModel, currentModel->scenes[j], filepaths[i].parent_path(), stagingDataPtr, stagingCurrentSize, meshes.back(), meshesMaterialURIs);
			}
		}
		else
		{
			std::cerr << "Parsing failed on file " << i << std::endl;
			assert(false);
		}
	}
	stagingCurrentSize = ALIGNED_SIZE(stagingCurrentSize, stagingBufferAlignment);

	//Prepare vertex data for upload
	uint64_t verticesByteSize{ 0 };
	uint64_t indicesByteSize{ 0 };
	for (auto& mesh : meshes)
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
	for (auto& mesh : meshes)
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

	for (auto& mesh : meshes)
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

	//staging offset, staging size, list index, layer index
	std::vector<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>> texDataTransfersData{};
	loadTexturesIntoStaging(stagingDataPtr, stagingCurrentSize, stagingBufferAlignment, texDataTransfersData, loadedTextures, meshes, meshesMaterialURIs);

	std::vector<VkImageMemoryBarrier> memBarriers{};
	for (uint32_t i{ 1 }; i < loadedTextures.getImageListCount(); ++i)
	{
		memBarriers.push_back(VkImageMemoryBarrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
													.srcAccessMask = 0,
													.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
													.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
													.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
													.image = loadedTextures.getImageHandle(i),
													.subresourceRange = loadedTextures.getImageListSubresourceRange(i) });
	}
	VkCommandBuffer CB{ commandBufferSet.beginRecording(FrameCommandBufferSet::MAIN_CB) };
	{
		BufferTools::cmdBufferCopy(CB, resourceStaging.getBufferHandle(), vertexBuffer.getBufferHandle(), copyRegionsVertexBuf.size(), copyRegionsVertexBuf.data());
		BufferTools::cmdBufferCopy(CB, resourceStaging.getBufferHandle(), indexBuffer.getBufferHandle(), copyRegionsIndexBuf.size(), copyRegionsIndexBuf.data());

		vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, memBarriers.size(), memBarriers.data());
		
		for (uint32_t i{ 0 }; i < texDataTransfersData.size(); ++i)
		{
			auto& op{ texDataTransfersData[i] };
			loadedTextures.cmdCopyDataFromBuffer(CB, std::get<2>(op), resourceStaging.getBufferHandle(), 1, &std::get<0>(op), &std::get<3>(op));
		}
		
		for (auto& barrier : memBarriers)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = 0;
			barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		vkCmdPipelineBarrier(CB, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, memBarriers.size(), memBarriers.data());
		loadedTextures.cmdCreateMipmaps(CB, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
	}
	commandBufferSet.endRecording(CB);

	VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &CB };
	ASSERT_ALWAYS(vkQueueSubmit(vulkanObjects->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE), 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS, "Vulkan", "Queue submission failed");
	ASSERT_ALWAYS(vkQueueWaitIdle(vulkanObjects->getQueue(VulkanObjectHandler::GRAPHICS_QUEUE_TYPE)) == VK_SUCCESS, "Vulkan", "Wait idle failed");

	commandBufferSet.resetBuffers();

	while (--modelCount >= 0)
	{
		cgltf_free(modelsData[modelCount]);
	}
	delete[] modelsData;
	return meshes;
}

void loadTexturesIntoStaging(uint8_t* const stagingDataPtr, 
	uint64_t& stagingCurrentSize, 
	uint32_t stagingBufferAlignment,
	std::vector<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>>& texDataTransfersData,	//staging offset, staging size, list index, layer index
	ImageListContainer& loadedTextures, 
	std::vector<StaticMesh>& meshes,
	std::vector<MaterialURIs>& meshesMaterialURIs)
{
	auto genImageList{ [](int materialTypeInd, 
						  const std::string& path,
						  ImageListContainer& loadedTextures,
						  std::map<std::string, std::pair<uint16_t, uint16_t>>& texPathsAndIndices,
						  RUnit& currentRUnit, 
						  std::vector<std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>>& texDataTransfersData,
						  uint8_t* const stagingDataPtr,
						  uint64_t& stagingCurrentSize,
						  uint32_t stagingBufferAlignment,
						  std::vector<std::unique_ptr<OIIO::ImageInput>>& images,
						  oneapi::tbb::task_group& group)
		{
			if (texPathsAndIndices.contains(path))
			{
				auto& matIndices{ currentRUnit.getMaterialIndices() };
				matIndices[materialTypeInd] = texPathsAndIndices[path];
			}
			else
			{
				if (path.empty())
				{
					auto& matIndices{ currentRUnit.getMaterialIndices() };
					matIndices[materialTypeInd].first = 0;
					matIndices[materialTypeInd].second = materialTypeInd;
					return;
				}

				VkFormat format{};
				uint32_t width{ 0 };
				uint32_t height{ 0 };

				auto imageIndex{ images.size() };
				images.push_back(OIIO::ImageInput::open(path));

				group.run([&images, imageIndex, stagingDataPtr, stagingCurrentSize]()
					{
						ASSERT_ALWAYS(images[imageIndex]->read_image(0, -1, OIIO::TypeDesc::UINT8, stagingDataPtr + stagingCurrentSize, sizeof(uint8_t) * 4) == true, "OIIO", "Could not load image.");
					});

				width = images.back()->spec().width;
				height = images.back()->spec().height;
				format = VK_FORMAT_R8G8B8A8_UNORM;

				ImageListContainer::ImageListContainerIndices indices{ loadedTextures.getNewImage(width, height, format) };
				texDataTransfersData.emplace_back();
				std::tuple<uint64_t, uint64_t, uint32_t, uint32_t>& transferData{ texDataTransfersData.back() };
				std::get<0>(transferData) = stagingCurrentSize;
				std::get<1>(transferData) = width * height * sizeof(uint8_t) * 4;
				std::get<2>(transferData) = indices.listIndex;
				std::get<3>(transferData) = indices.layerIndex;

				auto& matIndices{ currentRUnit.getMaterialIndices() };
				matIndices[materialTypeInd].first = indices.listIndex;
				matIndices[materialTypeInd].second = indices.layerIndex;

				texPathsAndIndices.emplace(path, std::pair<uint16_t, uint16_t>{ indices.listIndex, indices.layerIndex });

				stagingCurrentSize += std::get<1>(transferData);
				stagingCurrentSize = ALIGNED_SIZE(stagingCurrentSize, stagingBufferAlignment);
			}
		} 
	};

	std::map<std::string, std::pair<uint16_t, uint16_t>> texPathsAndIndices{};

	oneapi::tbb::task_group group{};

	std::vector<std::unique_ptr<OIIO::ImageInput>> images{};
	images.reserve(meshesMaterialURIs.size() * 4);

	for (int i{ 0 }, matInd{ 0 }; i < meshes.size(); ++i)
	{
		std::vector<RUnit>& rUnits{ meshes[i].getRUnits() };

		for (int j{ 0 }; j < rUnits.size(); ++j)
		{
			RUnit& currentRUnit{ rUnits[j] };
			MaterialURIs uris{ meshesMaterialURIs[matInd++] };
			genImageList(0, uris.bcURI, loadedTextures, texPathsAndIndices, currentRUnit, texDataTransfersData, stagingDataPtr, stagingCurrentSize, stagingBufferAlignment, images, group);
			genImageList(1, uris.nmURI, loadedTextures, texPathsAndIndices, currentRUnit, texDataTransfersData, stagingDataPtr, stagingCurrentSize, stagingBufferAlignment, images, group);
			genImageList(2, uris.mrURI, loadedTextures, texPathsAndIndices, currentRUnit, texDataTransfersData, stagingDataPtr, stagingCurrentSize, stagingBufferAlignment, images, group);
			genImageList(3, uris.emURI, loadedTextures, texPathsAndIndices, currentRUnit, texDataTransfersData, stagingDataPtr, stagingCurrentSize, stagingBufferAlignment, images, group);
		}
	}
	group.wait();
}

inline void processMeshData(cgltf_data* model,
	cgltf_scene& scene,
	const fs::path& workPath,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	StaticMesh& mesh,
	std::vector<MaterialURIs>& meshesMaterialURIs)
{
	for (int i{ 0 }; i < scene.nodes_count; ++i)
	{
		glm::mat4 nodeTransform{ 1.0 };
		processNode(model, scene.nodes[i], workPath, meshesMaterialURIs, stagingDataPtr, stagingCurrentSize, mesh, nodeTransform);
	}
}

inline void processNode(cgltf_data* model,
	cgltf_node* node,
	const fs::path& workPath,
	std::vector<MaterialURIs>& meshesMaterialURIs,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	StaticMesh& loadedMesh,
	const glm::mat4& nodeTransformL)
{
	float matr[16]{};
	cgltf_node_transform_local(node, matr);
	glm::mat4 nodeTransformW{ glm::make_mat4x4(matr) * nodeTransformL };

	cgltf_mesh* mesh{ node->mesh };

	if (mesh != nullptr)
	{
		for (int i{ 0 }; i < mesh->primitives_count; ++i)
		{
			ASSERT_ALWAYS(mesh->primitives->type == cgltf_primitive_type_triangles, "App", "Primitive type is not supported yet");

			RUnit& renderUnit{ loadedMesh.getRUnits().emplace_back() };
			renderUnit.setVertexSize(sizeof(StaticVertex));
			formVertexChunk<StaticVertex>(model, mesh->primitives + i, stagingDataPtr, stagingCurrentSize, renderUnit, nodeTransformW);

			renderUnit.setIndexSize(sizeof(uint32_t));
			formIndexChunk(model, mesh->primitives[i].indices, stagingDataPtr, stagingCurrentSize, renderUnit);

			cgltf_material* mat{ mesh->primitives[i].material };
			MaterialURIs mUri{};

			if (mat->pbr_metallic_roughness.base_color_texture.texture == nullptr)
				mUri.bcURI = std::string{};
			else
				mUri.bcURI = (workPath / mat->pbr_metallic_roughness.base_color_texture.texture->image->uri).generic_string();
			if (mat->normal_texture.texture == nullptr)
				mUri.nmURI = std::string{};
			else
				mUri.nmURI = (workPath / mat->normal_texture.texture->image->uri).generic_string();
			if (mat->pbr_metallic_roughness.metallic_roughness_texture.texture == nullptr)
				mUri.mrURI = std::string{};
			else
				mUri.mrURI = (workPath / mat->pbr_metallic_roughness.metallic_roughness_texture.texture->image->uri).generic_string();
			if (mat->emissive_texture.texture == nullptr)
				mUri.emURI = std::string{};
			else
				mUri.emURI = (workPath / mat->emissive_texture.texture->image->uri).generic_string();

			meshesMaterialURIs.push_back(mUri);
		}
	}
	for (int i{ 0 }; i < node->children_count; ++i)
	{
		processNode(model, node->children[i], workPath, meshesMaterialURIs, stagingDataPtr, stagingCurrentSize, loadedMesh, nodeTransformW);
	}
}

template<typename T>
inline void formVertexChunk(cgltf_data* model,
	cgltf_primitive* meshData,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	RUnit& renderUnit,
	const glm::mat4& nodeTransform) {};

template<>
inline void formVertexChunk<StaticVertex>(cgltf_data* model,
	cgltf_primitive* meshData,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	RUnit& renderUnit,
	const glm::mat4& nodeTransform)
{
	uint32_t flagcheck{ 0 };
	cgltf_attribute posAtrrib{};
	cgltf_attribute normAtrrib{};
	cgltf_attribute tangAtrrib{};
	cgltf_attribute texcAtrrib{};
	for (int i{ 0 }; i < meshData->attributes_count; ++i)
	{
		switch (meshData->attributes[i].type)
		{
			case cgltf_attribute_type_invalid:
				break;
			case cgltf_attribute_type_position:
				posAtrrib = meshData->attributes[i];
				flagcheck |= 1;
				break;
			case cgltf_attribute_type_normal:
				normAtrrib = meshData->attributes[i];
				flagcheck |= 2;
				break;
			case cgltf_attribute_type_tangent:
				tangAtrrib = meshData->attributes[i];
				flagcheck |= 4;
				break;
			case cgltf_attribute_type_texcoord:
				texcAtrrib = meshData->attributes[i];
				flagcheck |= 8;
				break;
			case cgltf_attribute_type_color:
				break;
			case cgltf_attribute_type_joints:
				break;
			case cgltf_attribute_type_weights:
				break;
			case cgltf_attribute_type_custom:
				break;
		}
	}
	if (!(flagcheck & 2))
		LOG_WARNING("{} accessor is not present.", "Normal");
	if (!(flagcheck & 4))
		LOG_WARNING("{} accessor is not present.", "Tangent");
	if (!(flagcheck & 8))
		LOG_WARNING("{} accessor is not present.", "Texture");

	int attrCount{ static_cast<int>(posAtrrib.data->count) };
	uint64_t chunkSize{ sizeof(StaticVertex) * attrCount };
	StaticVertex* vertexDataPtr{ reinterpret_cast<StaticVertex*>(stagingDataPtr + stagingCurrentSize) };
	renderUnit.setVertBufByteSize(chunkSize);
	renderUnit.setVertBufOffset(stagingCurrentSize);
	stagingCurrentSize += chunkSize;

	/*ASSERT_ALWAYS(posAtrrib.data->count == normAtrrib.data->count && normAtrrib.data->count == tangAtrrib.data->count && tangAtrrib.data->count == texcAtrrib.data->count,
		"App", "Not every vertex in a mesh has equal amount of attributes.");

	ASSERT_ALWAYS(
		posAtrrib.data->type == cgltf_type_vec3 &&
		posAtrrib.data->component_type == cgltf_component_type_r_32f &&
		normAtrrib.data->type == cgltf_type_vec3 &&
		normAtrrib.data->component_type == cgltf_component_type_r_32f &&
		tangAtrrib.data->type == cgltf_type_vec4 &&
		tangAtrrib.data->component_type == cgltf_component_type_r_32f &&
		texcAtrrib.data->type == cgltf_type_vec2 &&
		texcAtrrib.data->component_type == cgltf_component_type_r_32f,
		"App", "Attribute data is in an unsupported format."
	);*/

	const uint8_t* posData{ reinterpret_cast<uint8_t*>(posAtrrib.data->buffer_view->buffer->data) + posAtrrib.data->buffer_view->offset + posAtrrib.data->offset };
	const uint8_t* normData{ flagcheck & 2 ? reinterpret_cast<uint8_t*>(normAtrrib.data->buffer_view->buffer->data) + normAtrrib.data->buffer_view->offset + normAtrrib.data->offset : nullptr };
	const uint8_t* tangData{ flagcheck & 4 ? reinterpret_cast<uint8_t*>(tangAtrrib.data->buffer_view->buffer->data) + tangAtrrib.data->buffer_view->offset + tangAtrrib.data->offset : nullptr };
	const uint8_t* texCoordData{ flagcheck & 8 ? reinterpret_cast<uint8_t*>(texcAtrrib.data->buffer_view->buffer->data) + texcAtrrib.data->buffer_view->offset + texcAtrrib.data->offset : nullptr };
	for (uint64_t i{ 0 }; i < attrCount; ++i)
	{
		const float* posDataTyped{ reinterpret_cast<const float*>(posData) };
		vertexDataPtr->position = nodeTransform * glm::vec4{ *(posDataTyped + 0), *(posDataTyped + 1), *(posDataTyped + 2), 1.0f };
		vertexDataPtr->position.z = -vertexDataPtr->position.z;
		posData += posAtrrib.data->stride;

		if (flagcheck & 2)
		{
			const float* normDataTyped{ reinterpret_cast<const float*>(normData) };
			vertexDataPtr->normal = glm::packSnorm4x8(glm::vec4{ *(normDataTyped + 0), *(normDataTyped + 1), -*(normDataTyped + 2), 0.0f });
			normData += normAtrrib.data->stride;
		}
		else
		{
			vertexDataPtr->normal = glm::packSnorm4x8(glm::vec4{0.0f});
		}

		if (flagcheck & 4)
		{
			const float* tangDataTyped{ reinterpret_cast<const float*>(tangData) };
			vertexDataPtr->tangent = glm::packSnorm4x8(glm::vec4{ *(tangDataTyped + 0), *(tangDataTyped + 1), -*(tangDataTyped + 2), * (tangDataTyped + 3) });
			tangData += tangAtrrib.data->stride;
		}
		else
		{
			vertexDataPtr->tangent = glm::packSnorm4x8(glm::vec4{0.0f});
		}

		if (flagcheck & 8)
		{
			const float* texCoordDataTyped{ reinterpret_cast<const float*>(texCoordData) };
			vertexDataPtr->texCoords = glm::packHalf2x16(glm::vec2{ *texCoordDataTyped, *(texCoordDataTyped + 1) });
			texCoordData += texcAtrrib.data->stride;
		}
		else
		{
			vertexDataPtr->texCoords = glm::packHalf2x16(glm::vec2{0.0});
		}

		++vertexDataPtr;
	}
}

inline void formIndexChunk(cgltf_data* model,
	cgltf_accessor* indexAccessor,
	uint8_t* const stagingDataPtr,
	uint64_t& stagingCurrentSize,
	RUnit& renderUnit)
{
	uint64_t count{ indexAccessor->count };
	uint64_t offset{ indexAccessor->buffer_view->offset + indexAccessor->offset };
	uint32_t stride{ static_cast<uint32_t>(indexAccessor->stride) };

	uint64_t chunkSize{ sizeof(uint32_t) * count };
	uint32_t* indexDataPtr{ reinterpret_cast<uint32_t*>(stagingDataPtr + stagingCurrentSize) };
	renderUnit.setIndexBufByteSize(chunkSize);
	renderUnit.setIndexBufOffset(stagingCurrentSize);
	stagingCurrentSize += chunkSize;

	switch (indexAccessor->component_type)
	{
	case cgltf_component_type_r_32u:
	{
		const uint32_t* buf = reinterpret_cast<const uint32_t*>(&(reinterpret_cast<uint8_t*>(indexAccessor->buffer_view->buffer->data)[offset]));

		for (uint64_t i{ 0 }; i < count; ++i)
		{
			*(indexDataPtr++) = buf[i];
		}
		break;
	}
	case cgltf_component_type_r_16u:
	{
		const uint16_t* buf = reinterpret_cast<const uint16_t*>(&(reinterpret_cast<uint8_t*>(indexAccessor->buffer_view->buffer->data)[offset]));
		for (uint64_t i{ 0 }; i < count; ++i)
		{
			*(indexDataPtr++) = buf[i];
		}
		break;
	}
	case cgltf_component_type_r_8u:
	{
		const uint8_t* buf = reinterpret_cast<const uint8_t*>(&(reinterpret_cast<uint8_t*>(indexAccessor->buffer_view->buffer->data)[offset]));
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


#endif