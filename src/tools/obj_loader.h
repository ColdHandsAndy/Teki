#ifndef OBJ_LOADER_HEADER
#define OBJ_LOADER_HEADER

#include <cstdint>
#include <filesystem>
namespace fs = std::filesystem;

#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include <tiny_obj_loader.h>

#include "src/rendering/data_management/buffer_class.h"
#include "src/tools/asserter.h"
#include "src/tools/logging.h"

namespace LoaderOBJ
{

    enum VertexOBJ : uint32_t
    {
        POS_VERT = 1u,
        NORM_VERT = 2u,
        TEXC_VERT = 4u
    };
    inline VertexOBJ operator|(VertexOBJ a, VertexOBJ b)
    {
        return static_cast<VertexOBJ>(static_cast<int>(a) | static_cast<int>(b));
    }

    inline uint32_t loadOBJfile(fs::path filepath, Buffer& vertexBuffer, VertexOBJ flags, BufferBaseHostAccessible& stagingBase, CommandBufferSet& cmdBufferSet, VkQueue queue)
    {
        tinyobj::ObjReader reader;

        BufferMapped stagingBuffer{ stagingBase };

        if (!reader.ParseFromFile(filepath.generic_string()))
        {
            EASSERT(reader.Error().empty(), "tinyobjloader", reader.Error());
        }

        LOG_IF_WARNING(!reader.Warning().empty(), "tinyobjloader issued a warning:\n\t{}", reader.Warning());

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        std::vector<float> vertices{};

        uint32_t vertexCount{ 0 };

        EASSERT(shapes.size() == 1, "App", "Unsupported number of shapes in obj model");

        for (size_t s = 0; s < shapes.size(); ++s)
        {
            // Loop over faces(polygon)
            size_t index_offset = 0;
            for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); ++f)
            {
                size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                // Loop over vertices in the face.
                for (size_t v = 0; v < fv; ++v)
                {
                    // access to vertex
                    tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                    if (flags & POS_VERT)
                    {
                        tinyobj::real_t vx = -attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                        tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                        tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];
                        vertices.push_back(vx);
                        vertices.push_back(vy);
                        vertices.push_back(vz);
                    }

                    // Check if `normal_index` is zero or positive. negative = no normal data
                    if (flags & NORM_VERT)
                    {
                        if (idx.normal_index >= 0)
                        {
                            tinyobj::real_t nx = -attrib.normals[3 * size_t(idx.normal_index) + 0];
                            tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                            tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                            vertices.push_back(nx);
                            vertices.push_back(ny);
                            vertices.push_back(nz);
                        }
                        else
                        {
                            LOG_IF_WARNING(true, "[{}]: {} were not found and loaded", "tinyobjloader", "Normals");
                        }
                    }

                    // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                    if (flags & TEXC_VERT)
                    {
                        if (idx.texcoord_index >= 0)
                        {
                            tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                            tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                            vertices.push_back(tx);
                            vertices.push_back(ty);
                        }
                        else
                        {
                            LOG_IF_WARNING(true, "[{}]: {} were not found and loaded", "tinyobjloader", "Texture coordinates");
                        }
                    }

                    ++vertexCount;
                }
                index_offset += fv;
            }

            stagingBuffer.initialize(vertices.size() * sizeof(float));
            std::memcpy(stagingBuffer.getData(), vertices.data(), stagingBuffer.getSize());
        }

        vertexBuffer.initialize(stagingBuffer.getSize());
        VkCommandBuffer cb{ cmdBufferSet.beginTransientRecording() };
            VkBufferCopy copy{ .srcOffset = stagingBuffer.getOffset(), .dstOffset = vertexBuffer.getOffset(), .size = stagingBuffer.getSize() };
            BufferTools::cmdBufferCopy(cb, stagingBuffer.getBufferHandle(), vertexBuffer.getBufferHandle(), 1, &copy);
        cmdBufferSet.endRecording(cb);
        VkSubmitInfo submitInfo{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        return vertexCount;
    }

 }
#endif