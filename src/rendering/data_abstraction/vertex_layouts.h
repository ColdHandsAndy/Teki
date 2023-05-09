#ifndef VERTEX_LAYOUTS_HEADER
#define VERTEX_LAYOUTS_HEADER

#include <array>

#include "vulkan/vulkan.h"
#include "glm/glm.hpp"

typedef uint32_t packed4x8int;
typedef uint32_t packed2x16float;

struct StaticVertex
{
	glm::vec3 position{};
	packed4x8int normal{};
	packed4x8int tangent{};
	packed2x16float texCoords{};

	StaticVertex(const glm::vec3& pos, const glm::vec3& norm, const glm::vec3& tang, const glm::vec2& texC)
		: position{ pos }, normal{ glm::packSnorm4x8(glm::vec4{ norm, 0.0f }) }, tangent{ glm::packSnorm4x8(glm::vec4{ tang, 0.0f }) }, texCoords{ glm::packHalf2x16(texC) }
	{
	}
	StaticVertex(const float* pos, const float* norm, const float* tang, const float* texC)
		: position{ glm::vec3{*pos, *(pos + 1), *(pos + 2)} },
			normal{ glm::packSnorm4x8(glm::vec4{*norm, *(norm + 1), *(norm + 2), 0.0f}) },
				tangent{ glm::packSnorm4x8(glm::vec4{*tang, *(tang + 1), *(tang + 2), 0.0f}) },
					texCoords{ glm::packHalf2x16(glm::vec2{*texC, *(texC + 1)}) }
	{
	}

    static const VkVertexInputBindingDescription& getBindingDescription()
    {
        static const VkVertexInputBindingDescription bindingDescription
        {
            .binding = 0,
            .stride = sizeof(StaticVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };

        return bindingDescription;
    }

    static const std::array<VkVertexInputAttributeDescription, 4>& getAttributeDescriptions()
    {
        static const std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions
        {
        VkVertexInputAttributeDescription
        {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(StaticVertex, position)
        },
        VkVertexInputAttributeDescription
        {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32_UINT,
            .offset = offsetof(StaticVertex, normal)
        },
        VkVertexInputAttributeDescription
        {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32_UINT,
            .offset = offsetof(StaticVertex, tangent)
        },
        VkVertexInputAttributeDescription
        {
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32_UINT,
            .offset = offsetof(StaticVertex, texCoords)
        }
        };

        return attributeDescriptions;
    }
};


struct PosOnlyVertex
{
    glm::vec3 position{};

    PosOnlyVertex(const glm::vec3& pos)
        : position{ pos }
    {
    }
    PosOnlyVertex(const float* pos)
        : position{ glm::vec3{*pos, *(pos + 1), *(pos + 2)} }
    {
    }

    static const VkVertexInputBindingDescription& getBindingDescription()
    {
        static const VkVertexInputBindingDescription bindingDescription{
            .binding = 0, 
            .stride = sizeof(PosOnlyVertex), 
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX 
        };

        return bindingDescription;
    }

    static const std::array<VkVertexInputAttributeDescription, 1>& getAttributeDescriptions()
    {
        static const std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions
        {
            VkVertexInputAttributeDescription
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(PosOnlyVertex, position)
            }
        };

        return attributeDescriptions;
    }
};

struct PosTexVertex
{
    glm::vec3 position{};
    glm::vec2 texCord{};

    PosTexVertex(const glm::vec3& pos, const glm::vec2& texC)
        : position{ pos }, texCord{ texC }
    {
    }
    PosTexVertex(const float* pos, const float* texC)
        : position{ glm::vec3{*pos, *(pos + 1), *(pos + 2)} },
            texCord{ glm::vec2{*texC, *(texC + 1)} }
    {
    }

    static const VkVertexInputBindingDescription& getBindingDescription()
    {
        static const VkVertexInputBindingDescription bindingDescription{
            .binding = 0,
            .stride = sizeof(PosTexVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
        };

        return bindingDescription;
    }


    static const std::array<VkVertexInputAttributeDescription, 2>& getAttributeDescriptions()
    {
        static const std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions
        {
            VkVertexInputAttributeDescription
            {
                .location = 0,
                .binding = 0,
                .format = VK_FORMAT_R32G32B32_SFLOAT,
                .offset = offsetof(PosTexVertex, position)
            },
            VkVertexInputAttributeDescription
            {
                .location = 1,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(PosTexVertex, texCord)
            },
        };

        return attributeDescriptions;
    }
};

#endif
