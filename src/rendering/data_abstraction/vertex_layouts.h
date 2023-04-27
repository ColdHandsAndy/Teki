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
			normal{ glm::packSnorm4x8(glm::vec4{*norm, *(norm + 1), *(norm + 2), *(norm + 3)}) },
				tangent{ glm::packSnorm4x8(glm::vec4{*tang, *(tang + 1), *(tang + 2), *(tang + 3)}) },
					texCoords{ glm::packHalf2x16(glm::vec2{*texC, *(texC + 1)}) }
	{
	}

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(StaticVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(StaticVertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32_UINT;
        attributeDescriptions[1].offset = offsetof(StaticVertex, normal);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32_UINT;
        attributeDescriptions[2].offset = offsetof(StaticVertex, tangent);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32_UINT;
        attributeDescriptions[3].offset = offsetof(StaticVertex, texCoords);

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

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};

        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(PosOnlyVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(PosOnlyVertex, position);

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

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(PosTexVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(PosTexVertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(PosTexVertex, texCord);

        return attributeDescriptions;
    }
};

#endif
