#ifndef SHADER_OPERATIONS_HEADER
#define SHADER_OPERATIONS_HEADER

#include <vector>
#include <filesystem>
#include <fstream>

#include <vulkan/vulkan.h>

namespace fs = std::filesystem;

namespace ShaderOperations
{
	inline std::vector<char> getShaderCode(const fs::path & filepath)
	{
		std::ifstream stream{ filepath, std::ios::ate | std::ios::binary };
		EASSERT(stream.is_open(), "App", "Could not open shader file");

		size_t streamSize{ static_cast<size_t>(stream.tellg()) };
		size_t codeSize{ streamSize };
		if (static_cast<size_t>(streamSize) % 4 != 0)
			codeSize += (4 - static_cast<size_t>(streamSize) % 4);

		std::vector<char> code(codeSize);
		stream.seekg(std::ios_base::beg);
		stream.read(code.data(), streamSize);
		return code;
	}

	inline VkShaderModule createModule(VkDevice device, const std::vector<char>&code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		EASSERT(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) == VK_SUCCESS, "Vulkan", "Shader module creation failed.");
		return shaderModule;
	}
}

#endif