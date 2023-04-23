#ifndef MDO_FORMAT_HEADER
#define MDO_FORMAT_HEADER

#include <filesystem>
#include <fstream>
#include <cstdint>
#include <vector>
#include <map>

namespace fs = std::filesystem;

namespace MDO
{
	namespace offsets
	{
		enum DataOffsets : uint64_t 
		{
			FORMAT_NAME = 0,
			FORMAT_VERSION = 4,
			MESH_NUMBER = 16,
			DATA_FLAGS = 20,
			MESH_DATA = 540
		};
	}

	namespace dataflags
	{
		enum DataFlags : uint64_t
		{
			POSITION_BIT = 1,
			NORMAL_BIT = 2,
			TANGENT_BIT = 4,
			BITANGENT_BIT = 8,
			BONES_BIT = 16,
			TEX_COORDS_BIT = 32,
			INDICES_BIT = 64
		};
	}

	class ModelSpec
	{
	public:
		
		struct MeshSpec
		{
			struct VertexComponentBufferSpec
			{
				std::streampos streamPosition{};
				uint64_t bufferByteSize{};
				uint64_t componentsNum{};
				uint32_t elementsPerComponent{};
				size_t elementSize{};
			};

			uint32_t materialIndex{};
			std::map<uint64_t, VertexComponentBufferSpec> VertexComponentBufferMap{};
		};

	private:
		uint32_t m_meshNum{};
		std::vector<MeshSpec> m_meshSpecs{};

	private:
		ModelSpec() = default;

	public:
		ModelSpec(const fs::path& filepath);

		const MeshSpec& getMeshSpec(int index) const
		{
			return m_meshSpecs.at(index);
		}
		uint32_t getMeshNum() const
		{
			return m_meshNum;
		}
	};
}

#endif