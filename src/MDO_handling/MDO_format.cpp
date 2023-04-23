#include "src/MDO_handling/MDO_format.h"

#include <iostream>
#include <array>

MDO::ModelSpec::ModelSpec(const fs::path& filepath)
{
	std::ifstream ifstream{ filepath, std::ios::binary };

	{
		char formatName[4]{};
		std::array<uint32_t, 3> formatVersion{};

		ifstream.read(formatName, 4);
		ifstream.read(reinterpret_cast<char*>(formatVersion.data()), sizeof(uint32_t) * formatVersion.size());

		std::cout << "Model " << filepath.stem() << " has format " << std::string(formatName) << '(' << formatVersion[0] << '.' << formatVersion[1] << '.' << formatVersion[1] << ')' << std::endl;
	}

	uint32_t meshNum{};
	ifstream.read(reinterpret_cast<char*>(&meshNum), sizeof(meshNum));

	m_meshNum = meshNum;
	m_meshSpecs.resize(meshNum);

	uint64_t dataflags{};
	ifstream.read(reinterpret_cast<char*>(&dataflags), sizeof(dataflags));

	ifstream.seekg(MDO::offsets::MESH_DATA);

	for (unsigned int i{ 0 }; i < meshNum; ++i)
	{
		MDO::ModelSpec::MeshSpec& meshspec = m_meshSpecs[i];

		std::streampos meshDataStartPos{ ifstream.tellg() };

		uint32_t materialIndex{};
		ifstream.read(reinterpret_cast<char*>(&materialIndex), sizeof(materialIndex));

		meshspec.materialIndex = materialIndex;

		constexpr std::streamoff blockDataOffset{ sizeof(uint64_t) };


		std::streamoff posBlockOffset{ sizeof(materialIndex) };
		uint64_t posBlockByteSize{ 0 };
		if (dataflags & MDO::dataflags::POSITION_BIT)
		{
			uint64_t posBufferByteSize{ 0 };
			ifstream.seekg(meshDataStartPos + posBlockOffset);
			ifstream.read(reinterpret_cast<char*>(&posBufferByteSize), sizeof(posBufferByteSize));

			posBlockByteSize = blockDataOffset + posBufferByteSize;

			meshspec.VertexComponentBufferMap[MDO::dataflags::POSITION_BIT] =
				MDO::ModelSpec::MeshSpec::VertexComponentBufferSpec{ meshDataStartPos + posBlockOffset + blockDataOffset, posBufferByteSize, posBufferByteSize / sizeof(float) / 3, 3, sizeof(float)};
		}

		std::streamoff normalBlockOffset{ static_cast<std::streamoff>(posBlockOffset + posBlockByteSize)};
		uint64_t normalBlockByteSize{ 0 };
		if (dataflags & MDO::dataflags::NORMAL_BIT)
		{
			uint64_t normalBufferByteSize{ 0 };
			ifstream.seekg(meshDataStartPos + normalBlockOffset);
			ifstream.read(reinterpret_cast<char*>(&normalBufferByteSize), sizeof(normalBufferByteSize));

			normalBlockByteSize = blockDataOffset + normalBufferByteSize;

			meshspec.VertexComponentBufferMap[MDO::dataflags::NORMAL_BIT] =
				MDO::ModelSpec::MeshSpec::VertexComponentBufferSpec{ meshDataStartPos + normalBlockOffset + blockDataOffset, normalBufferByteSize, normalBufferByteSize / sizeof(float) / 3, 3, sizeof(float) };
		}

		std::streamoff tangentBlockOffset{ static_cast<std::streamoff>(normalBlockOffset + normalBlockByteSize) };
		uint64_t tangentBlockByteSize{ 0 };
		if (dataflags & MDO::dataflags::TANGENT_BIT)
		{
			uint64_t tangentBufferByteSize{ 0 };
			ifstream.seekg(meshDataStartPos + tangentBlockOffset);
			ifstream.read(reinterpret_cast<char*>(&tangentBufferByteSize), sizeof(tangentBufferByteSize));

			tangentBlockByteSize = blockDataOffset + tangentBufferByteSize;

			meshspec.VertexComponentBufferMap[MDO::dataflags::TANGENT_BIT] =
				MDO::ModelSpec::MeshSpec::VertexComponentBufferSpec{ meshDataStartPos + tangentBlockOffset + blockDataOffset, tangentBufferByteSize, tangentBufferByteSize / sizeof(float) / 3, 3, sizeof(float) };
		}

		std::streamoff bitangentBlockOffset{ static_cast<std::streamoff>(tangentBlockOffset + tangentBlockByteSize) };
		uint64_t bitangentBlockByteSize{ 0 };
		if (dataflags & MDO::dataflags::BITANGENT_BIT)
		{
			uint64_t bitangentBufferByteSize{ 0 };
			ifstream.seekg(meshDataStartPos + bitangentBlockOffset);
			ifstream.read(reinterpret_cast<char*>(&bitangentBufferByteSize), sizeof(bitangentBufferByteSize));

			bitangentBlockByteSize = blockDataOffset + bitangentBufferByteSize;

			meshspec.VertexComponentBufferMap[MDO::dataflags::BITANGENT_BIT] =
				MDO::ModelSpec::MeshSpec::VertexComponentBufferSpec{ meshDataStartPos + bitangentBlockOffset + blockDataOffset, bitangentBufferByteSize, bitangentBufferByteSize / sizeof(float) / 3, 3, sizeof(float) };
		}

		std::streamoff bonesBlockOffset{ static_cast<std::streamoff>(bitangentBlockOffset + bitangentBlockByteSize) };
		uint64_t bonesBlockByteSize{ 0 };
		if (dataflags & MDO::dataflags::BONES_BIT)
		{
			uint64_t bonesBufferByteSize{ 0 };
			ifstream.seekg(meshDataStartPos + bonesBlockOffset);
			ifstream.read(reinterpret_cast<char*>(&bonesBufferByteSize), sizeof(bonesBufferByteSize));

			//specify bone component later
		}

		std::streamoff texCoordsBlockOffset{ static_cast<std::streamoff>(bonesBlockOffset + bonesBlockByteSize) };
		uint64_t texCoordsBlockByteSize{ 0 };
		if (dataflags & MDO::dataflags::TEX_COORDS_BIT)
		{
			uint64_t texCoordsBufferByteSize{ 0 };
			ifstream.seekg(meshDataStartPos + texCoordsBlockOffset);
			ifstream.read(reinterpret_cast<char*>(&texCoordsBufferByteSize), sizeof(texCoordsBufferByteSize));

			texCoordsBlockByteSize = blockDataOffset + texCoordsBufferByteSize;

			meshspec.VertexComponentBufferMap[MDO::dataflags::TEX_COORDS_BIT] =
				MDO::ModelSpec::MeshSpec::VertexComponentBufferSpec{ meshDataStartPos + texCoordsBlockOffset + blockDataOffset, texCoordsBufferByteSize, texCoordsBufferByteSize / sizeof(float) / 2, 2, sizeof(float) };
		}

		std::streamoff indicesBlockOffset{ static_cast<std::streamoff>(texCoordsBlockOffset + texCoordsBlockByteSize) };
		uint64_t indicesBlockByteSize{ 0 };
		if (dataflags & MDO::dataflags::INDICES_BIT)
		{
			uint64_t indicesBufferByteSize{ 0 };
			ifstream.seekg(meshDataStartPos + indicesBlockOffset);
			ifstream.read(reinterpret_cast<char*>(&indicesBufferByteSize), sizeof(indicesBufferByteSize));

			indicesBlockByteSize = blockDataOffset + indicesBufferByteSize;

			meshspec.VertexComponentBufferMap[MDO::dataflags::INDICES_BIT] =
				MDO::ModelSpec::MeshSpec::VertexComponentBufferSpec{ meshDataStartPos + indicesBlockOffset + blockDataOffset, indicesBufferByteSize, indicesBufferByteSize / sizeof(uint32_t), 1, sizeof(uint32_t) };
		}

		ifstream.seekg(meshDataStartPos + indicesBlockOffset + static_cast<std::streamoff>(indicesBlockByteSize));
	}

	ifstream.close();
}