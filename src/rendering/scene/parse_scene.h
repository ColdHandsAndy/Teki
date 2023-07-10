#ifndef PARSE_SCENE_HEADER
#define PARSE_SCENE_HEADER

#include <fstream>
#include <filesystem>
#include <vector>

#include <nlohmann/json.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "src/tools/asserter.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace Scene
{
	inline void parseSceneData(fs::path sceneFile, std::vector<fs::path>& outPaths, std::vector<glm::mat4>& outMatrices, fs::path& outEnvPath)
	{
		std::ifstream f{ sceneFile };
		json scene{ json::parse(f, nullptr, false) };
		EASSERT(!scene.is_discarded(), "nlohmannJSON", "Parsing failed");

		json models{ scene["models"] };
		json modelMatrices{ scene["model matrices"] };

		size_t modelCount{ models.size() };
		size_t matrixCount{ modelMatrices.size() };

		std::vector<std::string> paths{};
		std::vector<glm::mat4> matrices{};
		std::string environmentPath{};

		for (int i{ 0 }; i < modelCount; ++i)
		{
			paths.push_back(models[i]["path"]);
			size_t index{ models[i]["model index"] };
			EASSERT(index < matrixCount, "Input", "Model matrix index is bigger than number of matrices.");
			float matr[16]{};
			for (int j{ 0 }; j < modelMatrices[index].size(); ++j)
			{
				matr[j] = modelMatrices[index][j];
			}
			matrices.push_back(glm::make_mat4x4(matr));
		}
		environmentPath = scene["environment"];

		for (int i{ 0 }; i < modelCount; ++i)
		{
			outPaths.push_back(paths[i]);
			outMatrices.push_back(matrices[i]);
		}
		outEnvPath = environmentPath;
	}
}

#endif