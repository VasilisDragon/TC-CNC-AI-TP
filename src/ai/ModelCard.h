#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ai
{

class ModelCard
{
public:
    enum class Backend
    {
        Torch,
        Onnx
    };

    struct Normalization
    {
        std::vector<double> mean;
        std::vector<double> std;
    };

    struct TrainingInfo
    {
        std::string framework;
        std::vector<std::string> versions;
    };

    struct DatasetInfo
    {
        std::string id;
        std::string sha256;
    };

    std::filesystem::path path;
    std::string schemaVersion;
    std::string modelType;
    std::size_t featureCount{0};
    std::vector<std::string> featureNames;
    Normalization normalization;
    TrainingInfo training;
    DatasetInfo dataset;
    std::string createdAt;

    [[nodiscard]] static std::optional<ModelCard> loadForModel(const std::filesystem::path& modelPath,
                                                               Backend backend,
                                                               std::string& errorOut);

private:
    [[nodiscard]] static std::optional<ModelCard> loadFromPath(const std::filesystem::path& cardPath,
                                                               Backend backend,
                                                               std::string& errorOut);
};

} // namespace ai

