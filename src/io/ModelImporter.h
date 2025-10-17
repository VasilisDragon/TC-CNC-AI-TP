#pragma once

#include "render/Model.h"

#include <filesystem>
#include <string>

namespace io
{

class ModelImporter
{
public:
    ModelImporter() = default;

    bool load(const std::filesystem::path& file,
              render::Model& outModel,
              std::string& error) const;
};

} // namespace io

