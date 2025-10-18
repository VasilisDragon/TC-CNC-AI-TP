#include "io/ModelImporter.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <QtCore/QString>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <unordered_set>
#include <utility>
#include <vector>

namespace io
{

namespace
{

constexpr unsigned int kPostProcessFlags =
    aiProcess_Triangulate |
    aiProcess_JoinIdenticalVertices |
    aiProcess_GenSmoothNormals |
    aiProcess_ImproveCacheLocality |
    aiProcess_RemoveRedundantMaterials |
    aiProcess_PreTransformVertices |
    aiProcess_SortByPType |
    aiProcess_CalcTangentSpace;

constexpr std::uintmax_t kMaxFileSizeBytes = 200 * 1024ull * 1024ull; // 200 MB
constexpr std::size_t kMaxTriangleCount = 5'000'000; // guard against runaway imports

std::string toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool isStepLikeExtension(const std::string& extLower)
{
    static const std::unordered_set<std::string> stepExtensions = {
        ".step", ".stp", ".iges", ".igs"
    };
    return stepExtensions.count(extLower) > 0;
}

bool isSupportedExtension(const std::string& extLower)
{
    static const std::unordered_set<std::string> supportedExtensions = {
        ".obj", ".stl"
    };
    return supportedExtensions.count(extLower) > 0;
}

render::Vertex makeVertex(const aiMesh* mesh, unsigned int index)
{
    render::Vertex vertex;
    const aiVector3D& position = mesh->mVertices[index];
    vertex.position = {position.x, position.y, position.z};

    if (mesh->HasNormals())
    {
        const aiVector3D& normal = mesh->mNormals[index];
        vertex.normal = {normal.x, normal.y, normal.z};
    }
    else
    {
        vertex.normal = {0.0f, 0.0f, 1.0f};
    }

    return vertex;
}

} // namespace

bool ModelImporter::load(const std::filesystem::path& file,
                         render::Model& outModel,
                         std::string& error) const
{
    namespace fs = std::filesystem;

    error.clear();

    if (!fs::exists(file) || !fs::is_regular_file(file))
    {
        error = "File does not exist.";
        return false;
    }

    std::error_code ec;
    const std::uintmax_t fileSize = fs::file_size(file, ec);
    if (!ec && fileSize > kMaxFileSizeBytes)
    {
        error = "File too large for import safeguard (limit 200 MB).";
        return false;
    }

    std::string extension = toLower(file.extension().string());

    if (isStepLikeExtension(extension))
    {
        error = "Use STEP/IGES export via neutral format later. Currently supported: OBJ/STL.";
        return false;
    }

    if (!isSupportedExtension(extension))
    {
        error = "Unsupported file extension. Supported formats: OBJ, STL.";
        return false;
    }

    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    importer.SetPropertyBool(AI_CONFIG_PP_PTV_KEEP_HIERARCHY, false);

    const auto utf8Raw = file.u8string();
    std::string utf8Path;
    utf8Path.reserve(utf8Raw.size());
    for (char8_t ch : utf8Raw)
    {
        utf8Path.push_back(static_cast<char>(ch));
    }

    const aiScene* scene = importer.ReadFile(utf8Path.c_str(), kPostProcessFlags);
    if (!scene || !scene->HasMeshes())
    {
        error = importer.GetErrorString();
        if (error.empty())
        {
            error = "Failed to load mesh.";
        }
        return false;
    }

    std::vector<render::Vertex> vertices;
    std::vector<render::Model::Index> indices;

    size_t estimatedVertexCount = 0;
    size_t estimatedIndexCount = 0;
    std::size_t triangleCount = 0;
    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
    {
        const aiMesh* mesh = scene->mMeshes[meshIndex];
        if (mesh)
        {
            estimatedVertexCount += mesh->mNumVertices;
            estimatedIndexCount += mesh->mNumFaces * 3;
            triangleCount += mesh->mNumFaces;
            if (triangleCount > kMaxTriangleCount)
            {
                error = "Mesh exceeds triangle safety limit (5M faces).";
                return false;
            }
        }
    }

    vertices.reserve(estimatedVertexCount);
    indices.reserve(estimatedIndexCount);

    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex)
    {
        const aiMesh* mesh = scene->mMeshes[meshIndex];
        if (!mesh || mesh->mNumVertices == 0 || mesh->mNumFaces == 0)
        {
            continue;
        }

        const auto baseIndex = static_cast<render::Model::Index>(vertices.size());

        for (unsigned int v = 0; v < mesh->mNumVertices; ++v)
        {
            vertices.push_back(makeVertex(mesh, v));
        }

        for (unsigned int f = 0; f < mesh->mNumFaces; ++f)
        {
            const aiFace& face = mesh->mFaces[f];
            if (face.mNumIndices < 3)
            {
                continue;
            }

            indices.push_back(baseIndex + static_cast<render::Model::Index>(face.mIndices[0]));
            indices.push_back(baseIndex + static_cast<render::Model::Index>(face.mIndices[1]));
            indices.push_back(baseIndex + static_cast<render::Model::Index>(face.mIndices[2]));
        }
    }

    if (vertices.empty() || indices.empty())
    {
        error = "No triangle data found in file.";
        return false;
    }

#if defined(_WIN32)
    outModel.setName(QString::fromStdWString(file.filename().wstring()));
#else
    outModel.setName(QString::fromStdString(file.filename().string()));
#endif
    outModel.setMeshData(std::move(vertices), std::move(indices));

    return true;
}

} // namespace io
