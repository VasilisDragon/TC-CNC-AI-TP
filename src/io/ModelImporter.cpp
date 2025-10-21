#include "io/ModelImporter.h"

#include "common/Enforce.h"

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <QtCore/QString>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <numbers>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef WITH_OCCT
#    include <BRepMesh_IncrementalMesh.hxx>
#    include <BRep_Tool.hxx>
#    include <IFSelect_ReturnStatus.hxx>
#    include <IGESControl_Reader.hxx>
#    include <Poly_Array1OfTriangle.hxx>
#    include <Poly_Triangulation.hxx>
#    include <STEPControl_Reader.hxx>
#    include <Standard_Failure.hxx>
#    include <TColgp_Array1OfPnt.hxx>
#    include <TopExp_Explorer.hxx>
#    include <TopAbs_ShapeEnum.hxx>
#    include <TopoDS.hxx>
#    include <TopoDS_Face.hxx>
#    include <TopLoc_Location.hxx>
#    include <TopAbs_Orientation.hxx>
#    include <gp_Pnt.hxx>
#    include <gp_Trsf.hxx>
#    include <gp_Vec.hxx>
#endif

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
// Limit meshes so a bad CAD export cannot wedge the desktop app; the value tracks a 5M triangle safety cap
// that still covers our largest production jobs.
constexpr std::size_t kMaxTriangleCount = 5'000'000; // guard against runaway imports
constexpr const char* kOcctEnableHint =
    "STEP/IGES import requires OpenCASCADE support. Reconfigure with -DWITH_OCCT=ON "
    "and ensure OpenCASCADE is available (e.g. install the vcpkg feature cnctc[occt-importer]).";

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
    if (extLower == ".obj" || extLower == ".stl")
    {
        return true;
    }
#ifdef WITH_OCCT
    if (isStepLikeExtension(extLower))
    {
        return true;
    }
#endif
    return false;
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

std::string toUtf8Path(const std::filesystem::path& path)
{
    const auto utf8Raw = path.u8string();
    std::string utf8Path;
    utf8Path.reserve(utf8Raw.size());
    for (char8_t ch : utf8Raw)
    {
        utf8Path.push_back(static_cast<char>(ch));
    }
    return utf8Path;
}

#ifdef WITH_OCCT
constexpr double kOcctChordTolerance_mm = 0.1;
constexpr double kOcctAngleTolerance_deg = 5.0;

const char* toString(IFSelect_ReturnStatus status)
{
    switch (status)
    {
    case IFSelect_RetDone:
        return "success";
    case IFSelect_RetError:
        return "error";
    case IFSelect_RetFail:
        return "failure";
    case IFSelect_RetVoid:
        return "void";
    default:
        return "unknown";
    }
}

TopoDS_Shape loadOcctShape(const std::filesystem::path& file,
                           const std::string& extensionLower,
                           std::string& error)
{
    const std::string utf8Path = toUtf8Path(file);

    if (extensionLower == ".step" || extensionLower == ".stp")
    {
        STEPControl_Reader reader;
        const IFSelect_ReturnStatus status = reader.ReadFile(utf8Path.c_str());
        if (status != IFSelect_RetDone)
        {
            error = "OpenCASCADE failed to read STEP file (" + std::string(toString(status)) + ").";
            return {};
        }

        const Standard_Integer transferred = reader.TransferRoots();
        if (transferred <= 0)
        {
            error = "STEP file did not contain transferable solids.";
            return {};
        }

        return reader.OneShape();
    }

    IGESControl_Reader reader;
    const IFSelect_ReturnStatus status = reader.ReadFile(utf8Path.c_str());
    if (status != IFSelect_RetDone)
    {
        error = "OpenCASCADE failed to read IGES file (" + std::string(toString(status)) + ").";
        return {};
    }

    const Standard_Integer transferred = reader.TransferRoots();
    if (transferred <= 0)
    {
        error = "IGES file did not contain transferable solids.";
        return {};
    }

    TopoDS_Shape shape = reader.OneShape();
    if (shape.IsNull())
    {
        error = "IGES file produced an empty shape.";
    }
    return shape;
}

bool tessellateShape(const TopoDS_Shape& shape,
                     std::vector<render::Vertex>& vertices,
                     std::vector<render::Model::Index>& indices,
                     std::string& error)
{
    const double angleTolRad = std::numbers::pi_v<double> * (kOcctAngleTolerance_deg / 180.0);
    BRepMesh_IncrementalMesh mesher(shape, kOcctChordTolerance_mm, false, angleTolRad, true);
    mesher.Perform();
    if (!mesher.IsDone())
    {
        error = "OpenCASCADE tessellation failed.";
        return false;
    }

    std::size_t triangleCount = 0;

    for (TopExp_Explorer exp(shape, TopAbs_FACE); exp.More(); exp.Next())
    {
        const TopoDS_Face face = TopoDS::Face(exp.Current());
        TopLoc_Location location;
        Handle(Poly_Triangulation) triangulation = BRep_Tool::Triangulation(face, location);
        if (triangulation.IsNull())
        {
            continue;
        }

        const Poly_Array1OfTriangle& triangles = triangulation->Triangles();
        const bool reversed = (face.Orientation() == TopAbs_REVERSED);
        const bool hasTransform = !location.IsIdentity();
        const gp_Trsf transform = location.Transformation();

        for (Standard_Integer triIndex = triangles.Lower(); triIndex <= triangles.Upper(); ++triIndex)
        {
            Standard_Integer i1, i2, i3;
            triangles(triIndex).Get(i1, i2, i3);
            if (reversed)
            {
                std::swap(i2, i3);
            }

            gp_Pnt p1 = triangulation->Node(i1);
            gp_Pnt p2 = triangulation->Node(i2);
            gp_Pnt p3 = triangulation->Node(i3);

            if (hasTransform)
            {
                p1.Transform(transform);
                p2.Transform(transform);
                p3.Transform(transform);
            }

            const gp_Vec edge1(p1, p2);
            const gp_Vec edge2(p1, p3);
            gp_Vec normalVec = edge1.Crossed(edge2);
            constexpr double kNormalEps = 1e-12;
            if (normalVec.SquareMagnitude() > kNormalEps)
            {
                normalVec.Normalize();
            }
            else
            {
                normalVec = gp_Vec(0.0, 0.0, 1.0);
            }

            QVector3D normal(static_cast<float>(normalVec.X()),
                             static_cast<float>(normalVec.Y()),
                             static_cast<float>(normalVec.Z()));
            if (!normal.isNull())
            {
                normal.normalize();
            }
            else
            {
                normal = {0.0f, 0.0f, 1.0f};
            }

            if (vertices.size() > std::numeric_limits<render::Model::Index>::max() - 3)
            {
                error = "Mesh exceeds index capacity.";
                return false;
            }

            const auto pushVertex = [&normal](const gp_Pnt& point) {
                render::Vertex vertex;
                vertex.position = {static_cast<float>(point.X()),
                                   static_cast<float>(point.Y()),
                                   static_cast<float>(point.Z())};
                vertex.normal = normal;
                return vertex;
            };

            const render::Model::Index baseIndex = static_cast<render::Model::Index>(vertices.size());
            vertices.push_back(pushVertex(p1));
            vertices.push_back(pushVertex(p2));
            vertices.push_back(pushVertex(p3));

            indices.push_back(baseIndex);
            indices.push_back(baseIndex + 1);
            indices.push_back(baseIndex + 2);

            ++triangleCount;
            if (triangleCount > kMaxTriangleCount)
            {
                error = "Mesh exceeds triangle safety limit (5M faces).";
                return false;
            }
        }
    }

    if (indices.empty())
    {
        error = "Tessellation produced no triangles.";
        return false;
    }

    return true;
}

bool loadWithOcct(const std::filesystem::path& file,
                  const std::string& extensionLower,
                  render::Model& outModel,
                  std::string& error)
{
    try
    {
        TopoDS_Shape shape = loadOcctShape(file, extensionLower, error);
        if (!error.empty() || shape.IsNull())
        {
            if (error.empty())
            {
                error = "OpenCASCADE returned an empty shape.";
            }
            return false;
        }

        std::vector<render::Vertex> vertices;
        std::vector<render::Model::Index> indices;
        vertices.reserve(32);
        indices.reserve(32);

        if (!tessellateShape(shape, vertices, indices, error))
        {
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
    catch (const Standard_Failure& failure)
    {
        const char* message = failure.GetMessageString();
        error = message && message[0] ? std::string("OpenCASCADE exception: ") + message
                                      : "OpenCASCADE exception during import.";
    }
    catch (const std::exception& ex)
    {
        error = std::string("Exception during OpenCASCADE import: ") + ex.what();
    }
    catch (...)
    {
        error = "Unknown exception during OpenCASCADE import.";
    }
    return false;
}
#endif

} // namespace

bool ModelImporter::load(const std::filesystem::path& file,
                         render::Model& outModel,
                         std::string& error) const
{
    namespace fs = std::filesystem;

    error.clear();
    ENFORCE(outModel.vertices().empty() && outModel.indices().empty(),
            "Destination model must be empty before import.");

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

    const std::string extension = toLower(file.extension().string());

#ifdef WITH_OCCT
    if (isStepLikeExtension(extension))
    {
        const bool ok = loadWithOcct(file, extension, outModel, error);
        if (!ok && error.empty())
        {
            error = "Failed to load CAD file.";
        }
        return ok;
    }
#else
    if (isStepLikeExtension(extension))
    {
        error = std::string("STEP/IGES import requires OpenCASCADE.\n") + kOcctEnableHint;
        return false;
    }
#endif

    if (!isSupportedExtension(extension))
    {
        error = "Unsupported file extension. Supported formats: OBJ, STL"
#ifdef WITH_OCCT
                ", STEP, IGES"
#endif
                ".";
        return false;
    }

    Assimp::Importer importer;
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    importer.SetPropertyBool(AI_CONFIG_PP_PTV_KEEP_HIERARCHY, false);

    const std::string utf8Path = toUtf8Path(file);

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
