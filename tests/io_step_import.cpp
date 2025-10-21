#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "doctest/doctest.h"

#include "io/ModelImporter.h"
#include "render/Model.h"

#include <filesystem>
#include <string>

#ifndef CNCTC_SOURCE_DIR
#    error "CNCTC_SOURCE_DIR must be defined"
#endif

#ifdef WITH_OCCT
#    include <BRepPrimAPI_MakeBox.hxx>
#    include <STEPControl_Writer.hxx>
#    include <STEPControl_StepModelType.hxx>
#    include <IFSelect_ReturnStatus.hxx>

namespace
{

std::filesystem::path generateTinyStep()
{
    const std::filesystem::path outputPath =
        std::filesystem::temp_directory_path() / "cnctc_tiny_block.step";

    BRepPrimAPI_MakeBox boxBuilder(8.0, 6.0, 4.0);
    const TopoDS_Shape shape = boxBuilder.Shape();

    STEPControl_Writer writer;
    const IFSelect_ReturnStatus transferStatus = writer.Transfer(shape, STEPControl_AsIs);
    CHECK(transferStatus == IFSelect_RetDone);

    const std::string outputUtf8 = outputPath.string();
    const IFSelect_ReturnStatus writeStatus = writer.Write(outputUtf8.c_str());
    CHECK(writeStatus == IFSelect_RetDone);

    CHECK(std::filesystem::exists(outputPath));
    return outputPath;
}

} // namespace

TEST_CASE("OCCT STEP importer tessellates tiny block")
{
    std::filesystem::path stepPath =
        std::filesystem::path(CNCTC_SOURCE_DIR) / "testdata" / "cad" / "tiny_block.step";

    io::ModelImporter importer;
    render::Model model;
    std::string error;

    bool loaded = std::filesystem::exists(stepPath)
                      ? importer.load(stepPath, model, error)
                      : false;
    if (!loaded)
    {
        const std::filesystem::path fallback = generateTinyStep();
        stepPath = fallback;
        error.clear();
        loaded = importer.load(stepPath, model, error);
    }

    CHECK(loaded);
    CHECK(error.empty());
    CHECK(model.isValid());

    const auto& vertices = model.vertices();
    const auto& indices = model.indices();

    CHECK(!vertices.empty());
    CHECK(!indices.empty());
    CHECK(indices.size() % 3 == 0);
    CHECK(indices.size() >= 36);

    const common::Bounds bounds = model.bounds();
    const QVector3D min = bounds.min;
    const QVector3D max = bounds.max;
    const QVector3D size = bounds.size();

    CHECK(static_cast<double>(min.x()) == doctest::Approx(0.0).epsilon(1e-4));
    CHECK(static_cast<double>(min.y()) == doctest::Approx(0.0).epsilon(1e-4));
    CHECK(static_cast<double>(min.z()) == doctest::Approx(0.0).epsilon(1e-4));

    CHECK(static_cast<double>(size.x()) == doctest::Approx(8.0).epsilon(1e-3));
    CHECK(static_cast<double>(size.y()) == doctest::Approx(6.0).epsilon(1e-3));
    CHECK(static_cast<double>(size.z()) == doctest::Approx(4.0).epsilon(1e-3));

    CHECK(static_cast<double>(max.x()) == doctest::Approx(8.0).epsilon(1e-3));
    CHECK(static_cast<double>(max.y()) == doctest::Approx(6.0).epsilon(1e-3));
    CHECK(static_cast<double>(max.z()) == doctest::Approx(4.0).epsilon(1e-3));
}
#else
TEST_CASE("OCCT STEP importer tessellates tiny block (skipped)")
{
    MESSAGE("WITH_OCCT not enabled; skipping STEP importer test.");
}
#endif
