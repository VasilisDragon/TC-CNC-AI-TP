#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "ai/FeatureExtractor.h"
#include "ai/OnnxAI.h"
#include "ai/TorchAI.h"
#include "render/Model.h"
#include "tp/GRBLPost.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"

#include <glm/vec3.hpp>

namespace
{

render::Model makeTriangleModel()
{
    render::Model model;
    std::vector<render::Vertex> vertices(3);
    vertices[0].position = QVector3D(0.0f, 0.0f, 0.0f);
    vertices[1].position = QVector3D(1.0f, 0.0f, 0.0f);
    vertices[2].position = QVector3D(0.0f, 1.0f, 0.0f);

    for (auto& v : vertices)
    {
        v.normal = QVector3D(0.0f, 0.0f, 1.0f);
    }

    std::vector<render::Model::Index> indices = {0, 1, 2};
    model.setMeshData(std::move(vertices), std::move(indices));
    return model;
}

} // namespace

TEST_CASE("FeatureExtractor flags invalid mesh")
{
    render::Model emptyModel;
    const auto features = ai::FeatureExtractor::computeGlobalFeatures(emptyModel);
    CHECK_FALSE(features.valid);
}

TEST_CASE("FeatureExtractor computes triangle metrics")
{
    const render::Model model = makeTriangleModel();
    const auto features = ai::FeatureExtractor::computeGlobalFeatures(model);
    CHECK(features.valid);
    CHECK(features.surfaceArea == doctest::Approx(0.5f));
    CHECK(features.volume == doctest::Approx(0.0f));
    CHECK(features.flatAreaRatio == doctest::Approx(1.0f));
    CHECK(features.steepAreaRatio == doctest::Approx(0.0f));
    CHECK(features.pocketDepth == doctest::Approx(0.0f));
}

TEST_CASE("TorchAI falls back when features invalid")
{
    render::Model emptyModel;
    tp::UserParams params;
    params.stepOver = 2.0;
    ai::TorchAI torchAi{std::filesystem::path{}};

    const ai::StrategyDecision decision = torchAi.predict(emptyModel, params);
    CHECK(decision.strat == ai::StrategyDecision::Strategy::Raster);
    CHECK(decision.stepOverMM == doctest::Approx(params.stepOver));
    CHECK_FALSE(torchAi.lastError().empty());
}

TEST_CASE("OnnxAI falls back when features invalid")
{
    render::Model emptyModel;
    tp::UserParams params;
    params.stepOver = 1.5;
    ai::OnnxAI onnxAi{std::filesystem::path{}};

    const ai::StrategyDecision decision = onnxAi.predict(emptyModel, params);
    CHECK(decision.strat == ai::StrategyDecision::Strategy::Raster);
    CHECK(decision.stepOverMM == doctest::Approx(params.stepOver));
    CHECK_FALSE(onnxAi.lastError().empty());
}

TEST_CASE("GRBLPost emits feed, unit, and tool moves")
{
    tp::Toolpath toolpath;
    toolpath.feed = 900.0;
    toolpath.spindle = 10000.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.machine.name = "Test Rig";
    toolpath.machine.rapidFeed_mm_min = 5000.0;
    toolpath.machine.maxFeed_mm_min = 1500.0;
    toolpath.rapidFeed = toolpath.machine.rapidFeed_mm_min;
    toolpath.stock = tp::makeDefaultStock();

    tp::Polyline line;
    line.motion = tp::MotionType::Cut;
    line.pts.push_back({glm::vec3(0.0f, 0.0f, 0.0f)});
    line.pts.push_back({glm::vec3(5.0f, 0.0f, -1.0f)});
    toolpath.passes.push_back(line);

    tp::UserParams params;
    params.feed = toolpath.feed;
    params.spindle = toolpath.spindle;
    params.machine = toolpath.machine;
    params.stock = toolpath.stock;

    tp::GRBLPost post;
    const std::string gcode = post.generate(toolpath, common::Unit::Millimeters, params);

    CHECK(gcode.find("G21") != std::string::npos); // metric units
    CHECK(gcode.find("F900.000") != std::string::npos);
    CHECK(gcode.find("M3 S10000") != std::string::npos);
    CHECK(gcode.find("G1 X5.000 Y0.000 Z-1.000") != std::string::npos);
}
