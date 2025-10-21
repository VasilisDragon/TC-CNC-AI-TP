#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "ai/FeatureExtractor.h"
#include "ai/ModelCard.h"
#include "ai/OnnxAI.h"
#include "ai/StrategySerialization.h"
#include "ai/TorchAI.h"
#include "render/Model.h"
#include "tp/GRBLPost.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"

#include <QtCore/QFile>
#include <QtCore/QIODevice>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QStringList>
#include <QtCore/QTemporaryDir>
#include <QtCore/QTextStream>

#include <glm/vec3.hpp>

#include <atomic>
#include <filesystem>
#include <vector>

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

const QStringList& featureNames()
{
    static const QStringList names = {
        QStringLiteral("bbox_x_mm"),
        QStringLiteral("bbox_y_mm"),
        QStringLiteral("bbox_z_mm"),
        QStringLiteral("surface_area_mm2"),
        QStringLiteral("volume_mm3"),
        QStringLiteral("slope_bin_0_15"),
        QStringLiteral("slope_bin_15_30"),
        QStringLiteral("slope_bin_30_45"),
        QStringLiteral("slope_bin_45_60"),
        QStringLiteral("slope_bin_60_90"),
        QStringLiteral("mean_curvature_rad"),
        QStringLiteral("curvature_variance_rad2"),
        QStringLiteral("flat_area_ratio"),
        QStringLiteral("steep_area_ratio"),
        QStringLiteral("pocket_depth_mm"),
        QStringLiteral("user_step_over_mm"),
        QStringLiteral("tool_diameter_mm")};
    return names;
}

std::filesystem::path toFsPath(const QString& path)
{
#ifdef _WIN32
    return std::filesystem::path(path.toStdWString());
#else
    return std::filesystem::path(path.toStdString());
#endif
}

bool writeJson(const QString& filePath, const QJsonObject& json)
{
    QFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        return false;
    }
    const QByteArray data = QJsonDocument(json).toJson(QJsonDocument::Compact);
    const bool ok = file.write(data) == data.size();
    file.close();
    return ok;
}

QJsonObject makeValidCard(const QString& modelType,
                          const QString& framework,
                          const QStringList& versions)
{
    const int featureCount = static_cast<int>(ai::FeatureExtractor::featureCount() + 2);

    QJsonObject root;
    root.insert(QStringLiteral("schema_version"), QStringLiteral("1.0.0"));
    root.insert(QStringLiteral("model_type"), modelType);

    QJsonObject features;
    features.insert(QStringLiteral("count"), featureCount);

    QJsonArray namesArray;
    for (const QString& name : featureNames())
    {
        namesArray.append(name);
    }
    features.insert(QStringLiteral("names"), namesArray);

    QJsonArray meanArray;
    QJsonArray stdArray;
    for (int i = 0; i < featureCount; ++i)
    {
        meanArray.append(0.0);
        stdArray.append(1.0);
    }

    QJsonObject normalize;
    normalize.insert(QStringLiteral("mean"), meanArray);
    normalize.insert(QStringLiteral("std"), stdArray);
    features.insert(QStringLiteral("normalize"), normalize);
    root.insert(QStringLiteral("features"), features);

    QJsonObject training;
    training.insert(QStringLiteral("framework"), framework);
    QJsonArray versionArray;
    for (const QString& version : versions)
    {
        versionArray.append(version);
    }
    training.insert(QStringLiteral("versions"), versionArray);
    root.insert(QStringLiteral("training"), training);

    QJsonObject dataset;
    dataset.insert(QStringLiteral("id"), QStringLiteral("synthetic_dataset"));
    dataset.insert(QStringLiteral("sha256"), QStringLiteral("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
    root.insert(QStringLiteral("dataset"), dataset);
    root.insert(QStringLiteral("created_at"), QStringLiteral("2025-01-01T00:00:00Z"));

    return root;
}

} // namespace

TEST_CASE("FeatureExtractor flags invalid mesh")
{
    render::Model model;
    const auto features = ai::FeatureExtractor::computeGlobalFeatures(model);
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

TEST_CASE("ModelCard validates happy path")
{
    QTemporaryDir dir;
    CHECK(dir.isValid());

    const QString modelFile = dir.filePath(QStringLiteral("sample.pt"));
    QFile binary(modelFile);
    CHECK(binary.open(QIODevice::WriteOnly));
    binary.write("torch");
    binary.close();

    const QString cardPath = dir.filePath(QStringLiteral("sample.pt.model.json"));
    const QJsonObject card = makeValidCard(QStringLiteral("torchscript"),
                                           QStringLiteral("PyTorch"),
                                           {QStringLiteral("2.3.0")});
    CHECK(writeJson(cardPath, card));

    std::string error;
    const auto loaded =
        ai::ModelCard::loadForModel(toFsPath(modelFile), ai::ModelCard::Backend::Torch, error);
    CHECK(loaded.has_value());
    CHECK(error.empty());
    if (loaded)
    {
        CHECK(loaded->featureCount == ai::FeatureExtractor::featureCount() + 2);
        CHECK(loaded->training.framework == "PyTorch");
        CHECK_FALSE(loaded->training.versions.empty());
    }
}

TEST_CASE("ModelCard rejects malformed normalization")
{
    QTemporaryDir dir;
    CHECK(dir.isValid());

    const QString modelFile = dir.filePath(QStringLiteral("sample.pt"));
    QFile binary(modelFile);
    CHECK(binary.open(QIODevice::WriteOnly));
    binary.write("torch");
    binary.close();

    QJsonObject card = makeValidCard(QStringLiteral("torchscript"),
                                     QStringLiteral("PyTorch"),
                                     {QStringLiteral("2.3.0")});
    QJsonObject features = card.value(QStringLiteral("features")).toObject();
    QJsonObject normalize = features.value(QStringLiteral("normalize")).toObject();
    normalize.insert(QStringLiteral("mean"), QJsonArray{1, 2, 3});
    features.insert(QStringLiteral("normalize"), normalize);
    card.insert(QStringLiteral("features"), features);
    CHECK(writeJson(dir.filePath(QStringLiteral("sample.pt.model.json")), card));

    std::string error;
    const auto loaded =
        ai::ModelCard::loadForModel(toFsPath(modelFile), ai::ModelCard::Backend::Torch, error);
    CHECK_FALSE(loaded.has_value());
    CHECK_FALSE(error.empty());
}

TEST_CASE("TorchAI falls back when features invalid")
{
    render::Model emptyModel;
    tp::UserParams params;
    params.stepOver = 2.0;
    ai::TorchAI torchAi{std::filesystem::path{}};

    const ai::StrategyDecision decision = torchAi.predict(emptyModel, params);
    CHECK(decision.steps.size() >= 2);
    CHECK(decision.steps.front().type == ai::StrategyStep::Type::Raster);
    CHECK(decision.steps.front().stepover == doctest::Approx(params.stepOver));
    CHECK_FALSE(torchAi.lastError().empty());
}

TEST_CASE("OnnxAI falls back when features invalid")
{
    render::Model emptyModel;
    tp::UserParams params;
    params.stepOver = 1.5;
    ai::OnnxAI onnxAi{std::filesystem::path{}};

    const ai::StrategyDecision decision = onnxAi.predict(emptyModel, params);
    CHECK(decision.steps.size() >= 2);
    CHECK(decision.steps.front().type == ai::StrategyStep::Type::Raster);
    CHECK(decision.steps.front().stepover == doctest::Approx(params.stepOver));
    CHECK_FALSE(onnxAi.lastError().empty());
}

TEST_CASE("StrategyDecision serialization round trip")
{
    ai::StrategyDecision decision;
    ai::StrategyStep rough;
    rough.type = ai::StrategyStep::Type::Raster;
    rough.stepover = 2.4;
    rough.stepdown = 1.0;
    rough.angle_deg = 45.0;
    rough.finish_pass = false;

    ai::StrategyStep finish = rough;
    finish.finish_pass = true;
    finish.stepover = 1.2;
    finish.stepdown = 0.5;
    finish.angle_deg = 90.0;

    decision.steps = {rough, finish};

    const QJsonObject json = ai::decisionToJson(decision);
    const ai::StrategyDecision restored = ai::decisionFromJson(json);
    CHECK(restored.steps.size() == decision.steps.size());
    for (std::size_t i = 0; i < decision.steps.size(); ++i)
    {
        const ai::StrategyStep& expected = decision.steps[i];
        const ai::StrategyStep& actual = restored.steps[i];
        CHECK(actual.type == expected.type);
        CHECK(actual.finish_pass == expected.finish_pass);
        CHECK(actual.stepover == doctest::Approx(expected.stepover));
        CHECK(actual.stepdown == doctest::Approx(expected.stepdown));
        CHECK(actual.angle_deg == doctest::Approx(expected.angle_deg));
    }
}

TEST_CASE("ToolpathGenerator honours override steps")
{
    render::Model model = makeTriangleModel();
    tp::UserParams params;
    params.stepOver = 1.0;
    params.maxDepthPerPass = 0.6;
    params.useStrategyOverride = true;
    params.stockAllowance_mm = 0.4;
    params.leaveStock_mm = params.stockAllowance_mm;
    params.strategyOverride.clear();

    ai::StrategyStep rough;
    rough.type = ai::StrategyStep::Type::Raster;
    rough.stepover = params.stepOver;
    rough.stepdown = params.maxDepthPerPass;
    rough.angle_deg = 0.0;
    rough.finish_pass = false;

    ai::StrategyStep finish = rough;
    finish.finish_pass = true;
    finish.stepover = params.stepOver * 0.5;
    finish.stepdown = params.maxDepthPerPass * 0.5;
    finish.angle_deg = 45.0;

    params.strategyOverride = {rough, finish};

    class NullAI : public ai::IPathAI
    {
    public:
        ai::StrategyDecision predict(const render::Model&, const tp::UserParams&) override { return {}; }
    } nullAi;

    tp::ToolpathGenerator generator;
    std::atomic<bool> cancel{false};
    tp::Toolpath toolpath = generator.generate(model, params, nullAi, cancel);
    CHECK(toolpath.strategySteps.size() == 2);
    CHECK(toolpath.strategySteps.front().finish_pass == false);
    CHECK(toolpath.strategySteps.back().finish_pass == true);
}

TEST_CASE("GRBLPost tags strategy step comments")
{
    tp::Toolpath toolpath;
    toolpath.feed = 900.0;
    toolpath.spindle = 10000.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.strategySteps.resize(2);

    tp::Polyline rough;
    rough.motion = tp::MotionType::Cut;
    rough.strategyStep = 0;
    rough.pts.push_back({glm::vec3(0.0f, 0.0f, 0.0f)});
    rough.pts.push_back({glm::vec3(5.0f, 0.0f, -1.0f)});
    toolpath.passes.push_back(rough);

    tp::Polyline finish = rough;
    finish.strategyStep = 1;
    finish.pts[0].p.y = 1.0f;
    finish.pts[1].p.y = 1.0f;
    toolpath.passes.push_back(finish);

    tp::UserParams params;
    params.strategyOverride = {ai::StrategyStep{}, ai::StrategyStep{}};

    tp::GRBLPost post;
    const std::string gcode = post.generate(toolpath, common::UnitSystem::Millimeters, params);
    CHECK(gcode.find("(STEP 1") != std::string::npos);
    CHECK(gcode.find("(STEP 2") != std::string::npos);
}

