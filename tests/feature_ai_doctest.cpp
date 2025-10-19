#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "ai/FeatureExtractor.h"
#include "ai/ModelCard.h"
#include "ai/OnnxAI.h"
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

#include <glm/vec3.hpp>
#include <filesystem>

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
    dataset.insert(QStringLiteral("id"), QStringLiteral("testset"));
    dataset.insert(QStringLiteral("sha256"),
                   QStringLiteral("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));
    root.insert(QStringLiteral("dataset"), dataset);
    root.insert(QStringLiteral("created_at"), QStringLiteral("2025-01-15T12:00:00Z"));

    return root;
}

} // namespace

TEST_CASE("ModelCard loads valid Torch schema")
{
    QTemporaryDir dir;
    CHECK(dir.isValid());
    if (!dir.isValid())
    {
        return;
    }

    const QString modelFile = dir.filePath(QStringLiteral("sample.pt"));
    QFile torchFile(modelFile);
    const bool openedTorch = torchFile.open(QIODevice::WriteOnly);
    CHECK(openedTorch);
    if (!openedTorch)
    {
        return;
    }
    torchFile.close();

    QJsonObject card = makeValidCard(QStringLiteral("torchscript"),
                                     QStringLiteral("PyTorch"),
                                     {QStringLiteral("2.1.0")});
    const bool savedTorchCard = writeJson(dir.filePath(QStringLiteral("sample.pt.model.json")), card);
    CHECK(savedTorchCard);
    if (!savedTorchCard)
    {
        return;
    }

    std::string error;
    const auto loaded = ai::ModelCard::loadForModel(toFsPath(modelFile),
                                                    ai::ModelCard::Backend::Torch,
                                                    error);
    CHECK(loaded.has_value());
    if (!loaded.has_value())
    {
        return;
    }
    CHECK(error.empty());
    CHECK(loaded->featureCount == ai::FeatureExtractor::featureCount() + 2);
    CHECK(loaded->normalization.mean.size() == loaded->featureCount);
    CHECK(loaded->training.framework.find("PyTorch") != std::string::npos);
}

TEST_CASE("ModelCard rejects normalization mismatch")
{
    QTemporaryDir dir;
    CHECK(dir.isValid());
    if (!dir.isValid())
    {
        return;
    }

    const QString modelFile = dir.filePath(QStringLiteral("broken.pt"));
    QFile brokenFile(modelFile);
    const bool openedBroken = brokenFile.open(QIODevice::WriteOnly);
    CHECK(openedBroken);
    if (!openedBroken)
    {
        return;
    }
    brokenFile.close();

    QJsonObject card = makeValidCard(QStringLiteral("torchscript"),
                                     QStringLiteral("PyTorch"),
                                     {QStringLiteral("2.1.0")});
    QJsonObject features = card.value(QStringLiteral("features")).toObject();
    QJsonObject normalize = features.value(QStringLiteral("normalize")).toObject();
    QJsonArray mean;
    mean.append(0.0);
    normalize.insert(QStringLiteral("mean"), mean);
    features.insert(QStringLiteral("normalize"), normalize);
    card.insert(QStringLiteral("features"), features);
    const bool savedBroken = writeJson(dir.filePath(QStringLiteral("broken.pt.model.json")), card);
    CHECK(savedBroken);
    if (!savedBroken)
    {
        return;
    }

    std::string error;
    const auto loaded = ai::ModelCard::loadForModel(toFsPath(modelFile),
                                                    ai::ModelCard::Backend::Torch,
                                                    error);
    CHECK_FALSE(loaded.has_value());
    CHECK_FALSE(error.empty());
    CHECK(error.find("features.normalize.mean") != std::string::npos);
}

TEST_CASE("ModelCard rejects ONNX framework mismatch")
{
    QTemporaryDir dir;
    CHECK(dir.isValid());
    if (!dir.isValid())
    {
        return;
    }

    const QString modelFile = dir.filePath(QStringLiteral("sample.onnx"));
    QFile onnxFile(modelFile);
    const bool openedOnnx = onnxFile.open(QIODevice::WriteOnly);
    CHECK(openedOnnx);
    if (!openedOnnx)
    {
        return;
    }
    onnxFile.close();

    QJsonObject card = makeValidCard(QStringLiteral("onnx"),
                                     QStringLiteral("TensorFlow"),
                                     {QStringLiteral("2.14.0")});
    const bool savedOnnxCard = writeJson(dir.filePath(QStringLiteral("sample.onnx.model.json")), card);
    CHECK(savedOnnxCard);
    if (!savedOnnxCard)
    {
        return;
    }

    std::string error;
    const auto loaded = ai::ModelCard::loadForModel(toFsPath(modelFile),
                                                    ai::ModelCard::Backend::Onnx,
                                                    error);
    CHECK_FALSE(loaded.has_value());
    CHECK_FALSE(error.empty());
    CHECK(error.find("framework") != std::string::npos);
}

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
