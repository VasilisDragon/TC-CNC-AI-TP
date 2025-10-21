#include "ai/IPathAI.h"
#include "common/Units.h"
#include "io/ModelImporter.h"
#include "render/Model.h"
#include "tp/GRBLPost.h"
#include "tp/Machine.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"

#include <QtGui/QVector3D>

#include <atomic>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

class FixedAI : public ai::IPathAI
{
public:
    FixedAI() = default;

    void setDecision(ai::StrategyDecision decision)
    {
        m_decision = decision;
    }

    ai::StrategyDecision predict(const render::Model&, const tp::UserParams&) override
    {
        return m_decision;
    }

private:
    ai::StrategyDecision m_decision{};
};

std::vector<std::string> splitLines(const std::string& text)
{
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        lines.push_back(line);
    }
    return lines;
}

void runScenario(tp::ToolpathGenerator& generator,
                 tp::UserParams& params,
                 FixedAI& ai,
                 const render::Model& model,
                 std::atomic<bool>& cancel,
                 const ai::StrategyDecision& decision)
{
    ai.setDecision(decision);
    tp::Toolpath toolpath = generator.generate(model, params, ai, cancel);
    assert(!toolpath.empty());

    bool hasCut = false;
    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (poly.motion == tp::MotionType::Cut && poly.pts.size() >= 2)
        {
            hasCut = true;
            break;
        }
    }
    assert(hasCut);

    tp::GRBLPost post;
    const std::string gcode = post.generate(toolpath, common::Unit::Millimeters, params);
    assert(!gcode.empty());

    const std::vector<std::string> lines = splitLines(gcode);
    assert(!lines.empty());
    assert(!lines.front().empty());
    assert(lines.front().find("(AIToolpathGenerator - GRBL Post)") == 0);
    assert(!lines.back().empty());
    assert(lines.back() == "M2");

    const std::filesystem::path tempDir = std::filesystem::temp_directory_path();
    const std::string suffix =
        (!decision.steps.empty() && decision.steps.front().type == ai::StrategyStep::Type::Waterline)
            ? "waterline"
            : "raster";
    const std::filesystem::path tempFile = tempDir / ("cnctc_headless_" + suffix + ".gcode");

    {
        std::ofstream out(tempFile, std::ios::binary);
        assert(out.good());
        out << gcode;
    }

    assert(std::filesystem::exists(tempFile));
    assert(std::filesystem::file_size(tempFile) > 0);
    std::filesystem::remove(tempFile);
}

} // namespace

int main()
{
#ifndef CNCTC_SOURCE_DIR
#error "CNCTC_SOURCE_DIR must be defined"
#endif

    const std::filesystem::path samplePath =
        std::filesystem::path(CNCTC_SOURCE_DIR) / "samples" / "sample_part.stl";

    io::ModelImporter importer;
    render::Model model;
    std::string error;
    const bool loaded = importer.load(samplePath, model, error);
    assert(loaded);
    assert(error.empty());
    assert(model.isValid());

    tp::UserParams params;
    params.enableRoughPass = false;
    params.stockAllowance_mm = 0.0;
    params.leaveStock_mm = 0.0;
    params.maxDepthPerPass = 1.0;
    params.stepOver = 2.0;
    params.machine = tp::makeDefaultMachine();
    params.stock = tp::makeDefaultStock();

    const auto bounds = model.bounds();
    params.stock.topZ_mm = static_cast<double>(bounds.max.z()) + 2.0;

    tp::ToolpathGenerator generator;
    FixedAI ai;
    std::atomic<bool> cancel{false};

    ai::StrategyDecision raster;
    ai::StrategyStep rasterStep;
    rasterStep.type = ai::StrategyStep::Type::Raster;
    rasterStep.stepover = params.stepOver;
    rasterStep.stepdown = params.maxDepthPerPass;
    rasterStep.finish_pass = true;
    raster.steps.push_back(rasterStep);
    runScenario(generator, params, ai, model, cancel, raster);

    ai::StrategyDecision waterline;
    ai::StrategyStep waterlineStep;
    waterlineStep.type = ai::StrategyStep::Type::Waterline;
    waterlineStep.stepover = params.stepOver;
    waterlineStep.stepdown = params.maxDepthPerPass;
    waterlineStep.finish_pass = true;
    waterline.steps.push_back(waterlineStep);
    runScenario(generator, params, ai, model, cancel, waterline);

    return 0;
}
