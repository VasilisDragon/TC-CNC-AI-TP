#include "ai/IPathAI.h"
#include "render/Model.h"
#include "tp/ToolpathGenerator.h"

#include <QtGui/QVector3D>

#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace
{

render::Model buildSlope(double width, double depth, int divisions, double slopeX, double slopeY)
{
    render::Model model;
    const int samples = divisions + 1;

    std::vector<render::Vertex> vertices(samples * samples);
    const double stepX = width / static_cast<double>(divisions);
    const double stepY = depth / static_cast<double>(divisions);

    for (int row = 0; row < samples; ++row)
    {
        for (int col = 0; col < samples; ++col)
        {
            const double x = static_cast<double>(col) * stepX;
            const double y = static_cast<double>(row) * stepY;
            const double z = slopeX * x + slopeY * y;
            render::Vertex vertex;
            vertex.position = QVector3D(static_cast<float>(x),
                                        static_cast<float>(y),
                                        static_cast<float>(z));
            vertex.normal = QVector3D(0.0f, 0.0f, 1.0f);
            vertices[row * samples + col] = vertex;
        }
    }

    std::vector<render::Model::Index> indices;
    indices.reserve(divisions * divisions * 6);
    for (int row = 0; row < divisions; ++row)
    {
        for (int col = 0; col < divisions; ++col)
        {
            const int base = row * samples + col;
            indices.push_back(static_cast<render::Model::Index>(base));
            indices.push_back(static_cast<render::Model::Index>(base + 1));
            indices.push_back(static_cast<render::Model::Index>(base + samples));
            indices.push_back(static_cast<render::Model::Index>(base + 1));
            indices.push_back(static_cast<render::Model::Index>(base + samples + 1));
            indices.push_back(static_cast<render::Model::Index>(base + samples));
        }
    }

    model.setMeshData(std::move(vertices), std::move(indices));
    return model;
}

class FixedWaterlineAI : public ai::IPathAI
{
public:
    FixedWaterlineAI()
    {
        ai::StrategyStep step;
        step.type = ai::StrategyStep::Type::Waterline;
        step.stepover = 0.0;
        step.stepdown = 0.0;
        step.finish_pass = true;
        m_decision.steps.push_back(step);
    }

    ai::StrategyDecision predict(const render::Model&, const tp::UserParams&) override { return m_decision; }

private:
    ai::StrategyDecision m_decision{};
};

} // namespace

int main()
{
    constexpr double kWidth = 60.0;
    constexpr double kDepth = 40.0;
    constexpr int kDivisions = 12;
    constexpr double kSlopeX = 0.03;
    constexpr double kSlopeY = 0.015;

    render::Model model = buildSlope(kWidth, kDepth, kDivisions, kSlopeX, kSlopeY);
    assert(model.isValid());

    tp::UserParams params;
    params.toolDiameter = 8.0;
    params.stepOver = 2.0;
    params.maxDepthPerPass = 1.2;
    params.enableRoughPass = false;
    params.enableFinishPass = true;
    params.leaveStock_mm = 0.15;
    params.stockAllowance_mm = params.leaveStock_mm;
    params.machine = tp::makeDefaultMachine();
    params.machine.safeZ_mm = 35.0;
    params.stock = tp::makeDefaultStock();
    params.stock.topZ_mm = kSlopeX * kWidth + kSlopeY * kDepth + 5.0;

    FixedWaterlineAI ai;
    tp::ToolpathGenerator generator;
    std::atomic<bool> cancel{false};
    tp::Toolpath toolpath = generator.generate(model, params, ai, cancel);
    assert(!toolpath.empty());

    double minClearance = std::numeric_limits<double>::infinity();
    bool sawSegment = false;
    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (poly.motion != tp::MotionType::Cut || poly.pts.size() < 2)
        {
            continue;
        }

        for (const tp::Vertex& vertex : poly.pts)
        {
            const double surfaceZ = kSlopeX * static_cast<double>(vertex.p.x)
                                    + kSlopeY * static_cast<double>(vertex.p.y);
            const double clearance = static_cast<double>(vertex.p.z) - surfaceZ;
            minClearance = std::min(minClearance, clearance);
        }
        sawSegment = true;
    }

    assert(sawSegment);
    const double tolerance = 5e-3;
    assert(minClearance > -tolerance);
    assert(!std::isfinite(minClearance) || minClearance + tolerance >= params.leaveStock_mm);
    return 0;
}
