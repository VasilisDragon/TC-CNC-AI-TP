#include "ai/IPathAI.h"
#include "render/Model.h"
#include "tp/ToolpathGenerator.h"

#include <QtGui/QVector3D>

#include <atomic>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

namespace
{

render::Model buildPocket(double width, double depth, double pocketDepth, int divisions)
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
            const bool insidePocket = (x > width * 0.2 && x < width * 0.8 && y > depth * 0.2 && y < depth * 0.8);
            const double z = insidePocket ? -pocketDepth : 0.0;

            render::Vertex vertex;
            vertex.position = QVector3D(static_cast<float>(x),
                                        static_cast<float>(y),
                                        static_cast<float>(z));
            vertex.normal = QVector3D(0.0f, 0.0f, insidePocket ? 1.0f : -1.0f);
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
        m_decision.strat = ai::StrategyDecision::Strategy::Waterline;
        m_decision.roughPass = false;
        m_decision.finishPass = true;
    }

    ai::StrategyDecision predict(const render::Model&, const tp::UserParams&) override
    {
        return m_decision;
    }

private:
    ai::StrategyDecision m_decision{};
};

glm::dvec3 toDVec3(const glm::vec3& v)
{
    return {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)};
}

double horizontalDistance(const glm::dvec3& a, const glm::dvec3& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

const tp::Polyline* findFirstCutPolyline(const tp::Toolpath& toolpath)
{
    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (poly.motion == tp::MotionType::Cut && poly.pts.size() >= 2)
        {
            return &poly;
        }
    }
    return nullptr;
}

const tp::Polyline* findFirstPlanarCut(const tp::Toolpath& toolpath)
{
    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (poly.motion != tp::MotionType::Cut || poly.pts.size() < 3)
        {
            continue;
        }

        double maxHorizontal = 0.0;
        double maxVertical = 0.0;
        for (std::size_t i = 0; i + 1 < poly.pts.size(); ++i)
        {
            const glm::dvec3 a = toDVec3(poly.pts[i].p);
            const glm::dvec3 b = toDVec3(poly.pts[i + 1].p);
            maxHorizontal = std::max(maxHorizontal, horizontalDistance(a, b));
            maxVertical = std::max(maxVertical, std::abs(a.z - b.z));
        }

        if (maxHorizontal > 1e-3 && maxVertical < 5e-4)
        {
            return &poly;
        }
    }
    return nullptr;
}

double signedAreaXY(const tp::Polyline& poly)
{
    if (poly.pts.size() < 3)
    {
        return 0.0;
    }

    double area = 0.0;
    for (std::size_t i = 0; i < poly.pts.size(); ++i)
    {
        const glm::dvec3 a = toDVec3(poly.pts[i].p);
        const glm::dvec3 b = toDVec3(poly.pts[(i + 1) % poly.pts.size()].p);
        area += a.x * b.y - b.x * a.y;
    }
    return area * 0.5;
}

} // namespace

int main()
{
    render::Model model = buildPocket(60.0, 60.0, 6.0, 20);
    assert(model.isValid());

    tp::UserParams baseParams;
    baseParams.enableRoughPass = false;
    baseParams.stockAllowance_mm = 0.0;
    baseParams.leaveStock_mm = 0.0;
    baseParams.stepOver = 3.0;
    baseParams.maxDepthPerPass = 1.5;
    baseParams.enableRamp = true;
    baseParams.enableHelical = true;
    baseParams.rampRadius = 5.0;
    baseParams.rampAngleDeg = 4.0;
    baseParams.leadInLength = 0.0;
    baseParams.leadOutLength = 0.0;
    baseParams.machine = tp::makeDefaultMachine();
    baseParams.machine.safeZ_mm = 28.0;
    baseParams.machine.clearanceZ_mm = 18.0;
    baseParams.stock = tp::makeDefaultStock();
    baseParams.stock.topZ_mm = 8.0;

    FixedWaterlineAI ai;
    tp::ToolpathGenerator generator;
    std::atomic<bool> cancel{false};

    tp::UserParams climbParams = baseParams;
    climbParams.cutDirection = tp::UserParams::CutDirection::Climb;
    tp::Toolpath climbPath = generator.generate(model, climbParams, ai, cancel);
    assert(!climbPath.empty());

    tp::UserParams conventionalParams = baseParams;
    conventionalParams.cutDirection = tp::UserParams::CutDirection::Conventional;
    tp::Toolpath conventionalPath = generator.generate(model, conventionalParams, ai, cancel);
    assert(!conventionalPath.empty());

    const tp::Polyline* helix = findFirstCutPolyline(climbPath);
    assert(helix != nullptr);
    assert(helix->pts.size() >= 6);
    double maxHorizontal = 0.0;
    bool nonIncreasingZ = true;
    for (std::size_t i = 0; i + 1 < helix->pts.size(); ++i)
    {
        const glm::dvec3 a = toDVec3(helix->pts[i].p);
        const glm::dvec3 b = toDVec3(helix->pts[i + 1].p);
        maxHorizontal = std::max(maxHorizontal, horizontalDistance(a, b));
        if (b.z > a.z + 5e-4)
        {
            nonIncreasingZ = false;
        }
    }
    assert(maxHorizontal > 1e-3);
    assert(nonIncreasingZ);

    const tp::Polyline* climbLoop = findFirstPlanarCut(climbPath);
    const tp::Polyline* conventionalLoop = findFirstPlanarCut(conventionalPath);
    assert(climbLoop != nullptr);
    assert(conventionalLoop != nullptr);
    const double areaClimb = signedAreaXY(*climbLoop);
    const double areaConventional = signedAreaXY(*conventionalLoop);
    assert(std::abs(areaClimb) > 1e-2);
    assert(areaClimb * areaConventional < 0.0);

    return 0;
}
