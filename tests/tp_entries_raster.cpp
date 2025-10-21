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

render::Model buildPlaneModel(double width, double depth, int divisions)
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
            const double z = 0.05 * x - 0.03 * y;

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

class FixedRasterAI : public ai::IPathAI
{
public:
    explicit FixedRasterAI(ai::StrategyDecision decision)
        : m_decision(std::move(decision))
    {
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

bool isPlanarCut(const tp::Polyline& poly)
{
    if (poly.motion != tp::MotionType::Cut || poly.pts.size() < 2)
    {
        return false;
    }

    double maxHorizontal = 0.0;
    bool sawDescending = false;
    for (std::size_t i = 0; i + 1 < poly.pts.size(); ++i)
    {
        const glm::dvec3 a = toDVec3(poly.pts[i].p);
        const glm::dvec3 b = toDVec3(poly.pts[i + 1].p);
        maxHorizontal = std::max(maxHorizontal, horizontalDistance(a, b));
        if (a.z > b.z + 1e-4)
        {
            sawDescending = true;
        }
    }

    const glm::dvec3 first = toDVec3(poly.pts.front().p);
    const glm::dvec3 last = toDVec3(poly.pts.back().p);
    const double dz = std::abs(first.z - last.z);
    return dz < 1e-3 && maxHorizontal > 1e-3 && !sawDescending;
}

const tp::Polyline* findFirstPlanarCut(const tp::Toolpath& toolpath)
{
    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (isPlanarCut(poly))
        {
            return &poly;
        }
    }
    return nullptr;
}

const tp::Polyline* findPlanarCutByY(const tp::Toolpath& toolpath, double targetY)
{
    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (!isPlanarCut(poly))
        {
            continue;
        }
        double sumY = 0.0;
        for (const tp::Vertex& vertex : poly.pts)
        {
            sumY += static_cast<double>(vertex.p.y);
        }
        const double avgY = sumY / static_cast<double>(poly.pts.size());
        if (std::abs(avgY - targetY) < 1e-3)
        {
            return &poly;
        }
    }
    return nullptr;
}

} // namespace

int main()
{
    render::Model model = buildPlaneModel(60.0, 40.0, 8);
    assert(model.isValid());

    tp::UserParams baseParams;
    baseParams.enableRoughPass = false;
    baseParams.stockAllowance_mm = 0.0;
    baseParams.leaveStock_mm = 0.0;
    baseParams.stepOver = 4.0;
    baseParams.maxDepthPerPass = 1.5;
    baseParams.enableRamp = true;
    baseParams.enableHelical = false;
    baseParams.rampAngleDeg = 5.0;
    baseParams.leadInLength = 4.0;
    baseParams.leadOutLength = 4.0;
    baseParams.machine = tp::makeDefaultMachine();
    baseParams.machine.safeZ_mm = 30.0;
    baseParams.machine.clearanceZ_mm = 18.0;
    baseParams.stock = tp::makeDefaultStock();
    baseParams.stock.topZ_mm = 12.0;

    ai::StrategyDecision decision;
    ai::StrategyStep step;
    step.type = ai::StrategyStep::Type::Raster;
    step.stepover = baseParams.stepOver;
    step.stepdown = baseParams.maxDepthPerPass;
    step.finish_pass = true;
    decision.steps.push_back(step);

    FixedRasterAI ai(decision);
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

    const tp::Polyline* rampPoly = findFirstCutPolyline(climbPath);
    assert(rampPoly != nullptr);
    bool sawHorizontal = false;
    bool sawDescending = false;
    for (std::size_t i = 0; i + 1 < rampPoly->pts.size(); ++i)
    {
        const glm::dvec3 a = toDVec3(rampPoly->pts[i].p);
        const glm::dvec3 b = toDVec3(rampPoly->pts[i + 1].p);
        if (horizontalDistance(a, b) > 1e-3)
        {
            sawHorizontal = true;
        }
        if (a.z > b.z + 1e-4)
        {
            sawDescending = true;
        }
    }
    assert(sawHorizontal);
    assert(sawDescending);

    const tp::Polyline* climbCut = findFirstPlanarCut(climbPath);
    assert(climbCut != nullptr);
    const double targetY = toDVec3(climbCut->pts.front().p).y;
    const tp::Polyline* conventionalCut = findPlanarCutByY(conventionalPath, targetY);
    assert(conventionalCut != nullptr);
    assert(climbCut->pts.size() >= 4);
    assert(climbCut->pts.size() == conventionalCut->pts.size());

    const glm::dvec3 climbFront = toDVec3(climbCut->pts.front().p);
    const glm::dvec3 climbBack = toDVec3(climbCut->pts.back().p);
    const glm::dvec3 convFront = toDVec3(conventionalCut->pts.front().p);
    const glm::dvec3 convBack = toDVec3(conventionalCut->pts.back().p);

    const glm::dvec2 climbVec{climbBack.x - climbFront.x, climbBack.y - climbFront.y};
    const glm::dvec2 convVec{convBack.x - convFront.x, convBack.y - convFront.y};
    const double climbLen = glm::length(climbVec);
    const double convLen = glm::length(convVec);
    assert(climbLen > 1e-3);
    assert(convLen > 1e-3);
    const double dot = (climbVec.x * convVec.x + climbVec.y * convVec.y) / (climbLen * convLen);
    assert(dot < -0.95);

    return 0;
}
