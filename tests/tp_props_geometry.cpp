#include "ai/IPathAI.h"
#include "render/Model.h"
#include "tp/GougeChecker.h"
#include "tp/Machine.h"
#include "tp/ToolpathGenerator.h"

#include <QtGui/QVector3D>

#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include "tp_props_helpers.h"

namespace
{

class FixedAI : public ai::IPathAI
{
public:
    explicit FixedAI(ai::StrategyDecision decision)
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

struct HeightfieldResult
{
    render::Model model;
    double minZ{0.0};
    double maxZ{0.0};
};

HeightfieldResult buildRandomHeightfield(std::mt19937& rng, double width, double depth, int divisions)
{
    HeightfieldResult result;

    const int samples = divisions + 1;
    std::vector<render::Vertex> vertices(samples * samples);
    double minZ = std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();

    const double baseHeight = tp_props::randomDouble(rng, 0.25, 1.25);
    const double slopeX = tp_props::randomDouble(rng, -0.02, 0.02);
    const double slopeY = tp_props::randomDouble(rng, -0.02, 0.02);
    const double amplitude = tp_props::randomDouble(rng, 0.05, 0.3);
    const double freqX = tp_props::randomDouble(rng, 0.1, 0.4);
    const double freqY = tp_props::randomDouble(rng, 0.1, 0.4);
    const double phaseX = tp_props::randomDouble(rng, 0.0, std::numbers::pi);
    const double phaseY = tp_props::randomDouble(rng, 0.0, std::numbers::pi);

    const double stepX = width / static_cast<double>(divisions);
    const double stepY = depth / static_cast<double>(divisions);

    for (int row = 0; row < samples; ++row)
    {
        for (int col = 0; col < samples; ++col)
        {
            const double x = static_cast<double>(col) * stepX;
            const double y = static_cast<double>(row) * stepY;
            const double z = baseHeight + slopeX * x + slopeY * y
                             + amplitude * std::sin(freqX * x + phaseX)
                             + amplitude * std::cos(freqY * y + phaseY);

            minZ = std::min(minZ, z);
            maxZ = std::max(maxZ, z);

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

    render::Model model;
    model.setMeshData(std::move(vertices), std::move(indices));

    result.model = std::move(model);
    result.minZ = minZ;
    result.maxZ = maxZ;
    return result;
}

bool isCutPolylineNearStock(const tp::Polyline& polyline, double stockTopMm, double tolerance)
{
    if (polyline.motion != tp::MotionType::Cut || polyline.pts.size() < 2)
    {
        return false;
    }

    const auto [minZ, maxZ] = tp_props::polylineZExtents(polyline);
    return maxZ <= stockTopMm + tolerance;
}

} // namespace

int main()
{
    constexpr int kIterations = 12;
    constexpr double kIntersectionTolerance = 1e-4;
    constexpr double kZTolerance = 1e-3;

    std::mt19937 rng(1337);

    for (int iteration = 0; iteration < kIterations; ++iteration)
    {
        const double width = tp_props::randomDouble(rng, 28.0, 48.0);
        const double depth = tp_props::randomDouble(rng, 28.0, 48.0);
        const int divisions = 12;

        HeightfieldResult surface = buildRandomHeightfield(rng, width, depth, divisions);
        assert(surface.model.isValid());

        tp::UserParams params;
        params.enableRoughPass = false;
        params.enableFinishPass = true;
        params.useHeightField = true;
        params.toolDiameter = tp_props::randomDouble(rng, 4.0, 10.0);
        params.stepOver = tp_props::randomDouble(rng, params.toolDiameter * 0.2, params.toolDiameter * 0.55);
        params.maxDepthPerPass = tp_props::randomDouble(rng, 0.35, 0.85);
        params.machine = tp::makeDefaultMachine();
        params.machine.clearanceZ_mm = surface.maxZ + tp_props::randomDouble(rng, 1.5, 2.8);
        params.machine.safeZ_mm = params.machine.clearanceZ_mm + tp_props::randomDouble(rng, 1.0, 2.5);
        params.stock = tp::makeDefaultStock();
        params.stock.topZ_mm = surface.maxZ + tp_props::randomDouble(rng, 0.2, 0.4);
        params.stock.originXYZ_mm = glm::dvec3{0.0};
        params.cutterType = (iteration % 2 == 0) ? tp::UserParams::CutterType::FlatEndmill
                                                 : tp::UserParams::CutterType::BallNose;

        const bool gougeEnabled = (iteration % 3 == 0);
        if (gougeEnabled)
        {
            params.leaveStock_mm = tp_props::randomDouble(rng, 0.1, 0.3);
        }
        else
        {
            params.leaveStock_mm = 0.0;
        }
        params.stockAllowance_mm = params.leaveStock_mm;

        ai::StrategyDecision decision;
        ai::StrategyStep step;
        step.type = (iteration % 2 == 0) ? ai::StrategyStep::Type::Raster : ai::StrategyStep::Type::Waterline;
        step.stepover = params.stepOver;
        step.stepdown = tp_props::randomDouble(rng, params.maxDepthPerPass * 0.5, params.maxDepthPerPass);
        step.angle_deg = tp_props::randomDouble(rng, 0.0, 180.0);
        step.finish_pass = true;
        decision.steps.push_back(step);

        FixedAI ai(decision);
        tp::ToolpathGenerator generator;
        std::atomic<bool> cancel{false};
        tp::Toolpath toolpath = generator.generate(surface.model, params, ai, cancel);
        assert(!toolpath.empty());

        for (const tp::Polyline& poly : toolpath.passes)
        {
            if (poly.pts.size() < 2)
            {
                continue;
            }
            const bool hasIntersection = tp_props::polylineHasSelfIntersections(poly, kIntersectionTolerance);
            assert(!hasIntersection);
        }

        const double clearanceZ = toolpath.machine.clearanceZ_mm;
        for (const tp::Polyline& poly : toolpath.passes)
        {
            if (poly.motion != tp::MotionType::Rapid)
            {
                continue;
            }

            for (const tp::Vertex& vertex : poly.pts)
            {
                const double z = static_cast<double>(vertex.p.z);
                assert(z + kZTolerance >= clearanceZ);
            }
        }

        double allowedStepdown = params.maxDepthPerPass;
        if (!toolpath.strategySteps.empty())
        {
            allowedStepdown = std::max(toolpath.strategySteps.front().stepdown, 0.05);
        }

        bool sawActualCut = false;
        double previousDepth = 0.0;

        for (const tp::Polyline& poly : toolpath.passes)
        {
            if (!isCutPolylineNearStock(poly, params.stock.topZ_mm, 1e-3))
            {
                continue;
            }

            const auto [minZ, maxZ] = tp_props::polylineZExtents(poly);
            if (!sawActualCut)
            {
                sawActualCut = true;
                previousDepth = minZ;
                continue;
            }

            if (minZ + kZTolerance < previousDepth)
            {
                const double drop = previousDepth - minZ;
                assert(drop <= allowedStepdown + kZTolerance);
                previousDepth = minZ;
            }
            else if (minZ > previousDepth + kZTolerance)
            {
                previousDepth = minZ;
            }
        }

        if (gougeEnabled)
        {
            tp::GougeChecker checker(surface.model);
            tp::GougeParams gougeParams;
            gougeParams.toolRadius = params.toolDiameter * 0.5;
            gougeParams.holderRadius = gougeParams.toolRadius + 5.0;
            gougeParams.leaveStock = params.leaveStock_mm;
            gougeParams.safetyZ = toolpath.machine.safeZ_mm;

            double minClearance = std::numeric_limits<double>::infinity();
            for (const tp::Polyline& poly : toolpath.passes)
            {
                if (!isCutPolylineNearStock(poly, params.stock.topZ_mm, 1e-3))
                {
                    continue;
                }

                std::vector<glm::vec3> path;
                path.reserve(poly.pts.size());
                for (const tp::Vertex& vertex : poly.pts)
                {
                    path.push_back(vertex.p);
                }

                const double clearance = checker.minClearanceAlong(path, gougeParams);
                if (std::isfinite(clearance))
                {
                    minClearance = std::min(minClearance, clearance);
                }
            }

            if (std::isfinite(minClearance))
            {
                assert(minClearance + 1e-4 >= 0.0);
            }
        }
    }

    return 0;
}
