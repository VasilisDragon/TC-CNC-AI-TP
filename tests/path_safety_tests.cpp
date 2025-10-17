#include "ai/IPathAI.h"
#include "render/Model.h"
#include "tp/Machine.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"

#include <QtGui/QVector3D>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

namespace
{

double planeZ(double x, double y)
{
    return 0.08 * x + 0.05 * y;
}

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
            const double z = planeZ(x, y);
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

class FixedAI : public ai::IPathAI
{
public:
    explicit FixedAI(ai::StrategyDecision decision)
        : m_decision(decision)
    {
    }

    ai::StrategyDecision predict(const render::Model&, const tp::UserParams&) override
    {
        return m_decision;
    }

private:
    ai::StrategyDecision m_decision{};
};

struct Segment
{
    glm::dvec3 a;
    glm::dvec3 b;
    double minX;
    double maxX;
    double minY;
    double maxY;
};

bool boxesOverlap(const Segment& lhs, const Segment& rhs, double tolerance)
{
    const bool overlapX = lhs.maxX >= rhs.minX - tolerance && lhs.minX <= rhs.maxX + tolerance;
    const bool overlapY = lhs.maxY >= rhs.minY - tolerance && lhs.minY <= rhs.maxY + tolerance;
    return overlapX && overlapY;
}

bool sharesEndpoint(const Segment& lhs, const Segment& rhs, double tolerance)
{
    const auto close = [tolerance](const glm::dvec3& p0, const glm::dvec3& p1) {
        return glm::length(p0 - p1) <= tolerance;
    };

    return close(lhs.a, rhs.a) || close(lhs.a, rhs.b) || close(lhs.b, rhs.a) || close(lhs.b, rhs.b);
}

glm::dvec3 toDVec3(const glm::vec3& v)
{
    return {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)};
}

} // namespace

int main()
{
    constexpr double kWidth = 50.0;
    constexpr double kDepth = 40.0;
    constexpr int kDivisions = 6;

    render::Model model = buildPlaneModel(kWidth, kDepth, kDivisions);
    assert(model.isValid());

    tp::UserParams params;
    params.enableRoughPass = false;
    params.stockAllowance_mm = 0.0;
    params.cutterType = tp::UserParams::CutterType::BallNose;
    params.stepOver = 2.5;
    params.maxDepthPerPass = 1.2;
    params.stock.topZ_mm = planeZ(kWidth, kDepth) + 5.0;
    params.machine = tp::makeDefaultMachine();

    ai::StrategyDecision decision;
    decision.strat = ai::StrategyDecision::Strategy::Raster;
    decision.stepOverMM = params.stepOver;
    decision.roughPass = false;
    decision.finishPass = true;

    FixedAI ai(decision);

    tp::ToolpathGenerator generator;
    std::atomic<bool> cancel{false};
    tp::Toolpath toolpath = generator.generate(model, params, ai, cancel);
    assert(!toolpath.empty());

    const double cutterOffset = std::max(0.0, params.toolDiameter * 0.5);
    const double tolerance = 1e-3;
    const double boxTolerance = 1e-4;
    bool sawCutSegment = false;

    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (poly.motion != tp::MotionType::Cut || poly.pts.size() < 2)
        {
            continue;
        }

        std::vector<Segment> segments;
        segments.reserve(poly.pts.size() - 1);

        for (std::size_t i = 0; i + 1 < poly.pts.size(); ++i)
        {
            const glm::dvec3 start = toDVec3(poly.pts[i].p);
            const glm::dvec3 end = toDVec3(poly.pts[i + 1].p);

            const std::array<double, 5> samples{0.0, 0.25, 0.5, 0.75, 1.0};
            for (double t : samples)
            {
                const glm::dvec3 point = start + (end - start) * t;
                const double surface = planeZ(point.x, point.y);
                const double minZ = surface + cutterOffset - tolerance;
                assert(point.z + 1e-6 >= minZ);
            }

            Segment seg;
            seg.a = start;
            seg.b = end;
            seg.minX = std::min(start.x, end.x);
            seg.maxX = std::max(start.x, end.x);
            seg.minY = std::min(start.y, end.y);
            seg.maxY = std::max(start.y, end.y);
            segments.push_back(seg);
            sawCutSegment = true;
        }

        for (std::size_t i = 0; i < segments.size(); ++i)
        {
            for (std::size_t j = i + 1; j < segments.size(); ++j)
            {
                if (j == i + 1)
                {
                    continue;
                }
                if (sharesEndpoint(segments[i], segments[j], boxTolerance))
                {
                    continue;
                }
                assert(!boxesOverlap(segments[i], segments[j], boxTolerance));
            }
        }
    }

    assert(sawCutSegment);
    return 0;
}
