#include "tp/ocl/OclAdapter.h"

#include "render/Model.h"
#include "tp/ToolpathGenerator.h"
#include "tp/TriangleGrid.h"
#include "tp/waterline/ZSlicer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <vector>
#include <sstream>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

namespace tp
{

namespace
{

float clampStepOver(double stepOverMm)
{
    constexpr double kMinStep = 0.1;
    return static_cast<float>(std::max(stepOverMm, kMinStep));
}

std::optional<double> sampleHeight(const TriangleGrid& grid,
                                   std::vector<std::uint32_t>& scratch,
                                   double x,
                                   double y,
                                   double eps)
{
    auto gather = [&](int radius) {
        grid.gatherCandidatesXY(x, y, radius, scratch);
        return !scratch.empty();
    };

    if (!gather(1))
    {
        if (!gather(2))
        {
            gather(3);
        }
    }

    double maxZ = -std::numeric_limits<double>::infinity();
    bool found = false;

    for (std::uint32_t index : scratch)
    {
        if (index >= grid.triangleCount())
        {
            continue;
        }

        const TriangleGrid::Triangle& tri = grid.triangle(index);
        if (!tri.validNormalZ || !tri.validBarycentric)
        {
            continue;
        }

        const double dx = x - tri.centroid.x;
        const double dy = y - tri.centroid.y;
        if ((dx * dx + dy * dy) > tri.boundingRadiusSq + eps)
        {
            continue;
        }

        if (x < tri.bboxMin.x - eps || x > tri.bboxMax.x + eps || y < tri.bboxMin.y - eps || y > tri.bboxMax.y + eps)
        {
            continue;
        }

        const double zCandidate = tri.planeHeightAt(x, y);
        if (!std::isfinite(zCandidate))
        {
            continue;
        }

        if (zCandidate < tri.minZ - eps || zCandidate > tri.maxZ + eps)
        {
            continue;
        }

        const glm::dvec3 point{x, y, zCandidate};
        if (!tri.barycentricContains(point, eps))
        {
            continue;
        }

        if (zCandidate > maxZ)
        {
            maxZ = zCandidate;
            found = true;
        }
    }

    scratch.clear();

    if (!found)
    {
        return std::nullopt;
    }
    return maxZ;
}

Toolpath makeEmptyToolpath(const UserParams& params)
{
    Toolpath path;
    path.feed = params.feed;
    path.spindle = params.spindle;
    return path;
}

} // namespace

bool OclAdapter::waterline(const render::Model& model,
                           const UserParams& params,
                           const Cutter& cutter,
                           Toolpath& out,
                           std::string& err)
{
    out = makeEmptyToolpath(params);
    err.clear();

    if (!model.isValid())
    {
        err = "Model is invalid.";
        return false;
    }

    const auto bounds = model.bounds();
    const double minZ = static_cast<double>(bounds.min.z());
    const double maxZ = static_cast<double>(bounds.max.z());

    if (maxZ - minZ <= 1e-4)
    {
        err = "Model bounds are too small for waterline generation.";
        return false;
    }

    const double stepDown = std::max(params.maxDepthPerPass, 0.1);
    const double toolRadius = (cutter.type == Cutter::Type::FlatEndmill) ? cutter.diameter * 0.5 : 0.0;

    waterline::ZSlicer slicer(model, 1e-4);

    std::size_t loopCount = 0;
    int levelCount = 0;

    const auto startTime = std::chrono::steady_clock::now();

    for (double planeZ = maxZ; planeZ >= minZ - 1e-6; planeZ -= stepDown)
    {
        const auto loops = slicer.slice(planeZ, toolRadius, cutter.type == Cutter::Type::FlatEndmill);
        if (loops.empty())
        {
            continue;
        }

        ++levelCount;
        for (const auto& loop : loops)
        {
            if (loop.size() < 3)
            {
                continue;
            }

            Polyline poly;
            poly.motion = MotionType::Cut;
            poly.pts.reserve(loop.size());
            for (const auto& pt : loop)
            {
                poly.pts.push_back({glm::vec3(static_cast<float>(pt.x),
                                              static_cast<float>(pt.y),
                                              static_cast<float>(pt.z))});
            }
            out.passes.push_back(std::move(poly));
            ++loopCount;
        }
    }

    if (out.passes.empty())
    {
        err = "Waterline generation produced no contours.";
        return false;
    }

    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime).count();
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(2);
    oss << "OCL waterline generated " << loopCount << " loops across " << levelCount
        << " levels in " << elapsed << " ms";
    err = oss.str();

    return true;
}

bool OclAdapter::rasterDropCutter(const render::Model& model,
                                  const UserParams& params,
                                  const Cutter& cutter,
                                  double rasterAngleDeg,
                                  Toolpath& out,
                                  std::string& err)
{
    out = makeEmptyToolpath(params);
    err.clear();

    if (!model.isValid())
    {
        err = "Model is invalid.";
        return false;
    }

    const auto& vertices = model.vertices();
    const auto& indices = model.indices();
    if (vertices.empty() || indices.size() < 3)
    {
        err = "Model contains no triangles.";
        return false;
    }

    const auto bounds = model.bounds();
    const double minX = static_cast<double>(bounds.min.x());
    const double maxX = static_cast<double>(bounds.max.x());
    const double minY = static_cast<double>(bounds.min.y());
    const double maxY = static_cast<double>(bounds.max.y());
    const double minZ = static_cast<double>(bounds.min.z());
    const double maxZ = static_cast<double>(bounds.max.z());

    if (std::abs(maxX - minX) < 1e-4 || std::abs(maxY - minY) < 1e-4)
    {
        err = "Model bounds are too small for raster generation.";
        return false;
    }

    const double rawStep = params.stepOver > 0.0 ? params.stepOver : params.toolDiameter * 0.5;
    const double step = static_cast<double>(clampStepOver(rawStep));
    const double angleRad = rasterAngleDeg * std::numbers::pi / 180.0;
    const double cosA = std::cos(angleRad);
    const double sinA = std::sin(angleRad);

    auto rotate2D = [cosA, sinA](double x, double y) -> std::pair<double, double> {
        return {x * cosA - y * sinA, x * sinA + y * cosA};
    };

    auto unrotate2D = [cosA, sinA](double xr, double yr) -> std::pair<double, double> {
        return {xr * cosA + yr * sinA, -xr * sinA + yr * cosA};
    };

    std::array<std::pair<double, double>, 4> corners = {
        std::make_pair(minX, minY),
        std::make_pair(maxX, minY),
        std::make_pair(maxX, maxY),
        std::make_pair(minX, maxY)
    };

    double minXRot = std::numeric_limits<double>::max();
    double maxXRot = std::numeric_limits<double>::lowest();
    double minYRot = std::numeric_limits<double>::max();
    double maxYRot = std::numeric_limits<double>::lowest();

    for (const auto& corner : corners)
    {
        const auto rotated = rotate2D(corner.first, corner.second);
        minXRot = std::min(minXRot, rotated.first);
        maxXRot = std::max(maxXRot, rotated.first);
        minYRot = std::min(minYRot, rotated.second);
        maxYRot = std::max(maxYRot, rotated.second);
    }

    const int rows = std::max(1, static_cast<int>(std::ceil((maxYRot - minYRot) / std::max(step, 1e-5))));
    const double tipOffset = cutter.type == Cutter::Type::BallNose ? cutter.diameter * 0.5 : 0.0;

    constexpr double kEps = 1e-5;

    TriangleGrid grid(model, std::max(0.5, step));
    if (grid.triangleCount() == 0)
    {
        err = "Raster drop-cutter produced no valid triangles.";
        return false;
    }

    std::vector<std::uint32_t> candidateScratch;
    candidateScratch.reserve(128);

    for (int row = 0; row <= rows; ++row)
    {
        const double yRot = std::min(minYRot + static_cast<double>(row) * step, maxYRot);
        const bool leftToRight = (row % 2) == 0;

        const double startXRot = leftToRight ? minXRot : maxXRot;
        const double endXRot = leftToRight ? maxXRot : minXRot;

        const auto startCutXY = unrotate2D(startXRot, yRot);
        const auto endCutXY = unrotate2D(endXRot, yRot);

        const auto startHeight = sampleHeight(grid, candidateScratch, startCutXY.first, startCutXY.second, kEps);
        const auto endHeight = sampleHeight(grid, candidateScratch, endCutXY.first, endCutXY.second, kEps);

        if (!startHeight && !endHeight)
        {
            continue;
        }

        double startZ = startHeight.value_or(endHeight.value_or(minZ));
        double endZ = endHeight.value_or(startZ);

        startZ = std::max(startZ - tipOffset, minZ);
        endZ = std::max(endZ - tipOffset, minZ);

        const glm::vec3 startCut{static_cast<float>(startCutXY.first),
                                 static_cast<float>(startCutXY.second),
                                 static_cast<float>(startZ)};
        const glm::vec3 endCut{static_cast<float>(endCutXY.first),
                               static_cast<float>(endCutXY.second),
                               static_cast<float>(endZ)};

        Polyline cut;
        cut.motion = MotionType::Cut;
        cut.pts.push_back({startCut});
        cut.pts.push_back({endCut});
        out.passes.push_back(std::move(cut));

    }

    if (out.passes.empty())
    {
        err = "Raster drop-cutter produced no passes.";
        return false;
    }

    return true;
}

} // namespace tp
