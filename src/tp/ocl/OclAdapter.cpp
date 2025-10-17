#include "tp/ocl/OclAdapter.h"

#include "render/Model.h"
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

glm::vec3 toGlm(const QVector3D& v)
{
    return glm::vec3{v.x(), v.y(), v.z()};
}

std::optional<float> surfaceHeightAt(const std::vector<glm::vec3>& vertices,
                                     const std::vector<render::Model::Index>& indices,
                                     float x,
                                     float y)
{
    float maxZ = -std::numeric_limits<float>::infinity();
    bool found = false;

    const float eps = 1e-5f;
    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const glm::vec3& a = vertices[indices[i]];
        const glm::vec3& b = vertices[indices[i + 1]];
        const glm::vec3& c = vertices[indices[i + 2]];

        const glm::vec3 normal = glm::cross(b - a, c - a);
        if (std::abs(normal.z) < eps)
        {
            continue;
        }

        const float z = a.z - (normal.x * (x - a.x) + normal.y * (y - a.y)) / normal.z;
        const glm::vec3 p{x, y, z};

        const glm::vec3 v0 = b - a;
        const glm::vec3 v1 = c - a;
        const glm::vec3 v2 = p - a;

        const float d00 = glm::dot(v0, v0);
        const float d01 = glm::dot(v0, v1);
        const float d11 = glm::dot(v1, v1);
        const float d20 = glm::dot(v2, v0);
        const float d21 = glm::dot(v2, v1);
        const float denom = d00 * d11 - d01 * d01;
        if (std::abs(denom) < eps)
        {
            continue;
        }

        const float v = (d11 * d20 - d01 * d21) / denom;
        const float w = (d00 * d21 - d01 * d20) / denom;
        const float u = 1.0f - v - w;

        if (u >= -eps && v >= -eps && w >= -eps && u <= 1.0f + eps && v <= 1.0f + eps && w <= 1.0f + eps)
        {
            if (z > maxZ)
            {
                maxZ = z;
                found = true;
            }
        }
    }

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

    std::vector<glm::vec3> glmVertices;
    glmVertices.reserve(vertices.size());
    for (const auto& v : vertices)
    {
        glmVertices.push_back(toGlm(v.position));
    }

    const auto bounds = model.bounds();
    const float minX = bounds.min.x();
    const float maxX = bounds.max.x();
    const float minY = bounds.min.y();
    const float maxY = bounds.max.y();
    const float minZ = bounds.min.z();
    const float maxZ = bounds.max.z();

    if (std::abs(maxX - minX) < 1e-4f || std::abs(maxY - minY) < 1e-4f)
    {
        err = "Model bounds are too small for raster generation.";
        return false;
    }

    const double rawStep = params.stepOver > 0.0 ? params.stepOver : params.toolDiameter * 0.5;
    const float step = clampStepOver(rawStep);
    const double angleRad = rasterAngleDeg * std::numbers::pi / 180.0;
    const float cosA = static_cast<float>(std::cos(angleRad));
    const float sinA = static_cast<float>(std::sin(angleRad));

    auto rotate2D = [cosA, sinA](float x, float y) -> std::pair<float, float> {
        return {x * cosA - y * sinA, x * sinA + y * cosA};
    };

    auto unrotate2D = [cosA, sinA](float xr, float yr) -> std::pair<float, float> {
        return {xr * cosA + yr * sinA, -xr * sinA + yr * cosA};
    };

    std::array<std::pair<float, float>, 4> corners = {
        std::make_pair(minX, minY),
        std::make_pair(maxX, minY),
        std::make_pair(maxX, maxY),
        std::make_pair(minX, maxY)
    };

    float minXRot = std::numeric_limits<float>::max();
    float maxXRot = std::numeric_limits<float>::lowest();
    float minYRot = std::numeric_limits<float>::max();
    float maxYRot = std::numeric_limits<float>::lowest();

    for (const auto& corner : corners)
    {
        const auto rotated = rotate2D(corner.first, corner.second);
        minXRot = std::min(minXRot, rotated.first);
        maxXRot = std::max(maxXRot, rotated.first);
        minYRot = std::min(minYRot, rotated.second);
        maxYRot = std::max(maxYRot, rotated.second);
    }

    const int rows = std::max(1, static_cast<int>(std::ceil((maxYRot - minYRot) / step)));
    const float tipOffset = cutter.type == Cutter::Type::BallNose ? static_cast<float>(cutter.diameter * 0.5) : 0.0f;

    for (int row = 0; row <= rows; ++row)
    {
        const float yRot = std::min(minYRot + static_cast<float>(row) * step, maxYRot);
        const bool leftToRight = (row % 2) == 0;

        const float startXRot = leftToRight ? minXRot : maxXRot;
        const float endXRot = leftToRight ? maxXRot : minXRot;

        const auto startCutXY = unrotate2D(startXRot, yRot);
        const auto endCutXY = unrotate2D(endXRot, yRot);

        const auto startHeight = surfaceHeightAt(glmVertices, indices, startCutXY.first, startCutXY.second);
        const auto endHeight = surfaceHeightAt(glmVertices, indices, endCutXY.first, endCutXY.second);

        if (!startHeight && !endHeight)
        {
            continue;
        }

        float startZ = startHeight ? *startHeight : (endHeight ? *endHeight : minZ);
        float endZ = endHeight ? *endHeight : startZ;

        startZ = std::max(startZ - tipOffset, minZ);
        endZ = std::max(endZ - tipOffset, minZ);

        const glm::vec3 startCut{startCutXY.first, startCutXY.second, startZ};
        const glm::vec3 endCut{endCutXY.first, endCutXY.second, endZ};

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
