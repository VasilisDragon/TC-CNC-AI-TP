#include "tp/GougeChecker.h"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace
{

constexpr double kEpsilon = 1e-6;

glm::vec3 toVec3(const QVector3D& v)
{
    return {v.x(), v.y(), v.z()};
}

double pointTriangleDistanceSquared(const glm::vec3& point,
                                    const glm::vec3& a,
                                    const glm::vec3& b,
                                    const glm::vec3& c,
                                    glm::vec3& outClosest)
{
    const glm::vec3 ab = b - a;
    const glm::vec3 ac = c - a;
    const glm::vec3 ap = point - a;

    const double d1 = glm::dot(ab, ap);
    const double d2 = glm::dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        outClosest = a;
        return glm::dot(ap, ap);
    }

    const glm::vec3 bp = point - b;
    const double d3 = glm::dot(ab, bp);
    const double d4 = glm::dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3)
    {
        outClosest = b;
        return glm::dot(bp, bp);
    }

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        const double v = d1 / (d1 - d3);
        outClosest = a + static_cast<float>(v) * ab;
        const glm::vec3 diff = point - outClosest;
        return glm::dot(diff, diff);
    }

    const glm::vec3 cp = point - c;
    const double d5 = glm::dot(ab, cp);
    const double d6 = glm::dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6)
    {
        outClosest = c;
        return glm::dot(cp, cp);
    }

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        const double w = d2 / (d2 - d6);
        outClosest = a + static_cast<float>(w) * ac;
        const glm::vec3 diff = point - outClosest;
        return glm::dot(diff, diff);
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        outClosest = b + static_cast<float>(w) * (c - b);
        const glm::vec3 diff = point - outClosest;
        return glm::dot(diff, diff);
    }

    const glm::vec3 n = glm::cross(ab, ac);
    const double denom = glm::dot(n, n);
        if (denom <= kEpsilon)
        {
            outClosest = a;
            const glm::vec3 diff = point - a;
            return glm::dot(diff, diff);
    }

    const double v = glm::dot(glm::cross(ap, ac), n) / denom;
    const double w = glm::dot(glm::cross(ab, ap), n) / denom;
    const double u = 1.0 - v - w;
    outClosest = static_cast<float>(u) * a + static_cast<float>(v) * b + static_cast<float>(w) * c;
    const glm::vec3 diff = point - outClosest;
    return glm::dot(diff, diff);
}

int clampIndex(int value, int maxValue)
{
    if (value < 0)
    {
        return 0;
    }
    if (value >= maxValue)
    {
        return maxValue - 1;
    }
    return value;
}

} // namespace

namespace tp
{

GougeChecker::GougeChecker(const render::Model& model)
{
    const auto& vertices = model.vertices();
    const auto& indices = model.indices();
    if (vertices.empty() || indices.empty())
    {
        m_grid.resize(1);
        return;
    }

    m_triangles.reserve(indices.size() / 3);
    m_minBounds = toVec3(vertices.front().position);
    m_maxBounds = m_minBounds;

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const auto ia = static_cast<std::size_t>(indices[i]);
        const auto ib = static_cast<std::size_t>(indices[i + 1]);
        const auto ic = static_cast<std::size_t>(indices[i + 2]);
        if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size())
        {
            continue;
        }

        Triangle triangle;
        triangle.a = toVec3(vertices[ia].position);
        triangle.b = toVec3(vertices[ib].position);
        triangle.c = toVec3(vertices[ic].position);
        triangle.normal = glm::normalize(glm::cross(triangle.b - triangle.a, triangle.c - triangle.a));

        triangle.minBounds = glm::min(glm::min(triangle.a, triangle.b), triangle.c);
        triangle.maxBounds = glm::max(glm::max(triangle.a, triangle.b), triangle.c);

        m_minBounds = glm::min(m_minBounds, triangle.minBounds);
        m_maxBounds = glm::max(m_maxBounds, triangle.maxBounds);

        m_triangles.push_back(triangle);
    }

    if (m_triangles.empty())
    {
        m_grid.resize(1);
        return;
    }

    const double spanX = static_cast<double>(m_maxBounds.x - m_minBounds.x);
    const double spanY = static_cast<double>(m_maxBounds.y - m_minBounds.y);
    const std::size_t triCount = m_triangles.size();

    const double targetResolution = std::max(4.0, std::sqrt(static_cast<double>(triCount)));
    m_cellsX = std::max(1, static_cast<int>(std::round(targetResolution)));
    m_cellsY = m_cellsX;

    if (spanX < kEpsilon)
    {
        m_cellsX = 1;
    }
    if (spanY < kEpsilon)
    {
        m_cellsY = 1;
    }

    m_grid.assign(static_cast<std::size_t>(m_cellsX * m_cellsY), {});

    const double cellSizeX = (m_cellsX > 0 && spanX >= kEpsilon) ? spanX / static_cast<double>(m_cellsX) : 1.0;
    const double cellSizeY = (m_cellsY > 0 && spanY >= kEpsilon) ? spanY / static_cast<double>(m_cellsY) : 1.0;
    m_invCellSizeX = cellSizeX > kEpsilon ? 1.0 / cellSizeX : 0.0;
    m_invCellSizeY = cellSizeY > kEpsilon ? 1.0 / cellSizeY : 0.0;

    for (std::uint32_t index = 0; index < m_triangles.size(); ++index)
    {
        const Triangle& tri = m_triangles[index];
        const double minX = static_cast<double>(tri.minBounds.x);
        const double maxX = static_cast<double>(tri.maxBounds.x);
        const double minY = static_cast<double>(tri.minBounds.y);
        const double maxY = static_cast<double>(tri.maxBounds.y);

        const int cellMinX = clampIndex(static_cast<int>(std::floor((minX - m_minBounds.x) * m_invCellSizeX)), m_cellsX);
        const int cellMaxX = clampIndex(static_cast<int>(std::floor((maxX - m_minBounds.x) * m_invCellSizeX)), m_cellsX);
        const int cellMinY = clampIndex(static_cast<int>(std::floor((minY - m_minBounds.y) * m_invCellSizeY)), m_cellsY);
        const int cellMaxY = clampIndex(static_cast<int>(std::floor((maxY - m_minBounds.y) * m_invCellSizeY)), m_cellsY);

        for (int ix = cellMinX; ix <= cellMaxX; ++ix)
        {
            for (int iy = cellMinY; iy <= cellMaxY; ++iy)
            {
                const std::size_t cellIndex = static_cast<std::size_t>(iy * m_cellsX + ix);
                m_grid[cellIndex].push_back(index);
            }
        }
    }
}

std::vector<std::uint32_t> GougeChecker::gatherCandidates(const Vec3& point) const
{
    if (m_grid.size() <= 1 || m_cellsX <= 1 || m_cellsY <= 1)
    {
        std::vector<std::uint32_t> all(m_triangles.size());
        std::iota(all.begin(), all.end(), 0u);
        return all;
    }

    const double localX = (point.x - m_minBounds.x) * m_invCellSizeX;
    const double localY = (point.y - m_minBounds.y) * m_invCellSizeY;

    int baseX = clampIndex(static_cast<int>(std::floor(localX)), m_cellsX);
    int baseY = clampIndex(static_cast<int>(std::floor(localY)), m_cellsY);

    std::vector<std::uint32_t> candidates;
    candidates.reserve(64);

    for (int dx = -1; dx <= 1; ++dx)
    {
        const int cx = baseX + dx;
        if (cx < 0 || cx >= m_cellsX)
        {
            continue;
        }
        for (int dy = -1; dy <= 1; ++dy)
        {
            const int cy = baseY + dy;
            if (cy < 0 || cy >= m_cellsY)
            {
                continue;
            }
            const std::size_t cellIndex = static_cast<std::size_t>(cy * m_cellsX + cx);
            const auto& bucket = m_grid[cellIndex];
            candidates.insert(candidates.end(), bucket.begin(), bucket.end());
        }
    }

    if (candidates.empty())
    {
        candidates.resize(m_triangles.size());
        std::iota(candidates.begin(), candidates.end(), 0u);
        return candidates;
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

GougeChecker::ClosestHit GougeChecker::closestPoint(const Vec3& point) const
{
    ClosestHit result;
    if (m_triangles.empty())
    {
        return result;
    }

    const auto candidates = gatherCandidates(point);
    double bestDist2 = std::numeric_limits<double>::infinity();
    glm::vec3 bestPoint{0.0f};
    bool found = false;

    for (std::uint32_t index : candidates)
    {
        if (index >= m_triangles.size())
        {
            continue;
        }
        const Triangle& tri = m_triangles[index];

        const double verticalComponent = std::abs(tri.normal.z);
        if (verticalComponent <= 0.1)
        {
            continue;
        }

        glm::vec3 candidatePoint{0.0f};
        const double dist2 = pointTriangleDistanceSquared(point, tri.a, tri.b, tri.c, candidatePoint);
        if (candidatePoint.z > static_cast<double>(point.z) + 1e-4)
        {
            continue;
        }
        if (dist2 < bestDist2)
        {
            bestDist2 = dist2;
            bestPoint = candidatePoint;
            found = true;
        }
    }

    if (!found)
    {
        return result;
    }

    result.hit = true;
    result.distance = std::sqrt(bestDist2);
    result.closestPoint = bestPoint;
    return result;
}

std::optional<double> GougeChecker::surfaceHeightAt(const Vec3& sample) const
{
    const ClosestHit hit = closestPoint(sample);
    if (!hit.hit)
    {
        return std::nullopt;
    }
    return static_cast<double>(hit.closestPoint.z);
}

double GougeChecker::minClearanceAlong(const std::vector<Vec3>& path, const GougeParams& params) const
{
    if (path.size() < 2 || m_triangles.empty())
    {
        return std::numeric_limits<double>::infinity();
    }

    const double sampleSpacing = std::max(0.5, params.toolRadius * 0.5);
    double minClearance = std::numeric_limits<double>::infinity();
    bool sawSample = false;

    for (std::size_t i = 0; i + 1 < path.size(); ++i)
    {
        const Vec3& start = path[i];
        const Vec3& end = path[i + 1];
        const double length = glm::length(end - start);
        const int samples = std::max(1, static_cast<int>(std::ceil(length / sampleSpacing)));

        for (int s = 0; s <= samples; ++s)
        {
            const double t = static_cast<double>(s) / static_cast<double>(samples);
            const Vec3 sample = start + static_cast<float>(t) * (end - start);
            const ClosestHit hit = closestPoint(sample);
            if (!hit.hit)
            {
                continue;
            }
            const double clearance = static_cast<double>(sample.z) - static_cast<double>(hit.closestPoint.z);
            minClearance = std::min(minClearance, clearance);
            sawSample = true;
        }
    }

    if (!sawSample)
    {
        return std::numeric_limits<double>::infinity();
    }
    return minClearance;
}

GougeChecker::AdjustResult GougeChecker::adjustZForLeaveStock(const std::vector<Vec3>& path,
                                                              const GougeParams& params) const
{
    AdjustResult result;
    if (path.empty())
    {
        result.adjustedPath = path;
        result.minClearance = std::numeric_limits<double>::infinity();
        return result;
    }

    result.adjustedPath = path;
    const double targetLeaveStock = std::max(0.0, params.leaveStock);
    const double initialClearance = minClearanceAlong(result.adjustedPath, params);
    double effectiveClearance = std::isfinite(initialClearance) ? initialClearance : 0.0;
    result.minClearance = initialClearance;

    if (targetLeaveStock <= kEpsilon)
    {
        result.ok = true;
        return result;
    }

    if (effectiveClearance + 1e-4 >= targetLeaveStock)
    {
        result.ok = true;
        result.minClearance = initialClearance;
        return result;
    }

    double deficit = targetLeaveStock - effectiveClearance;
    double maxZ = std::numeric_limits<double>::lowest();
    for (const Vec3& point : result.adjustedPath)
    {
        maxZ = std::max(maxZ, static_cast<double>(point.z));
    }

    if (params.safetyZ > 0.0)
    {
        const double available = params.safetyZ - maxZ;
        if (available <= 1e-4 || available + 1e-4 < deficit)
        {
            result.ok = false;
            result.message = "clearance would exceed safety Z";
            return result;
        }
        deficit = std::min(deficit, available);
    }

    for (Vec3& point : result.adjustedPath)
    {
        point.z = static_cast<float>(static_cast<double>(point.z) + deficit);
    }
    result.adjusted = deficit > 1e-6;
    result.minClearance = effectiveClearance + deficit;
    result.ok = true;
    return result;
}

} // namespace tp
