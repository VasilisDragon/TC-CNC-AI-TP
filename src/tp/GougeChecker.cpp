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

double pointTriangleDistanceSquared(const glm::dvec3& point,
                                    const glm::dvec3& a,
                                    const glm::dvec3& b,
                                    const glm::dvec3& c,
                                    glm::dvec3& outClosest)
{
    const glm::dvec3 ab = b - a;
    const glm::dvec3 ac = c - a;
    const glm::dvec3 ap = point - a;

    const double d1 = glm::dot(ab, ap);
    const double d2 = glm::dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        outClosest = a;
        return glm::dot(ap, ap);
    }

    const glm::dvec3 bp = point - b;
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
        outClosest = a + v * ab;
        const glm::dvec3 diff = point - outClosest;
        return glm::dot(diff, diff);
    }

    const glm::dvec3 cp = point - c;
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
        outClosest = a + w * ac;
        const glm::dvec3 diff = point - outClosest;
        return glm::dot(diff, diff);
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        outClosest = b + w * (c - b);
        const glm::dvec3 diff = point - outClosest;
        return glm::dot(diff, diff);
    }

    const glm::dvec3 n = glm::cross(ab, ac);
    const double denom = glm::dot(n, n);
    if (denom <= kEpsilon)
    {
        outClosest = a;
        const glm::dvec3 diff = point - a;
        return glm::dot(diff, diff);
    }

    const double v = glm::dot(glm::cross(ap, ac), n) / denom;
    const double w = glm::dot(glm::cross(ab, ap), n) / denom;
    const double u = 1.0 - v - w;
    outClosest = u * a + v * b + w * c;
    const glm::dvec3 diff = point - outClosest;
    return glm::dot(diff, diff);
}

} // namespace

namespace tp
{

GougeChecker::GougeChecker(const render::Model& model)
    : m_grid(model, 0.0)
{
    m_candidateScratch.reserve(128);
}

GougeChecker::ClosestHit GougeChecker::closestPoint(const Vec3& point) const
{
    ClosestHit result;
    if (m_grid.empty())
    {
        return result;
    }

    glm::dvec3 query(static_cast<double>(point.x), static_cast<double>(point.y), static_cast<double>(point.z));

    auto gatherWithRadius = [&](int radius) {
        m_grid.gatherCandidatesXY(query.x, query.y, radius, m_candidateScratch);
        return !m_candidateScratch.empty();
    };

    if (!gatherWithRadius(1))
    {
        if (!gatherWithRadius(2))
        {
            gatherWithRadius(3);
        }
    }

    double bestDist2 = std::numeric_limits<double>::infinity();
    glm::dvec3 bestPoint{0.0};
    bool found = false;

    for (std::uint32_t index : m_candidateScratch)
    {
        if (index >= m_grid.triangleCount())
        {
            continue;
        }
        const TriangleGrid::Triangle& tri = m_grid.triangle(index);

        const double verticalComponent = std::abs(tri.normal.z);
        if (verticalComponent <= 0.1)
        {
            continue;
        }

        glm::dvec3 candidatePoint{0.0};
        const double dist2 = pointTriangleDistanceSquared(query, tri.v0, tri.v1, tri.v2, candidatePoint);
        if (candidatePoint.z > query.z + 1e-4)
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
        m_candidateScratch.clear();
        return result;
    }

    result.hit = true;
    result.distance = std::sqrt(bestDist2);
    result.closestPoint =
        Vec3(static_cast<float>(bestPoint.x), static_cast<float>(bestPoint.y), static_cast<float>(bestPoint.z));
    m_candidateScratch.clear();
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
    if (path.size() < 2 || m_grid.empty())
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
