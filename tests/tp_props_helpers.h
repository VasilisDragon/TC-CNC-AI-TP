#pragma once

#include "tp/Toolpath.h"

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace tp_props
{

inline double randomDouble(std::mt19937& rng, double minValue, double maxValue)
{
    std::uniform_real_distribution<double> dist(minValue, maxValue);
    return dist(rng);
}

inline glm::dvec3 toDVec3(const glm::vec3& value)
{
    return {static_cast<double>(value.x), static_cast<double>(value.y), static_cast<double>(value.z)};
}

struct Segment2D
{
    glm::dvec2 a;
    glm::dvec2 b;
    double minX;
    double maxX;
    double minY;
    double maxY;
};

inline Segment2D makeSegment(const glm::dvec3& a, const glm::dvec3& b)
{
    Segment2D segment;
    segment.a = {a.x, a.y};
    segment.b = {b.x, b.y};
    segment.minX = std::min(segment.a.x, segment.b.x);
    segment.maxX = std::max(segment.a.x, segment.b.x);
    segment.minY = std::min(segment.a.y, segment.b.y);
    segment.maxY = std::max(segment.a.y, segment.b.y);
    return segment;
}

inline bool boxesOverlap(const Segment2D& lhs, const Segment2D& rhs, double tolerance)
{
    const bool overlapX = lhs.maxX >= rhs.minX - tolerance && lhs.minX <= rhs.maxX + tolerance;
    const bool overlapY = lhs.maxY >= rhs.minY - tolerance && lhs.minY <= rhs.maxY + tolerance;
    return overlapX && overlapY;
}

inline bool sharesEndpoint(const Segment2D& lhs, const Segment2D& rhs, double tolerance)
{
    const auto close = [tolerance](const glm::dvec2& p0, const glm::dvec2& p1) {
        return glm::distance(p0, p1) <= tolerance;
    };
    return close(lhs.a, rhs.a) || close(lhs.a, rhs.b) || close(lhs.b, rhs.a) || close(lhs.b, rhs.b);
}

inline double cross2D(const glm::dvec2& a, const glm::dvec2& b)
{
    return a.x * b.y - a.y * b.x;
}

inline bool segmentsIntersectStrict(const Segment2D& lhs, const Segment2D& rhs, double tolerance)
{
    if (!boxesOverlap(lhs, rhs, tolerance))
    {
        return false;
    }
    if (sharesEndpoint(lhs, rhs, tolerance))
    {
        return false;
    }

    const glm::dvec2 d1 = lhs.b - lhs.a;
    const glm::dvec2 d2 = rhs.b - rhs.a;
    const double denom = cross2D(d1, d2);
    const glm::dvec2 diff = rhs.a - lhs.a;

    if (std::abs(denom) <= tolerance)
    {
        const double crossDiff = cross2D(diff, d1);
        if (std::abs(crossDiff) > tolerance)
        {
            return false;
        }

        const double lengthSq = glm::dot(d1, d1);
        if (lengthSq <= tolerance * tolerance)
        {
            return false;
        }

        const double projectionStart = glm::dot(diff, d1) / lengthSq;
        const double projectionEnd = projectionStart + glm::dot(d2, d1) / lengthSq;
        const double minProj = std::min(projectionStart, projectionEnd);
        const double maxProj = std::max(projectionStart, projectionEnd);
        return maxProj > tolerance && minProj < 1.0 - tolerance;
    }

    const double t = cross2D(diff, d2) / denom;
    const double u = cross2D(diff, d1) / denom;
    return t > tolerance && t < 1.0 - tolerance && u > tolerance && u < 1.0 - tolerance;
}

inline bool polylineHasSelfIntersections(const tp::Polyline& polyline, double tolerance)
{
    if (polyline.pts.size() < 4)
    {
        return false;
    }

    std::vector<Segment2D> segments;
    segments.reserve(polyline.pts.size());

    glm::dvec3 previous = toDVec3(polyline.pts.front().p);
    for (std::size_t i = 1; i < polyline.pts.size(); ++i)
    {
        const glm::dvec3 current = toDVec3(polyline.pts[i].p);
        const glm::dvec2 offset{current.x - previous.x, current.y - previous.y};
        if (glm::length(offset) <= tolerance)
        {
            previous = current;
            continue;
        }
        segments.push_back(makeSegment(previous, current));
        previous = current;
    }

    const std::size_t segmentCount = segments.size();
    if (segmentCount < 2)
    {
        return false;
    }

    for (std::size_t i = 0; i < segmentCount; ++i)
    {
        for (std::size_t j = i + 1; j < segmentCount; ++j)
        {
            if (j <= i + 1)
            {
                continue;
            }
            if (i == 0 && j == segmentCount - 1)
            {
                if (sharesEndpoint(segments[i], segments[j], tolerance))
                {
                    continue;
                }
            }
            if (segmentsIntersectStrict(segments[i], segments[j], tolerance))
            {
                return true;
            }
        }
    }

    return false;
}

inline std::pair<double, double> polylineZExtents(const tp::Polyline& polyline)
{
    double minZ = std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();

    for (const tp::Vertex& vertex : polyline.pts)
    {
        const double z = static_cast<double>(vertex.p.z);
        minZ = std::min(minZ, z);
        maxZ = std::max(maxZ, z);
    }

    return {minZ, maxZ};
}

} // namespace tp_props
