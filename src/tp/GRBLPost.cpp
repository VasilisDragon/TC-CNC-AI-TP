#include "tp/GRBLPost.h"

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <vector>

namespace
{

constexpr const char* kNL = "\r\n";
constexpr double kZPlaneTolerance = 1e-4;
constexpr double kDegenerateDistance = 1e-6;
constexpr double kMinSweepRadians = 1e-4;
constexpr double kFullCircleGuard = 1e-3;

struct ArcCommand
{
    std::size_t endIndex{0};
    glm::dvec2 center{0.0};
    bool clockwise{false};
};

double toUnits(double valueMm, common::Unit units)
{
    return (units == common::Unit::Inches)
               ? common::fromMillimeters(valueMm, common::Unit::Inches)
               : valueMm;
}

std::string unitPrefix(common::Unit units)
{
    return (units == common::Unit::Inches) ? "G20" : "G21";
}

glm::dvec3 toDVec3(const glm::vec3& v)
{
    return {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)};
}

double cross2(const glm::dvec2& a, const glm::dvec2& b)
{
    return a.x * b.y - a.y * b.x;
}

std::vector<glm::dvec3> sanitizePolyline(const tp::Polyline& poly)
{
    std::vector<glm::dvec3> points;
    points.reserve(poly.pts.size());
    for (const auto& vertex : poly.pts)
    {
        const glm::dvec3 candidate = toDVec3(vertex.p);
        if (!points.empty())
        {
            const glm::dvec3 delta = candidate - points.back();
            if (glm::length(delta) <= kDegenerateDistance)
            {
                continue;
            }
        }
        points.push_back(candidate);
    }
    return points;
}

bool circleFromPoints(const glm::dvec2& a,
                      const glm::dvec2& b,
                      const glm::dvec2& c,
                      glm::dvec2& outCenter,
                      double& outRadius)
{
    const double d = 2.0 * ((a.x * (b.y - c.y)) + (b.x * (c.y - a.y)) + (c.x * (a.y - b.y)));
    if (std::abs(d) <= std::numeric_limits<double>::epsilon())
    {
        return false;
    }

    const double aSq = a.x * a.x + a.y * a.y;
    const double bSq = b.x * b.x + b.y * b.y;
    const double cSq = c.x * c.x + c.y * c.y;

    const double ux = (aSq * (b.y - c.y) + bSq * (c.y - a.y) + cSq * (a.y - b.y)) / d;
    const double uy = (aSq * (c.x - b.x) + bSq * (a.x - c.x) + cSq * (b.x - a.x)) / d;

    outCenter = {ux, uy};
    outRadius = glm::length(a - outCenter);
    return std::isfinite(outRadius) && outRadius > kDegenerateDistance;
}

bool tryFitArc(const std::vector<glm::dvec3>& points,
               std::size_t start,
               std::size_t end,
               double maxChordError,
               ArcCommand& out)
{
    if (end <= start + 1)
    {
        return false;
    }

    const glm::dvec3& start3 = points[start];
    const glm::dvec3& end3 = points[end];
    if (glm::length(end3 - start3) <= kDegenerateDistance)
    {
        return false;
    }

    glm::dvec2 p0{start3.x, start3.y};
    glm::dvec2 pn{end3.x, end3.y};

    std::size_t pivotIndex = start + 1;
    double maxArea = 0.0;
    for (std::size_t idx = start + 1; idx < end; ++idx)
    {
        const glm::dvec2 pi{points[idx].x, points[idx].y};
        const double area = std::abs(cross2(pi - p0, pn - p0));
        if (area > maxArea)
        {
            maxArea = area;
            pivotIndex = idx;
        }
    }

    if (maxArea <= std::numeric_limits<double>::epsilon())
    {
        return false;
    }

    glm::dvec2 center{0.0};
    double radius = 0.0;
    if (!circleFromPoints(p0, glm::dvec2(points[pivotIndex].x, points[pivotIndex].y), pn, center, radius))
    {
        return false;
    }

    double maxRadialError = 0.0;
    std::vector<glm::dvec2> vectors;
    vectors.reserve(end - start + 1);

    for (std::size_t idx = start; idx <= end; ++idx)
    {
        const glm::dvec2 pi{points[idx].x, points[idx].y};
        const glm::dvec2 vec = pi - center;
        const double dist = glm::length(vec);
        if (!std::isfinite(dist) || dist <= kDegenerateDistance)
        {
            return false;
        }

        maxRadialError = std::max(maxRadialError, std::abs(dist - radius));
        if (maxRadialError > maxChordError)
        {
            return false;
        }
        vectors.push_back(vec);
    }

    for (std::size_t idx = start; idx < end; ++idx)
    {
        const glm::dvec2 pi{points[idx].x, points[idx].y};
        const glm::dvec2 pj{points[idx + 1].x, points[idx + 1].y};
        const double chord = glm::length(pj - pi);
        if (chord <= kDegenerateDistance)
        {
            continue;
        }

        const double term = radius * radius - (chord * chord * 0.25);
        if (term < 0.0)
        {
            return false;
        }
        const double sagitta = radius - std::sqrt(term);
        if (sagitta > maxChordError + 1e-9)
        {
            return false;
        }
    }

    double crossSum = 0.0;
    for (std::size_t i = 0; i + 1 < vectors.size(); ++i)
    {
        crossSum += cross2(vectors[i], vectors[i + 1]);
    }

    if (std::abs(crossSum) <= std::numeric_limits<double>::epsilon())
    {
        return false;
    }

    const bool clockwise = (crossSum < 0.0);
    const auto angleOf = [](const glm::dvec2& v) { return std::atan2(v.y, v.x); };

    double previousAngle = angleOf(vectors.front());
    double cumulative = 0.0;
    for (std::size_t i = 1; i < vectors.size(); ++i)
    {
        double angle = angleOf(vectors[i]);
        double delta = angle - previousAngle;
        if (clockwise)
        {
            while (delta >= 0.0)
            {
                delta -= 2.0 * std::numbers::pi;
            }
        }
        else
        {
            while (delta <= 0.0)
            {
                delta += 2.0 * std::numbers::pi;
            }
        }
        cumulative += delta;
        previousAngle = angle;
    }

    const double sweep = std::abs(cumulative);
    if (sweep < kMinSweepRadians || sweep >= (2.0 * std::numbers::pi - kFullCircleGuard))
    {
        return false;
    }

    out.endIndex = end;
    out.center = center;
    out.clockwise = clockwise;
    return true;
}

void emitLinear(std::ostringstream& out,
                const char* code,
                const glm::dvec3& point,
                common::Unit units)
{
    out << code
        << " X" << toUnits(point.x, units)
        << " Y" << toUnits(point.y, units)
        << " Z" << toUnits(point.z, units)
        << kNL;
}

void emitArc(std::ostringstream& out,
             const ArcCommand& arc,
             const glm::dvec3& start,
             const glm::dvec3& end,
             common::Unit units)
{
    const char* code = arc.clockwise ? "G2" : "G3";
    const double i = arc.center.x - start.x;
    const double j = arc.center.y - start.y;
    out << code
        << " X" << toUnits(end.x, units)
        << " Y" << toUnits(end.y, units)
        << " Z" << toUnits(end.z, units)
        << " I" << toUnits(i, units)
        << " J" << toUnits(j, units)
        << kNL;
}

} // namespace

namespace tp
{

std::string GRBLPost::name() const
{
    return "GRBL";
}

std::string GRBLPost::generate(const tp::Toolpath& toolpath,
                               common::Unit units,
                               const tp::UserParams& params)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);

    out << "(AIToolpathGenerator - GRBL Post)" << kNL;
    out << unitPrefix(units) << " ; units" << kNL;
    out << "G90 ; absolute positioning" << kNL;
    const Machine& machine = toolpath.machine;
    const double rapidUnits = toUnits(machine.rapidFeed_mm_min, units);
    const double maxFeedUnits = toUnits(machine.maxFeed_mm_min, units);
    out << "(Machine: " << machine.name
        << ", rapid " << rapidUnits << (units == common::Unit::Inches ? " in/min" : " mm/min")
        << ", max feed " << maxFeedUnits << (units == common::Unit::Inches ? " in/min" : " mm/min") << ")" << kNL;
    out << "M3 S" << toolpath.spindle << " ; spindle on" << kNL;

    const double feedUnits = toUnits(toolpath.feed, units);
    const double maxChordError = std::max(0.0, params.post.maxArcChordError_mm);
    const bool arcsEnabled = (maxChordError > 0.0);

    for (const tp::Polyline& poly : toolpath.passes)
    {
        std::vector<glm::dvec3> points = sanitizePolyline(poly);
        if (points.size() < 2)
        {
            continue;
        }

        const bool isCut = (poly.motion == MotionType::Cut);
        const char* linearCode = isCut ? "G1" : "G0";
        if (isCut)
        {
            out << "F" << feedUnits << kNL;
        }

        emitLinear(out, linearCode, points.front(), units);

        for (std::size_t i = 1; i < points.size();)
        {
            const glm::dvec3& prev = points[i - 1];
            const glm::dvec3& current = points[i];

            if (!isCut || !arcsEnabled)
            {
                emitLinear(out, linearCode, current, units);
                ++i;
                continue;
            }

            if (std::abs(current.z - prev.z) > kZPlaneTolerance)
            {
                emitLinear(out, linearCode, current, units);
                ++i;
                continue;
            }

            std::size_t runLimit = i + 1;
            while (runLimit < points.size() && std::abs(points[runLimit].z - prev.z) <= kZPlaneTolerance)
            {
                ++runLimit;
            }

            ArcCommand bestArc{};
            bool hasArc = false;
            for (std::size_t end = i + 1; end < runLimit; ++end)
            {
                ArcCommand candidate{};
                if (tryFitArc(points, i - 1, end, maxChordError, candidate))
                {
                    bestArc = candidate;
                    hasArc = true;
                }
                else
                {
                    break;
                }
            }

            if (hasArc)
            {
                const std::size_t endIndex = bestArc.endIndex;
                emitArc(out, bestArc, points[i - 1], points[endIndex], units);
                i = endIndex + 1;
            }
            else
            {
                emitLinear(out, linearCode, current, units);
                ++i;
            }
        }
    }

    out << "M5 ; spindle off" << kNL;
    out << "M2" << kNL;
    return out.str();
}

} // namespace tp
