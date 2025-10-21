#include "tp/GCodePostBase.h"

#include "ai/IPathAI.h"

#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <limits>
#include <numbers>
#include <vector>

namespace
{

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

} // namespace

namespace tp
{

std::string GCodePostBase::generate(const tp::Toolpath& toolpath,
                                    common::UnitSystem units,
                                    const tp::UserParams& params)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);

    const double maxChordError = std::max(0.0, params.post.maxArcChordError_mm);
    const bool arcsEnabled = supportsArcs() && allowArcs(params) && maxChordError > 0.0;
    const double feedUnits = toUnits(toolpath.feed, units);
    const std::string nl(newline());

    TemplateContext headerContext;
    buildHeaderContext(headerContext, toolpath, units, params, arcsEnabled);
    const std::string header = TemplateEngine::render(headerTemplate(), headerContext);
    out << header;
    if (!header.empty() && !header.ends_with(nl))
    {
        out << nl;
    }

    int currentStep = -1;
    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (poly.strategyStep != currentStep)
        {
            currentStep = poly.strategyStep;
            const std::size_t stepIndex = static_cast<std::size_t>(std::max(currentStep, 0));
            if (currentStep >= 0 && stepIndex < toolpath.strategySteps.size())
            {
                TemplateContext stepContext;
                buildStepContext(stepContext, toolpath.strategySteps[stepIndex], stepIndex);
                const std::string stepBlock = TemplateEngine::render(stepBlockTemplate(), stepContext);
                if (!stepBlock.empty())
                {
                    out << stepBlock;
                    if (!stepBlock.ends_with(nl))
                    {
                        out << nl;
                    }
                }
            }
        }

        emitPolyline(out, poly, units, feedUnits, arcsEnabled, maxChordError);
    }

    TemplateContext footerContext;
    buildFooterContext(footerContext, toolpath, units, params, arcsEnabled);
    const std::string footer = TemplateEngine::render(footerTemplate(), footerContext);
    out << footer;
    if (!footer.empty() && !footer.ends_with(nl))
    {
        out << nl;
    }

    return out.str();
}

std::string_view GCodePostBase::stepBlockTemplate() const
{
    static constexpr std::string_view kDefaultStepBlock =
        "(STEP {{step_number}} {{step_label}} {{pass_kind}} stepover={{stepover_mm}}mm stepdown={{stepdown_mm}}mm"
        "{{#if has_angle}} angle={{angle_deg}}deg{{/if}})";
    return kDefaultStepBlock;
}

void GCodePostBase::buildHeaderContext(TemplateContext& context,
                                       const tp::Toolpath& toolpath,
                                       common::UnitSystem units,
                                       const tp::UserParams& params,
                                       bool arcsEnabled) const
{
    context.set("post_name", name());
    context.set("unit_code", (units == common::UnitSystem::Inches) ? "G20" : "G21");
    context.set("unit_suffix", (units == common::UnitSystem::Inches) ? "in/min" : "mm/min");
    context.set("positioning_mode", std::string(positioningMode()));
    context.setBool("has_plane", !planeCode().empty());
    context.set("plane_code", std::string(planeCode()));
    context.setBool("has_feed_mode", !feedMode().empty());
    context.set("feed_mode", std::string(feedMode()));
    context.setBool("has_work_offset", !workOffset().empty());
    context.set("work_offset", std::string(workOffset()));
    context.setBool("spindle_supported", spindleSupported());
    context.setBool("spindle_requested", toolpath.spindle > 0.0);
    context.set("spindle_speed", formatNumber(toolpath.spindle));
    context.set("feed_rate", formatNumber(toUnits(toolpath.feed, units)));
    context.set("rapid_feed", formatNumber(toUnits(toolpath.machine.rapidFeed_mm_min, units)));
    context.set("max_feed", formatNumber(toUnits(toolpath.machine.maxFeed_mm_min, units)));
    {
        std::ostringstream summary;
        summary.setf(std::ios::fixed);
        summary << "(Machine: " << toolpath.machine.name
                << ", rapid " << formatNumber(toUnits(toolpath.machine.rapidFeed_mm_min, units))
                << " " << (units == common::UnitSystem::Inches ? "in/min" : "mm/min")
                << ", max feed " << formatNumber(toUnits(toolpath.machine.maxFeed_mm_min, units))
                << " " << (units == common::UnitSystem::Inches ? "in/min" : "mm/min")
                << ")";
        context.set("machine_summary", summary.str());
    }
    context.setBool("arcs_enabled", arcsEnabled);
    context.set("spindle_on_code", std::string(spindleOnCode()));
    context.set("spindle_off_code", std::string(spindleOffCode()));
    context.set("program_end_code", std::string(programEndCode()));
    context.setBool("has_toolpath", !toolpath.empty());
    context.setBool("has_strategy_steps", !toolpath.strategySteps.empty());
    context.setBool("has_user_arcs", params.post.maxArcChordError_mm > 0.0);
}

void GCodePostBase::buildFooterContext(TemplateContext& context,
                                       const tp::Toolpath& toolpath,
                                       common::UnitSystem units,
                                       const tp::UserParams& params,
                                       bool arcsEnabled) const
{
    (void)params;
    (void)units;
    (void)arcsEnabled;
    context.setBool("spindle_supported", spindleSupported());
    context.setBool("spindle_requested", toolpath.spindle > 0.0);
    context.set("spindle_speed", formatNumber(toolpath.spindle));
    context.set("spindle_off_code", std::string(spindleOffCode()));
    context.set("program_end_code", std::string(programEndCode()));
}

void GCodePostBase::buildStepContext(TemplateContext& context,
                                     const ai::StrategyStep& step,
                                     std::size_t stepIndex) const
{
    context.set("step_number", std::to_string(stepIndex + 1));
    context.set("step_label", (step.type == ai::StrategyStep::Type::Raster) ? "Raster" : "Waterline");
    context.set("pass_kind", step.finish_pass ? "finish" : "rough");
    context.set("stepover_mm", formatNumber(step.stepover));
    context.set("stepdown_mm", formatNumber(step.stepdown));
    context.setBool("has_angle", step.type == ai::StrategyStep::Type::Raster);
    context.set("angle_deg", formatNumber(step.angle_deg, 1));
}

bool GCodePostBase::allowArcs(const tp::UserParams& params) const
{
    return params.post.maxArcChordError_mm > 0.0;
}

void GCodePostBase::emitFeedCommand(std::ostringstream& out, double feedUnits) const
{
    out << "F" << formatNumber(feedUnits) << newline();
}

void GCodePostBase::emitLinearMove(std::ostringstream& out,
                                   const glm::dvec3& point,
                                   MotionType motion,
                                   common::UnitSystem units,
                                   double /*feedUnits*/) const
{
    const char* code = (motion == MotionType::Cut) ? "G1" : "G0";
    out << code
        << " X" << formatNumber(toUnits(point.x, units))
        << " Y" << formatNumber(toUnits(point.y, units))
        << " Z" << formatNumber(toUnits(point.z, units))
        << newline();
}

void GCodePostBase::emitArcMove(std::ostringstream& out,
                                bool clockwise,
                                const glm::dvec3& start,
                                const glm::dvec3& end,
                                const glm::dvec2& center,
                                common::UnitSystem units,
                                double /*feedUnits*/) const
{
    const char* code = clockwise ? "G2" : "G3";
    const double i = center.x - start.x;
    const double j = center.y - start.y;
    out << code
        << " X" << formatNumber(toUnits(end.x, units))
        << " Y" << formatNumber(toUnits(end.y, units))
        << " Z" << formatNumber(toUnits(end.z, units))
        << " I" << formatNumber(toUnits(i, units))
        << " J" << formatNumber(toUnits(j, units))
        << newline();
}

double GCodePostBase::toUnits(double valueMm, common::UnitSystem units)
{
    return (units == common::UnitSystem::Inches)
               ? common::fromMillimeters(valueMm, common::UnitSystem::Inches)
               : valueMm;
}

std::string GCodePostBase::formatNumber(double value, int precision)
{
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(precision) << value;
    return oss.str();
}

void GCodePostBase::emitPolyline(std::ostringstream& out,
                                 const tp::Polyline& poly,
                                 common::UnitSystem units,
                                 double feedUnits,
                                 bool arcsEnabled,
                                 double maxChordError) const
{
    const std::vector<glm::dvec3> points = sanitizePolyline(poly);
    if (points.size() < 2)
    {
        return;
    }

    const bool isCut = (poly.motion == MotionType::Cut);
    if (isCut)
    {
        emitFeedCommand(out, feedUnits);
    }

    emitLinearMove(out, points.front(), poly.motion, units, feedUnits);

    for (std::size_t i = 1; i < points.size();)
    {
        const glm::dvec3& prev = points[i - 1];
        const glm::dvec3& current = points[i];

        if (!isCut || !arcsEnabled || std::abs(current.z - prev.z) > kZPlaneTolerance)
        {
            emitLinearMove(out, current, poly.motion, units, feedUnits);
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
            emitArcMove(out, bestArc.clockwise, points[i - 1], points[endIndex], bestArc.center, units, feedUnits);
            i = endIndex + 1;
        }
        else
        {
            emitLinearMove(out, current, poly.motion, units, feedUnits);
            ++i;
        }
    }
}

} // namespace tp
