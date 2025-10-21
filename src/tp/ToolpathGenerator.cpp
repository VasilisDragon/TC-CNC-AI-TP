// ToolpathGenerator.cpp orchestrates the toolpath planning pipeline, translating AI proposals and user
// preferences into motion primitives that keep benchtop CNC hardware safe without wasting cycle time.
// The heuristics live here so the reasoning behind each pass (clearance, ramping, caching) stays close
// to the code that executes it.

#include "tp/ToolpathGenerator.h"

#include "common/Enforce.h"
#include "common/log.h"
#include "render/Model.h"
#include "tp/heightfield/HeightField.h"
#include "tp/heightfield/UniformGrid.h"
#include "tp/GougeChecker.h"
#include "tp/ocl/OclAdapter.h"
#include "tp/waterline/ZSlicer.h"

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <memory>
#include <limits>
#include <mutex>
#include <numbers>
#include <numeric>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <algorithm>

#include <QtCore/QString>

namespace tp
{

namespace
{
// Safety heuristics tuned from shop runs; they bias toward conservative approach feeds so the desktop
// machines we target clear clamps and ease into material without chatter.
constexpr double kMinClearanceOffset = 0.25;
constexpr double kMinSafeGap = 0.5;
constexpr double kPositionEpsilon = 1e-4;
// Default ramp angle keeps entry moves <3° so cutters ease into the stock; bounds match safe limits for
// aluminium on hobby-class routers but still allow aggressive overrides for rigid machines.
constexpr double kDefaultRampAngleDeg = 3.0;
constexpr double kMinRampAngleDeg = 0.5;
constexpr double kMaxRampAngleDeg = 45.0;
// Lateral ramp factor limits prevent the machine from racing ahead while plunging, so users cannot pick
// ratios that exceed the travel envelope that our jerk-limited planners can follow.
constexpr double kMinRampHorizontalFactor = 0.25;
constexpr double kMaxRampHorizontalFactor = 6.0;

class ScopedTimer
{
public:
    using Callback = std::function<void(const QString&, double, bool)>;

    ScopedTimer(QString label, Callback callback, const std::atomic<bool>* cancelFlag = nullptr)
        : m_label(std::move(label))
        , m_callback(std::move(callback))
        , m_cancel(cancelFlag)
        , m_start(std::chrono::steady_clock::now())
    {
    }

    ~ScopedTimer()
    {
        const auto elapsed = std::chrono::steady_clock::now() - m_start;
        const double ms = std::chrono::duration<double, std::milli>(elapsed).count();
        const bool cancelled = m_cancel != nullptr && m_cancel->load(std::memory_order_relaxed);
        if (m_callback)
        {
            m_callback(m_label, ms, cancelled);
        }
        else
        {
            if (cancelled)
            {
                LOG_INFO(Tp, QStringLiteral("%1 cancelled after %2 ms").arg(m_label).arg(ms, 0, 'f', 2));
            }
            else
            {
                LOG_INFO(Tp, QStringLiteral("%1 took %2 ms").arg(m_label).arg(ms, 0, 'f', 2));
            }
        }
    }

private:
    QString m_label;
    Callback m_callback;
    const std::atomic<bool>* m_cancel{nullptr};
    std::chrono::steady_clock::time_point m_start;
};

float clampStepOver(double stepOverMm)
{
    constexpr double kMinStep = 0.1;
    return static_cast<float>(std::max(stepOverMm, kMinStep));
}

glm::dvec3 toDVec3(const glm::vec3& v)
{
    return glm::dvec3{static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)};
}

glm::vec3 toVec3(const glm::dvec3& v)
{
    return glm::vec3{static_cast<float>(v.x), static_cast<float>(v.y), static_cast<float>(v.z)};
}

bool nearlyEqual(const glm::dvec3& a, const glm::dvec3& b)
{
    return glm::length(a - b) <= kPositionEpsilon;
}

void appendPolyline(std::vector<Polyline>& passes,
                    MotionType motion,
                    std::initializer_list<glm::dvec3> points)
{
    if (points.size() < 2)
    {
        return;
    }

    Polyline poly;
    poly.motion = motion;

    auto it = points.begin();
    glm::dvec3 prev = *it;
    poly.pts.push_back({toVec3(prev)});
    ++it;

    for (; it != points.end(); ++it)
    {
        if (nearlyEqual(prev, *it))
        {
            continue;
        }
        prev = *it;
        poly.pts.push_back({toVec3(prev)});
    }

    if (poly.pts.size() >= 2)
    {
        passes.push_back(std::move(poly));
    }
}

glm::dvec2 normalize2D(const glm::dvec2& dir)
{
    const double len = glm::length(dir);
    if (len <= kPositionEpsilon)
    {
        return {1.0, 0.0};
    }
    return dir / len;
}

void pruneSequentialDuplicates(std::vector<glm::dvec3>& points)
{
    if (points.size() < 2)
    {
        return;
    }

    std::vector<glm::dvec3> pruned;
    pruned.reserve(points.size());
    glm::dvec3 prev = points.front();
    pruned.push_back(prev);

    for (std::size_t i = 1; i < points.size(); ++i)
    {
        if (nearlyEqual(prev, points[i]))
        {
            continue;
        }
        prev = points[i];
        pruned.push_back(prev);
    }

    points = std::move(pruned);
}

void appendCutPolyline(std::vector<Polyline>& passes, const std::vector<glm::dvec3>& points)
{
    if (points.size() < 2)
    {
        return;
    }

    Polyline poly;
    poly.motion = MotionType::Cut;

    glm::dvec3 prev = points.front();
    poly.pts.push_back({toVec3(prev)});
    for (std::size_t i = 1; i < points.size(); ++i)
    {
        if (nearlyEqual(prev, points[i]))
        {
            continue;
        }
        prev = points[i];
        poly.pts.push_back({toVec3(prev)});
    }

    if (poly.pts.size() >= 2)
    {
        passes.push_back(std::move(poly));
    }
}

glm::dvec2 selectDirection2D(const std::vector<glm::dvec3>& points, bool forward)
{
    if (points.size() < 2)
    {
        return {1.0, 0.0};
    }

    if (forward)
    {
        const glm::dvec3& origin = points.front();
        for (std::size_t i = 1; i < points.size(); ++i)
        {
            glm::dvec3 delta = points[i] - origin;
            delta.z = 0.0;
            const double len = glm::length(delta);
            if (len > kPositionEpsilon)
            {
                return {delta.x / len, delta.y / len};
            }
        }
    }
    else
    {
        const glm::dvec3& origin = points.back();
        for (std::size_t i = points.size() - 1; i > 0; --i)
        {
            glm::dvec3 delta = origin - points[i - 1];
            delta.z = 0.0;
            const double len = glm::length(delta);
            if (len > kPositionEpsilon)
            {
                return {delta.x / len, delta.y / len};
            }
        }
    }

    return {1.0, 0.0};
}

glm::dvec3 offsetPoint(const glm::dvec3& origin,
                       const glm::dvec2& dir,
                       double distance,
                       double targetZ,
                       bool invertDirection)
{
    const double scale = invertDirection ? -distance : distance;
    return {origin.x + dir.x * scale,
            origin.y + dir.y * scale,
            targetZ};
}

double computeRampDistance(double verticalDrop,
                           double rampAngleRad,
                           double minHorizontal,
                           double maxHorizontal)
{
    if (verticalDrop <= kPositionEpsilon)
    {
        return 0.0;
    }

    const double safeAngle = std::clamp(rampAngleRad,
                                        kMinRampAngleDeg * std::numbers::pi / 180.0,
                                        kMaxRampAngleDeg * std::numbers::pi / 180.0);
    const double tanValue = std::tan(std::max(safeAngle, 1e-3));
    double horizontal = (tanValue > 1e-6) ? (verticalDrop / tanValue) : maxHorizontal;
    if (!std::isfinite(horizontal))
    {
        horizontal = maxHorizontal;
    }
    horizontal = std::clamp(horizontal, minHorizontal, maxHorizontal);
    return horizontal;
}

double horizontalDistance(const glm::dvec3& a, const glm::dvec3& b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

std::vector<glm::dvec3> buildHelicalEntry(const glm::dvec3& targetPoint,
                                          const glm::dvec2& nominalDir,
                                          double clearanceZ,
                                          double entryDrop,
                                          double rampAngleRad,
                                          double radius)
{
    if (entryDrop <= kPositionEpsilon || radius <= kPositionEpsilon)
    {
        return {};
    }

    const double tanValue = std::tan(std::max(rampAngleRad, 1e-3));
    const double circumference = 2.0 * std::numbers::pi * radius;
    double verticalPerTurn = (tanValue > 1e-6) ? circumference * tanValue : entryDrop;
    if (!std::isfinite(verticalPerTurn) || verticalPerTurn <= 1e-6)
    {
        verticalPerTurn = entryDrop;
    }

    const double baseTurns = std::max(entryDrop / verticalPerTurn, 0.25);
    const double totalTurns = std::min(baseTurns + 0.25, 6.0); // clamp for runtime safety
    const double thetaStart = totalTurns * 2.0 * std::numbers::pi;
    const double thetaEnd = 0.0;
    const double thetaSpan = thetaStart - thetaEnd;

    const int segmentsPerTurn = 18;
    int totalSegments = static_cast<int>(std::ceil(totalTurns * segmentsPerTurn));
    totalSegments = std::clamp(totalSegments, 12, 360);

    const glm::dvec2 tangent = normalize2D(nominalDir);
    glm::dvec2 normal{-tangent.y, tangent.x};
    normal = normalize2D(normal);
    glm::dvec2 radialBase = -tangent;

    std::vector<glm::dvec3> helix;
    helix.reserve(static_cast<std::size_t>(totalSegments) + 1);

    for (int i = 0; i <= totalSegments; ++i)
    {
        const double progress = static_cast<double>(i) / static_cast<double>(totalSegments);
        const double theta = thetaStart - thetaSpan * progress;
        const double cosTheta = std::cos(theta);
        const double sinTheta = std::sin(theta);
        const double scale = radius * (1.0 - progress);

        const glm::dvec2 radial = radialBase * cosTheta + normal * sinTheta;
        glm::dvec3 point;
        point.x = targetPoint.x + radial.x * scale;
        point.y = targetPoint.y + radial.y * scale;
        point.z = clearanceZ - entryDrop * progress;
        helix.push_back(point);
    }

    if (helix.empty() || !nearlyEqual(helix.back(), targetPoint))
    {
        helix.push_back(targetPoint);
    }

    pruneSequentialDuplicates(helix);
    return helix;
}

void applyMachineMotion(Toolpath& toolpath,
                        const Machine& machine,
                        const Stock& stock,
                        const UserParams& params)
{
    if (toolpath.passes.empty())
    {
        return;
    }

    const double stockTop = stock.topZ_mm;
    double clearanceZ = std::max(machine.clearanceZ_mm, stockTop + kMinClearanceOffset);
    double safeZ = std::max(machine.safeZ_mm, clearanceZ + kMinSafeGap);
    if (clearanceZ >= safeZ)
    {
        clearanceZ = std::max(stockTop + kMinClearanceOffset, safeZ - kMinSafeGap);
        safeZ = clearanceZ + kMinSafeGap;
    }

    const double requestedRamp = std::isfinite(params.rampAngleDeg) ? params.rampAngleDeg : kDefaultRampAngleDeg;
    const double rampAngleRad = std::clamp(requestedRamp, kMinRampAngleDeg, kMaxRampAngleDeg) * std::numbers::pi / 180.0;
    const double safeToolDiameter = std::max(params.toolDiameter, 0.1);
    const double minHorizontal = std::max(kMinRampHorizontalFactor * safeToolDiameter, 0.25);
    const double maxHorizontal = std::max(kMaxRampHorizontalFactor * safeToolDiameter, minHorizontal * 2.0);
    const bool enableRamp = params.enableRamp;
    const bool enableHelical = params.enableHelical;
    const double leadIn = std::max(params.leadInLength, 0.0);
    const double leadOut = std::max(params.leadOutLength, 0.0);
    const double rampRadius = (params.rampRadius > kPositionEpsilon)
                                  ? params.rampRadius
                                  : safeToolDiameter * 0.5;

    std::vector<Polyline> result;
    result.reserve(toolpath.passes.size() * 5);

    glm::dvec3 lastSafe{};
    bool haveLast = false;

    for (const Polyline& poly : toolpath.passes)
    {
        if (poly.motion != MotionType::Cut || poly.pts.size() < 2)
        {
            continue;
        }

        std::vector<glm::dvec3> cutPoints;
        cutPoints.reserve(poly.pts.size() + 2);
        for (const Vertex& vertex : poly.pts)
        {
            cutPoints.emplace_back(toDVec3(vertex.p));
        }

        glm::dvec2 entryDir = selectDirection2D(cutPoints, true);
        glm::dvec2 exitDir = selectDirection2D(cutPoints, false);

        std::vector<glm::dvec3> pathPoints;
        pathPoints.reserve(cutPoints.size() + 2);

        if (leadIn > kPositionEpsilon)
        {
            const glm::dvec3 leadStart = offsetPoint(cutPoints.front(), entryDir, leadIn, cutPoints.front().z, true);
            pathPoints.push_back(leadStart);
        }

        pathPoints.insert(pathPoints.end(), cutPoints.begin(), cutPoints.end());

        if (leadOut > kPositionEpsilon)
        {
            const glm::dvec3 leadEnd = offsetPoint(cutPoints.back(), exitDir, leadOut, cutPoints.back().z, false);
            pathPoints.push_back(leadEnd);
        }

        pruneSequentialDuplicates(pathPoints);
        if (pathPoints.size() < 2)
        {
            continue;
        }

        entryDir = selectDirection2D(pathPoints, true);
        exitDir = selectDirection2D(pathPoints, false);

        const glm::dvec3& entryPoint = pathPoints.front();
        const glm::dvec3& exitPoint = pathPoints.back();
        const double entryDrop = std::max(0.0, clearanceZ - entryPoint.z);
        const double exitDrop = std::max(0.0, clearanceZ - exitPoint.z);

        std::vector<glm::dvec3> entryPath;
        glm::dvec3 entryClear{entryPoint.x, entryPoint.y, (entryDrop > kPositionEpsilon) ? clearanceZ : entryPoint.z};

        if (entryDrop > kPositionEpsilon)
        {
            if (enableHelical)
            {
                entryPath = buildHelicalEntry(entryPoint, entryDir, clearanceZ, entryDrop, rampAngleRad, rampRadius);
            }

            if (entryPath.empty())
            {
                if (enableRamp)
                {
                    const double entryHorizontal = computeRampDistance(entryDrop, rampAngleRad, minHorizontal, maxHorizontal);
                    entryClear = offsetPoint(entryPoint, entryDir, entryHorizontal, clearanceZ, true);
                }
                else
                {
                    entryClear = {entryPoint.x, entryPoint.y, clearanceZ};
                }
                entryPath = {entryClear, entryPoint};
            }
            else
            {
                entryClear = entryPath.front();
            }
        }

        pruneSequentialDuplicates(entryPath);

        glm::dvec3 entrySafe{entryClear.x, entryClear.y, safeZ};

        if (!haveLast)
        {
            appendPolyline(result, MotionType::Rapid, {entrySafe, entryClear});
        }
        else
        {
            appendPolyline(result, MotionType::Rapid, {lastSafe, entrySafe});
            appendPolyline(result, MotionType::Rapid, {entrySafe, entryClear});
        }

        appendCutPolyline(result, entryPath);
        appendCutPolyline(result, pathPoints);

        std::vector<glm::dvec3> exitPath;
        glm::dvec3 exitClear{exitPoint.x, exitPoint.y, (exitDrop > kPositionEpsilon) ? clearanceZ : exitPoint.z};

        if (exitDrop > kPositionEpsilon)
        {
            if (enableRamp)
            {
                const double exitHorizontal = computeRampDistance(exitDrop, rampAngleRad, minHorizontal, maxHorizontal);
                exitClear = offsetPoint(exitPoint, exitDir, exitHorizontal, clearanceZ, false);
            }
            exitPath = {exitPoint, exitClear};
        }

        pruneSequentialDuplicates(exitPath);
        appendCutPolyline(result, exitPath);

        glm::dvec3 exitSafe{exitClear.x, exitClear.y, safeZ};
        appendPolyline(result, MotionType::Rapid, {exitClear, exitSafe});

        lastSafe = exitSafe;
        haveLast = true;
    }

    toolpath.passes = std::move(result);
}

glm::dvec3 reorderPassRange(std::vector<Polyline>& polylines,
                            std::size_t begin,
                            std::size_t end,
                            const glm::dvec3* seedPosition)
{
    if (begin >= end)
    {
        return seedPosition ? *seedPosition : glm::dvec3{};
    }

    const std::size_t count = end - begin;
    if (count == 0)
    {
        return seedPosition ? *seedPosition : glm::dvec3{};
    }

    if (count == 1)
    {
        const Polyline& poly = polylines[begin];
        if (poly.pts.empty())
        {
            return seedPosition ? *seedPosition : glm::dvec3{};
        }
        return toDVec3(poly.pts.back().p);
    }

    std::vector<std::size_t> order;
    order.reserve(count);
    std::vector<bool> used(count, false);

    auto startPoint = [&](std::size_t rel) -> glm::dvec3 {
        const Polyline& poly = polylines[begin + rel];
        return poly.pts.empty() ? glm::dvec3{} : toDVec3(poly.pts.front().p);
    };
    auto endPoint = [&](std::size_t rel) -> glm::dvec3 {
        const Polyline& poly = polylines[begin + rel];
        return poly.pts.empty() ? glm::dvec3{} : toDVec3(poly.pts.back().p);
    };

    auto chooseClosest = [&](const glm::dvec3& from) -> std::size_t {
        double bestDist = std::numeric_limits<double>::max();
        std::size_t bestIndex = count;
        for (std::size_t i = 0; i < count; ++i)
        {
            if (used[i])
            {
                continue;
            }

            const double dist = horizontalDistance(from, startPoint(i));
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIndex = i;
            }
        }
        return bestIndex;
    };

    std::size_t current = 0;
    if (seedPosition)
    {
        const std::size_t candidate = chooseClosest(*seedPosition);
        if (candidate < count)
        {
            current = candidate;
        }
    }
    else
    {
        double bestMetric = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < count; ++i)
        {
            const glm::dvec3 start = startPoint(i);
            const double metric = std::abs(start.x) + std::abs(start.y);
            if (metric < bestMetric)
            {
                bestMetric = metric;
                current = i;
            }
        }
    }

    used[current] = true;
    order.push_back(current);
    glm::dvec3 cursor = endPoint(current);

    while (order.size() < count)
    {
        std::size_t next = chooseClosest(cursor);
        if (next >= count)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                if (!used[i])
                {
                    next = i;
                    break;
                }
            }
        }
        used[next] = true;
        order.push_back(next);
        cursor = endPoint(next);
    }

    std::vector<Polyline> reordered;
    reordered.reserve(count);
    for (std::size_t index : order)
    {
        reordered.push_back(std::move(polylines[begin + index]));
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        polylines[begin + i] = std::move(reordered[i]);
    }

    return cursor;
}

std::function<void(int)> makePassProgressCallback(const std::function<void(int)>& callback,
                                                  std::size_t passIndex,
                                                  std::size_t passCount)
{
    if (!callback || passCount == 0)
    {
        return {};
    }

    const double start = (static_cast<double>(passIndex) / static_cast<double>(passCount)) * 100.0;
    const double span = 100.0 / static_cast<double>(passCount);

    return [callback, start, span](int localPercent) {
        const int clamped = std::clamp(localPercent, 0, 100);
        const double normalized = static_cast<double>(clamped) / 100.0;
        double value = start + span * normalized;
        if (value >= 100.0)
        {
            value = 99.0;
        }
        callback(static_cast<int>(value));
    };
}

double cutterOffsetFor(const UserParams& params)
{
    if (params.cutterType == UserParams::CutterType::BallNose)
    {
        return std::max(0.0, params.toolDiameter * 0.5);
    }
    return 0.0;
}

double normalizeAngleDeg(double angle)
{
    double normalized = std::fmod(angle, 360.0);
    if (normalized < 0.0)
    {
        normalized += 360.0;
    }
    return normalized;
}

double selectRasterAngleDeg(const UserParams& params,
                            const ai::StrategyStep& step,
                            bool preferUserAngle)
{
    const double userAngle = params.rasterAngleDeg;
    const double aiAngle = (step.type == ai::StrategyStep::Type::Raster) ? step.angle_deg : 0.0;

    if (preferUserAngle && std::abs(userAngle) > 1e-6)
    {
        return normalizeAngleDeg(userAngle);
    }
    if (std::abs(aiAngle) > 1e-6)
    {
        return normalizeAngleDeg(aiAngle);
    }
    return normalizeAngleDeg(userAngle);
}

double computeHeightFieldResolution(double stepOverMm)
{
    const double clamped = std::max(stepOverMm, 0.1);
    return std::max(0.1, std::min(clamped * 0.5, 0.5));
}

class HeightFieldCache
{
public:
    struct Entry
    {
        const render::Model* model{nullptr};
        double resolution{0.5};
        std::size_t vertexCount{0};
        std::size_t indexCount{0};
        std::shared_ptr<heightfield::HeightField> field;
    };

    static HeightFieldCache& instance()
    {
        static HeightFieldCache cache;
        return cache;
    }

    std::shared_ptr<heightfield::HeightField> acquire(const render::Model& model,
                                                      double resolution,
                                                      const std::atomic<bool>& cancelFlag,
                                                      std::string& logMessage,
                                                      bool& reused)
    {
        reused = false;
        logMessage.clear();

        const std::size_t vertexCount = model.vertices().size();
        const std::size_t indexCount = model.indices().size();

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            for (const Entry& entry : m_entries)
            {
                if (entry.model == &model
                    && std::abs(entry.resolution - resolution) < 1e-6
                    && entry.vertexCount == vertexCount
                    && entry.indexCount == indexCount
                    && entry.field
                    && entry.field->isValid())
                {
                    reused = true;
                    std::ostringstream oss;
                    oss.setf(std::ios::fixed);
                    oss.precision(2);
                    oss << "Height field cache hit (" << entry.field->columns() << "x" << entry.field->rows()
                        << " @ " << resolution << " mm)";
                    logMessage = oss.str();
                    return entry.field;
                }
            }
        }

        if (cancelFlag.load(std::memory_order_relaxed))
        {
            return nullptr;
        }

        heightfield::UniformGrid grid(model, resolution);

        if (cancelFlag.load(std::memory_order_relaxed))
        {
            return nullptr;
        }

        auto field = std::make_shared<heightfield::HeightField>();
        heightfield::HeightField::BuildStats stats;
        if (!field->build(grid, resolution, cancelFlag, &stats))
        {
            return nullptr;
        }

        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(2);
        oss << "Height field built (" << field->columns() << "x" << field->rows()
            << " @ " << resolution << " mm, valid " << stats.validSamples << "/" << stats.totalSamples
            << ") in " << stats.buildMilliseconds << " ms";
        logMessage = oss.str();

        Entry newEntry;
        newEntry.model = &model;
        newEntry.resolution = resolution;
        newEntry.vertexCount = vertexCount;
        newEntry.indexCount = indexCount;
        newEntry.field = field;

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = std::remove_if(m_entries.begin(), m_entries.end(), [&](const Entry& entry) {
                return entry.model == &model && std::abs(entry.resolution - resolution) < 1e-6;
            });
            m_entries.erase(it, m_entries.end());
            m_entries.push_back(std::move(newEntry));
        }

        return field;
    }

private:
    HeightFieldCache() = default;

    std::mutex m_mutex;
    std::vector<Entry> m_entries;
};

void finalizeToolpath(Toolpath& toolpath, const UserParams& params)
{
    Stock stock = params.stock;
    stock.ensureValid();

    Machine machine = params.machine;
    machine.ensureValid();

    const double clearanceFloor = stock.topZ_mm + kMinClearanceOffset;
    if (machine.clearanceZ_mm < clearanceFloor)
    {
        machine.clearanceZ_mm = clearanceFloor;
    }
    if (machine.safeZ_mm <= machine.clearanceZ_mm + kMinSafeGap * 0.5)
    {
        machine.safeZ_mm = machine.clearanceZ_mm + kMinSafeGap;
    }

    toolpath.feed = (machine.maxFeed_mm_min > 0.0)
                        ? std::min(params.feed, machine.maxFeed_mm_min)
                        : params.feed;
    toolpath.spindle = (machine.maxSpindleRPM > 0.0)
                           ? std::min(params.spindle, machine.maxSpindleRPM)
                           : params.spindle;
    toolpath.rapidFeed = machine.rapidFeed_mm_min;
    toolpath.machine = machine;
    toolpath.stock = stock;

    applyMachineMotion(toolpath, machine, stock, params);
}

} // namespace

const char* ToolpathGenerator::passLabel(const PassProfile& profile)
{
    return (profile.kind == PassProfile::Kind::Rough) ? "Roughing" : "Finishing";
}

std::string ToolpathGenerator::makePassLog(const PassProfile& profile, const std::string& message)
{
    if (message.empty())
    {
        return {};
    }

    std::ostringstream oss;
    oss << passLabel(profile) << ": " << message;
    return oss.str();
}

std::vector<ToolpathGenerator::PassProfile> ToolpathGenerator::buildPassPlan(const UserParams& params,
                                                                             const ai::StrategyDecision& decision)
{
    // Clamp derived parameters to the safe range we measured on benchtop routers so AI overrides cannot
    // push cutters beyond the torque envelope when generating roughing passes.
    const double safeToolDiameter = std::max(params.toolDiameter, 0.1);
    const double userStepOver = (params.stepOver > 0.0) ? params.stepOver : safeToolDiameter * 0.4;
    const double finishStep = std::clamp(userStepOver, 0.1, safeToolDiameter * 0.45);
    double roughStep = std::max({params.stepOver, finishStep, safeToolDiameter * 0.65});
    roughStep = std::clamp(roughStep, finishStep + 0.05, safeToolDiameter);
    if (roughStep - finishStep < 0.05)
    {
        roughStep = std::min(safeToolDiameter, finishStep * 1.5);
    }

    const double baseStepDown = std::max(params.maxDepthPerPass, 0.1);
    const double finishStepDown = std::max(0.1, baseStepDown * 0.5);
    const double allowanceNominal = std::clamp(params.stockAllowance_mm, 0.0, safeToolDiameter);

    std::vector<PassProfile> plan;
    plan.reserve(decision.steps.size());

    auto pushProfile = [&](ai::StrategyStep step, std::size_t index) {
        const bool isFinish = step.finish_pass;
        const bool enabled = isFinish ? params.enableFinishPass : params.enableRoughPass;
        if (!enabled)
        {
            return;
        }
        if (!isFinish && allowanceNominal <= 1e-6)
        {
            return;
        }

        if (step.stepover <= 0.0)
        {
            step.stepover = isFinish ? finishStep : roughStep;
        }
        const double maxAllowedOver = isFinish ? safeToolDiameter * 0.6 : safeToolDiameter;
        step.stepover = std::clamp(step.stepover, 0.05, maxAllowedOver);
        ENFORCE(step.stepover > 0.0, "Strategy normalization must produce positive stepover.");

        if (step.stepdown <= 0.0)
        {
            step.stepdown = isFinish ? finishStepDown : baseStepDown;
        }
        else
        {
            step.stepdown = std::max(0.05, step.stepdown);
        }
        ENFORCE(step.stepdown > 0.0, "Strategy normalization must produce positive stepdown.");

        if (step.type == ai::StrategyStep::Type::Raster)
        {
            if (std::abs(step.angle_deg) <= 1e-6)
            {
                step.angle_deg = params.rasterAngleDeg;
            }
            step.angle_deg = normalizeAngleDeg(step.angle_deg);
        }
        else
        {
            step.angle_deg = 0.0;
        }

        PassProfile profile;
        profile.kind = isFinish ? PassProfile::Kind::Finish : PassProfile::Kind::Rough;
        profile.step = step;
        profile.allowance = isFinish ? 0.0 : allowanceNominal;
        profile.index = index;
        plan.push_back(std::move(profile));
    };

    for (std::size_t i = 0; i < decision.steps.size(); ++i)
    {
        pushProfile(decision.steps[i], i);
    }

    if (plan.empty())
    {
        ai::StrategyStep fallbackStep;
        fallbackStep.type = ai::StrategyStep::Type::Raster;
        fallbackStep.stepover = finishStep;
        fallbackStep.stepdown = finishStepDown;
        fallbackStep.angle_deg = normalizeAngleDeg(params.rasterAngleDeg);
        fallbackStep.finish_pass = true;

        PassProfile profile;
        profile.kind = PassProfile::Kind::Finish;
        profile.step = fallbackStep;
        profile.allowance = 0.0;
        profile.index = 0;
        plan.push_back(std::move(profile));
    }
    else
    {
        for (std::size_t i = 0; i < plan.size(); ++i)
        {
            plan[i].index = i;
        }
    }

    return plan;
}

Toolpath ToolpathGenerator::generate(const render::Model& model,
                                     const UserParams& params,
                                     ai::IPathAI& ai,
                                     const std::atomic<bool>& cancelFlag,
                                     const std::function<void(int)>& progressCallback,
                                     ai::StrategyDecision* outDecision,
                                     std::string* bannerMessage) const
{
    Toolpath toolpath;

    ENFORCE(params.toolDiameter > 0.0, "Tool diameter must be specified before toolpath generation.");

    if (!model.isValid())
    {
        finalizeToolpath(toolpath, params);
        return toolpath;
    }

    if (cancelFlag.load(std::memory_order_relaxed))
    {
        return Toolpath{};
    }

    if (progressCallback)
    {
        progressCallback(0);
    }

    const bool useOverride = params.useStrategyOverride && !params.strategyOverride.empty();
    ai::StrategyDecision decision;
    if (useOverride)
    {
        decision.steps = params.strategyOverride;
    }
    else
    {
        decision = ai.predict(model, params);
    }
    auto passPlan = buildPassPlan(params, decision);
    if (passPlan.empty())
    {
        if (outDecision)
        {
            outDecision->steps.clear();
        }
        finalizeToolpath(toolpath, params);
        if (progressCallback)
        {
            progressCallback(100);
        }
        if (bannerMessage)
        {
            bannerMessage->clear();
        }
        return toolpath;
    }

    ai::StrategyDecision appliedDecision;
    appliedDecision.steps.reserve(passPlan.size());
    for (const auto& profile : passPlan)
    {
        appliedDecision.steps.push_back(profile.step);
    }
    if (outDecision)
    {
        *outDecision = appliedDecision;
    }

    Toolpath aggregated;
    aggregated.strategySteps = appliedDecision.steps;
    std::string bannerText;
    std::vector<std::pair<std::size_t, std::size_t>> passRanges;
    passRanges.reserve(passPlan.size());

#if TP_WITH_OCL
    bool usedOcl = false;
    if (passPlan.size() == 1 && passPlan.front().allowance <= 1e-6)
    {
        const auto& profile = passPlan.front();
        std::string oclError;
        Toolpath oclToolpath;
        Cutter cutter;
        cutter.length = std::max(3.0 * params.toolDiameter, params.toolDiameter + 5.0);
        cutter.diameter = params.toolDiameter;
        const bool isWaterline = (profile.step.type == ai::StrategyStep::Type::Waterline);
        cutter.type = isWaterline ? Cutter::Type::BallNose : Cutter::Type::FlatEndmill;

        UserParams oclParams = params;
        oclParams.stepOver = profile.step.stepover;

        const auto oclStart = std::chrono::steady_clock::now();
        if (isWaterline)
        {
            if (OclAdapter::waterline(model, oclParams, cutter, oclToolpath, oclError))
            {
                usedOcl = true;
            }
        }
        else
        {
            if (OclAdapter::rasterDropCutter(model,
                                             oclParams,
                                             cutter,
                                             profile.step.angle_deg,
                                             oclToolpath,
                                             oclError))
            {
                usedOcl = true;
            }
        }

        if (usedOcl && !oclToolpath.empty())
        {
            aggregated = std::move(oclToolpath);
            for (auto& poly : aggregated.passes)
            {
                poly.strategyStep = static_cast<int>(profile.index);
            }
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - oclStart)
                                     .count();
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(2);
            oss << passLabel(profile) << ": OCL path generated in " << elapsed << " ms";
            bannerText = oss.str();
            passRanges.emplace_back(0, aggregated.passes.size());
        }
        else if (!oclError.empty())
        {
            bannerText = "OCL error: " + oclError;
        }
    }
#endif

    if (cancelFlag.load(std::memory_order_relaxed))
    {
        return Toolpath{};
    }

    if (aggregated.empty())
    {
        for (std::size_t passIndex = 0; passIndex < passPlan.size(); ++passIndex)
        {
            if (cancelFlag.load(std::memory_order_relaxed))
            {
                return Toolpath{};
            }

            const auto& profile = passPlan[passIndex];
            auto subProgress = makePassProgressCallback(progressCallback, passIndex, passPlan.size());
            std::string passLog;
            Toolpath passToolpath;

            if (profile.step.type == ai::StrategyStep::Type::Waterline)
            {
                passToolpath = generateWaterlineSlicer(model,
                                                       params,
                                                       profile,
                                                       cancelFlag,
                                                       subProgress,
                                                       &passLog);
            }
            else if (params.useHeightField)
            {
                passToolpath = generateRasterTopography(model,
                                                        params,
                                                        profile,
                                                        cancelFlag,
                                                        subProgress,
                                                        &passLog);
            }

            if (passToolpath.empty())
            {
                passToolpath = generateFallbackRaster(model,
                                                      params,
                                                      profile,
                                                      cancelFlag,
                                                      subProgress);
            }

            if (cancelFlag.load(std::memory_order_relaxed))
            {
                return Toolpath{};
            }

            if (!passLog.empty())
            {
                if (!bannerText.empty())
                {
                    bannerText += " | ";
                }
                bannerText += passLog;
            }

            if (!passToolpath.passes.empty())
            {
                for (auto& poly : passToolpath.passes)
                {
                    poly.strategyStep = static_cast<int>(profile.index);
                }
                const std::size_t offset = aggregated.passes.size();
                aggregated.passes.insert(aggregated.passes.end(),
                                         std::make_move_iterator(passToolpath.passes.begin()),
                                         std::make_move_iterator(passToolpath.passes.end()));
                passRanges.emplace_back(offset, aggregated.passes.size());
            }
        }
    }

    if (aggregated.passes.empty())
    {
        finalizeToolpath(aggregated, params);
        if (progressCallback)
        {
            progressCallback(100);
        }
        if (bannerMessage && !bannerText.empty())
        {
            *bannerMessage = std::move(bannerText);
        }
        return aggregated;
    }

    glm::dvec3 seed{};
    bool haveSeed = false;
    for (const auto& range : passRanges)
    {
        const glm::dvec3* seedPtr = haveSeed ? &seed : nullptr;
        seed = reorderPassRange(aggregated.passes, range.first, range.second, seedPtr);
        haveSeed = true;
    }

    applyLeaveStockAdjustment(aggregated, model, params);
    finalizeToolpath(aggregated, params);

    if (progressCallback)
    {
        progressCallback(100);
    }

    if (bannerMessage && !bannerText.empty())
    {
        *bannerMessage = std::move(bannerText);
    }

    return aggregated;
}

Toolpath ToolpathGenerator::generateRasterTopography(const render::Model& model,
                                                     const UserParams& params,
                                                     const PassProfile& profile,
                                                     const std::atomic<bool>& cancelFlag,
                                                     const std::function<void(int)>& progressCallback,
                                                     std::string* logMessage) const
{
    Toolpath toolpath;
    toolpath.feed = params.feed;
    toolpath.spindle = params.spindle;

    const auto bounds = model.bounds();
    const double minX = static_cast<double>(bounds.min.x());
    const double maxX = static_cast<double>(bounds.max.x());
    const double minY = static_cast<double>(bounds.min.y());
    const double maxY = static_cast<double>(bounds.max.y());

    if (std::abs(maxX - minX) < 1e-6 || std::abs(maxY - minY) < 1e-6)
    {
        return toolpath;
    }

    const double rowSpacing = std::max(0.1, profile.step.stepover);
    const double resolution = computeHeightFieldResolution(profile.step.stepover);

    bool reused = false;
    bool completed = false;
    const QString timerLabel = QStringLiteral("Raster pass (row=%1 mm, res=%2 mm)")
                                   .arg(rowSpacing, 0, 'f', 3)
                                   .arg(resolution, 0, 'f', 3);

    ScopedTimer timer(timerLabel,
                      [&](const QString& label, double ms, bool cancelled) {
                          const std::size_t polyCount = toolpath.passes.size();
                          if (cancelled)
                          {
                              LOG_INFO(Tp, QStringLiteral("%1 cancelled after %2 ms (polylines=%3)")
                                                .arg(label)
                                                .arg(ms, 0, 'f', 2)
                                                .arg(static_cast<qulonglong>(polyCount)));
                              return;
                          }
                          if (!completed)
                          {
                              LOG_WARN(Tp, QStringLiteral("%1 aborted after %2 ms (polylines=%3, heightfield=%4). "
                                                          "Review strategy settings before retrying.")
                                                .arg(label)
                                                .arg(ms, 0, 'f', 2)
                                                .arg(static_cast<qulonglong>(polyCount))
                                                .arg(reused ? QStringLiteral("reused")
                                                            : QStringLiteral("rebuilt")));
                              return;
                          }
                          LOG_INFO(Tp, QStringLiteral("%1 finished in %2 ms (polylines=%3, heightfield=%4)")
                                            .arg(label)
                                            .arg(ms, 0, 'f', 2)
                                            .arg(static_cast<qulonglong>(polyCount))
                                            .arg(reused ? QStringLiteral("reused") : QStringLiteral("rebuilt")));
                      },
                      &cancelFlag);

    std::string cacheLog;
    auto heightField = HeightFieldCache::instance().acquire(model, resolution, cancelFlag, cacheLog, reused);
    if (logMessage)
    {
        *logMessage = makePassLog(profile, cacheLog);
    }
    if (!heightField || !heightField->isValid())
    {
        return Toolpath{};
    }

    const double cutterOffset = cutterOffsetFor(params);
    const double topZ = params.stock.topZ_mm;
    const double maxDepthPerPass = std::max(profile.step.stepdown, 0.1);

    const double angleDeg = selectRasterAngleDeg(params, profile.step, true);
    const double angleRad = angleDeg * std::numbers::pi / 180.0;
    const double cosA = std::cos(angleRad);
    const double sinA = std::sin(angleRad);

    const auto rotate2D = [cosA, sinA](double x, double y) -> std::pair<double, double> {
        return {x * cosA - y * sinA, x * sinA + y * cosA};
    };

    const auto unrotate2D = [cosA, sinA](double xr, double yr) -> std::pair<double, double> {
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

    const double spanXRot = std::max(1e-6, maxXRot - minXRot);
    const double spanYRot = std::max(1e-6, maxYRot - minYRot);

    const int rows = std::max(1, static_cast<int>(std::ceil(spanYRot / rowSpacing)));
    const int totalIterations = rows + 1;

    struct SamplePoint
    {
        double x;
        double y;
        double z;
    };

    std::vector<SamplePoint> segmentPoints;
    segmentPoints.reserve(256);

    const auto flushSegment = [&](std::vector<SamplePoint>& points) {
        if (points.size() < 2)
        {
            points.clear();
            return;
        }

        double minZ = topZ;
        for (const auto& p : points)
        {
            minZ = std::min(minZ, p.z);
        }

        std::vector<double> levels;
        double currentLevel = topZ - maxDepthPerPass;
        while (currentLevel > minZ + 1e-6)
        {
            levels.push_back(currentLevel);
            currentLevel -= maxDepthPerPass;
        }
        levels.push_back(minZ);

        for (double level : levels)
        {
            Polyline poly;
            poly.motion = MotionType::Cut;
            poly.strategyStep = static_cast<int>(profile.index);
            poly.pts.reserve(points.size());

            for (const auto& p : points)
            {
                double cutZ = (level == minZ) ? p.z : std::max(p.z, level);
                poly.pts.push_back({glm::vec3(static_cast<float>(p.x),
                                              static_cast<float>(p.y),
                                              static_cast<float>(cutZ))});
            }

            if (params.cutDirection == UserParams::CutDirection::Conventional)
            {
                std::reverse(poly.pts.begin(), poly.pts.end());
            }

            toolpath.passes.push_back(std::move(poly));
        }

        points.clear();
    };

    for (int row = 0; row <= rows; ++row)
    {
        if (cancelFlag.load(std::memory_order_relaxed))
        {
            return Toolpath{};
        }

        const double yRot = std::min(minYRot + static_cast<double>(row) * rowSpacing, maxYRot);
        const bool leftToRight = (row % 2) == 0;
        const double startXRot = leftToRight ? minXRot : maxXRot;
        const double endXRot = leftToRight ? maxXRot : minXRot;
        const double spanX = std::abs(endXRot - startXRot);
        const int steps = std::max(1, static_cast<int>(std::ceil(spanX / resolution)));

        segmentPoints.clear();

        for (int step = 0; step <= steps; ++step)
        {
            if (cancelFlag.load(std::memory_order_relaxed))
            {
                return Toolpath{};
            }

            const double t = static_cast<double>(step) / static_cast<double>(steps);
            double xRot = 0.0;
            if (leftToRight)
            {
                xRot = std::min(startXRot + t * spanX, maxXRot);
            }
            else
            {
                xRot = std::max(startXRot - t * spanX, minXRot);
            }

            const auto xy = unrotate2D(xRot, yRot);
            const double sampleX = xy.first;
            const double sampleY = xy.second;

            double sampleZ = 0.0;
            if (heightField->interpolate(sampleX, sampleY, sampleZ))
            {
                double targetZ = sampleZ + cutterOffset + profile.allowance;
                targetZ = std::min(targetZ, topZ);
                segmentPoints.push_back({sampleX, sampleY, targetZ});
            }
            else
            {
                flushSegment(segmentPoints);
            }
        }

        flushSegment(segmentPoints);

        if (progressCallback)
        {
            const int percent = std::clamp(static_cast<int>(((row + 1) * 100.0) / totalIterations), 0, 99);
            progressCallback(percent);
        }
    }

    if (progressCallback)
    {
        progressCallback(100);
    }

    completed = true;
    return toolpath;
}

Toolpath ToolpathGenerator::generateWaterlineSlicer(const render::Model& model,
                                                    const UserParams& params,
                                                    const PassProfile& profile,
                                                    const std::atomic<bool>& cancelFlag,
                                                    const std::function<void(int)>& progressCallback,
                                                    std::string* logMessage) const
{
    Toolpath toolpath;
    toolpath.feed = params.feed;
    toolpath.spindle = params.spindle;

    if (!model.isValid())
    {
        return toolpath;
    }

    const auto bounds = model.bounds();
    const double minZ = static_cast<double>(bounds.min.z());
    const double maxZ = static_cast<double>(bounds.max.z());

    if (maxZ - minZ <= 1e-4)
    {
        return toolpath;
    }

    const double stepDown = std::max(profile.step.stepdown, 0.1);
    const double allowance = profile.allowance;
    const double topZ = params.stock.topZ_mm;
    const double toolRadius = (params.cutterType == UserParams::CutterType::FlatEndmill)
                                  ? params.toolDiameter * 0.5
                                  : 0.0;

    tp::waterline::ZSlicer slicer(model, 1e-4);

    std::size_t loopCount = 0;
    int levelCount = 0;
    double elapsedMs = 0.0;
    bool completed = false;

    const QString timerLabel = QStringLiteral("Waterline pass (step=%1 mm, allowance=%2 mm)")
                                   .arg(stepDown, 0, 'f', 3)
                                   .arg(allowance, 0, 'f', 3);

    {
        ScopedTimer timer(timerLabel,
                          [&](const QString& label, double ms, bool cancelled) {
                              elapsedMs = ms;
                              if (cancelled)
                              {
                                  LOG_INFO(Tp, QStringLiteral("%1 cancelled after %2 ms (loops=%3)")
                                                    .arg(label)
                                                    .arg(ms, 0, 'f', 2)
                                                    .arg(static_cast<qulonglong>(loopCount)));
                                  return;
                              }
                              if (!completed)
                              {
                                  LOG_WARN(Tp, QStringLiteral("%1 aborted after %2 ms (loops=%3, levels=%4). "
                                                              "Inspect stock limits and retry.")
                                                    .arg(label)
                                                    .arg(ms, 0, 'f', 2)
                                                    .arg(static_cast<qulonglong>(loopCount))
                                                    .arg(levelCount));
                                  return;
                              }
                              LOG_INFO(Tp, QStringLiteral("%1 finished in %2 ms (loops=%3, levels=%4)")
                                                .arg(label)
                                                .arg(ms, 0, 'f', 2)
                                                .arg(static_cast<qulonglong>(loopCount))
                                                .arg(levelCount));
                          },
                          &cancelFlag);

        const double totalSpan = maxZ - minZ;
        const int totalLevels = std::max(1, static_cast<int>(std::ceil(totalSpan / stepDown))) + 1;

        int processedLevels = 0;
        const bool applyOffset = (params.cutterType == UserParams::CutterType::FlatEndmill);

        for (double planeZ = maxZ; planeZ >= minZ - 1e-6; planeZ -= stepDown)
        {
            if (cancelFlag.load(std::memory_order_relaxed))
            {
                return Toolpath{};
            }

            const auto loops = slicer.slice(planeZ, toolRadius, applyOffset);
            if (!loops.empty())
            {
                ++levelCount;
                for (const auto& loop : loops)
                {
                    if (loop.size() < 3)
                    {
                        continue;
                    }

                    Polyline poly;
                    poly.motion = MotionType::Cut;
                    poly.strategyStep = static_cast<int>(profile.index);
                    poly.pts.reserve(loop.size());
                    for (const auto& pt : loop)
                    {
                        const double targetZ = std::min(static_cast<double>(pt.z) + allowance, topZ);
                        poly.pts.push_back({glm::vec3(static_cast<float>(pt.x),
                                                       static_cast<float>(pt.y),
                                                       static_cast<float>(targetZ))});
                    }
                    if (params.cutDirection == UserParams::CutDirection::Conventional)
                    {
                        std::reverse(poly.pts.begin(), poly.pts.end());
                    }
                    toolpath.passes.push_back(std::move(poly));
                    ++loopCount;
                }
            }

            ++processedLevels;
            if (progressCallback)
            {
                const int percent = std::clamp(static_cast<int>((processedLevels * 100.0) / totalLevels), 0, 99);
                progressCallback(percent);
            }
        }

        if (progressCallback)
        {
            progressCallback(100);
        }

        if (toolpath.passes.empty())
        {
            return toolpath;
        }

        completed = true;
    }

    if (logMessage)
    {
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(2);
        oss << "Waterline slicer generated " << loopCount << " loops across " << levelCount
            << " levels in " << elapsedMs << " ms";
        *logMessage = makePassLog(profile, oss.str());
    }

    return toolpath;
}

Toolpath ToolpathGenerator::generateFallbackRaster(const render::Model& model,
                                                   const UserParams& params,
                                                   const PassProfile& profile,
                                                   const std::atomic<bool>& cancelFlag,
                                                   const std::function<void(int)>& progressCallback) const
{
    Toolpath toolpath;
    toolpath.feed = params.feed;
    toolpath.spindle = params.spindle;

    const auto bounds = model.bounds();
    const float minX = bounds.min.x();
    const float maxX = bounds.max.x();
    const float minY = bounds.min.y();
    const float maxY = bounds.max.y();
    const float minZ = bounds.min.z();

    if (std::abs(maxX - minX) < 1e-4f || std::abs(maxY - minY) < 1e-4f)
    {
        return toolpath;
    }

    const double allowance = profile.allowance;
    const double topZ = params.stock.topZ_mm;
    const float cutPlane = static_cast<float>(std::min(static_cast<double>(minZ) + allowance, topZ));
    const float step = clampStepOver(profile.step.stepover);

    const double angleDeg = selectRasterAngleDeg(params, profile.step, false);
    const double angleRad = angleDeg * std::numbers::pi / 180.0;
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
    const int totalIterations = rows + 1;

    for (int row = 0; row <= rows; ++row)
    {
        if (cancelFlag.load(std::memory_order_relaxed))
        {
            return Toolpath{};
        }

        const float yRot = std::min(minYRot + static_cast<float>(row) * step, maxYRot);
        const bool leftToRight = (row % 2) == 0;

        const float startXRot = leftToRight ? minXRot : maxXRot;
        const float endXRot = leftToRight ? maxXRot : minXRot;

        const auto startCutXY = unrotate2D(startXRot, yRot);
        const auto endCutXY = unrotate2D(endXRot, yRot);

        const glm::vec3 startCut{startCutXY.first, startCutXY.second, cutPlane};
        const glm::vec3 endCut{endCutXY.first, endCutXY.second, cutPlane};

        Polyline cut;
        cut.motion = MotionType::Cut;
        cut.strategyStep = static_cast<int>(profile.index);
        cut.pts.push_back({startCut});
        cut.pts.push_back({endCut});
        if (params.cutDirection == UserParams::CutDirection::Conventional)
        {
            std::reverse(cut.pts.begin(), cut.pts.end());
        }
        toolpath.passes.push_back(std::move(cut));

        if (progressCallback)
        {
            const int percent = std::clamp(static_cast<int>(((row + 1) * 100.0) / totalIterations), 0, 99);
            progressCallback(percent);
        }
    }

    if (progressCallback)
    {
        progressCallback(100);
    }

    return toolpath;
}

void ToolpathGenerator::applyLeaveStockAdjustment(Toolpath& toolpath,
                                                  const render::Model& model,
                                                  const UserParams& params) const
{
    if (toolpath.passes.empty() || params.leaveStock_mm <= 1e-6)
    {
        return;
    }

    GougeChecker checker(model);

    for (Polyline& poly : toolpath.passes)
    {
        if (poly.motion != MotionType::Cut || poly.pts.size() < 2)
        {
            continue;
        }

        for (Vertex& vertex : poly.pts)
        {
            GougeChecker::Vec3 sample = vertex.p;
            sample.z = static_cast<float>(params.stock.topZ_mm + 1.0);
            const auto surfaceZOpt = checker.surfaceHeightAt(sample);
            if (!surfaceZOpt)
            {
                continue;
            }

            double desiredZ = *surfaceZOpt + params.leaveStock_mm;
            if (params.machine.safeZ_mm > 0.0)
            {
                desiredZ = std::min(desiredZ, params.machine.safeZ_mm);
            }

            const double currentZ = static_cast<double>(vertex.p.z);
            if (desiredZ <= currentZ + 1e-6)
            {
                continue;
            }

            vertex.p.z = static_cast<float>(desiredZ);
        }
    }
}

} // namespace tp
