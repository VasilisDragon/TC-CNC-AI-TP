#include "tp/ToolpathGenerator.h"

#include "render/Model.h"
#include "tp/heightfield/HeightField.h"
#include "tp/heightfield/UniformGrid.h"
#include "tp/ocl/OclAdapter.h"
#include "tp/waterline/ZSlicer.h"

#include <array>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <limits>
#include <mutex>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>

#include <glm/geometric.hpp>
#include <glm/glm.hpp>

#include <algorithm>

namespace tp
{

namespace
{

constexpr double kMinClearanceOffset = 0.25;
constexpr double kMinSafeGap = 0.5;
constexpr double kPositionEpsilon = 1e-4;

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

void applyMachineMotion(Toolpath& toolpath, const Machine& machine, const Stock& stock)
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

    std::vector<Polyline> result;
    result.reserve(toolpath.passes.size() * 4);

    glm::dvec3 lastSafe{};
    glm::dvec3 lastClear{};
    bool haveLast = false;

    for (const Polyline& poly : toolpath.passes)
    {
        if (poly.motion != MotionType::Cut || poly.pts.size() < 2)
        {
            continue;
        }

        std::vector<glm::dvec3> cutPoints;
        cutPoints.reserve(poly.pts.size());
        for (const Vertex& vertex : poly.pts)
        {
            cutPoints.emplace_back(toDVec3(vertex.p));
        }

        const glm::dvec3& startCut = cutPoints.front();
        const glm::dvec3& endCut = cutPoints.back();

        glm::dvec3 startClear{startCut.x, startCut.y, clearanceZ};
        glm::dvec3 endClear{endCut.x, endCut.y, clearanceZ};
        glm::dvec3 startSafe{startCut.x, startCut.y, safeZ};
        glm::dvec3 endSafe{endCut.x, endCut.y, safeZ};

        if (!haveLast)
        {
            appendPolyline(result, MotionType::Link, {startSafe, startClear});
        }
        else
        {
            appendPolyline(result, MotionType::Link, {lastSafe, lastClear});
            appendPolyline(result, MotionType::Link, {lastClear, startClear});
        }

        appendPolyline(result, MotionType::Link, {startClear, startCut});

        Polyline cutPoly;
        cutPoly.motion = MotionType::Cut;
        cutPoly.pts.reserve(cutPoints.size());
        for (const auto& pt : cutPoints)
        {
            cutPoly.pts.push_back({toVec3(pt)});
        }
        result.push_back(std::move(cutPoly));

        appendPolyline(result, MotionType::Link, {endCut, endClear});
        appendPolyline(result, MotionType::Rapid, {endClear, endSafe});

        lastClear = endClear;
        lastSafe = endSafe;
        haveLast = true;
    }

    toolpath.passes = std::move(result);
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
                            const ai::StrategyDecision& decision,
                            bool preferUserAngle)
{
    const double userAngle = params.rasterAngleDeg;
    const double aiAngle = decision.rasterAngleDeg;

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

    applyMachineMotion(toolpath, machine, stock);
}

} // namespace

Toolpath ToolpathGenerator::generate(const render::Model& model,
                                     const UserParams& params,
                                     ai::IPathAI& ai,
                                     const std::atomic<bool>& cancelFlag,
                                     const std::function<void(int)>& progressCallback,
                                     ai::StrategyDecision* outDecision,
                                     std::string* bannerMessage) const
{
    Toolpath toolpath;

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

    ai::StrategyDecision decision = ai.predict(model, params);
    if (outDecision)
    {
        *outDecision = decision;
    }

    Toolpath generated;
    std::string bannerText;

#if TP_WITH_OCL
    bool usedOcl = false;
    std::string oclError;
    Toolpath oclToolpath;
    Cutter cutter;
    cutter.length = std::max(3.0 * params.toolDiameter, params.toolDiameter + 5.0);
    cutter.diameter = params.toolDiameter;
    cutter.type = (decision.strat == ai::StrategyDecision::Strategy::Waterline)
                      ? Cutter::Type::BallNose
                      : Cutter::Type::FlatEndmill;

    UserParams oclParams = params;
    if (decision.stepOverMM > 0.0)
    {
        oclParams.stepOver = decision.stepOverMM;
    }

    const auto oclStart = std::chrono::steady_clock::now();
    if (decision.strat == ai::StrategyDecision::Strategy::Waterline)
    {
        if (OclAdapter::waterline(model, oclParams, cutter, oclToolpath, oclError))
        {
            usedOcl = true;
        }
    }
    else if (decision.strat == ai::StrategyDecision::Strategy::Raster)
    {
        if (OclAdapter::rasterDropCutter(model, oclParams, cutter, decision.rasterAngleDeg, oclToolpath, oclError))
        {
            usedOcl = true;
        }
    }

    if (usedOcl && !oclToolpath.empty())
    {
        generated = std::move(oclToolpath);
        generated.feed = params.feed;
        generated.spindle = params.spindle;
        if (decision.strat == ai::StrategyDecision::Strategy::Waterline)
        {
            if (!oclError.empty())
            {
                if (!bannerText.empty())
                {
                    bannerText += " | ";
                }
                bannerText += oclError;
            }
        }
        else
        {
            const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - oclStart)
                                     .count();
            std::ostringstream oss;
            oss.precision(2);
            oss << std::fixed << "OCL drop-cutter path generated in " << elapsed << " ms";
            bannerText = oss.str();
        }
    }
    else if (!oclError.empty())
    {
        if (!bannerText.empty())
        {
            bannerText += " | ";
        }
        bannerText += "OCL error: " + oclError;
    }
#endif

    if (generated.empty())
    {
        if (decision.strat == ai::StrategyDecision::Strategy::Waterline)
        {
            std::string slicerLog;
            generated = generateWaterlineSlicer(model, params, decision, cancelFlag, progressCallback, &slicerLog);
            if (!slicerLog.empty())
            {
                if (!bannerText.empty())
                {
                    bannerText += " | ";
                }
                bannerText += slicerLog;
            }
        }
        else if (params.useHeightField)
        {
            std::string hfLog;
            generated = generateRasterTopography(model, params, decision, cancelFlag, progressCallback, &hfLog);
            if (!hfLog.empty())
            {
                if (!bannerText.empty())
                {
                    bannerText += " | ";
                }
                bannerText += hfLog;
            }
        }
    }

    if (generated.empty())
    {
        generated = generateFallbackRaster(model, params, decision, cancelFlag, progressCallback);
    }

    finalizeToolpath(generated, params);

    if (progressCallback)
    {
        progressCallback(100);
    }

    if (bannerMessage && !bannerText.empty())
    {
        *bannerMessage = std::move(bannerText);
    }

    return generated;
}

Toolpath ToolpathGenerator::generateRasterTopography(const render::Model& model,
                                                     const UserParams& params,
                                                     const ai::StrategyDecision& decision,
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

    const double stepOver = (decision.stepOverMM > 0.0) ? decision.stepOverMM : params.stepOver;
    const double rowSpacing = std::max(0.1, stepOver);
    const double resolution = computeHeightFieldResolution(stepOver);

    bool reused = false;
    std::string cacheLog;
    auto heightField = HeightFieldCache::instance().acquire(model, resolution, cancelFlag, cacheLog, reused);
    if (logMessage)
    {
        *logMessage = cacheLog;
    }
    if (!heightField || !heightField->isValid())
    {
        return Toolpath{};
    }

    const double cutterOffset = cutterOffsetFor(params);
    const double topZ = params.stock.topZ_mm;
    const double maxDepthPerPass = std::max(params.maxDepthPerPass, 0.1);

    const double angleDeg = selectRasterAngleDeg(params, decision, true);
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
            poly.pts.reserve(points.size());

            for (const auto& p : points)
            {
                double cutZ = (level == minZ) ? p.z : std::max(p.z, level);
                poly.pts.push_back({glm::vec3(static_cast<float>(p.x),
                                              static_cast<float>(p.y),
                                              static_cast<float>(cutZ))});
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
                double targetZ = sampleZ + cutterOffset;
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

    return toolpath;
}

Toolpath ToolpathGenerator::generateWaterlineSlicer(const render::Model& model,
                                                    const UserParams& params,
                                                    const ai::StrategyDecision& decision,
                                                    const std::atomic<bool>& cancelFlag,
                                                    const std::function<void(int)>& progressCallback,
                                                    std::string* logMessage) const
{
    (void)decision;
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

    const double stepDown = std::max(params.maxDepthPerPass, 0.1);
    const double toolRadius = (params.cutterType == UserParams::CutterType::FlatEndmill)
                                  ? params.toolDiameter * 0.5
                                  : 0.0;

    tp::waterline::ZSlicer slicer(model, 1e-4);

    std::size_t loopCount = 0;
    int levelCount = 0;

    const auto startTime = std::chrono::steady_clock::now();

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
                poly.pts.reserve(loop.size());
                for (const auto& pt : loop)
                {
                    poly.pts.push_back({glm::vec3(static_cast<float>(pt.x),
                                                  static_cast<float>(pt.y),
                                                  static_cast<float>(pt.z))});
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

    if (logMessage)
    {
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startTime)
                                 .count();
        std::ostringstream oss;
        oss.setf(std::ios::fixed);
        oss.precision(2);
        oss << "Waterline slicer generated " << loopCount << " loops across " << levelCount
            << " levels in " << elapsed << " ms";
        *logMessage = oss.str();
    }

    return toolpath;
}

Toolpath ToolpathGenerator::generateFallbackRaster(const render::Model& model,
                                                   const UserParams& params,
                                                   const ai::StrategyDecision& decision,
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

    const float cutPlane = minZ + 0.5f;
    const double rawStep = (decision.stepOverMM > 0.0) ? decision.stepOverMM : params.stepOver;
    const float step = clampStepOver(rawStep);

    const double angleDeg = selectRasterAngleDeg(params, decision, false);
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
        cut.pts.push_back({startCut});
        cut.pts.push_back({endCut});
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

} // namespace tp
