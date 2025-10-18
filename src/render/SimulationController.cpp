#include "render/SimulationController.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
constexpr double kMinSpeedMmPerS = 1.0;
constexpr double kDefaultRapidMmPerMin = 9000.0;

double distanceBetween(const QVector3D& a, const QVector3D& b)
{
    return std::sqrt(std::pow(b.x() - a.x(), 2.0) + std::pow(b.y() - a.y(), 2.0) + std::pow(b.z() - a.z(), 2.0));
}

} // namespace

namespace render
{

SimulationController::SimulationController(QObject* parent)
    : QObject(parent)
{
    m_timer.setInterval(16);
    m_timer.setTimerType(Qt::PreciseTimer);
    connect(&m_timer, &QTimer::timeout, this, &SimulationController::onTick);
}

void SimulationController::setToolpath(std::shared_ptr<tp::Toolpath> toolpath)
{
    stop();
    m_toolpath = std::move(toolpath);
    rebuildSegments();
}

void SimulationController::setToolDiameter(double diameterMm)
{
    m_toolDiameter = std::max(0.1, diameterMm);
    emitPosition(hasPath());
}

void SimulationController::play()
{
    if (m_segments.empty())
    {
        return;
    }

    if (m_state == State::Stopped)
    {
        m_currentTime = 0.0;
        m_currentSegment = 0;
        emitPosition(true);
        Q_EMIT progressChanged(progress());
    }

    if (m_state == State::Playing)
    {
        return;
    }

    m_state = State::Playing;
    m_elapsed.restart();
    m_timer.start();
    Q_EMIT stateChanged(m_state);
}

void SimulationController::pause()
{
    if (m_state == State::Playing)
    {
        m_timer.stop();
        m_state = State::Paused;
        Q_EMIT stateChanged(m_state);
    }
}

void SimulationController::stop()
{
    if (m_state == State::Stopped && m_currentTime == 0.0)
    {
        emitPosition(hasPath());
        Q_EMIT progressChanged(0.0);
        return;
    }

    m_timer.stop();
    m_state = State::Stopped;
    m_currentTime = 0.0;
    m_currentSegment = 0;
    emitPosition(hasPath());
    Q_EMIT progressChanged(0.0);
    Q_EMIT stateChanged(m_state);
}

void SimulationController::setSpeedMultiplier(double multiplier)
{
    const double clamped = std::clamp(multiplier, 0.1, 8.0);
    m_speedMultiplier = clamped;
}

void SimulationController::setProgress(double normalized)
{
    if (m_segments.empty())
    {
        return;
    }

    const double clamped = std::clamp(normalized, 0.0, 1.0);
    m_currentTime = clamped * m_totalDuration;
    updateSegmentFromTime();
    emitPosition(true);
    Q_EMIT progressChanged(progress());
}

double SimulationController::progress() const noexcept
{
    if (m_totalDuration <= std::numeric_limits<double>::epsilon())
    {
        return 0.0;
    }
    return std::clamp(m_currentTime / m_totalDuration, 0.0, 1.0);
}

void SimulationController::onTick()
{
    if (m_state != State::Playing || m_segments.empty())
    {
        return;
    }

    const double elapsedSeconds = std::max(0.0, m_elapsed.restart() / 1000.0);
    const double delta = elapsedSeconds * m_speedMultiplier;
    m_currentTime += delta;

    if (m_currentTime >= m_totalDuration)
    {
        m_currentTime = m_totalDuration;
        updateSegmentFromTime();
        emitPosition(true);
        Q_EMIT progressChanged(progress());
        stop();
        return;
    }

    updateSegmentFromTime();
    emitPosition(true);
    Q_EMIT progressChanged(progress());
}

void SimulationController::rebuildSegments()
{
    m_segments.clear();
    m_totalDuration = 0.0;
    m_currentTime = 0.0;
    m_currentSegment = 0;

    if (!m_toolpath || m_toolpath->empty())
    {
        Q_EMIT progressChanged(0.0);
        emitPosition(false);
        return;
    }

    const double cutFeed = std::max(kMinSpeedMmPerS, m_toolpath->feed / 60.0);
    const double rapidSource = (m_toolpath->machine.rapidFeed_mm_min > 0.0)
                                   ? m_toolpath->machine.rapidFeed_mm_min
                                   : kDefaultRapidMmPerMin;
    const double rapidFeed = std::max(kMinSpeedMmPerS, rapidSource / 60.0);
    m_cutSpeed = cutFeed;
    m_rapidSpeed = rapidFeed;

    double cumulative = 0.0;
    for (const tp::Polyline& poly : m_toolpath->passes)
    {
        if (poly.pts.size() < 2)
        {
            continue;
        }

        for (std::size_t i = 1; i < poly.pts.size(); ++i)
        {
            const tp::Vertex& prev = poly.pts[i - 1];
            const tp::Vertex& curr = poly.pts[i];

            QVector3D start(prev.p.x, prev.p.y, prev.p.z);
            QVector3D end(curr.p.x, curr.p.y, curr.p.z);
            const double length = distanceBetween(start, end);
            if (length <= std::numeric_limits<double>::epsilon())
            {
                continue;
            }

            const bool rapid = poly.motion != tp::MotionType::Cut;
            const double speed = rapid ? m_rapidSpeed : m_cutSpeed;
            const double duration = std::max(length / speed, 0.0);

            Segment segment;
            segment.start = start;
            segment.end = end;
            segment.rapid = rapid;
            segment.length = length;
            segment.duration = duration;
            segment.cumulativeStart = cumulative;
            segment.motion = poly.motion;

            m_segments.push_back(segment);
            cumulative += duration;
        }
    }

    m_totalDuration = cumulative;
    if (m_totalDuration <= std::numeric_limits<double>::epsilon())
    {
        m_segments.clear();
        Q_EMIT progressChanged(0.0);
        emitPosition(false);
        return;
    }

    Q_EMIT progressChanged(0.0);
    emitPosition(true);
}

void SimulationController::updateSegmentFromTime()
{
    if (m_segments.empty())
    {
        m_currentSegment = 0;
        return;
    }

    if (m_currentSegment >= m_segments.size())
    {
        m_currentSegment = m_segments.size() - 1;
    }

    while (m_currentSegment + 1 < m_segments.size() &&
           m_currentTime >= m_segments[m_currentSegment].cumulativeStart + m_segments[m_currentSegment].duration)
    {
        ++m_currentSegment;
    }

    while (m_currentSegment > 0 &&
           m_currentTime < m_segments[m_currentSegment].cumulativeStart)
    {
        --m_currentSegment;
    }
}

void SimulationController::emitPosition(bool visible)
{
    if (m_segments.empty() || !visible)
    {
        Q_EMIT positionChanged(QVector3D(), false, false, static_cast<float>(m_toolDiameter * 0.5));
        return;
    }

    const Segment& segment = m_segments[m_currentSegment];
    const double startTime = segment.cumulativeStart;
    const double localTime = std::clamp(m_currentTime - startTime, 0.0, segment.duration);
    const double t = segment.duration > 0.0 ? (localTime / segment.duration) : 0.0;

    const QVector3D position = segment.start + static_cast<float>(t) * (segment.end - segment.start);
    Q_EMIT positionChanged(position, segment.rapid, true, static_cast<float>(currentRadius()));
}

double SimulationController::currentRadius() const noexcept
{
    return std::max(0.2, m_toolDiameter * 0.5);
}

} // namespace render
