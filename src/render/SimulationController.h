#pragma once

#include "tp/Toolpath.h"

#include <QtCore/QElapsedTimer>
#include <QtCore/QMetaType>
#include <QtCore/QObject>
#include <QtCore/QTimer>
#include <QtGui/QVector3D>

#include <memory>
#include <vector>

namespace render
{

class SimulationController : public QObject
{
    Q_OBJECT

public:
    enum class State
    {
        Stopped,
        Playing,
        Paused
    };

    explicit SimulationController(QObject* parent = nullptr);

    void setToolpath(std::shared_ptr<tp::Toolpath> toolpath);
    void setToolDiameter(double diameterMm);

    void play();
    void pause();
    void stop();

    void setSpeedMultiplier(double multiplier);
    [[nodiscard]] double speedMultiplier() const noexcept { return m_speedMultiplier; }

    void setProgress(double normalized);
    [[nodiscard]] double progress() const noexcept;

    [[nodiscard]] State state() const noexcept { return m_state; }
    [[nodiscard]] bool hasPath() const noexcept { return !m_segments.empty(); }

Q_SIGNALS:
    void positionChanged(const QVector3D& position, bool rapid, bool visible, float radius);
    void progressChanged(double normalized);
    void stateChanged(render::SimulationController::State state);

private Q_SLOTS:
    void onTick();

private:
    struct Segment
    {
        QVector3D start;
        QVector3D end;
        bool rapid{false};
        tp::MotionType motion{tp::MotionType::Cut};
        double length{0.0};
        double duration{0.0};
        double cumulativeStart{0.0};
    };

    void rebuildSegments();
    void updateSegmentFromTime();
    void emitPosition(bool visible);
    double currentRadius() const noexcept;

    std::shared_ptr<tp::Toolpath> m_toolpath;
    std::vector<Segment> m_segments;

    QTimer m_timer;
    QElapsedTimer m_elapsed;

    double m_speedMultiplier{1.0};
    double m_totalDuration{0.0};
    double m_currentTime{0.0};
    double m_toolDiameter{6.0};
    double m_cutSpeed{20.0};  // mm/s
    double m_rapidSpeed{120.0}; // mm/s

    std::size_t m_currentSegment{0};

    State m_state{State::Stopped};
};

} // namespace render

Q_DECLARE_METATYPE(render::SimulationController::State)
