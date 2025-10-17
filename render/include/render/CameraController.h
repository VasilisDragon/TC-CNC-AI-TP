#pragma once

#include "common/math.h"

#include <QtCore/QPoint>
#include <QtCore/QSize>
#include <QtGui/QMatrix4x4>
#include <QtGui/QVector3D>

namespace render
{

class CameraController
{
public:
    void setBounds(const common::Bounds& bounds);
    void setViewportSize(const QSize& size);

    void beginOrbit(const QPoint& position);
    void updateOrbit(const QPoint& position);
    void endOrbit();

    void beginPan(const QPoint& position);
    void updatePan(const QPoint& position);
    void endPan();

    void applyZoom(float deltaSteps);

    [[nodiscard]] const QMatrix4x4& viewMatrix() const;
    [[nodiscard]] const QMatrix4x4& projectionMatrix() const;
    [[nodiscard]] QVector3D cameraPosition() const;
    [[nodiscard]] float distance() const { return m_distance; }

    void reset();
    void setViewAngles(float yawRadians, float pitchRadians);
    void setDistance(float distance);

private:
    void updateViewMatrix() const;
    void updateProjectionMatrix() const;

    common::Bounds m_bounds;
    QSize m_viewportSize{1, 1};

    QVector3D m_target{0.0f, 0.0f, 0.0f};
    float m_distance{5.0f};
    float m_yaw{0.0f};
    float m_pitch{-0.5f};

    QPoint m_lastCursor{};
    bool m_isOrbiting{false};
    bool m_isPanning{false};

    mutable bool m_viewDirty{true};
    mutable bool m_projectionDirty{true};
    mutable QMatrix4x4 m_viewMatrix;
    mutable QMatrix4x4 m_projectionMatrix;
};

} // namespace render
