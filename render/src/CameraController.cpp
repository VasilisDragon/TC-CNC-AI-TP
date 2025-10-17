#include "render/CameraController.h"

#include <QtGui/QVector2D>

#include <algorithm>
#include <cmath>

namespace render
{

namespace
{
constexpr float kMinDistance = 0.1f;
constexpr float kMaxDistance = 1'000.0f;
constexpr float kOrbitSensitivity = 0.005f;
constexpr float kPanSensitivity = 0.0025f;
constexpr float kZoomFactor = 0.1f;
constexpr float kPitchLimit = 1.55334306f; // ~89 degrees in radians
}

void CameraController::setBounds(const common::Bounds& bounds)
{
    m_bounds = bounds;
    m_target = bounds.center();
    const QVector3D size = bounds.size();
    const float radius = std::max({size.x(), size.y(), size.z()});
    m_distance = std::max(radius * 2.0f, 1.0f);
    m_viewDirty = true;
}

void CameraController::setViewportSize(const QSize& size)
{
    if (size.width() <= 0 || size.height() <= 0)
    {
        return;
    }
    m_viewportSize = size;
    m_projectionDirty = true;
}

void CameraController::beginOrbit(const QPoint& position)
{
    m_isOrbiting = true;
    m_lastCursor = position;
}

void CameraController::updateOrbit(const QPoint& position)
{
    if (!m_isOrbiting)
    {
        return;
    }
    const QPoint delta = position - m_lastCursor;
    m_lastCursor = position;

    m_yaw -= delta.x() * kOrbitSensitivity;
    m_pitch -= delta.y() * kOrbitSensitivity;

    m_pitch = std::clamp(m_pitch, -kPitchLimit, kPitchLimit);

    m_viewDirty = true;
}

void CameraController::beginPan(const QPoint& position)
{
    m_isPanning = true;
    m_lastCursor = position;
}

void CameraController::endOrbit()
{
    m_isOrbiting = false;
}

void CameraController::endPan()
{
    m_isPanning = false;
}

void CameraController::updatePan(const QPoint& position)
{
    if (!m_isPanning)
    {
        return;
    }
    const QPoint delta = position - m_lastCursor;
    m_lastCursor = position;

    const float aspect = static_cast<float>(m_viewportSize.width()) / std::max(1, m_viewportSize.height());
    const float scale = m_distance * kPanSensitivity;

    QVector3D forward = cameraPosition() - m_target;
    forward.normalize();
    QVector3D right = QVector3D::crossProduct(forward, {0.0f, 0.0f, 1.0f});
    if (right.lengthSquared() < 0.0001f)
    {
        right = {1.0f, 0.0f, 0.0f};
    }
    right.normalize();
    QVector3D up = QVector3D::crossProduct(right, forward);

    m_target += (-right * delta.x() * scale * aspect) + (up * delta.y() * scale);

    m_viewDirty = true;
}

void CameraController::applyZoom(float deltaSteps)
{
    const float factor = 1.0f + deltaSteps * kZoomFactor;
    m_distance = std::clamp(m_distance * factor, kMinDistance, kMaxDistance);
    m_viewDirty = true;
}

const QMatrix4x4& CameraController::viewMatrix() const
{
    updateViewMatrix();
    return m_viewMatrix;
}

const QMatrix4x4& CameraController::projectionMatrix() const
{
    updateProjectionMatrix();
    return m_projectionMatrix;
}

QVector3D CameraController::cameraPosition() const
{
    const float cosPitch = std::cos(m_pitch);
    const float sinPitch = std::sin(m_pitch);
    const float cosYaw = std::cos(m_yaw);
    const float sinYaw = std::sin(m_yaw);

    QVector3D offset;
    offset.setX(m_distance * cosPitch * cosYaw);
    offset.setY(m_distance * cosPitch * sinYaw);
    offset.setZ(m_distance * sinPitch);

    return m_target + offset;
}

void CameraController::reset()
{
    setBounds(m_bounds);
    m_yaw = 0.0f;
    m_pitch = -0.5f;
    m_viewDirty = true;
}

void CameraController::setViewAngles(float yawRadians, float pitchRadians)
{
    m_yaw = yawRadians;
    m_pitch = std::clamp(pitchRadians, -kPitchLimit, kPitchLimit);
    m_viewDirty = true;
}

void CameraController::setDistance(float distance)
{
    m_distance = std::clamp(distance, kMinDistance, kMaxDistance);
    m_viewDirty = true;
}

void CameraController::updateViewMatrix() const
{
    if (!m_viewDirty)
    {
        return;
    }
    m_viewMatrix.setToIdentity();
    m_viewMatrix.lookAt(cameraPosition(), m_target, QVector3D{0.0f, 0.0f, 1.0f});
    m_viewDirty = false;
}

void CameraController::updateProjectionMatrix() const
{
    if (!m_projectionDirty)
    {
        return;
    }
    const float aspect = static_cast<float>(m_viewportSize.width()) / std::max(1, m_viewportSize.height());
    m_projectionMatrix = common::perspectiveRadians(45.0f * static_cast<float>(M_PI) / 180.0f, aspect, 0.1f, 5'000.0f);
    m_projectionDirty = false;
}

} // namespace render
