#pragma once

#include <QtGui/QMatrix4x4>
#include <QtGui/QVector3D>

namespace common
{

struct Bounds
{
    QVector3D min{0.0f, 0.0f, 0.0f};
    QVector3D max{0.0f, 0.0f, 0.0f};

    [[nodiscard]] QVector3D center() const
    {
        return (min + max) * 0.5f;
    }

    [[nodiscard]] QVector3D size() const
    {
        return max - min;
    }
};

QMatrix4x4 perspectiveRadians(float fovRadians, float aspect, float nearPlane, float farPlane);

} // namespace common

