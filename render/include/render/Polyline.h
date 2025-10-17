#pragma once

#include <QtGui/QVector3D>

#include <vector>

namespace render
{

enum class PolylineType
{
    Cut,
    Rapid
};

struct Polyline
{
    PolylineType type{PolylineType::Cut};
    QVector3D color{1.0f, 1.0f, 1.0f};
    std::vector<QVector3D> points;
};

} // namespace render

