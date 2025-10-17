#pragma once

#include "common/math.h"

#include <QtCore/QByteArray>
#include <QtCore/QString>
#include <QtGui/QVector3D>

#include <vector>

namespace render
{

struct Vertex
{
    QVector3D position;
    QVector3D normal;
};

class Model
{
public:
    using Index = quint32;

    Model() = default;

    void setName(QString name);
    [[nodiscard]] const QString& name() const;

    void setMeshData(std::vector<Vertex> vertices, std::vector<Index> indices);

    [[nodiscard]] const std::vector<Vertex>& vertices() const;
    [[nodiscard]] const std::vector<Index>& indices() const;
    [[nodiscard]] bool isValid() const;

    [[nodiscard]] common::Bounds bounds() const;

    [[nodiscard]] QByteArray toObjFormat() const;

private:
    QString m_name;
    std::vector<Vertex> m_vertices;
    std::vector<Index> m_indices;
};

} // namespace render

