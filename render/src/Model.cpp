// Model.cpp holds the lightweight geometry container shared across the render and toolpath modules, so
// we keep serialization and bounding-box math close to the data to minimize coupling to Qt helpers.
#include "render/Model.h"

#include <QtCore/QTextStream>

#include <limits>

namespace render
{

void Model::setName(QString name)
{
    m_name = std::move(name);
}

const QString& Model::name() const
{
    return m_name;
}

void Model::setMeshData(std::vector<Vertex> vertices, std::vector<Model::Index> indices)
{
    m_vertices = std::move(vertices);
    m_indices = std::move(indices);
}

const std::vector<Vertex>& Model::vertices() const
{
    return m_vertices;
}

const std::vector<Model::Index>& Model::indices() const
{
    return m_indices;
}

bool Model::isValid() const
{
    return !m_vertices.empty() && !m_indices.empty();
}

common::Bounds Model::bounds() const
{
    common::Bounds result;
    if (m_vertices.empty())
    {
        return result;
    }

    QVector3D minPoint(std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max(),
                       std::numeric_limits<float>::max());
    QVector3D maxPoint(std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest(),
                       std::numeric_limits<float>::lowest());

    for (const Vertex& vertex : m_vertices)
    {
        minPoint.setX(std::min(minPoint.x(), vertex.position.x()));
        minPoint.setY(std::min(minPoint.y(), vertex.position.y()));
        minPoint.setZ(std::min(minPoint.z(), vertex.position.z()));

        maxPoint.setX(std::max(maxPoint.x(), vertex.position.x()));
        maxPoint.setY(std::max(maxPoint.y(), vertex.position.y()));
        maxPoint.setZ(std::max(maxPoint.z(), vertex.position.z()));
    }

    result.min = minPoint;
    result.max = maxPoint;
    return result;
}

QByteArray Model::toObjFormat() const
{
    QByteArray result;
    QTextStream stream(&result);

    stream << "# Exported from AIToolpathGenerator\n";
    stream << "o " << (m_name.isEmpty() ? QStringLiteral("model") : m_name) << '\n';

    for (const Vertex& vertex : m_vertices)
    {
        stream << "v " << vertex.position.x() << ' ' << vertex.position.y() << ' ' << vertex.position.z() << '\n';
    }

    for (const Vertex& vertex : m_vertices)
    {
        stream << "vn " << vertex.normal.x() << ' ' << vertex.normal.y() << ' ' << vertex.normal.z() << '\n';
    }

    for (std::size_t i = 0; i + 2 < m_indices.size(); i += 3)
    {
        const int i0 = static_cast<int>(m_indices[i]) + 1;
        const int i1 = static_cast<int>(m_indices[i + 1]) + 1;
        const int i2 = static_cast<int>(m_indices[i + 2]) + 1;
        stream << "f " << i0 << "//" << i0 << ' ' << i1 << "//" << i1 << ' ' << i2 << "//" << i2 << '\n';
    }

    stream.flush();
    return result;
}

} // namespace render
