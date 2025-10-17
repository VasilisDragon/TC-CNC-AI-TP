#include "tp/heightfield/UniformGrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace tp::heightfield
{

namespace
{
constexpr double kEpsilon = 1e-9;

[[nodiscard]] inline int clampIndex(int value, int max)
{
    if (value < 0)
    {
        return 0;
    }
    if (value > max)
    {
        return max;
    }
    return value;
}

} // namespace

UniformGrid::Triangle UniformGrid::makeTriangle(const render::Model& model, std::size_t triIndex)
{
    const std::size_t base = triIndex * 3;
    const auto& indices = model.indices();
    const auto& vertices = model.vertices();

    const render::Model::Index i0 = indices.at(base + 0);
    const render::Model::Index i1 = indices.at(base + 1);
    const render::Model::Index i2 = indices.at(base + 2);

    Triangle tri;
    tri.v0 = glm::dvec3(vertices.at(i0).position.x(), vertices.at(i0).position.y(), vertices.at(i0).position.z());
    tri.v1 = glm::dvec3(vertices.at(i1).position.x(), vertices.at(i1).position.y(), vertices.at(i1).position.z());
    tri.v2 = glm::dvec3(vertices.at(i2).position.x(), vertices.at(i2).position.y(), vertices.at(i2).position.z());

    tri.bboxMin = glm::min(glm::min(tri.v0, tri.v1), tri.v2);
    tri.bboxMax = glm::max(glm::max(tri.v0, tri.v1), tri.v2);
    return tri;
}

UniformGrid::UniformGrid(const render::Model& model, double cellSizeMm)
{
    const auto bounds = model.bounds();
    m_boundsMin = glm::dvec2(bounds.min.x(), bounds.min.y());
    m_boundsMax = glm::dvec2(bounds.max.x(), bounds.max.y());

    m_cellSize = std::max(0.1, cellSizeMm);

    const double extentX = std::max(m_boundsMax.x - m_boundsMin.x, m_cellSize);
    const double extentY = std::max(m_boundsMax.y - m_boundsMin.y, m_cellSize);

    m_columns = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(extentX / m_cellSize)));
    m_rows = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(extentY / m_cellSize)));

    m_cells.resize(m_columns * m_rows);

    const std::size_t triangleCount = model.indices().size() / 3;
    m_triangles.reserve(triangleCount);

    for (std::size_t tri = 0; tri < triangleCount; ++tri)
    {
        Triangle triangle = makeTriangle(model, tri);
        m_triangles.push_back(triangle);

        const double minXRel = (triangle.bboxMin.x - m_boundsMin.x) / m_cellSize;
        const double maxXRel = (triangle.bboxMax.x - m_boundsMin.x) / m_cellSize;
        const double minYRel = (triangle.bboxMin.y - m_boundsMin.y) / m_cellSize;
        const double maxYRel = (triangle.bboxMax.y - m_boundsMin.y) / m_cellSize;

        const int ixMin = clampIndex(static_cast<int>(std::floor(minXRel)), static_cast<int>(m_columns) - 1);
        const int ixMax = clampIndex(static_cast<int>(std::floor(maxXRel + kEpsilon)), static_cast<int>(m_columns) - 1);
        const int iyMin = clampIndex(static_cast<int>(std::floor(minYRel)), static_cast<int>(m_rows) - 1);
        const int iyMax = clampIndex(static_cast<int>(std::floor(maxYRel + kEpsilon)), static_cast<int>(m_rows) - 1);

        for (int iy = iyMin; iy <= iyMax; ++iy)
        {
            for (int ix = ixMin; ix <= ixMax; ++ix)
            {
                const std::size_t cellIndex = static_cast<std::size_t>(iy) * m_columns + static_cast<std::size_t>(ix);
                m_cells[cellIndex].push_back(static_cast<std::uint32_t>(tri));
            }
        }
    }
}

bool UniformGrid::intersect(const Triangle& tri, double x, double y, double& zOut) const
{
    // Project to XY plane and use barycentric coordinates.
    const glm::dvec2 p{x, y};
    const glm::dvec2 a{tri.v0.x, tri.v0.y};
    const glm::dvec2 b{tri.v1.x, tri.v1.y};
    const glm::dvec2 c{tri.v2.x, tri.v2.y};

    const double denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
    if (std::abs(denom) < kEpsilon)
    {
        return false;
    }

    const double alpha = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denom;
    const double beta = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denom;
    const double gamma = 1.0 - alpha - beta;

    if (alpha < -kEpsilon || beta < -kEpsilon || gamma < -kEpsilon)
    {
        return false;
    }

    const double z = alpha * tri.v0.z + beta * tri.v1.z + gamma * tri.v2.z;
    zOut = z;
    return true;
}

bool UniformGrid::sampleMaxZAtXY(double x, double y, double& zOut) const
{
    if (x < m_boundsMin.x - kEpsilon || x > m_boundsMax.x + kEpsilon || y < m_boundsMin.y - kEpsilon
        || y > m_boundsMax.y + kEpsilon)
    {
        return false;
    }

    const double relX = (x - m_boundsMin.x) / m_cellSize;
    const double relY = (y - m_boundsMin.y) / m_cellSize;

    int ix = clampIndex(static_cast<int>(std::floor(relX)), static_cast<int>(m_columns) - 1);
    int iy = clampIndex(static_cast<int>(std::floor(relY)), static_cast<int>(m_rows) - 1);

    const auto evaluateCell = [this, x, y, &zOut](int cellX, int cellY, double& currentMax) -> bool {
        const std::size_t cellIndex = static_cast<std::size_t>(cellY) * m_columns + static_cast<std::size_t>(cellX);
        const auto& indices = m_cells[cellIndex];
        bool hit = false;
        for (std::uint32_t triIdx : indices)
        {
            const Triangle& tri = m_triangles[triIdx];
            if (x < tri.bboxMin.x - kEpsilon || x > tri.bboxMax.x + kEpsilon || y < tri.bboxMin.y - kEpsilon
                || y > tri.bboxMax.y + kEpsilon)
            {
                continue;
            }

            double zCandidate = 0.0;
            if (intersect(tri, x, y, zCandidate) && zCandidate > currentMax)
            {
                currentMax = zCandidate;
                hit = true;
            }
        }
        if (hit)
        {
            zOut = currentMax;
        }
        return hit;
    };

    double maxZ = -std::numeric_limits<double>::infinity();
    if (evaluateCell(ix, iy, maxZ))
    {
        return true;
    }

    // Check immediate neighbours if the primary cell did not contain an intersection.
    const int maxXIndex = static_cast<int>(m_columns) - 1;
    const int maxYIndex = static_cast<int>(m_rows) - 1;
    for (int ny = std::max(0, iy - 1); ny <= std::min(maxYIndex, iy + 1); ++ny)
    {
        for (int nx = std::max(0, ix - 1); nx <= std::min(maxXIndex, ix + 1); ++nx)
        {
            if (nx == ix && ny == iy)
            {
                continue;
            }
            if (evaluateCell(nx, ny, maxZ))
            {
                return true;
            }
        }
    }

    return false;
}

} // namespace tp::heightfield

