#include "tp/heightfield/UniformGrid.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

#include "common/logging.h"

#include <QtCore/QString>

#include <glm/geometric.hpp>

namespace tp::heightfield
{

namespace
{
constexpr double kEpsilon = 1e-9;
template <typename Vec>
inline auto lengthSquared(const Vec& v) -> decltype(glm::dot(v, v))
{
    return glm::dot(v, v);
}


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

QString formatBytes(std::size_t bytes)
{
    constexpr double kKibi = 1024.0;
    constexpr double kMebi = 1024.0 * 1024.0;
    if (bytes >= static_cast<std::size_t>(kMebi))
    {
        return QStringLiteral("%1 MiB").arg(bytes / kMebi, 0, 'f', 2);
    }
    if (bytes >= static_cast<std::size_t>(kKibi))
    {
        return QStringLiteral("%1 KiB").arg(bytes / kKibi, 0, 'f', 2);
    }
    return QStringLiteral("%1 B").arg(bytes);
}

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

    tri.centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0;
    tri.bboxMin = glm::min(glm::min(tri.v0, tri.v1), tri.v2);
    tri.bboxMax = glm::max(glm::max(tri.v0, tri.v1), tri.v2);
    tri.maxZ = std::max({tri.v0.z, tri.v1.z, tri.v2.z});
    tri.minZ = std::min({tri.v0.z, tri.v1.z, tri.v2.z});

    const double r0 = lengthSquared(tri.v0 - tri.centroid);
    const double r1 = lengthSquared(tri.v1 - tri.centroid);
    const double r2 = lengthSquared(tri.v2 - tri.centroid);
    tri.boundingRadiusSq = std::max({r0, r1, r2});
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

    const std::size_t triangleCount = model.indices().size() / 3;
    m_triangles.reserve(triangleCount);

    std::vector<std::uint32_t> cellCounts(m_columns * m_rows, 0);

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
                ++cellCounts[cellIndex];
            }
        }
    }

    std::vector<std::uint32_t> offsets(m_columns * m_rows, 0);
    std::exclusive_scan(cellCounts.begin(), cellCounts.end(), offsets.begin(), 0u);

    const std::uint32_t totalEntries = offsets.back() + cellCounts.back();
    m_cellTriangleIndices.resize(totalEntries);
    m_cellRanges.resize(m_columns * m_rows);

    std::vector<std::uint32_t> writeCursor = offsets;

    for (std::uint32_t tri = 0; tri < static_cast<std::uint32_t>(m_triangles.size()); ++tri)
    {
        const Triangle& triangle = m_triangles[tri];

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
                const std::uint32_t cursor = writeCursor[cellIndex]++;
                m_cellTriangleIndices[cursor] = tri;
            }
        }
    }

    for (std::size_t cell = 0; cell < m_cellRanges.size(); ++cell)
    {
        const std::uint32_t offset = offsets[cell];
        const std::uint32_t count = cellCounts[cell];
        m_cellRanges[cell] = CellRange{offset, count};

        auto begin = m_cellTriangleIndices.begin() + offset;
        auto end = begin + count;
        std::sort(begin, end, [this](std::uint32_t lhs, std::uint32_t rhs) {
            const double lhsMax = m_triangles[lhs].maxZ;
            const double rhsMax = m_triangles[rhs].maxZ;
            if (std::abs(lhsMax - rhsMax) < kEpsilon)
            {
                return lhs < rhs;
            }
            return lhsMax > rhsMax;
        });
    }

    const std::size_t triangleBytes = m_triangles.size() * sizeof(Triangle);
    const std::size_t indexBytes = m_cellTriangleIndices.size() * sizeof(std::uint32_t);
    const std::size_t rangeBytes = m_cellRanges.size() * sizeof(CellRange);
    const QString summary = QStringLiteral("UniformGrid: %1x%2 cells (%3 total) for %4 triangles. Memory ~ %5 "
                                           "(triangles=%6, indices=%7, ranges=%8)")
                                .arg(m_columns)
                                .arg(m_rows)
                                .arg(m_columns * m_rows)
                                .arg(m_triangles.size())
                                .arg(formatBytes(triangleBytes + indexBytes + rangeBytes))
                                .arg(formatBytes(triangleBytes))
                                .arg(formatBytes(indexBytes))
                                .arg(formatBytes(rangeBytes));
    common::logInfo(summary);
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
        const CellRange& range = m_cellRanges[cellIndex];
        const std::uint32_t* begin = m_cellTriangleIndices.data() + range.offset;
        const std::uint32_t* end = begin + range.count;
        bool hit = false;
        for (const std::uint32_t* it = begin; it != end; ++it)
        {
            const Triangle& tri = m_triangles[*it];

            if (tri.maxZ + kEpsilon < currentMax)
            {
                break;
            }

            const double dx = x - tri.centroid.x;
            const double dy = y - tri.centroid.y;
            if ((dx * dx + dy * dy) > tri.boundingRadiusSq + kEpsilon)
            {
                continue;
            }

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
