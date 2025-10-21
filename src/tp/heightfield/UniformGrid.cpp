#include "tp/heightfield/UniformGrid.h"

#include "common/log.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <QtCore/QString>

namespace tp::heightfield
{

namespace
{

constexpr double kEpsilon = 1e-9;
constexpr double kBarycentricEpsilon = 1e-7;

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

} // namespace

UniformGrid::UniformGrid(const render::Model& model, double cellSizeMm)
    : m_grid(model, std::max(0.1, cellSizeMm))
    , m_cellSize(std::max(0.1, cellSizeMm))
{
    m_queryBuffer.reserve(128);

    const std::size_t cellCount = std::max<std::size_t>(1, m_grid.cellCount());
    const std::size_t triangleBytes = m_grid.triangleCount() * sizeof(TriangleGrid::Triangle);
    const std::size_t indexBytes = m_grid.cellIndexCount() * sizeof(std::uint32_t);
    const QString summary = QStringLiteral("UniformGrid: %1x%2 cells (%3 total) for %4 triangles. Memory ~ %5 "
                                           "(triangles=%6, indices=%7)")
                                .arg(std::max(1, m_grid.cellsX()))
                                .arg(std::max(1, m_grid.cellsY()))
                                .arg(static_cast<qulonglong>(cellCount))
                                .arg(static_cast<qulonglong>(m_grid.triangleCount()))
                                .arg(formatBytes(triangleBytes + indexBytes))
                                .arg(formatBytes(triangleBytes))
                                .arg(formatBytes(indexBytes));
    LOG_INFO(Tp, summary);
}

bool UniformGrid::intersect(const TriangleGrid::Triangle& tri, double x, double y, double& zOut) const
{
    if (!tri.validNormalZ || !tri.validBarycentric)
    {
        return false;
    }

    const double z = tri.planeHeightAt(x, y);
    if (!std::isfinite(z))
    {
        return false;
    }
    if (z < tri.minZ - kEpsilon || z > tri.maxZ + kEpsilon)
    {
        return false;
    }

    const glm::dvec3 point{x, y, z};
    if (!tri.barycentricContains(point, kBarycentricEpsilon))
    {
        return false;
    }

    zOut = z;
    return true;
}

bool UniformGrid::sampleMaxZAtXY(double x, double y, double& zOut) const
{
    if (x < minX() - kEpsilon || x > maxX() + kEpsilon || y < minY() - kEpsilon || y > maxY() + kEpsilon)
    {
        return false;
    }

    const auto evaluate = [&](int radius) -> bool {
        m_grid.gatherCandidatesXY(x, y, radius, m_queryBuffer);
        if (m_queryBuffer.empty())
        {
            return false;
        }

        std::sort(m_queryBuffer.begin(), m_queryBuffer.end(), [this](std::uint32_t lhs, std::uint32_t rhs) {
            const double lhsMax = m_grid.triangle(lhs).maxZ;
            const double rhsMax = m_grid.triangle(rhs).maxZ;
            if (std::abs(lhsMax - rhsMax) < kEpsilon)
            {
                return lhs < rhs;
            }
            return lhsMax > rhsMax;
        });

        double currentMax = -std::numeric_limits<double>::infinity();
        bool hit = false;

        for (std::uint32_t idx : m_queryBuffer)
        {
            const TriangleGrid::Triangle& tri = m_grid.triangle(idx);

            if (tri.maxZ + kEpsilon < currentMax)
            {
                continue;
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
        m_queryBuffer.clear();
        return hit;
    };

    if (evaluate(0))
    {
        return true;
    }
    return evaluate(1);
}

} // namespace tp::heightfield
