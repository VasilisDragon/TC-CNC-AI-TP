#include "sim/StockGrid.h"

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

constexpr double kDegenerateLength = 1e-6;
constexpr double kEpsilon = 1e-9;

glm::dvec3 toDVec3(const glm::vec3& v)
{
    return {static_cast<double>(v.x), static_cast<double>(v.y), static_cast<double>(v.z)};
}

glm::dvec3 toDVec3(const QVector3D& v)
{
    return {static_cast<double>(v.x()), static_cast<double>(v.y()), static_cast<double>(v.z())};
}

double cross2(const glm::dvec2& a, const glm::dvec2& b)
{
    return a.x * b.y - a.y * b.x;
}

bool triangleProjectContains(const glm::dvec3& p,
                             const glm::dvec3& a,
                             const glm::dvec3& b,
                             const glm::dvec3& c,
                             glm::dvec3& bary)
{
    const glm::dvec2 p2{p.x, p.y};
    const glm::dvec2 a2{a.x, a.y};
    const glm::dvec2 b2{b.x, b.y};
    const glm::dvec2 c2{c.x, c.y};

    const glm::dvec2 v0 = b2 - a2;
    const glm::dvec2 v1 = c2 - a2;
    const glm::dvec2 v2 = p2 - a2;

    const double denom = v0.x * v1.y - v1.x * v0.y;
    if (std::abs(denom) <= kEpsilon)
    {
        return false;
    }

    const double invDenom = 1.0 / denom;
    const double u = (v2.x * v1.y - v1.x * v2.y) * invDenom;
    const double v = (v0.x * v2.y - v2.x * v0.y) * invDenom;

    if (u < -kEpsilon || v < -kEpsilon || (u + v) > 1.0 + kEpsilon)
    {
        return false;
    }

    bary = {1.0 - u - v, u, v};
    return true;
}

} // namespace

namespace sim
{

StockGrid::StockGrid(const render::Model& model, double cellSizeMm, double marginMm)
    : m_cellSize(std::max(0.05, cellSizeMm))
    , m_margin(std::max(0.0, marginMm))
{
    const common::Bounds bounds = model.bounds();
    const glm::dvec3 minBounds = toDVec3(bounds.min) - glm::dvec3(m_margin);
    const glm::dvec3 maxBounds = toDVec3(bounds.max) + glm::dvec3(m_margin);
    m_origin = minBounds;

    glm::dvec3 extent = maxBounds - minBounds;
    extent.x = std::max(extent.x, m_cellSize);
    extent.y = std::max(extent.y, m_cellSize);
    extent.z = std::max(extent.z, m_cellSize);

    m_dims.x = std::max(1, static_cast<int>(std::ceil(extent.x / m_cellSize)));
    m_dims.y = std::max(1, static_cast<int>(std::ceil(extent.y / m_cellSize)));
    m_dims.z = std::max(1, static_cast<int>(std::ceil(extent.z / m_cellSize)));

    m_totalCells = static_cast<std::size_t>(m_dims.x) * static_cast<std::size_t>(m_dims.y) * static_cast<std::size_t>(m_dims.z);
    m_cells.resize(m_totalCells, 1);
    m_remainingCells = m_totalCells;

    m_targetSurface.resize(static_cast<std::size_t>(m_dims.x) * static_cast<std::size_t>(m_dims.y),
                           std::numeric_limits<double>::quiet_NaN());

    computeTargetSurface(model);
}

void StockGrid::initializeOccupancy()
{
    std::fill(m_cells.begin(), m_cells.end(), 1u);
    m_removedCells = 0;
    m_remainingCells = m_totalCells;
}

bool StockGrid::inBounds(int x, int y, int z) const noexcept
{
    return x >= 0 && x < m_dims.x && y >= 0 && y < m_dims.y && z >= 0 && z < m_dims.z;
}

double StockGrid::cellCenterX(int ix) const noexcept
{
    return m_origin.x + (static_cast<double>(ix) + 0.5) * m_cellSize;
}

double StockGrid::cellCenterY(int iy) const noexcept
{
    return m_origin.y + (static_cast<double>(iy) + 0.5) * m_cellSize;
}

double StockGrid::cellCenterZ(int iz) const noexcept
{
    return m_origin.z + (static_cast<double>(iz) + 0.5) * m_cellSize;
}

std::size_t StockGrid::cellIndex(int ix, int iy, int iz) const noexcept
{
    return (static_cast<std::size_t>(iz) * static_cast<std::size_t>(m_dims.y) + static_cast<std::size_t>(iy)) * static_cast<std::size_t>(m_dims.x)
           + static_cast<std::size_t>(ix);
}

std::size_t StockGrid::columnIndex(int ix, int iy) const noexcept
{
    return static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_dims.x) + static_cast<std::size_t>(ix);
}

void StockGrid::computeTargetSurface(const render::Model& model)
{
    const auto& vertices = model.vertices();
    const auto& indices = model.indices();

    if (vertices.empty() || indices.empty())
    {
        return;
    }

    const std::size_t triangleCount = indices.size() / 3;
    for (std::size_t t = 0; t < triangleCount; ++t)
    {
        const render::Model::Index i0 = indices[t * 3];
        const render::Model::Index i1 = indices[t * 3 + 1];
        const render::Model::Index i2 = indices[t * 3 + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size())
        {
            continue;
        }

        const glm::dvec3 v0 = toDVec3(vertices[i0].position);
        const glm::dvec3 v1 = toDVec3(vertices[i1].position);
        const glm::dvec3 v2 = toDVec3(vertices[i2].position);

        const glm::dvec3 normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
        if (!std::isfinite(normal.x) || glm::dot(normal, normal) <= kEpsilon)
        {
            continue;
        }

        const double triMinX = std::min({v0.x, v1.x, v2.x});
        const double triMaxX = std::max({v0.x, v1.x, v2.x});
        const double triMinY = std::min({v0.y, v1.y, v2.y});
        const double triMaxY = std::max({v0.y, v1.y, v2.y});

        const int ix0 = std::clamp(static_cast<int>(std::floor((triMinX - m_origin.x) / m_cellSize)), 0, m_dims.x - 1);
        const int ix1 = std::clamp(static_cast<int>(std::ceil((triMaxX - m_origin.x) / m_cellSize)), 0, m_dims.x - 1);
        const int iy0 = std::clamp(static_cast<int>(std::floor((triMinY - m_origin.y) / m_cellSize)), 0, m_dims.y - 1);
        const int iy1 = std::clamp(static_cast<int>(std::ceil((triMaxY - m_origin.y) / m_cellSize)), 0, m_dims.y - 1);

        for (int ix = ix0; ix <= ix1; ++ix)
        {
            const double x = cellCenterX(ix);
            for (int iy = iy0; iy <= iy1; ++iy)
            {
                const double y = cellCenterY(iy);
                const glm::dvec3 sample{x, y, 0.0};
                glm::dvec3 bary{0.0};
                if (!triangleProjectContains(sample, v0, v1, v2, bary))
                {
                    continue;
                }

                const double z = bary.x * v0.z + bary.y * v1.z + bary.z * v2.z;
                const std::size_t idx = columnIndex(ix, iy);
                double& slot = m_targetSurface[idx];
                if (!std::isfinite(slot))
                {
                    slot = z;
                }
                else
                {
                    slot = std::max(slot, z);
                }
            }
        }
    }
}

void StockGrid::removeSample(const glm::dvec3& position, double radius, bool ballNose)
{
    if (radius <= 0.0)
    {
        return;
    }

    const double influence = radius + m_cellSize * 1.1;
    const double minX = position.x - influence;
    const double maxX = position.x + influence;
    const double minY = position.y - influence;
    const double maxY = position.y + influence;

    const int ix0 = std::clamp(static_cast<int>(std::floor((minX - m_origin.x) / m_cellSize)), 0, m_dims.x - 1);
    const int ix1 = std::clamp(static_cast<int>(std::ceil((maxX - m_origin.x) / m_cellSize)), 0, m_dims.x - 1);
    const int iy0 = std::clamp(static_cast<int>(std::floor((minY - m_origin.y) / m_cellSize)), 0, m_dims.y - 1);
    const int iy1 = std::clamp(static_cast<int>(std::ceil((maxY - m_origin.y) / m_cellSize)), 0, m_dims.y - 1);

    const double radiusSq = radius * radius;

    for (int ix = ix0; ix <= ix1; ++ix)
    {
        const double cx = cellCenterX(ix);
        const double dx = cx - position.x;
        for (int iy = iy0; iy <= iy1; ++iy)
        {
            const double cy = cellCenterY(iy);
            const double dy = cy - position.y;
            const double distSq = dx * dx + dy * dy;
            if (distSq > (radiusSq + m_cellSize * m_cellSize))
            {
                continue;
            }

            const double dist = std::sqrt(std::max(0.0, distSq));
            if (dist > radius + m_cellSize)
            {
                continue;
            }

            double removalThreshold = position.z;
            if (ballNose)
            {
                if (dist > radius + kEpsilon)
                {
                    continue;
                }
                const double inner = std::max(0.0, radiusSq - distSq);
                const double capHeight = radius - std::sqrt(inner);
                removalThreshold = position.z + capHeight;
            }

            const double normalized = (removalThreshold - m_origin.z) / m_cellSize - 0.5;
            int zStart = static_cast<int>(std::ceil(normalized));
            if (zStart < 0)
            {
                zStart = 0;
            }

            const std::size_t columnIdx = columnIndex(ix, iy);
            const double targetHeight = m_targetSurface[columnIdx];
            if (std::isfinite(targetHeight))
            {
                const double targetNormalized = (targetHeight - m_origin.z) / m_cellSize - 0.5;
                const int targetStart = static_cast<int>(std::ceil(targetNormalized));
                if (targetStart > zStart)
                {
                    zStart = targetStart;
                }
            }

            if (zStart >= m_dims.z)
            {
                continue;
            }

            for (int iz = zStart; iz < m_dims.z; ++iz)
            {
                const std::size_t idx = cellIndex(ix, iy, iz);
                if (m_cells[idx] == 0)
                {
                    continue;
                }
                m_cells[idx] = 0;
                ++m_removedCells;
                if (m_remainingCells > 0)
                {
                    --m_remainingCells;
                }
            }
        }
    }
}

void StockGrid::subtractToolpath(const tp::Toolpath& toolpath, const tp::UserParams& params)
{
    initializeOccupancy();

    const double radius = std::max(0.05, params.toolDiameter * 0.5);
    const bool ballNose = (params.cutterType == tp::UserParams::CutterType::BallNose);
    const double step = std::max(0.1, m_cellSize * 0.5);

    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (poly.motion != tp::MotionType::Cut || poly.pts.size() < 2)
        {
            continue;
        }

        for (std::size_t i = 1; i < poly.pts.size(); ++i)
        {
            const glm::dvec3 start = toDVec3(poly.pts[i - 1].p);
            const glm::dvec3 end = toDVec3(poly.pts[i].p);
            const double length = glm::length(end - start);
            if (length <= kDegenerateLength)
            {
                removeSample(start, radius, ballNose);
                continue;
            }

            const int segments = std::max(1, static_cast<int>(std::ceil(length / step)));
            for (int s = 0; s <= segments; ++s)
            {
                const double t = static_cast<double>(s) / static_cast<double>(segments);
                const glm::dvec3 position = start + (end - start) * t;
                removeSample(position, radius, ballNose);
            }
        }
    }
}

double StockGrid::columnStockHeight(int ix, int iy) const noexcept
{
    for (int iz = m_dims.z - 1; iz >= 0; --iz)
    {
        const std::size_t idx = cellIndex(ix, iy, iz);
        if (m_cells[idx] != 0)
        {
            return cellCenterZ(iz);
        }
    }
    return m_origin.z - 0.5 * m_cellSize;
}

StockGridSummary StockGrid::summarize() const
{
    StockGridSummary summary;
    summary.cellSize = m_cellSize;
    summary.origin = m_origin;
    summary.dims = m_dims;
    summary.percentRemoved = (m_totalCells == 0) ? 0.0 : (static_cast<double>(m_removedCells) / static_cast<double>(m_totalCells)) * 100.0;
    summary.removedFraction = (m_totalCells == 0) ? 0.0 : static_cast<double>(m_removedCells) / static_cast<double>(m_totalCells);

    double sumError = 0.0;
    double minError = std::numeric_limits<double>::infinity();
    double maxError = -std::numeric_limits<double>::infinity();

    const std::size_t columns = static_cast<std::size_t>(m_dims.x) * static_cast<std::size_t>(m_dims.y);
    summary.samples.reserve(columns);

    for (int iy = 0; iy < m_dims.y; ++iy)
    {
        for (int ix = 0; ix < m_dims.x; ++ix)
        {
            const std::size_t idx = columnIndex(ix, iy);
            const double target = m_targetSurface[idx];
            if (!std::isfinite(target))
            {
                continue;
            }

            double stock = columnStockHeight(ix, iy);
            if (stock < target)
            {
                stock = target;
            }

            const double error = stock - target;

            StockGridSummary::ColumnSample sample;
            sample.position = {cellCenterX(ix), cellCenterY(iy), stock};
            sample.error = error;
            summary.samples.push_back(sample);

            sumError += error;
            minError = std::min(minError, error);
            maxError = std::max(maxError, error);
        }
    }

    summary.columnCount = summary.samples.size();
    if (summary.columnCount > 0)
    {
        summary.averageError = sumError / static_cast<double>(summary.columnCount);
        summary.minError = minError;
        summary.maxError = maxError;
    }
    else
    {
        summary.averageError = 0.0;
        summary.minError = 0.0;
        summary.maxError = 0.0;
    }

    return summary;
}

} // namespace sim
