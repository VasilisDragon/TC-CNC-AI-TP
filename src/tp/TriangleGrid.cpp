#include "tp/TriangleGrid.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <glm/geometric.hpp>

namespace
{

constexpr double kEpsilon = 1e-9;

inline double lengthSquared(const glm::dvec3& v)
{
    return glm::dot(v, v);
}

} // namespace

namespace tp
{

double TriangleGrid::Triangle::planeHeightAt(double x, double y) const
{
    if (!validNormalZ)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (-planeD - normal.x * x - normal.y * y) * invNormalZ;
}

bool TriangleGrid::Triangle::barycentricContains(const glm::dvec3& point, double eps) const
{
    if (!validBarycentric)
    {
        return false;
    }

    const glm::dvec3 v2 = point - v0;
    const double d20 = glm::dot(v2, edge0);
    const double d21 = glm::dot(v2, edge1);

    const double v = (dot11 * d20 - dot01 * d21) * invDet;
    const double w = (dot00 * d21 - dot01 * d20) * invDet;
    const double u = 1.0 - v - w;

    return u >= -eps && v >= -eps && w >= -eps && u <= 1.0 + eps && v <= 1.0 + eps && w <= 1.0 + eps;
}

TriangleGrid::TriangleGrid(const render::Model& model, double targetCellSizeMm)
{
    build(model, targetCellSizeMm);
}

void TriangleGrid::build(const render::Model& model, double targetCellSizeMm)
{
    m_triangles.clear();
    m_cellRanges.clear();
    m_cellIndices.clear();
    m_visitMarks.clear();
    m_visitStamp = 1;

    const auto& vertices = model.vertices();
    const auto& indices = model.indices();
    if (vertices.empty() || indices.size() < 3)
    {
        m_boundsMin = glm::dvec2(0.0);
        m_boundsMax = glm::dvec2(0.0);
        m_cellsX = 1;
        m_cellsY = 1;
        m_cellSizeX = 1.0;
        m_cellSizeY = 1.0;
        m_invCellSizeX = 0.0;
        m_invCellSizeY = 0.0;
        return;
    }

    const auto bounds = model.bounds();
    m_boundsMin = glm::dvec2(bounds.min.x(), bounds.min.y());
    m_boundsMax = glm::dvec2(bounds.max.x(), bounds.max.y());

    const std::size_t triangleCount = indices.size() / 3;
    m_triangles.reserve(triangleCount);

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const auto ia = static_cast<std::size_t>(indices[i]);
        const auto ib = static_cast<std::size_t>(indices[i + 1]);
        const auto ic = static_cast<std::size_t>(indices[i + 2]);
        if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size())
        {
            continue;
        }

        Triangle tri;
        tri.v0 = glm::dvec3(vertices[ia].position.x(), vertices[ia].position.y(), vertices[ia].position.z());
        tri.v1 = glm::dvec3(vertices[ib].position.x(), vertices[ib].position.y(), vertices[ib].position.z());
        tri.v2 = glm::dvec3(vertices[ic].position.x(), vertices[ic].position.y(), vertices[ic].position.z());

        tri.edge0 = tri.v1 - tri.v0;
        tri.edge1 = tri.v2 - tri.v0;
        tri.normal = glm::cross(tri.edge0, tri.edge1);

        const double normalLengthSq = lengthSquared(tri.normal);
        if (normalLengthSq <= kEpsilon)
        {
            continue;
        }
        tri.normal /= std::sqrt(normalLengthSq);

        tri.centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0;
        tri.bboxMin = glm::min(glm::min(tri.v0, tri.v1), tri.v2);
        tri.bboxMax = glm::max(glm::max(tri.v0, tri.v1), tri.v2);
        tri.maxZ = std::max({tri.v0.z, tri.v1.z, tri.v2.z});
        tri.minZ = std::min({tri.v0.z, tri.v1.z, tri.v2.z});

        const double r0 = lengthSquared(tri.v0 - tri.centroid);
        const double r1 = lengthSquared(tri.v1 - tri.centroid);
        const double r2 = lengthSquared(tri.v2 - tri.centroid);
        tri.boundingRadiusSq = std::max({r0, r1, r2});

        tri.dot00 = glm::dot(tri.edge0, tri.edge0);
        tri.dot01 = glm::dot(tri.edge0, tri.edge1);
        tri.dot11 = glm::dot(tri.edge1, tri.edge1);
        const double denom = tri.dot00 * tri.dot11 - tri.dot01 * tri.dot01;
        if (std::abs(denom) > kEpsilon)
        {
            tri.invDet = 1.0 / denom;
            tri.validBarycentric = true;
        }

        tri.planeD = -glm::dot(tri.normal, tri.v0);
        if (std::abs(tri.normal.z) > kEpsilon)
        {
            tri.invNormalZ = 1.0 / tri.normal.z;
            tri.validNormalZ = true;
        }

        m_triangles.push_back(tri);
    }

    if (m_triangles.empty())
    {
        m_cellsX = 1;
        m_cellsY = 1;
        m_cellSizeX = 1.0;
        m_cellSizeY = 1.0;
        m_invCellSizeX = 0.0;
        m_invCellSizeY = 0.0;
        return;
    }

    const double spanX = std::max(m_boundsMax.x - m_boundsMin.x, kEpsilon);
    const double spanY = std::max(m_boundsMax.y - m_boundsMin.y, kEpsilon);

    if (targetCellSizeMm > kEpsilon)
    {
        m_cellsX = std::max(1, static_cast<int>(std::ceil(spanX / targetCellSizeMm)));
        m_cellsY = std::max(1, static_cast<int>(std::ceil(spanY / targetCellSizeMm)));
    }
    else
    {
        const double approx = std::sqrt(static_cast<double>(m_triangles.size()));
        int base = std::max(1, static_cast<int>(std::round(approx)));
        const double aspect = spanY > kEpsilon ? spanX / spanY : 1.0;
        if (aspect >= 1.0)
        {
            m_cellsX = base;
            m_cellsY = std::max(1, static_cast<int>(std::round(base / aspect)));
        }
        else
        {
            m_cellsY = base;
            m_cellsX = std::max(1, static_cast<int>(std::round(base / std::max(aspect, kEpsilon))));
        }
    }

    m_cellsX = std::max(1, m_cellsX);
    m_cellsY = std::max(1, m_cellsY);

    m_cellSizeX = (m_cellsX > 0) ? spanX / static_cast<double>(m_cellsX) : spanX;
    m_cellSizeY = (m_cellsY > 0) ? spanY / static_cast<double>(m_cellsY) : spanY;

    if (m_cellSizeX < kEpsilon)
    {
        m_cellSizeX = spanX;
    }
    if (m_cellSizeY < kEpsilon)
    {
        m_cellSizeY = spanY;
    }

    m_invCellSizeX = (m_cellSizeX > kEpsilon) ? 1.0 / m_cellSizeX : 0.0;
    m_invCellSizeY = (m_cellSizeY > kEpsilon) ? 1.0 / m_cellSizeY : 0.0;

    const std::size_t cellCount = static_cast<std::size_t>(m_cellsX) * static_cast<std::size_t>(m_cellsY);
    std::vector<std::uint32_t> cellCounts(cellCount, 0);

    auto cellIndexFor = [&](double value, double origin, double invSize, int cells) -> int {
        if (invSize <= 0.0 || cells <= 1)
        {
            return 0;
        }
        const double rel = (value - origin) * invSize;
        return clampIndex(static_cast<int>(std::floor(rel)), cells);
    };

    for (std::uint32_t triIndex = 0; triIndex < static_cast<std::uint32_t>(m_triangles.size()); ++triIndex)
    {
        const Triangle& tri = m_triangles[triIndex];

        int ixMin = 0;
        int ixMax = m_cellsX - 1;
        int iyMin = 0;
        int iyMax = m_cellsY - 1;

        if (m_invCellSizeX > 0.0 && m_cellsX > 1)
        {
            const double minXRel = (tri.bboxMin.x - m_boundsMin.x) * m_invCellSizeX;
            const double maxXRel = (tri.bboxMax.x - m_boundsMin.x) * m_invCellSizeX;
            ixMin = clampIndex(static_cast<int>(std::floor(minXRel)), m_cellsX);
            ixMax = clampIndex(static_cast<int>(std::floor(maxXRel + kEpsilon)), m_cellsX);
        }

        if (m_invCellSizeY > 0.0 && m_cellsY > 1)
        {
            const double minYRel = (tri.bboxMin.y - m_boundsMin.y) * m_invCellSizeY;
            const double maxYRel = (tri.bboxMax.y - m_boundsMin.y) * m_invCellSizeY;
            iyMin = clampIndex(static_cast<int>(std::floor(minYRel)), m_cellsY);
            iyMax = clampIndex(static_cast<int>(std::floor(maxYRel + kEpsilon)), m_cellsY);
        }

        for (int iy = iyMin; iy <= iyMax; ++iy)
        {
            for (int ix = ixMin; ix <= ixMax; ++ix)
            {
                const std::size_t cellIndex = static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_cellsX)
                                              + static_cast<std::size_t>(ix);
                ++cellCounts[cellIndex];
            }
        }
    }

    std::vector<std::uint32_t> offsets(cellCount, 0);
    std::exclusive_scan(cellCounts.begin(), cellCounts.end(), offsets.begin(), 0u);

    const std::uint32_t totalEntries = offsets.back() + cellCounts.back();
    m_cellIndices.resize(totalEntries);
    m_cellRanges.resize(cellCount);

    std::vector<std::uint32_t> writeCursor = offsets;

    for (std::uint32_t triIndex = 0; triIndex < static_cast<std::uint32_t>(m_triangles.size()); ++triIndex)
    {
        const Triangle& tri = m_triangles[triIndex];

        int ixMin = 0;
        int ixMax = m_cellsX - 1;
        int iyMin = 0;
        int iyMax = m_cellsY - 1;

        if (m_invCellSizeX > 0.0 && m_cellsX > 1)
        {
            const double minXRel = (tri.bboxMin.x - m_boundsMin.x) * m_invCellSizeX;
            const double maxXRel = (tri.bboxMax.x - m_boundsMin.x) * m_invCellSizeX;
            ixMin = clampIndex(static_cast<int>(std::floor(minXRel)), m_cellsX);
            ixMax = clampIndex(static_cast<int>(std::floor(maxXRel + kEpsilon)), m_cellsX);
        }

        if (m_invCellSizeY > 0.0 && m_cellsY > 1)
        {
            const double minYRel = (tri.bboxMin.y - m_boundsMin.y) * m_invCellSizeY;
            const double maxYRel = (tri.bboxMax.y - m_boundsMin.y) * m_invCellSizeY;
            iyMin = clampIndex(static_cast<int>(std::floor(minYRel)), m_cellsY);
            iyMax = clampIndex(static_cast<int>(std::floor(maxYRel + kEpsilon)), m_cellsY);
        }

        for (int iy = iyMin; iy <= iyMax; ++iy)
        {
            for (int ix = ixMin; ix <= ixMax; ++ix)
            {
                const std::size_t cellIndex = static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_cellsX)
                                              + static_cast<std::size_t>(ix);
                const std::uint32_t cursor = writeCursor[cellIndex]++;
                m_cellIndices[cursor] = triIndex;
            }
        }
    }

    for (std::size_t cell = 0; cell < cellCount; ++cell)
    {
        const std::uint32_t offset = offsets[cell];
        const std::uint32_t count = cellCounts[cell];
        m_cellRanges[cell] = CellRange{offset, count};

        auto begin = m_cellIndices.begin() + offset;
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

    m_visitMarks.resize(m_triangles.size(), 0);
}

int TriangleGrid::clampIndex(int value, int maxExclusive)
{
    if (maxExclusive <= 1)
    {
        return 0;
    }
    if (value < 0)
    {
        return 0;
    }
    if (value >= maxExclusive)
    {
        return maxExclusive - 1;
    }
    return value;
}

void TriangleGrid::gatherCellRange(int ixMin,
                                   int iyMin,
                                   int ixMax,
                                   int iyMax,
                                   std::vector<std::uint32_t>& out) const
{
    if (m_triangles.empty())
    {
        out.clear();
        return;
    }

    ++m_visitStamp;
    if (m_visitStamp == 0)
    {
        std::fill(m_visitMarks.begin(), m_visitMarks.end(), 0);
        m_visitStamp = 1;
    }

    out.clear();

    ixMin = std::max(0, ixMin);
    iyMin = std::max(0, iyMin);
    ixMax = std::min(m_cellsX - 1, ixMax);
    iyMax = std::min(m_cellsY - 1, iyMax);

    for (int iy = iyMin; iy <= iyMax; ++iy)
    {
        for (int ix = ixMin; ix <= ixMax; ++ix)
        {
            const std::size_t cellIndex = static_cast<std::size_t>(iy) * static_cast<std::size_t>(m_cellsX)
                                          + static_cast<std::size_t>(ix);
            const CellRange& range = m_cellRanges[cellIndex];
            const std::uint32_t* begin = m_cellIndices.data() + range.offset;
            const std::uint32_t* end = begin + range.count;
            for (const std::uint32_t* it = begin; it != end; ++it)
            {
                const std::uint32_t idx = *it;
                if (idx >= m_triangles.size())
                {
                    continue;
                }
                if (m_visitMarks[idx] == m_visitStamp)
                {
                    continue;
                }
                m_visitMarks[idx] = m_visitStamp;
                out.push_back(idx);
            }
        }
    }

    if (out.empty())
    {
        out.resize(m_triangles.size());
        std::iota(out.begin(), out.end(), 0u);
    }
}

void TriangleGrid::gatherCandidatesXY(double x, double y, int radius, std::vector<std::uint32_t>& out) const
{
    if (m_triangles.empty())
    {
        out.clear();
        return;
    }

    if (m_cellsX <= 1 || m_cellsY <= 1 || m_invCellSizeX <= 0.0 || m_invCellSizeY <= 0.0)
    {
        out.resize(m_triangles.size());
        std::iota(out.begin(), out.end(), 0u);
        return;
    }

    const double relX = (x - m_boundsMin.x) * m_invCellSizeX;
    const double relY = (y - m_boundsMin.y) * m_invCellSizeY;

    int ix = clampIndex(static_cast<int>(std::floor(relX)), m_cellsX);
    int iy = clampIndex(static_cast<int>(std::floor(relY)), m_cellsY);

    radius = std::max(0, radius);
    gatherCellRange(ix - radius, iy - radius, ix + radius, iy + radius, out);
}

void TriangleGrid::gatherCandidatesAABB(double minX,
                                        double minY,
                                        double maxX,
                                        double maxY,
                                        std::vector<std::uint32_t>& out) const
{
    if (m_triangles.empty())
    {
        out.clear();
        return;
    }

    if (m_cellsX <= 1 || m_cellsY <= 1 || m_invCellSizeX <= 0.0 || m_invCellSizeY <= 0.0)
    {
        out.resize(m_triangles.size());
        std::iota(out.begin(), out.end(), 0u);
        return;
    }

    const double minRelX = (minX - m_boundsMin.x) * m_invCellSizeX;
    const double maxRelX = (maxX - m_boundsMin.x) * m_invCellSizeX;
    const double minRelY = (minY - m_boundsMin.y) * m_invCellSizeY;
    const double maxRelY = (maxY - m_boundsMin.y) * m_invCellSizeY;

    const int ixMin = clampIndex(static_cast<int>(std::floor(minRelX)), m_cellsX);
    const int ixMax = clampIndex(static_cast<int>(std::floor(maxRelX + kEpsilon)), m_cellsX);
    const int iyMin = clampIndex(static_cast<int>(std::floor(minRelY)), m_cellsY);
    const int iyMax = clampIndex(static_cast<int>(std::floor(maxRelY + kEpsilon)), m_cellsY);

    gatherCellRange(ixMin, iyMin, ixMax, iyMax, out);
}

} // namespace tp

