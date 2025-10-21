#pragma once

#include "render/Model.h"
#include "tp/TriangleGrid.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tp::heightfield
{

class UniformGrid
{
public:
    UniformGrid(const render::Model& model, double cellSizeMm);

    [[nodiscard]] bool sampleMaxZAtXY(double x, double y, double& zOut) const;

    [[nodiscard]] double minX() const noexcept { return m_grid.boundsMin().x; }
    [[nodiscard]] double minY() const noexcept { return m_grid.boundsMin().y; }
    [[nodiscard]] double maxX() const noexcept { return m_grid.boundsMax().x; }
    [[nodiscard]] double maxY() const noexcept { return m_grid.boundsMax().y; }
    [[nodiscard]] double cellSize() const noexcept { return m_cellSize; }
    [[nodiscard]] std::size_t columns() const noexcept
    {
        return static_cast<std::size_t>(std::max(1, m_grid.cellsX()));
    }
    [[nodiscard]] std::size_t rows() const noexcept
    {
        return static_cast<std::size_t>(std::max(1, m_grid.cellsY()));
    }

private:
    [[nodiscard]] bool intersect(const TriangleGrid::Triangle& tri, double x, double y, double& zOut) const;

    TriangleGrid m_grid;
    double m_cellSize{1.0};
    mutable std::vector<std::uint32_t> m_queryBuffer;
};

} // namespace tp::heightfield
