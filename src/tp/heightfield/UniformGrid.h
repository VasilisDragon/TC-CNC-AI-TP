#pragma once

#include "render/Model.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

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

    [[nodiscard]] double minX() const noexcept { return m_boundsMin.x; }
    [[nodiscard]] double minY() const noexcept { return m_boundsMin.y; }
    [[nodiscard]] double maxX() const noexcept { return m_boundsMax.x; }
    [[nodiscard]] double maxY() const noexcept { return m_boundsMax.y; }
    [[nodiscard]] double cellSize() const noexcept { return m_cellSize; }
    [[nodiscard]] std::size_t columns() const noexcept { return m_columns; }
    [[nodiscard]] std::size_t rows() const noexcept { return m_rows; }

private:
    struct Triangle
    {
        glm::dvec3 v0;
        glm::dvec3 v1;
        glm::dvec3 v2;
        glm::dvec3 centroid;
        glm::dvec3 bboxMin;
        glm::dvec3 bboxMax;
        double maxZ{0.0};
        double minZ{0.0};
        double boundingRadiusSq{0.0};
    };

    struct CellRange
    {
        std::uint32_t offset{0};
        std::uint32_t count{0};
    };

    [[nodiscard]] static Triangle makeTriangle(const render::Model& model, std::size_t index);
    [[nodiscard]] bool intersect(const Triangle& tri, double x, double y, double& zOut) const;

    glm::dvec2 m_boundsMin{0.0};
    glm::dvec2 m_boundsMax{0.0};
    double m_cellSize{1.0};
    std::size_t m_columns{0};
    std::size_t m_rows{0};

    std::vector<Triangle> m_triangles;
    std::vector<CellRange> m_cellRanges;
    std::vector<std::uint32_t> m_cellTriangleIndices;
};

} // namespace tp::heightfield
