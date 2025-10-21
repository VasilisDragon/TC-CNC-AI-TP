#pragma once

#include "render/Model.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <vector>

namespace tp
{

class TriangleGrid
{
public:
    struct Triangle
    {
        glm::dvec3 v0{0.0};
        glm::dvec3 v1{0.0};
        glm::dvec3 v2{0.0};
        glm::dvec3 normal{0.0};
        glm::dvec3 bboxMin{0.0};
        glm::dvec3 bboxMax{0.0};
        glm::dvec3 centroid{0.0};
        glm::dvec3 edge0{0.0};
        glm::dvec3 edge1{0.0};
        double dot00{0.0};
        double dot01{0.0};
        double dot11{0.0};
        double invDet{0.0};
        double planeD{0.0};
        double maxZ{0.0};
        double minZ{0.0};
        double boundingRadiusSq{0.0};
        double invNormalZ{0.0};
        bool validBarycentric{false};
        bool validNormalZ{false};

        [[nodiscard]] double planeHeightAt(double x, double y) const;
        [[nodiscard]] bool barycentricContains(const glm::dvec3& point, double eps) const;
    };

    TriangleGrid() = default;
    TriangleGrid(const render::Model& model, double targetCellSizeMm);

    void build(const render::Model& model, double targetCellSizeMm);

    [[nodiscard]] bool empty() const noexcept { return m_triangles.empty(); }
    [[nodiscard]] std::size_t triangleCount() const noexcept { return m_triangles.size(); }
    [[nodiscard]] const Triangle& triangle(std::uint32_t index) const { return m_triangles[index]; }

    [[nodiscard]] const glm::dvec2& boundsMin() const noexcept { return m_boundsMin; }
    [[nodiscard]] const glm::dvec2& boundsMax() const noexcept { return m_boundsMax; }
    [[nodiscard]] int cellsX() const noexcept { return m_cellsX; }
    [[nodiscard]] int cellsY() const noexcept { return m_cellsY; }
    [[nodiscard]] std::size_t cellCount() const noexcept { return m_cellRanges.size(); }
    [[nodiscard]] std::size_t cellIndexCount() const noexcept { return m_cellIndices.size(); }

    void gatherCandidatesXY(double x, double y, int radius, std::vector<std::uint32_t>& out) const;
    void gatherCandidatesAABB(double minX,
                              double minY,
                              double maxX,
                              double maxY,
                              std::vector<std::uint32_t>& out) const;

private:
    struct CellRange
    {
        std::uint32_t offset{0};
        std::uint32_t count{0};
    };

    void gatherCellRange(int ixMin, int iyMin, int ixMax, int iyMax, std::vector<std::uint32_t>& out) const;
    [[nodiscard]] static int clampIndex(int value, int maxExclusive);

    std::vector<Triangle> m_triangles;
    glm::dvec2 m_boundsMin{0.0};
    glm::dvec2 m_boundsMax{0.0};
    int m_cellsX{1};
    int m_cellsY{1};
    double m_cellSizeX{1.0};
    double m_cellSizeY{1.0};
    double m_invCellSizeX{0.0};
    double m_invCellSizeY{0.0};
    std::vector<CellRange> m_cellRanges;
    std::vector<std::uint32_t> m_cellIndices;

    mutable std::vector<std::uint32_t> m_visitMarks;
    mutable std::uint32_t m_visitStamp{1};
};

} // namespace tp
