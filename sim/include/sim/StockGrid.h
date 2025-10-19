#pragma once

#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"

#include "render/Model.h"

#include <glm/vec3.hpp>

#include <QtGui/QVector3D>

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace sim
{

struct StockGridSummary
{
    struct ColumnSample
    {
        glm::dvec3 position{};
        double error{0.0};
    };

    double percentRemoved{0.0};
    double removedFraction{0.0};
    double averageError{0.0};
    double maxError{0.0};
    double minError{0.0};
    double cellSize{0.0};
    std::size_t columnCount{0};
    glm::dvec3 origin{};
    glm::ivec3 dims{};
    std::vector<ColumnSample> samples;
};

class StockGrid
{
public:
    StockGrid(const render::Model& model,
              double cellSizeMm,
              double marginMm);

    void subtractToolpath(const tp::Toolpath& toolpath, const tp::UserParams& params);

    [[nodiscard]] StockGridSummary summarize() const;

private:
    [[nodiscard]] bool inBounds(int x, int y, int z) const noexcept;
    [[nodiscard]] double cellCenterX(int ix) const noexcept;
    [[nodiscard]] double cellCenterY(int iy) const noexcept;
    [[nodiscard]] double cellCenterZ(int iz) const noexcept;
    [[nodiscard]] std::size_t cellIndex(int ix, int iy, int iz) const noexcept;
    [[nodiscard]] std::size_t columnIndex(int ix, int iy) const noexcept;

    void computeTargetSurface(const render::Model& model);
    void initializeOccupancy();
    void removeSample(const glm::dvec3& position, double radius, bool ballNose);
    [[nodiscard]] double columnStockHeight(int ix, int iy) const noexcept;

    double m_cellSize{0.5};
    double m_margin{1.0};
    glm::dvec3 m_origin{0.0};
    glm::ivec3 m_dims{0};

    std::vector<std::uint8_t> m_cells;
    std::size_t m_totalCells{0};
    std::size_t m_removedCells{0};
    std::size_t m_remainingCells{0};

    std::vector<double> m_targetSurface;
};

} // namespace sim
