#pragma once

#include "tp/heightfield/UniformGrid.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace tp::heightfield
{

class HeightField
{
public:
    struct BuildStats
    {
        double buildMilliseconds{0.0};
        std::size_t validSamples{0};
        std::size_t totalSamples{0};
    };

    HeightField() = default;

    bool build(const UniformGrid& grid,
               double resolutionMm,
               const std::atomic<bool>& cancelFlag,
               BuildStats* stats = nullptr);

    [[nodiscard]] bool isValid() const noexcept { return m_valid; }
    [[nodiscard]] double minX() const noexcept { return m_minX; }
    [[nodiscard]] double minY() const noexcept { return m_minY; }
    [[nodiscard]] double maxX() const noexcept { return m_maxX; }
    [[nodiscard]] double maxY() const noexcept { return m_maxY; }
    [[nodiscard]] double resolution() const noexcept { return m_resolution; }
    [[nodiscard]] std::size_t columns() const noexcept { return m_columns; }
    [[nodiscard]] std::size_t rows() const noexcept { return m_rows; }

    [[nodiscard]] bool interpolate(double x, double y, double& zOut) const;
    [[nodiscard]] bool sampleAt(std::size_t col, std::size_t row, double& zOut) const;
    [[nodiscard]] bool hasSample(std::size_t col, std::size_t row) const;
    [[nodiscard]] const std::vector<std::uint8_t>& coverageMask() const noexcept { return m_coverage; }

private:
    inline std::size_t offset(std::size_t col, std::size_t row) const noexcept
    {
        return row * m_columns + col;
    }

    double m_minX{0.0};
    double m_minY{0.0};
    double m_maxX{0.0};
    double m_maxY{0.0};
    double m_resolution{1.0};
    std::size_t m_columns{0};
    std::size_t m_rows{0};
    bool m_valid{false};

    std::vector<double> m_samples;
    std::vector<std::uint8_t> m_coverage;
};

} // namespace tp::heightfield

