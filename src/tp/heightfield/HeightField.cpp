#include "tp/heightfield/HeightField.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <limits>
#include <numeric>

namespace tp::heightfield
{

namespace
{
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kEpsilon = 1e-9;
} // namespace

bool HeightField::build(const UniformGrid& grid,
                        double resolutionMm,
                        const std::atomic<bool>& cancelFlag,
                        BuildStats* stats)
{
    m_resolution = std::max(0.1, resolutionMm);
    m_minX = grid.minX();
    m_minY = grid.minY();
    m_maxX = grid.maxX();
    m_maxY = grid.maxY();

    const double extentX = std::max(m_maxX - m_minX, m_resolution);
    const double extentY = std::max(m_maxY - m_minY, m_resolution);

    m_columns = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(extentX / m_resolution)));
    m_rows = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(extentY / m_resolution)));

    m_samples.assign(m_columns * m_rows, kNan);
    m_coverage.assign(m_columns * m_rows, 0);

    const auto startTime = std::chrono::steady_clock::now();

    std::vector<std::size_t> rowIndices(m_rows);
    std::iota(rowIndices.begin(), rowIndices.end(), 0);

    std::atomic<std::size_t> validCounter{0};

    std::for_each(std::execution::par, rowIndices.begin(), rowIndices.end(), [&](std::size_t row) {
        if (cancelFlag.load(std::memory_order_relaxed))
        {
            return;
        }

        const double y = m_minY + static_cast<double>(row) * m_resolution;
        for (std::size_t col = 0; col < m_columns; ++col)
        {
            if (cancelFlag.load(std::memory_order_relaxed))
            {
                return;
            }

            const double x = m_minX + static_cast<double>(col) * m_resolution;
            double z = 0.0;
            if (grid.sampleMaxZAtXY(x, y, z))
            {
                m_samples[offset(col, row)] = z;
                m_coverage[offset(col, row)] = 1;
                validCounter.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    if (cancelFlag.load(std::memory_order_relaxed))
    {
        m_valid = false;
        return false;
    }

    if (stats)
    {
        const auto elapsed = std::chrono::steady_clock::now() - startTime;
        stats->buildMilliseconds = std::chrono::duration<double, std::milli>(elapsed).count();
        stats->validSamples = validCounter.load(std::memory_order_relaxed);
        stats->totalSamples = m_columns * m_rows;
    }

    m_valid = true;
    return true;
}

bool HeightField::sampleAt(std::size_t col, std::size_t row, double& zOut) const
{
    if (!m_valid || col >= m_columns || row >= m_rows)
    {
        return false;
    }

    const double value = m_samples[offset(col, row)];
    if (std::isnan(value))
    {
        return false;
    }

    zOut = value;
    return true;
}

bool HeightField::hasSample(std::size_t col, std::size_t row) const
{
    if (!m_valid || col >= m_columns || row >= m_rows)
    {
        return false;
    }
    return m_coverage[offset(col, row)] != 0;
}

bool HeightField::interpolate(double x, double y, double& zOut) const
{
    if (!m_valid)
    {
        return false;
    }

    if (x < m_minX - kEpsilon || x > m_minX + m_resolution * static_cast<double>(m_columns - 1) + kEpsilon
        || y < m_minY - kEpsilon || y > m_minY + m_resolution * static_cast<double>(m_rows - 1) + kEpsilon)
    {
        return false;
    }

    const double tx = (x - m_minX) / m_resolution;
    const double ty = (y - m_minY) / m_resolution;

    const double fx = std::clamp(tx, 0.0, static_cast<double>(m_columns - 1));
    const double fy = std::clamp(ty, 0.0, static_cast<double>(m_rows - 1));

    const std::size_t ix = static_cast<std::size_t>(std::floor(fx));
    const std::size_t iy = static_cast<std::size_t>(std::floor(fy));

    if (ix >= m_columns - 1 || iy >= m_rows - 1)
    {
        // Require four neighbours for bilinear interpolation.
        if (!sampleAt(ix, iy, zOut))
        {
            return false;
        }
        return true;
    }

    const double localX = fx - static_cast<double>(ix);
    const double localY = fy - static_cast<double>(iy);

    double z00 = 0.0;
    double z10 = 0.0;
    double z01 = 0.0;
    double z11 = 0.0;

    if (!sampleAt(ix, iy, z00) || !sampleAt(ix + 1, iy, z10) || !sampleAt(ix, iy + 1, z01)
        || !sampleAt(ix + 1, iy + 1, z11))
    {
        return false;
    }

    const double z0 = z00 * (1.0 - localX) + z10 * localX;
    const double z1 = z01 * (1.0 - localX) + z11 * localX;
    zOut = z0 * (1.0 - localY) + z1 * localY;
    return true;
}

} // namespace tp::heightfield

