#include "tp/heightfield/HeightField.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <functional>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

#include "common/log.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QString>
#include <QtCore/QStringView>
#include <QtCore/QStringList>

#include <cstdlib>
#include <atomic>

namespace tp::heightfield
{

namespace
{
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kEpsilon = 1e-9;

class ScopedTimer
{
public:
    using Callback = std::function<void(const QString&, double, bool)>;

    ScopedTimer(QString label, Callback callback, const std::atomic<bool>* cancelFlag = nullptr)
        : m_label(std::move(label))
        , m_callback(std::move(callback))
        , m_cancel(cancelFlag)
        , m_start(std::chrono::steady_clock::now())
    {
    }

    ~ScopedTimer()
    {
        const auto elapsed = std::chrono::steady_clock::now() - m_start;
        const double ms = std::chrono::duration<double, std::milli>(elapsed).count();
        const bool cancelled = m_cancel != nullptr && m_cancel->load(std::memory_order_relaxed);
        if (m_callback)
        {
            m_callback(m_label, ms, cancelled);
        }
        else
        {
            if (cancelled)
            {
                LOG_INFO(Tp, QStringLiteral("%1 cancelled after %2 ms").arg(m_label).arg(ms, 0, 'f', 2));
            }
            else
            {
                LOG_INFO(Tp, QStringLiteral("%1 took %2 ms").arg(m_label).arg(ms, 0, 'f', 2));
            }
        }
    }

private:
    QString m_label;
    Callback m_callback;
    const std::atomic<bool>* m_cancel{nullptr};
    std::chrono::steady_clock::time_point m_start;
};

int parseThreadOverrideFromArgs()
{
    if (!QCoreApplication::instance())
    {
        return 0;
    }

    const QStringList args = QCoreApplication::arguments();
    for (int i = 0; i < args.size(); ++i)
    {
        const QString& arg = args.at(i);
        if (arg == QStringLiteral("--threads") && i + 1 < args.size())
        {
            bool ok = false;
            const int value = args.at(i + 1).toInt(&ok);
            if (ok)
            {
                return value;
            }
        }
        else if (arg.startsWith(QStringLiteral("--threads=")))
        {
            const QStringView value = QStringView(arg).mid(QStringLiteral("--threads=").size());
            bool ok = false;
            const int parsed = value.toInt(&ok);
            if (ok)
            {
                return parsed;
            }
        }
    }
    return 0;
}

int threadOverride()
{
    static const int overrideValue = []() {
        int value = 0;
        if (const char* env = std::getenv("CNCTC_THREADS"))
        {
            char* endPtr = nullptr;
            const long envValue = std::strtol(env, &endPtr, 10);
            if (endPtr != env && envValue > 0 && envValue <= std::numeric_limits<int>::max())
            {
                value = static_cast<int>(envValue);
            }
        }

        if (value <= 0)
        {
            const int argValue = parseThreadOverrideFromArgs();
            if (argValue > 0)
            {
                value = argValue;
            }
        }

        return std::max(0, value);
    }();
    return overrideValue;
}

struct RowRange
{
    std::size_t begin{0};
    std::size_t end{0};
};

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

    const auto hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    const int userOverride = threadOverride();
    const std::size_t effectiveThreads = static_cast<std::size_t>(std::max<int>(1, (userOverride > 0) ? userOverride
                                                                                                      : static_cast<int>(hardwareThreads)));

    const QString timerLabel = QStringLiteral("HeightField build (%1x%2 @ %3 mm, threads=%4)")
                                   .arg(static_cast<qulonglong>(m_columns))
                                   .arg(static_cast<qulonglong>(m_rows))
                                   .arg(m_resolution, 0, 'f', 3)
                                   .arg(static_cast<qulonglong>(effectiveThreads));

    std::atomic<std::size_t> validCounter{0};
    double elapsedMs = 0.0;

    {
        ScopedTimer timer(timerLabel,
                          [&](const QString& label, double ms, bool cancelled) {
                              elapsedMs = ms;
                              const std::size_t coverageCount = validCounter.load(std::memory_order_relaxed);
                              const std::size_t total = m_columns * m_rows;
                              const double ratio =
                                  (total > 0) ? (static_cast<double>(coverageCount) / static_cast<double>(total)) : 0.0;
                              const std::size_t gridBytes =
                                  m_samples.size() * sizeof(double) + m_coverage.size() * sizeof(std::uint8_t);

                              if (cancelled)
                              {
                                  LOG_INFO(Tp, QStringLiteral("%1 cancelled after %2 ms (valid=%3/%4, %5%% coverage)")
                                                    .arg(label)
                                                    .arg(ms, 0, 'f', 2)
                                                    .arg(static_cast<qulonglong>(coverageCount))
                                                    .arg(static_cast<qulonglong>(total))
                                                    .arg(ratio * 100.0, 0, 'f', 1));
                                  return;
                              }
                              LOG_INFO(Tp,
                                       QStringLiteral(
                                           "%1 completed in %2 ms (valid=%3/%4, %5%% coverage, grid=%6 bytes)")
                                           .arg(label)
                                           .arg(ms, 0, 'f', 2)
                                           .arg(static_cast<qulonglong>(coverageCount))
                                           .arg(static_cast<qulonglong>(total))
                                           .arg(ratio * 100.0, 0, 'f', 1)
                                           .arg(static_cast<qulonglong>(gridBytes)));
                          },
                          &cancelFlag);

        std::vector<RowRange> workItems;
        const std::size_t totalRows = m_rows;
        const std::size_t baseChunk = (totalRows >= effectiveThreads)
                                          ? std::max<std::size_t>(1, totalRows / (effectiveThreads * 4))
                                          : 1;
        const std::size_t chunkSize = std::max<std::size_t>(16, baseChunk);

        for (std::size_t row = 0; row < totalRows; row += chunkSize)
        {
            workItems.push_back(RowRange{row, std::min(totalRows, row + chunkSize)});
        }

        const bool runParallel = effectiveThreads > 1 && workItems.size() > 1;

        const auto worker = [&](const RowRange& range) {
            std::size_t localValid = 0;
            for (std::size_t row = range.begin; row < range.end; ++row)
            {
                if (cancelFlag.load(std::memory_order_relaxed))
                {
                    break;
                }

                const double y = m_minY + static_cast<double>(row) * m_resolution;
                const std::size_t rowOffset = row * m_columns;
                for (std::size_t col = 0; col < m_columns; ++col)
                {
                    if (cancelFlag.load(std::memory_order_relaxed))
                    {
                        break;
                    }

                    const double x = m_minX + static_cast<double>(col) * m_resolution;
                    double z = 0.0;
                    if (grid.sampleMaxZAtXY(x, y, z))
                    {
                        const std::size_t sampleIndex = rowOffset + col;
                        m_samples[sampleIndex] = z;
                        m_coverage[sampleIndex] = 1;
                        ++localValid;
                    }
                }
            }
            return localValid;
        };

        if (runParallel)
        {
            std::for_each(std::execution::par, workItems.begin(), workItems.end(), [&](const RowRange& range) {
                const std::size_t local = worker(range);
                if (local > 0)
                {
                    validCounter.fetch_add(local, std::memory_order_relaxed);
                }
            });
        }
        else
        {
            for (const RowRange& range : workItems)
            {
                const std::size_t local = worker(range);
                if (local > 0)
                {
                    validCounter.fetch_add(local, std::memory_order_relaxed);
                }
                if (cancelFlag.load(std::memory_order_relaxed))
                {
                    break;
                }
            }
        }
    }

    if (cancelFlag.load(std::memory_order_relaxed))
    {
        m_valid = false;
        return false;
    }

    if (stats)
    {
        stats->buildMilliseconds = elapsedMs;
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
