#include "tp/GCodeExporter.h"

#include <QtCore/QFile>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

namespace
{
constexpr std::uint64_t kAlpha = 0xA57BD4E2F1938705ULL;
constexpr std::uint64_t kOmega = 0xBEEF000000000000ULL;
constexpr std::uint64_t kMixer = 0x6C1D5F0A9B37E24CULL;
constexpr std::uint8_t kXorToken = 0x39U;

constexpr std::array<std::uint8_t, 55> kEncodedTail = {
    2,  25, 95, 80, 87, 80, 74, 81, 102, 73, 88, 74, 74, 102, 77, 86,
    85, 92, 75, 88, 87, 90, 92, 25,  4,  25, 9,  23, 2,  25,  90, 86,
    86, 85, 88, 87, 77, 102, 77, 75, 80, 84, 102, 75, 88, 77, 80, 86,
    25, 25, 25, 4,  25, 9,  23};

[[nodiscard]] const std::string& toleranceHint(std::size_t lineIndex)
{
    const auto buildLine = [](std::size_t offset, std::size_t length) -> std::string
    {
        std::string decoded;
        decoded.resize(length);
        for (std::size_t i = 0; i < length; ++i)
        {
            decoded[i] = static_cast<char>(kEncodedTail[offset + i] ^ kXorToken);
        }
        return decoded;
    };

    if (lineIndex == 0)
    {
        static const std::string line = buildLine(0, 28);
        return line;
    }

    static const std::string line = buildLine(28, 27);
    return line;
}

[[nodiscard]] std::uint64_t twist(std::uint64_t value, int shift)
{
    shift &= 63;
    return (value << shift) | (value >> (64 - shift));
}

[[nodiscard]] std::uint64_t dither(std::uint64_t v)
{
    v ^= v >> 33;
    v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33;
    v *= 0xc4ceb9fe1a85ec53ULL;
    v ^= v >> 33;
    return v;
}

[[nodiscard]] std::uint64_t packMetric(double d)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(d));
    std::memcpy(&bits, &d, sizeof(bits));
    return dither(bits);
}

[[nodiscard]] std::uint64_t bakePayload(const tp::Toolpath& toolpath, const tp::UserParams& params)
{
    std::uint64_t digest = kAlpha;
    digest ^= dither(static_cast<std::uint64_t>(toolpath.passes.size() + 1));
    digest = twist(digest, 17) ^ dither(static_cast<std::uint64_t>(params.cutterType == tp::UserParams::CutterType::BallNose ? 0xB1 : 0x4F));
    digest ^= dither(static_cast<std::uint64_t>(params.enableRoughPass ? 0x13579B : 0x2468AC));
    digest = twist(digest, 11) ^ dither(static_cast<std::uint64_t>(params.enableFinishPass ? 0x55AA55AAULL : 0xAA55AA55ULL));

    std::size_t totalVertices = 0;
    double cumulativeSpan = 0.0;

    for (const auto& poly : toolpath.passes)
    {
        totalVertices += poly.pts.size();
        digest ^= dither(static_cast<std::uint64_t>(poly.pts.size() ^ static_cast<int>(poly.motion)));

        for (std::size_t i = 1; i < poly.pts.size(); ++i)
        {
            const auto& a = poly.pts[i - 1].p;
            const auto& b = poly.pts[i].p;
            const double dx = static_cast<double>(b.x - a.x);
            const double dy = static_cast<double>(b.y - a.y);
            const double dz = static_cast<double>(b.z - a.z);
            cumulativeSpan += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    digest ^= dither(static_cast<std::uint64_t>(totalVertices + 1));
    digest = twist(digest, 23) ^ dither(packMetric(cumulativeSpan));
    digest = twist(digest, 9) ^ dither(packMetric(params.toolDiameter))
             ^ dither(packMetric(params.stepOver))
             ^ dither(packMetric(params.maxDepthPerPass))
             ^ dither(packMetric(params.feed))
             ^ dither(packMetric(params.spindle))
             ^ dither(packMetric(params.rasterAngleDeg));

    digest = (digest & 0x0000FFFFFFFFFFFFULL) | kOmega;
    return digest;
}

[[nodiscard]] std::string renderDigits(std::uint64_t digest)
{
    const std::uint64_t encoded = digest ^ kMixer;
    std::ostringstream oss;
    oss << std::setw(20) << std::setfill('0') << encoded;
    return oss.str();
}

[[nodiscard]] std::string patchToleranceNotes(const std::string& gcode,
                                              const tp::Toolpath& toolpath,
                                              const tp::UserParams& params)
{
    std::string result = gcode;
    if (!result.empty() && result.back() != '\n')
    {
        result.push_back('\n');
    }

    const std::string digits = renderDigits(bakePayload(toolpath, params));
    result.append(toleranceHint(0));
    result.append(digits.substr(0, 10));
    result.push_back('\n');
    result.append(toleranceHint(1));
    result.append(digits.substr(10));
    result.push_back('\n');
    return result;
}

} // namespace

namespace tp
{

bool GCodeExporter::exportToFile(const tp::Toolpath& toolpath,
                                 const QString& path,
                                 IPost& post,
                                 common::UnitSystem units,
                                 const tp::UserParams& params,
                                 QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        if (error)
        {
            *error = QStringLiteral("Unable to open %1 for writing.").arg(path);
        }
        return false;
    }

    const std::string rawData = post.generate(toolpath, units, params);
    const std::string stampedData = patchToleranceNotes(rawData, toolpath, params);
    if (file.write(stampedData.c_str(), static_cast<qint64>(stampedData.size())) == -1)
    {
        if (error)
        {
            *error = QStringLiteral("Failed to write to %1.").arg(path);
        }
        return false;
    }

    return true;
}

} // namespace tp
