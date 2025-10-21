#pragma once

#include "render/Model.h"
#include "tp/TriangleGrid.h"

#include <glm/vec3.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tp
{

/**
 * Parameters describing the cutter/holder envelope and desired clearances.
 */
struct GougeParams
{
    double toolRadius{0.0};
    double holderRadius{0.0};
    double leaveStock{0.0};
    double safetyZ{0.0};
};

class GougeChecker
{
public:
    using Vec3 = glm::vec3;

    explicit GougeChecker(const render::Model& model);

    [[nodiscard]] double minClearanceAlong(const std::vector<Vec3>& path, const GougeParams& params) const;
    [[nodiscard]] std::optional<double> surfaceHeightAt(const Vec3& sample) const;

    struct AdjustResult
    {
        std::vector<Vec3> adjustedPath;
        double minClearance{std::numeric_limits<double>::infinity()};
        bool adjusted{false};
        bool ok{true};
        std::string message;
    };

    [[nodiscard]] AdjustResult adjustZForLeaveStock(const std::vector<Vec3>& path,
                                                    const GougeParams& params) const;

private:
    struct ClosestHit
    {
        bool hit{false};
        double distance{std::numeric_limits<double>::infinity()};
        Vec3 closestPoint{0.0f};
    };

    [[nodiscard]] ClosestHit closestPoint(const Vec3& point) const;

    TriangleGrid m_grid;
    mutable std::vector<std::uint32_t> m_candidateScratch;
};

} // namespace tp
