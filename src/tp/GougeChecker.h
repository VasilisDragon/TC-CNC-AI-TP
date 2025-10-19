#pragma once

#include "render/Model.h"

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
    struct Triangle
    {
        Vec3 a;
        Vec3 b;
        Vec3 c;
        Vec3 normal;
        Vec3 minBounds;
        Vec3 maxBounds;
    };

    struct ClosestHit
    {
        bool hit{false};
        double distance{std::numeric_limits<double>::infinity()};
        Vec3 closestPoint{0.0f};
    };

    [[nodiscard]] ClosestHit closestPoint(const Vec3& point) const;
    [[nodiscard]] std::vector<std::uint32_t> gatherCandidates(const Vec3& point) const;

    std::vector<Triangle> m_triangles;
    Vec3 m_minBounds{0.0f};
    Vec3 m_maxBounds{0.0f};
    int m_cellsX{1};
    int m_cellsY{1};
    double m_invCellSizeX{0.0};
    double m_invCellSizeY{0.0};
    std::vector<std::vector<std::uint32_t>> m_grid;
};

} // namespace tp
