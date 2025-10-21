#pragma once

#include "render/Model.h"

#include <glm/vec3.hpp>

#include <vector>

namespace tp::waterline
{

class ZSlicer
{
public:
    explicit ZSlicer(const render::Model& model, double toleranceMm = 1e-4);

    enum class SliceMode
    {
        Sequential,
        Parallel
    };

    std::vector<std::vector<glm::dvec3>> slice(double planeZ,
                                               double toolRadius,
                                               bool applyOffsetForFlat,
                                               SliceMode mode) const;

    std::vector<std::vector<glm::dvec3>> slice(double planeZ,
                                               double toolRadius,
                                               bool applyOffsetForFlat) const
    {
        return slice(planeZ, toolRadius, applyOffsetForFlat, SliceMode::Parallel);
    }

    [[nodiscard]] double minZ() const noexcept { return m_minZ; }
    [[nodiscard]] double maxZ() const noexcept { return m_maxZ; }

private:
    struct Triangle
    {
        glm::dvec3 v0;
        glm::dvec3 v1;
        glm::dvec3 v2;
        double minZ{0.0};
        double maxZ{0.0};
    };

    double m_tolerance{1e-4};
    double m_minZ{0.0};
    double m_maxZ{0.0};
    std::vector<Triangle> m_triangles;
};

#ifdef TP_ENABLE_ZSLICER_BENCHMARK
void runZSlicerBenchmark(const ZSlicer& slicer,
                         double planeZ,
                         double toolRadius,
                         bool applyOffsetForFlat,
                         std::size_t iterations = 12);
#endif

} // namespace tp::waterline
