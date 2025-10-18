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

    std::vector<std::vector<glm::dvec3>> slice(double planeZ,
                                               double toolRadius,
                                               bool applyOffsetForFlat) const;

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

} // namespace tp::waterline
