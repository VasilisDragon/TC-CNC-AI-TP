#pragma once

#include <glm/vec3.hpp>

namespace tp
{

struct Stock
{
    enum class Shape
    {
        Block
    };

    Shape shape{Shape::Block};
    glm::dvec3 sizeXYZ_mm{0.0};
    glm::dvec3 originXYZ_mm{0.0};
    double topZ_mm{0.0};
    double margin_mm{0.0};

    void ensureValid();
};

Stock makeDefaultStock();

} // namespace tp

