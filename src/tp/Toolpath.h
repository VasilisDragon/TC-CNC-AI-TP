#pragma once

#include "tp/Machine.h"
#include "tp/Stock.h"

#include <glm/vec3.hpp>

#include <vector>

namespace tp
{

enum class MotionType
{
    Cut,
    Link,
    Rapid
};

struct Vertex
{
    glm::vec3 p{0.0f};
};

struct Polyline
{
    std::vector<Vertex> pts;
    MotionType motion{MotionType::Cut};

    [[nodiscard]] bool isRapid() const noexcept
    {
        return motion != MotionType::Cut;
    }
};

struct Toolpath
{
    std::vector<Polyline> passes;
    double feed{0.0};
    double spindle{0.0};
    double rapidFeed{0.0};
    Machine machine{};
    Stock stock{};

    [[nodiscard]] bool empty() const noexcept
    {
        return passes.empty();
    }
};

} // namespace tp
