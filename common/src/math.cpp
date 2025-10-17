#include "common/math.h"

#include <cmath>

namespace common
{

QMatrix4x4 perspectiveRadians(float fovRadians, float aspect, float nearPlane, float farPlane)
{
    QMatrix4x4 matrix;
    constexpr float radToDeg = 180.0f / 3.14159265358979323846f;
    matrix.perspective(fovRadians * radToDeg, aspect, nearPlane, farPlane);
    return matrix;
}

} // namespace common
