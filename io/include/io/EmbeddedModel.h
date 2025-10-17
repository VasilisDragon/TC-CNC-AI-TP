#pragma once

#include <string_view>

namespace io
{

inline constexpr std::string_view kEmbeddedModelName = "embedded_calibration";

inline constexpr std::string_view kEmbeddedModelObj = R"(# Embedded sanity mesh
o embedded_calibration
v -10.0 -10.0 0.0
v 10.0 -10.0 0.0
v 10.0 10.0 0.0
v -10.0 10.0 0.0
v 0.0 0.0 5.0
vn 0.0 0.0 1.0
vn 0.0 0.0 1.0
vn 0.0 0.0 1.0
vn 0.0 0.0 1.0
vn 0.0 0.8944 0.4472
f 1//1 2//2 3//3
f 1//1 3//3 4//4
f 1//5 2//5 5//5
f 2//5 3//5 5//5
f 3//5 4//5 5//5
f 4//5 1//5 5//5
)";

} // namespace io

