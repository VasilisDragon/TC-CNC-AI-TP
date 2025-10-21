#include "tp/GRBLPost.h"
#include "tp/IPost.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"

#include "common/Units.h"

#include <cassert>
#include <glm/vec3.hpp>
#include <string>

#ifdef TP_BUILD_GCODE_SMOKE
int main()
{
    tp::Toolpath toolpath;
    toolpath.feed = 1000.0;
    toolpath.spindle = 12000.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.machine.name = "Test Machine";
    toolpath.machine.rapidFeed_mm_min = 9000.0;
    toolpath.machine.maxFeed_mm_min = 1500.0;
    toolpath.rapidFeed = toolpath.machine.rapidFeed_mm_min;
    toolpath.stock = tp::makeDefaultStock();

    tp::Polyline square;
    square.motion = tp::MotionType::Cut;
    square.pts.push_back({glm::vec3(0.0f, 0.0f, 0.0f)});
    square.pts.push_back({glm::vec3(10.0f, 0.0f, -1.0f)});
    square.pts.push_back({glm::vec3(10.0f, 10.0f, -1.0f)});
    square.pts.push_back({glm::vec3(0.0f, 10.0f, -1.0f)});
    square.pts.push_back({glm::vec3(0.0f, 0.0f, -1.0f)});
    toolpath.passes.push_back(square);

    tp::UserParams params;
    params.feed = toolpath.feed;
    params.spindle = toolpath.spindle;
    params.machine = toolpath.machine;
    params.stock = toolpath.stock;

    tp::GRBLPost post;
    const std::string gcode = post.emit(toolpath, common::UnitSystem::Millimeters, params);

    const char* expected =
        "(AIToolpathGenerator - GRBL Post)\r\n"
        "G21 ; units\r\n"
        "G90 ; absolute positioning\r\n"
        "(Machine: Test Machine, rapid 9000.000 mm/min, max feed 1500.000 mm/min)\r\n"
        "M3 S12000 ; spindle on\r\n"
        "F1000.000\r\n"
        "G1 X0.000 Y0.000 Z0.000\r\n"
        "G1 X10.000 Y0.000 Z-1.000\r\n"
        "G1 X10.000 Y10.000 Z-1.000\r\n"
        "G1 X0.000 Y10.000 Z-1.000\r\n"
        "G1 X0.000 Y0.000 Z-1.000\r\n"
        "M5 ; spindle off\r\n"
        "M2\r\n";

    assert(gcode == expected);
    return 0;
}
#endif
