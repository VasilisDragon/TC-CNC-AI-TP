#include "tp/FanucPost.h"
#include "tp/GRBLPost.h"
#include "tp/HeidenhainPost.h"
#include "tp/MarlinPost.h"
#include "tp/Toolpath.h"

#include "common/Units.h"
#include "common/log.h"

#include <glm/vec3.hpp>

#include <sstream>
#include <string>

namespace
{

tp::Toolpath makeSampleToolpath()
{
    tp::Toolpath toolpath;
    toolpath.feed = 1200.0;
    toolpath.spindle = 10'000.0;
    toolpath.rapidFeed = 2'500.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.machine.name = "TemplateHarness";
    toolpath.machine.rapidFeed_mm_min = 2'500.0;
    toolpath.machine.maxFeed_mm_min = 3'000.0;
    toolpath.stock = tp::makeDefaultStock();

    tp::Polyline poly;
    poly.motion = tp::MotionType::Cut;
    poly.pts.push_back({glm::vec3(0.0f, 0.0f, 0.0f)});
    poly.pts.push_back({glm::vec3(10.0f, 0.0f, -1.0f)});
    poly.pts.push_back({glm::vec3(10.0f, 10.0f, -1.0f)});
    toolpath.passes.push_back(poly);

    return toolpath;
}

tp::UserParams makeParams()
{
    tp::UserParams params;
    params.post.maxArcChordError_mm = 0.1;
    return params;
}

bool verifyContains(const std::string& text, const std::string& needle, const char* label)
{
    if (text.find(needle) != std::string::npos)
    {
        return true;
    }
    std::ostringstream oss;
    oss << "Post template validation failed: missing '" << needle << "' in " << label
        << " output. Update the template to emit the expected token.\n"
        << text;
    LOG_ERR(Tp, oss.str());
    return false;
}

bool verifyNotContains(const std::string& text, const std::string& needle, const char* label)
{
    if (text.find(needle) == std::string::npos)
    {
        return true;
    }
    std::ostringstream oss;
    oss << "Post template validation failed: unexpected '" << needle << "' in " << label
        << " output. Remove the extra token from the generator.\n"
        << text;
    LOG_ERR(Tp, oss.str());
    return false;
}

} // namespace

int main()
{
    const tp::Toolpath toolpath = makeSampleToolpath();
    tp::UserParams params = makeParams();

    {
        tp::GRBLPost post;
        const std::string gcode = post.generate(toolpath, common::UnitSystem::Millimeters, params);
        if (!verifyContains(gcode, "(AIToolpathGenerator - GRBL Post)", "GRBL")
            || !verifyContains(gcode, "G21 ; units", "GRBL")
            || !verifyContains(gcode, "M5 ; spindle off", "GRBL")
            || !verifyContains(gcode, "M2", "GRBL")
            || !verifyContains(gcode, "G1 X10.000", "GRBL"))
        {
            return 1;
        }
    }

    {
        tp::FanucPost post;
        const std::string gcode = post.generate(toolpath, common::UnitSystem::Millimeters, params);
        if (!verifyContains(gcode, "G54", "Fanuc")
            || !verifyContains(gcode, "G90", "Fanuc")
            || !verifyContains(gcode, "G17", "Fanuc")
            || !verifyContains(gcode, "G94", "Fanuc")
            || !verifyContains(gcode, "M30", "Fanuc")
            || !verifyContains(gcode, "M3 S10000.000", "Fanuc"))
        {
            return 2;
        }
    }

    {
        tp::MarlinPost post;
        const std::string gcode = post.generate(toolpath, common::UnitSystem::Millimeters, params);
        if (!verifyContains(gcode, "; AIToolpathGenerator - Marlin Post", "Marlin")
            || !verifyContains(gcode, "; Requested spindle 10000.000 but controller has no spindle", "Marlin")
            || !verifyContains(gcode, "; Arcs enabled (G2/G3)", "Marlin")
            || !verifyNotContains(gcode, "M3", "Marlin")
            || !verifyContains(gcode, "M84", "Marlin"))
        {
            return 3;
        }

        params.post.maxArcChordError_mm = 0.0;
        const std::string linearized = post.generate(toolpath, common::UnitSystem::Millimeters, params);
        if (!verifyContains(linearized, "; Arcs disabled (linearized)", "Marlin"))
        {
            return 4;
        }
        params.post.maxArcChordError_mm = 0.1;
    }

    {
        tp::HeidenhainPost post;
        const std::string gcode = post.generate(toolpath, common::UnitSystem::Millimeters, params);
        if (!verifyContains(gcode, "BEGIN PGM AIHeidenhain MM", "Heidenhain")
            || !verifyContains(gcode, "; Machine: TemplateHarness", "Heidenhain")
            || !verifyContains(gcode, "L X10.000 Y0.000 Z-1.000 F1200.000", "Heidenhain")
            || !verifyContains(gcode, "END PGM", "Heidenhain")
            || !verifyNotContains(gcode, "G1", "Heidenhain")
            || !verifyNotContains(gcode, "G2", "Heidenhain")
            || !verifyContains(gcode, "; Arcs emitted as linear moves", "Heidenhain"))
        {
            return 5;
        }
    }

    return 0;
}
