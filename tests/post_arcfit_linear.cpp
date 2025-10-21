#include "tp/GRBLPost.h"
#include "tp/Toolpath.h"

#include "common/Units.h"

#include <cassert>
#include <sstream>
#include <string>

namespace
{

tp::Toolpath buildLShapedToolpath()
{
    tp::Toolpath toolpath;
    toolpath.feed = 900.0;
    toolpath.spindle = 12'000.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.rapidFeed = toolpath.machine.rapidFeed_mm_min;
    toolpath.stock = tp::makeDefaultStock();

    tp::Polyline path;
    path.motion = tp::MotionType::Cut;
    path.pts.push_back({glm::vec3(0.0f, 0.0f, -0.5f)});
    path.pts.push_back({glm::vec3(15.0f, 0.0f, -0.5f)});
    path.pts.push_back({glm::vec3(15.0f, 10.0f, -0.5f)});
    toolpath.passes.push_back(path);
    return toolpath;
}

int countArcCommands(const std::string& gcode)
{
    std::istringstream stream(gcode);
    std::string line;
    int arcs = 0;
    while (std::getline(stream, line))
    {
        std::istringstream lineStream(line);
        std::string token;
        if (!(lineStream >> token))
        {
            continue;
        }
        if (token == "G2" || token == "G3")
        {
            ++arcs;
        }
    }
    return arcs;
}

} // namespace

int main()
{
    tp::Toolpath toolpath = buildLShapedToolpath();
    tp::UserParams params;
    params.post.maxArcChordError_mm = 0.05;

    tp::GRBLPost post;
    const std::string gcode = post.generate(toolpath, common::UnitSystem::Millimeters, params);

    const int arcMoves = countArcCommands(gcode);
    assert(arcMoves == 0);

    return 0;
}
