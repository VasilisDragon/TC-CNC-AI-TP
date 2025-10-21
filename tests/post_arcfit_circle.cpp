#include "tp/GRBLPost.h"
#include "tp/Toolpath.h"

#include "common/Units.h"

#include <cassert>
#include <cmath>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace
{

tp::Toolpath buildCircleToolpath(double radiusMm, int samples)
{
    tp::Toolpath toolpath;
    toolpath.feed = 1200.0;
    toolpath.spindle = 18'000.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.machine.name = "ArcFitHarness";
    toolpath.machine.rapidFeed_mm_min = 9'000.0;
    toolpath.machine.maxFeed_mm_min = 2'400.0;
    toolpath.rapidFeed = toolpath.machine.rapidFeed_mm_min;
    toolpath.stock = tp::makeDefaultStock();

    tp::Polyline circle;
    circle.motion = tp::MotionType::Cut;

    for (int i = 0; i <= samples; ++i)
    {
        const double angle = (2.0 * std::numbers::pi * static_cast<double>(i)) / static_cast<double>(samples);
        const double x = radiusMm * std::cos(angle);
        const double y = radiusMm * std::sin(angle);
        circle.pts.push_back({glm::vec3(static_cast<float>(x),
                                        static_cast<float>(y),
                                        -1.5f)});
    }

    toolpath.passes.push_back(circle);
    return toolpath;
}

int countMotionCommands(const std::string& gcode, const std::string& code)
{
    std::istringstream stream(gcode);
    std::string line;
    int count = 0;
    while (std::getline(stream, line))
    {
        std::istringstream lineStream(line);
        std::string token;
        if (!(lineStream >> token))
        {
            continue;
        }
        if (token == code)
        {
            ++count;
        }
    }
    return count;
}

} // namespace

int main()
{
    constexpr double kRadius = 20.0;
    constexpr int kSamples = 48;

    tp::Toolpath toolpath = buildCircleToolpath(kRadius, kSamples);
    tp::UserParams params;
    params.post.maxArcChordError_mm = 0.05;

    tp::GRBLPost post;
    const std::string gcode = post.generate(toolpath, common::UnitSystem::Millimeters, params);

    const int arcMoves = countMotionCommands(gcode, "G2") + countMotionCommands(gcode, "G3");
    assert(arcMoves >= 1 && arcMoves <= 2);

    const int linearMoves = countMotionCommands(gcode, "G1");
    assert(linearMoves <= 3);

    return 0;
}
