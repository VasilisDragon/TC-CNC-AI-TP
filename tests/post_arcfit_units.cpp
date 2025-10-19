#include "tp/GRBLPost.h"
#include "tp/Toolpath.h"

#include "common/Units.h"

#include <algorithm>
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
    toolpath.feed = 1600.0;
    toolpath.spindle = 15'000.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.machine.name = "ArcUnitHarness";
    toolpath.machine.maxFeed_mm_min = 3'000.0;
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
                                        -2.0f)});
    }

    toolpath.passes.push_back(circle);
    return toolpath;
}

std::vector<std::string> extractMotionCodes(const std::string& gcode)
{
    std::istringstream stream(gcode);
    std::string line;
    std::vector<std::string> codes;

    while (std::getline(stream, line))
    {
        std::istringstream lineStream(line);
        std::string token;
        if (!(lineStream >> token))
        {
            continue;
        }
        if (token.empty() || token[0] != 'G')
        {
            continue;
        }
        if (token == "G20" || token == "G21" || token == "G90")
        {
            continue;
        }
        codes.push_back(token);
    }

    return codes;
}

} // namespace

int main()
{
    tp::Toolpath toolpath = buildCircleToolpath(12.5, 36);
    tp::UserParams params;
    params.post.maxArcChordError_mm = 0.05;

    tp::GRBLPost post;
    const std::string mmGcode = post.generate(toolpath, common::Unit::Millimeters, params);
    const std::string inchGcode = post.generate(toolpath, common::Unit::Inches, params);

    const std::vector<std::string> mmCodes = extractMotionCodes(mmGcode);
    const std::vector<std::string> inchCodes = extractMotionCodes(inchGcode);

    assert(mmCodes.size() == inchCodes.size());
    assert(!mmCodes.empty());

    for (std::size_t i = 0; i < mmCodes.size(); ++i)
    {
        assert(mmCodes[i] == inchCodes[i]);
    }

    const int arcMoves = static_cast<int>(std::count(mmCodes.begin(), mmCodes.end(), "G2")
                                          + std::count(mmCodes.begin(), mmCodes.end(), "G3"));
    assert(arcMoves >= 1);

    return 0;
}
