#include "tests_core/TestRegistry.h"

#include "common/Units.h"
#include "doctest/doctest.h"
#include "tp/GRBLPost.h"
#include "tp/Toolpath.h"

#include <glm/vec3.hpp>

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace
{

std::vector<std::string> collectLinearMoves(const std::string& gcode)
{
    std::istringstream stream(gcode);
    std::string line;
    std::vector<std::string> moves;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.rfind("G0 ", 0) == 0 || line.rfind("G1 ", 0) == 0)
        {
            moves.push_back(line);
        }
    }
    return moves;
}

double axisValue(const std::string& line, char axis)
{
    const std::size_t axisPos = line.find(axis);
    DOCTEST_CHECK(axisPos != std::string::npos);
    if (axisPos == std::string::npos)
    {
        return 0.0;
    }
    const std::size_t start = axisPos + 1;
    const std::size_t end = line.find(' ', start);
    return std::stod(line.substr(start, end - start));
}

tp::Toolpath makeToolpath()
{
    tp::Toolpath toolpath;
    toolpath.feed = 1800.0;
    toolpath.spindle = 16000.0;
    toolpath.machine = tp::makeDefaultMachine();
    toolpath.machine.name = "Unit Harness";
    toolpath.machine.rapidFeed_mm_min = 7200.0;
    toolpath.machine.maxFeed_mm_min = 2500.0;
    toolpath.rapidFeed = toolpath.machine.rapidFeed_mm_min;
    toolpath.stock = tp::makeDefaultStock();

    tp::Polyline cut;
    cut.motion = tp::MotionType::Cut;
    cut.pts.push_back({glm::vec3(0.0f, 0.0f, 0.0f)});
    cut.pts.push_back({glm::vec3(10.0f, 5.0f, -1.5f)});
    toolpath.passes.push_back(cut);

    return toolpath;
}

double mmToInches(double mm)
{
    return common::fromMillimeters(mm, common::UnitSystem::Inches);
}

double toPostPrecision(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << value;
    return std::stod(stream.str());
}

} // namespace

TESTS_CORE_TEST_CASE(unit_system_round_trip_is_stable, "fast")
{
    const std::vector<double> samplesMm{0.0, 0.0254, 1.0, 12.7, 42.1234, 254.0};
    for (const double mm : samplesMm)
    {
        const double inches = common::fromMillimeters(mm, common::UnitSystem::Inches);
        const double roundTrip = common::toMillimeters(inches, common::UnitSystem::Inches);
        DOCTEST_CHECK(roundTrip == doctest::Approx(mm).epsilon(1e-6));
    }
}

TESTS_CORE_TEST_CASE(grbl_post_respects_selected_units, "fast")
{
    const tp::Toolpath toolpath = makeToolpath();
    tp::UserParams params;
    params.feed = toolpath.feed;
    params.spindle = toolpath.spindle;
    params.machine = toolpath.machine;
    params.stock = toolpath.stock;

    tp::GRBLPost post;
    const std::string mmGcode = post.generate(toolpath, common::UnitSystem::Millimeters, params);
    const std::string inchGcode = post.generate(toolpath, common::UnitSystem::Inches, params);

    DOCTEST_CHECK(mmGcode.find("G21 ; units") != std::string::npos);
    DOCTEST_CHECK(inchGcode.find("G20 ; units") != std::string::npos);

    const std::vector<std::string> mmMoves = collectLinearMoves(mmGcode);
    const std::vector<std::string> inchMoves = collectLinearMoves(inchGcode);
    DOCTEST_CHECK(mmMoves.size() == inchMoves.size());
    DOCTEST_CHECK(mmMoves.size() >= 2);
    if (mmMoves.size() != inchMoves.size() || mmMoves.size() < 2)
    {
        return;
    }

    DOCTEST_CHECK(axisValue(mmMoves[0], 'X') == doctest::Approx(0.0));
    DOCTEST_CHECK(axisValue(mmMoves[0], 'Y') == doctest::Approx(0.0));
    DOCTEST_CHECK(axisValue(mmMoves[0], 'Z') == doctest::Approx(0.0));
    DOCTEST_CHECK(axisValue(inchMoves[0], 'X') == doctest::Approx(0.0));
    DOCTEST_CHECK(axisValue(inchMoves[0], 'Y') == doctest::Approx(0.0));
    DOCTEST_CHECK(axisValue(inchMoves[0], 'Z') == doctest::Approx(0.0));

    DOCTEST_CHECK(axisValue(mmMoves[1], 'X') == doctest::Approx(10.0));
    DOCTEST_CHECK(axisValue(mmMoves[1], 'Y') == doctest::Approx(5.0));
    DOCTEST_CHECK(axisValue(mmMoves[1], 'Z') == doctest::Approx(-1.5));

    DOCTEST_CHECK(axisValue(inchMoves[1], 'X')
                  == doctest::Approx(toPostPrecision(mmToInches(10.0))).epsilon(1e-6));
    DOCTEST_CHECK(axisValue(inchMoves[1], 'Y')
                  == doctest::Approx(toPostPrecision(mmToInches(5.0))).epsilon(1e-6));
    DOCTEST_CHECK(axisValue(inchMoves[1], 'Z')
                  == doctest::Approx(toPostPrecision(mmToInches(-1.5))).epsilon(1e-6));

    std::ostringstream feedInches;
    feedInches << std::fixed << std::setprecision(3) << mmToInches(toolpath.feed);
    DOCTEST_CHECK(mmGcode.find("F1800.000") != std::string::npos);
    DOCTEST_CHECK(inchGcode.find("F" + feedInches.str()) != std::string::npos);
}
