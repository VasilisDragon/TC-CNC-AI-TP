#include "tp/GRBLPost.h"

#include <QtCore/QLocale>

#include <sstream>
#include <iomanip>

namespace
{

constexpr const char* kNL = "\r\n";

double toUnits(double valueMm, common::Unit units)
{
    return (units == common::Unit::Inches)
               ? common::fromMillimeters(valueMm, common::Unit::Inches)
               : valueMm;
}

std::string unitPrefix(common::Unit units)
{
    return (units == common::Unit::Inches) ? "G20" : "G21";
}

} // namespace

namespace tp
{

std::string GRBLPost::name() const
{
    return "GRBL";
}

std::string GRBLPost::emit(const tp::Toolpath& toolpath,
                           common::Unit units,
                           const tp::UserParams& params)
{
    (void)params;
    std::ostringstream out;
    out << std::fixed << std::setprecision(3);

    out << "(AIToolpathGenerator - GRBL Post)" << kNL;
    out << unitPrefix(units) << " ; units" << kNL;
    out << "G90 ; absolute positioning" << kNL;
    const Machine& machine = toolpath.machine;
    const double rapidUnits = (units == common::Unit::Inches)
                                  ? common::fromMillimeters(machine.rapidFeed_mm_min, common::Unit::Inches)
                                  : machine.rapidFeed_mm_min;
    const double maxFeedUnits = (units == common::Unit::Inches)
                                    ? common::fromMillimeters(machine.maxFeed_mm_min, common::Unit::Inches)
                                    : machine.maxFeed_mm_min;
    out << "(Machine: " << machine.name
        << ", rapid " << rapidUnits << (units == common::Unit::Inches ? " in/min" : " mm/min")
        << ", max feed " << maxFeedUnits << (units == common::Unit::Inches ? " in/min" : " mm/min") << ")" << kNL;
    out << "M3 S" << toolpath.spindle << " ; spindle on" << kNL;

    const double feedUnits = (units == common::Unit::Inches)
                                 ? common::fromMillimeters(toolpath.feed, common::Unit::Inches)
                                 : toolpath.feed;

    for (const tp::Polyline& poly : toolpath.passes)
    {
        if (poly.pts.size() < 2)
        {
            continue;
        }

        const bool isCut = (poly.motion == MotionType::Cut);
        const char* code = isCut ? "G1" : "G0";
        if (isCut)
        {
            out << "F" << feedUnits << kNL;
        }

        for (const tp::Vertex& vertex : poly.pts)
        {
            const auto& p = vertex.p;
            out << code << " X" << toUnits(p.x, units)
                << " Y" << toUnits(p.y, units)
                << " Z" << toUnits(p.z, units)
                << kNL;
        }
    }

    out << "M5 ; spindle off" << kNL;
    out << "M2" << kNL;
    return out.str();
}

} // namespace tp
