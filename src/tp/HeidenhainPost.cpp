#include "tp/HeidenhainPost.h"

namespace
{

constexpr std::string_view kHeidenhainHeader =
    "BEGIN PGM AI{{post_name}} {{unit_keyword}}\n"
    "; Machine: {{machine_plain}}\n"
    "; Rapid {{rapid_feed}} {{unit_suffix}}, Max feed {{max_feed}} {{unit_suffix}}\n"
    "; Feed {{feed_rate}} {{unit_suffix}}\n"
    "{{#if spindle_requested}}; Spindle {{spindle_speed}}\n{{/if}}"
    "{{#if arcs_enabled}}; Arcs retained where possible\n{{else}}; Arcs emitted as linear moves\n{{/if}}";

constexpr std::string_view kHeidenhainFooter =
    "{{#if spindle_requested}}; {{spindle_off_code}} (stop spindle)\n{{/if}}"
    "{{program_end_code}}\n";

constexpr std::string_view kHeidenhainStep =
    "; Step {{step_number}} {{step_label}} {{pass_kind}} stepover={{stepover_mm}}mm "
    "stepdown={{stepdown_mm}}mm{{#if has_angle}} angle={{angle_deg}}deg{{/if}}";

} // namespace

namespace tp
{

std::string HeidenhainPost::name() const
{
    return "Heidenhain";
}

std::string_view HeidenhainPost::headerTemplate() const
{
    return kHeidenhainHeader;
}

std::string_view HeidenhainPost::footerTemplate() const
{
    return kHeidenhainFooter;
}

std::string_view HeidenhainPost::stepBlockTemplate() const
{
    return kHeidenhainStep;
}

std::string_view HeidenhainPost::newline() const
{
    return "\n";
}

bool HeidenhainPost::supportsArcs() const
{
    return false;
}

void HeidenhainPost::emitFeedCommand(std::ostringstream& out, double /*feedUnits*/) const
{
    (void)out;
    // Feed rate is embedded in the move commands for this post.
}

void HeidenhainPost::emitLinearMove(std::ostringstream& out,
                                    const glm::dvec3& point,
                                    MotionType motion,
                                    common::UnitSystem units,
                                    double feedUnits) const
{
    out << "L"
        << " X" << formatNumber(toUnits(point.x, units))
        << " Y" << formatNumber(toUnits(point.y, units))
        << " Z" << formatNumber(toUnits(point.z, units));
    if (motion == MotionType::Cut)
    {
        out << " F" << formatNumber(feedUnits);
    }
    else
    {
        out << " FMAX";
    }
    out << newline();
}

std::string_view HeidenhainPost::programEndCode() const
{
    return "END PGM";
}

void HeidenhainPost::buildHeaderContext(TemplateContext& context,
                                        const tp::Toolpath& toolpath,
                                        common::UnitSystem units,
                                        const tp::UserParams& params,
                                        bool arcsEnabled) const
{
    GCodePostBase::buildHeaderContext(context, toolpath, units, params, arcsEnabled);
    context.set("unit_keyword", (units == common::UnitSystem::Inches) ? "INCH" : "MM");
    context.set("machine_plain", toolpath.machine.name);
}

} // namespace tp

