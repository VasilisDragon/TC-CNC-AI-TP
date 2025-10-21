#include "tp/MarlinPost.h"

namespace
{

constexpr std::string_view kMarlinHeader =
    "; AIToolpathGenerator - {{post_name}} Post\n"
    "{{unit_code}} ; units\n"
    "{{positioning_mode}} ; absolute positioning\n"
    "; {{machine_summary}}\n"
    "{{#if spindle_requested}}; Requested spindle {{spindle_speed}} but controller has no spindle\n{{/if}}"
    "{{#if arcs_enabled}}; Arcs enabled (G2/G3)\n{{else}}; Arcs disabled (linearized)\n{{/if}}";

constexpr std::string_view kMarlinFooter =
    "M400 ; wait for moves to finish\n"
    "{{program_end_code}} ; disable motors\n";

} // namespace

namespace tp
{

std::string MarlinPost::name() const
{
    return "Marlin";
}

std::string_view MarlinPost::headerTemplate() const
{
    return kMarlinHeader;
}

std::string_view MarlinPost::footerTemplate() const
{
    return kMarlinFooter;
}

std::string_view MarlinPost::newline() const
{
    return "\n";
}

bool MarlinPost::spindleSupported() const
{
    return false;
}

std::string_view MarlinPost::programEndCode() const
{
    return "M84";
}

} // namespace tp

