#include "tp/FanucPost.h"

namespace
{

constexpr std::string_view kFanucHeader =
    "(AIToolpathGenerator - {{post_name}} Post)\r\n"
    "{{work_offset}}\r\n"
    "{{unit_code}}\r\n"
    "{{plane_code}}\r\n"
    "{{positioning_mode}}\r\n"
    "{{feed_mode}}\r\n"
    "{{machine_summary}}\r\n"
    "{{#if spindle_supported}}{{#if spindle_requested}}{{spindle_on_code}} S{{spindle_speed}}\r\n{{/if}}{{/if}}"
    "{{#if spindle_supported}}{{#unless spindle_requested}}{{spindle_on_code}}\r\n{{/unless}}{{/if}}"
    "{{#unless spindle_supported}}(Spindle not supported)\r\n{{/unless}}";

constexpr std::string_view kFanucFooter =
    "{{#if spindle_supported}}{{spindle_off_code}}\r\n{{/if}}"
    "{{program_end_code}}\r\n";

} // namespace

namespace tp
{

std::string FanucPost::name() const
{
    return "Fanuc";
}

std::string_view FanucPost::headerTemplate() const
{
    return kFanucHeader;
}

std::string_view FanucPost::footerTemplate() const
{
    return kFanucFooter;
}

std::string_view FanucPost::workOffset() const
{
    return "G54";
}

std::string_view FanucPost::programEndCode() const
{
    return "M30";
}

} // namespace tp

