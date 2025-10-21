#include "tp/GRBLPost.h"

namespace
{

constexpr std::string_view kHeaderTemplate =
    "(AIToolpathGenerator - {{post_name}} Post)\r\n"
    "{{unit_code}} ; units\r\n"
    "{{positioning_mode}} ; absolute positioning\r\n"
    "{{machine_summary}}\r\n"
    "{{#if spindle_supported}}{{#if spindle_requested}}{{spindle_on_code}} S{{spindle_speed}} ; spindle on\r\n{{/if}}{{/if}}"
    "{{#if spindle_supported}}{{#unless spindle_requested}}{{spindle_on_code}} ; spindle on\r\n{{/unless}}{{/if}}"
    "{{#unless spindle_supported}}; Spindle not supported\r\n{{/unless}}";

constexpr std::string_view kFooterTemplate =
    "{{#if spindle_supported}}{{spindle_off_code}} ; spindle off\r\n{{/if}}"
    "{{program_end_code}}";

} // namespace

namespace tp
{

std::string GRBLPost::name() const
{
    return "GRBL";
}

std::string_view GRBLPost::headerTemplate() const
{
    return kHeaderTemplate;
}

std::string_view GRBLPost::footerTemplate() const
{
    return kFooterTemplate;
}

} // namespace tp

