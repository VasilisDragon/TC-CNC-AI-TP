#pragma once

#include "tp/GCodePostBase.h"

#include <string_view>

namespace tp
{

class MarlinPost : public GCodePostBase
{
public:
    MarlinPost() = default;

    std::string name() const override;

protected:
    std::string_view headerTemplate() const override;
    std::string_view footerTemplate() const override;
    std::string_view newline() const override;
    bool spindleSupported() const override;
    std::string_view programEndCode() const override;
};

} // namespace tp

