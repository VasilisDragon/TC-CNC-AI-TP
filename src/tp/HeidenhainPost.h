#pragma once

#include "tp/GCodePostBase.h"

#include <sstream>
#include <string_view>

namespace tp
{

class HeidenhainPost : public GCodePostBase
{
public:
    HeidenhainPost() = default;

    std::string name() const override;

protected:
    std::string_view headerTemplate() const override;
    std::string_view footerTemplate() const override;
    std::string_view stepBlockTemplate() const override;
    std::string_view newline() const override;
    bool supportsArcs() const override;
    void emitFeedCommand(std::ostringstream& out, double feedUnits) const override;
    void emitLinearMove(std::ostringstream& out,
                        const glm::dvec3& point,
                        MotionType motion,
                        common::UnitSystem units,
                        double feedUnits) const override;
    std::string_view programEndCode() const override;
    void buildHeaderContext(TemplateContext& context,
                            const tp::Toolpath& toolpath,
                            common::UnitSystem units,
                            const tp::UserParams& params,
                            bool arcsEnabled) const override;
};

} // namespace tp
