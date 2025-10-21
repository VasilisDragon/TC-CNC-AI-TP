#pragma once

#include "tp/IPost.h"
#include "tp/TemplateEngine.h"
#include "tp/Toolpath.h"

#include "common/Units.h"

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>

namespace ai
{
struct StrategyStep;
} // namespace ai

namespace tp
{

class GCodePostBase : public IPost
{
public:
    std::string generate(const tp::Toolpath& toolpath,
                         common::UnitSystem units,
                         const tp::UserParams& params) override;

protected:
    virtual std::string_view headerTemplate() const = 0;
    virtual std::string_view footerTemplate() const = 0;
    virtual std::string_view stepBlockTemplate() const;

    virtual void buildHeaderContext(TemplateContext& context,
                                    const tp::Toolpath& toolpath,
                                    common::UnitSystem units,
                                    const tp::UserParams& params,
                                    bool arcsEnabled) const;
    virtual void buildFooterContext(TemplateContext& context,
                                    const tp::Toolpath& toolpath,
                                    common::UnitSystem units,
                                    const tp::UserParams& params,
                                    bool arcsEnabled) const;
    virtual void buildStepContext(TemplateContext& context,
                                  const ai::StrategyStep& step,
                                  std::size_t stepIndex) const;

    virtual bool spindleSupported() const { return true; }
    virtual bool supportsArcs() const { return true; }
    virtual bool allowArcs(const tp::UserParams& params) const;

    virtual std::string_view positioningMode() const { return "G90"; }
    virtual std::string_view planeCode() const { return "G17"; }
    virtual std::string_view feedMode() const { return "G94"; }
    virtual std::string_view workOffset() const { return {}; }
    virtual std::string_view spindleOnCode() const { return "M3"; }
    virtual std::string_view spindleOffCode() const { return "M5"; }
    virtual std::string_view programEndCode() const { return "M2"; }
    virtual std::string_view newline() const { return "\r\n"; }

    virtual void emitFeedCommand(std::ostringstream& out, double feedUnits) const;
    virtual void emitLinearMove(std::ostringstream& out,
                                const glm::dvec3& point,
                                MotionType motion,
                                common::UnitSystem units,
                                double feedUnits) const;
    virtual void emitArcMove(std::ostringstream& out,
                             bool clockwise,
                             const glm::dvec3& start,
                             const glm::dvec3& end,
                             const glm::dvec2& center,
                             common::UnitSystem units,
                             double feedUnits) const;

    static double toUnits(double valueMm, common::UnitSystem units);
    static std::string formatNumber(double value, int precision = 3);

private:
    void emitPolyline(std::ostringstream& out,
                      const tp::Polyline& poly,
                      common::UnitSystem units,
                      double feedUnits,
                      bool arcsEnabled,
                      double maxChordError) const;
};

} // namespace tp

