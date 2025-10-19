#pragma once

#include "tp/Toolpath.h"
#include "ai/IPathAI.h"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace render
{
class Model;
}

namespace tp
{

struct UserParams
{
    enum class CutterType
    {
        FlatEndmill,
        BallNose
    };

    double toolDiameter{6.0};
    double stepOver{3.0};
    double maxDepthPerPass{1.0};
    double feed{800.0};
    double spindle{12'000.0};
    double rasterAngleDeg{0.0};
    bool enableRoughPass{true};
    bool enableFinishPass{true};
    double stockAllowance_mm{0.3};
    double leaveStock_mm{0.3};
    double rampAngleDeg{3.0};
    bool useHeightField{true};
    CutterType cutterType{CutterType::FlatEndmill};
    Stock stock{makeDefaultStock()};
    Machine machine{makeDefaultMachine()};
};

class ToolpathGenerator
{
public:
    ToolpathGenerator() = default;

    Toolpath generate(const render::Model& model,
                      const UserParams& params,
                      ai::IPathAI& ai,
                      const std::atomic<bool>& cancelFlag,
                      const std::function<void(int)>& progressCallback = {},
                      ai::StrategyDecision* outDecision = nullptr,
                      std::string* bannerMessage = nullptr) const;

private:
    struct PassProfile
    {
        enum class Kind
        {
            Rough,
            Finish
        };

        Kind kind{Kind::Finish};
        double stepOver{0.0};
        double maxStepDown{0.0};
        double allowance{0.0};
    };

    static const char* passLabel(const PassProfile& profile);
    static std::string makePassLog(const PassProfile& profile, const std::string& message);
    static std::vector<PassProfile> buildPassPlan(const UserParams& params,
                                                  const ai::StrategyDecision& decision);

    Toolpath generateRasterTopography(const render::Model& model,
                                      const UserParams& params,
                                      const ai::StrategyDecision& decision,
                                      const PassProfile& profile,
                                      const std::atomic<bool>& cancelFlag,
                                      const std::function<void(int)>& progressCallback,
                                      std::string* logMessage) const;

    Toolpath generateWaterlineSlicer(const render::Model& model,
                                     const UserParams& params,
                                     const ai::StrategyDecision& decision,
                                     const PassProfile& profile,
                                     const std::atomic<bool>& cancelFlag,
                                     const std::function<void(int)>& progressCallback,
                                      std::string* logMessage) const;

    Toolpath generateFallbackRaster(const render::Model& model,
                                    const UserParams& params,
                                    const ai::StrategyDecision& decision,
                                    const PassProfile& profile,
                                    const std::atomic<bool>& cancelFlag,
                                    const std::function<void(int)>& progressCallback) const;
    void applyLeaveStockAdjustment(Toolpath& toolpath, const render::Model& model, const UserParams& params) const;
};

} // namespace tp
