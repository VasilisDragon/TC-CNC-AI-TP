#pragma once

#include "tp/Toolpath.h"
#include "ai/IPathAI.h"

#include <atomic>
#include <functional>
#include <string>

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
    Toolpath generateRasterTopography(const render::Model& model,
                                      const UserParams& params,
                                      const ai::StrategyDecision& decision,
                                      const std::atomic<bool>& cancelFlag,
                                      const std::function<void(int)>& progressCallback,
                                      std::string* logMessage) const;

    Toolpath generateWaterlineSlicer(const render::Model& model,
                                     const UserParams& params,
                                     const ai::StrategyDecision& decision,
                                     const std::atomic<bool>& cancelFlag,
                                     const std::function<void(int)>& progressCallback,
                                     std::string* logMessage) const;

    Toolpath generateFallbackRaster(const render::Model& model,
                                    const UserParams& params,
                                    const ai::StrategyDecision& decision,
                                    const std::atomic<bool>& cancelFlag,
                                    const std::function<void(int)>& progressCallback) const;
};

} // namespace tp
