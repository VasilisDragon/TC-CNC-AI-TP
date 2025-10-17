#pragma once

#include <QtCore/QMetaType>

#include <memory>

namespace render
{
class Model;
}

namespace tp
{
struct UserParams;
}

namespace ai
{

struct StrategyDecision
{
    enum class Strategy
    {
        Raster = 0,
        Waterline = 1
    };

    Strategy strat{Strategy::Raster};
    double rasterAngleDeg{0.0};
    double stepOverMM{0.0};
    bool roughPass{true};
    bool finishPass{true};
};

class IPathAI
{
public:
    virtual ~IPathAI() = default;

    virtual StrategyDecision predict(const render::Model& model,
                                     const tp::UserParams& params) = 0;
};

} // namespace ai

Q_DECLARE_METATYPE(ai::StrategyDecision)
