#pragma once

#include <QtCore/QMetaType>

#include <memory>
#include <vector>

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

struct StrategyStep
{
    enum class Type
    {
        Raster = 0,
        Waterline = 1
    };

    Type type{Type::Raster};
    double stepover{0.0};
    double stepdown{0.0};
    double angle_deg{0.0};
    bool finish_pass{false};
};

struct StrategyDecision
{
    std::vector<StrategyStep> steps;
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
