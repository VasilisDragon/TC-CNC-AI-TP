#include "ai/StrategySerialization.h"

#include <QtCore/QJsonValue>

namespace ai
{

QJsonObject stepToJson(const StrategyStep& step)
{
    QJsonObject obj;
    obj.insert(QStringLiteral("type"), static_cast<int>(step.type));
    obj.insert(QStringLiteral("stepover"), step.stepover);
    obj.insert(QStringLiteral("stepdown"), step.stepdown);
    obj.insert(QStringLiteral("angle_deg"), step.angle_deg);
    obj.insert(QStringLiteral("finish"), step.finish_pass);
    return obj;
}

StrategyStep stepFromJson(const QJsonObject& object, bool* ok)
{
    StrategyStep step;
    bool valid = object.contains(QStringLiteral("type"));
    const int typeValue = object.value(QStringLiteral("type")).toInt(-1);
    if (typeValue == static_cast<int>(StrategyStep::Type::Raster))
    {
        step.type = StrategyStep::Type::Raster;
    }
    else if (typeValue == static_cast<int>(StrategyStep::Type::Waterline))
    {
        step.type = StrategyStep::Type::Waterline;
    }
    else
    {
        valid = false;
    }

    step.stepover = object.value(QStringLiteral("stepover")).toDouble(step.stepover);
    step.stepdown = object.value(QStringLiteral("stepdown")).toDouble(step.stepdown);
    step.angle_deg = object.value(QStringLiteral("angle_deg")).toDouble(step.angle_deg);
    step.finish_pass = object.value(QStringLiteral("finish")).toBool(step.finish_pass);

    if (ok)
    {
        *ok = valid;
    }
    return step;
}

QJsonArray stepsToJson(const std::vector<StrategyStep>& steps)
{
    QJsonArray array;
    for (const StrategyStep& step : steps)
    {
        array.append(stepToJson(step));
    }
    return array;
}

std::vector<StrategyStep> stepsFromJson(const QJsonArray& array)
{
    std::vector<StrategyStep> steps;
    steps.reserve(array.size());
    for (const QJsonValue& value : array)
    {
        if (!value.isObject())
        {
            continue;
        }
        bool ok = false;
        StrategyStep step = stepFromJson(value.toObject(), &ok);
        if (ok)
        {
            steps.push_back(step);
        }
    }
    return steps;
}

QJsonObject decisionToJson(const StrategyDecision& decision)
{
    QJsonObject obj;
    obj.insert(QStringLiteral("steps"), stepsToJson(decision.steps));
    return obj;
}

StrategyDecision decisionFromJson(const QJsonObject& object)
{
    StrategyDecision decision;
    if (object.contains(QStringLiteral("steps")) && object.value(QStringLiteral("steps")).isArray())
    {
        decision.steps = stepsFromJson(object.value(QStringLiteral("steps")).toArray());
    }
    return decision;
}

} // namespace ai
