#pragma once

#include "ai/IPathAI.h"

#include <QtCore/QJsonArray>
#include <QtCore/QJsonObject>

namespace ai
{

QJsonObject stepToJson(const StrategyStep& step);
StrategyStep stepFromJson(const QJsonObject& object, bool* ok = nullptr);

QJsonArray stepsToJson(const std::vector<StrategyStep>& steps);
std::vector<StrategyStep> stepsFromJson(const QJsonArray& array);

QJsonObject decisionToJson(const StrategyDecision& decision);
StrategyDecision decisionFromJson(const QJsonObject& object);

} // namespace ai

