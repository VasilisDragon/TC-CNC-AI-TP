#pragma once

#include "common/Units.h"

#include <QString>

namespace common
{

class Tool
{
public:
    QString id;
    QString name;
    QString type;
    double diameterMm{0.0};
    QString notes;

    [[nodiscard]] bool isValid() const noexcept;
    [[nodiscard]] double recommendedStepOverMm() const noexcept;
    [[nodiscard]] double recommendedMaxDepthMm() const noexcept;
    [[nodiscard]] QString displayLabel(Unit unit) const;
};

} // namespace common

