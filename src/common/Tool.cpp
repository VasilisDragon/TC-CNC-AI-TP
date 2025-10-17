#include "common/Tool.h"

#include <QtCore/QStringBuilder>

namespace common
{

bool Tool::isValid() const noexcept
{
    return !id.trimmed().isEmpty() && !name.trimmed().isEmpty() && diameterMm > 0.0;
}

double Tool::recommendedStepOverMm() const noexcept
{
    return diameterMm * 0.4;
}

double Tool::recommendedMaxDepthMm() const noexcept
{
    return diameterMm * 0.5;
}

QString Tool::displayLabel(Unit unit) const
{
    QString label = name;
    if (diameterMm > 0.0)
    {
        label += QStringLiteral(" (") % formatLength(diameterMm, unit, unit == Unit::Inches ? 3 : 2) % QLatin1Char(')');
    }
    return label;
}

} // namespace common

