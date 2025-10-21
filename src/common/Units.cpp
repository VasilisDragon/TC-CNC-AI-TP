#include "common/Units.h"

#include <QtCore/QLocale>

namespace common
{

namespace
{
constexpr double kMmPerInch = 25.4;
}

double convertLength(double value, UnitSystem from, UnitSystem to)
{
    if (from == to)
    {
        return value;
    }
    if (from == UnitSystem::Inches && to == kInternalUnitSystem)
    {
        return value * kMmPerInch;
    }
    if (from == kInternalUnitSystem && to == UnitSystem::Inches)
    {
        return value / kMmPerInch;
    }
    return value;
}

double toMillimeters(double value, UnitSystem from)
{
    return convertLength(value, from, kInternalUnitSystem);
}

double fromMillimeters(double valueMm, UnitSystem to)
{
    return convertLength(valueMm, kInternalUnitSystem, to);
}

QString unitName(UnitSystem unit)
{
    switch (unit)
    {
    case kInternalUnitSystem: return QStringLiteral("Millimeters");
    case UnitSystem::Inches: return QStringLiteral("Inches");
    }
    return QStringLiteral("Millimeters");
}

QString unitSuffix(UnitSystem unit)
{
    switch (unit)
    {
    case kInternalUnitSystem: return QStringLiteral("mm");
    case UnitSystem::Inches: return QStringLiteral("in");
    }
    return QStringLiteral("mm");
}

QString feedSuffix(UnitSystem unit)
{
    switch (unit)
    {
    case kInternalUnitSystem: return QStringLiteral("mm/min");
    case UnitSystem::Inches: return QStringLiteral("in/min");
    }
    return QStringLiteral("mm/min");
}

QString unitKey(UnitSystem unit)
{
    switch (unit)
    {
    case kInternalUnitSystem: return QStringLiteral("mm");
    case UnitSystem::Inches: return QStringLiteral("inch");
    }
    return QStringLiteral("mm");
}

UnitSystem unitFromString(const QString& text, UnitSystem fallback)
{
    const QString lower = text.trimmed().toLower();
    if (lower == QStringLiteral("mm") || lower == QStringLiteral("millimeters") || lower == QStringLiteral("millimetres"))
    {
        return kInternalUnitSystem;
    }
    if (lower == QStringLiteral("inch") || lower == QStringLiteral("inches") || lower == QStringLiteral("in"))
    {
        return UnitSystem::Inches;
    }
    return fallback;
}

QString formatLength(double valueMm, UnitSystem unit, int precision)
{
    const double displayValue = fromMillimeters(valueMm, unit);
    const QString suffix = unitSuffix(unit);
    return QStringLiteral("%1 %2")
        .arg(QLocale().toString(displayValue, 'f', precision))
        .arg(suffix);
}

} // namespace common
