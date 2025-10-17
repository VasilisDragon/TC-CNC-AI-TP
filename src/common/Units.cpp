#include "common/Units.h"

#include <QtCore/QLocale>

namespace common
{

namespace
{
constexpr double kMmPerInch = 25.4;
}

double convertLength(double value, Unit from, Unit to)
{
    if (from == to)
    {
        return value;
    }
    if (from == Unit::Inches && to == Unit::Millimeters)
    {
        return value * kMmPerInch;
    }
    if (from == Unit::Millimeters && to == Unit::Inches)
    {
        return value / kMmPerInch;
    }
    return value;
}

double toMillimeters(double value, Unit from)
{
    return convertLength(value, from, Unit::Millimeters);
}

double fromMillimeters(double valueMm, Unit to)
{
    return convertLength(valueMm, Unit::Millimeters, to);
}

QString unitName(Unit unit)
{
    switch (unit)
    {
    case Unit::Millimeters: return QStringLiteral("Millimeters");
    case Unit::Inches: return QStringLiteral("Inches");
    }
    return QStringLiteral("Millimeters");
}

QString unitSuffix(Unit unit)
{
    switch (unit)
    {
    case Unit::Millimeters: return QStringLiteral("mm");
    case Unit::Inches: return QStringLiteral("in");
    }
    return QStringLiteral("mm");
}

QString feedSuffix(Unit unit)
{
    switch (unit)
    {
    case Unit::Millimeters: return QStringLiteral("mm/min");
    case Unit::Inches: return QStringLiteral("in/min");
    }
    return QStringLiteral("mm/min");
}

QString unitKey(Unit unit)
{
    switch (unit)
    {
    case Unit::Millimeters: return QStringLiteral("mm");
    case Unit::Inches: return QStringLiteral("inch");
    }
    return QStringLiteral("mm");
}

Unit unitFromString(const QString& text, Unit fallback)
{
    const QString lower = text.trimmed().toLower();
    if (lower == QStringLiteral("mm") || lower == QStringLiteral("millimeters") || lower == QStringLiteral("millimetres"))
    {
        return Unit::Millimeters;
    }
    if (lower == QStringLiteral("inch") || lower == QStringLiteral("inches") || lower == QStringLiteral("in"))
    {
        return Unit::Inches;
    }
    return fallback;
}

QString formatLength(double valueMm, Unit unit, int precision)
{
    const double displayValue = fromMillimeters(valueMm, unit);
    const QString suffix = unitSuffix(unit);
    return QStringLiteral("%1 %2")
        .arg(QLocale().toString(displayValue, 'f', precision))
        .arg(suffix);
}

} // namespace common

