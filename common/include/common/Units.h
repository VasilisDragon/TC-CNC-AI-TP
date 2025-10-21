#pragma once

#include <QString>

namespace common
{

enum class UnitSystem
{
    Millimeters,
    Inches
};

constexpr UnitSystem kInternalUnitSystem = UnitSystem::Millimeters;

double convertLength(double value, UnitSystem from, UnitSystem to);
double toMillimeters(double value, UnitSystem from);
double fromMillimeters(double valueMm, UnitSystem to);

QString unitName(UnitSystem unit);
QString unitSuffix(UnitSystem unit);
QString feedSuffix(UnitSystem unit);
QString unitKey(UnitSystem unit);

UnitSystem unitFromString(const QString& text, UnitSystem fallback = UnitSystem::Millimeters);
QString formatLength(double valueMm, UnitSystem unit, int precision = 3);

} // namespace common
