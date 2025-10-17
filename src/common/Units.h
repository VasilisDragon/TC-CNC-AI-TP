#pragma once

#include <QString>

namespace common
{

enum class Unit
{
    Millimeters,
    Inches
};

double convertLength(double value, Unit from, Unit to);
double toMillimeters(double value, Unit from);
double fromMillimeters(double valueMm, Unit to);

QString unitName(Unit unit);
QString unitSuffix(Unit unit);
QString feedSuffix(Unit unit);
QString unitKey(Unit unit);

Unit unitFromString(const QString& text, Unit fallback = Unit::Millimeters);
QString formatLength(double valueMm, Unit unit, int precision = 3);

} // namespace common

