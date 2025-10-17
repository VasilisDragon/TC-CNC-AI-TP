#include "tp/Machine.h"

#include <algorithm>

namespace tp
{

namespace
{
constexpr double kMinFeed = 0.0;
constexpr double kMinSpindle = 0.0;
}

void Machine::ensureValid()
{
    rapidFeed_mm_min = std::max(rapidFeed_mm_min, kMinFeed);
    maxFeed_mm_min = std::max(maxFeed_mm_min, kMinFeed);
    maxSpindleRPM = std::max(maxSpindleRPM, kMinSpindle);
    clearanceZ_mm = std::max(clearanceZ_mm, 0.0);
    safeZ_mm = std::max(safeZ_mm, clearanceZ_mm);
}

Machine makeDefaultMachine()
{
    Machine machine;
    machine.name = "Generic Router";
    machine.ensureValid();
    return machine;
}

} // namespace tp

