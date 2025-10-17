#pragma once

#include <string>

namespace tp
{

struct Machine
{
    std::string name;
    double rapidFeed_mm_min{3'000.0};
    double maxFeed_mm_min{2'000.0};
    double maxSpindleRPM{12'000.0};
    double clearanceZ_mm{5.0};
    double safeZ_mm{15.0};

    void ensureValid();
};

Machine makeDefaultMachine();

} // namespace tp

