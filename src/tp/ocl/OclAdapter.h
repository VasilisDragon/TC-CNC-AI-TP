#pragma once

#include "tp/Toolpath.h"

#include <string>

namespace render
{
class Model;
}

namespace tp
{

struct UserParams;

struct Cutter
{
    enum class Type
    {
        FlatEndmill,
        BallNose
    };

    Type type{Type::FlatEndmill};
    double diameter{0.0};
    double length{0.0};
};

class OclAdapter
{
public:
    static bool waterline(const render::Model& model,
                          const UserParams& params,
                          const Cutter& cutter,
                          Toolpath& out,
                          std::string& err);

    static bool rasterDropCutter(const render::Model& model,
                                 const UserParams& params,
                                 const Cutter& cutter,
                                 double rasterAngleDeg,
                                 Toolpath& out,
                                 std::string& err);
};

} // namespace tp
