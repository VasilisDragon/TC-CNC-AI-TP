#pragma once

#include "tp/IPost.h"

#include <string>

namespace tp
{

class GRBLPost : public IPost
{
public:
    GRBLPost() = default;

    std::string name() const override;
    std::string generate(const tp::Toolpath& toolpath,
                     common::Unit units,
                     const tp::UserParams& params) override;
};

} // namespace tp

