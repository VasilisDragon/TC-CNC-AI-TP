#pragma once

#include "common/Units.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"

#include <QtCore/QMetaType>

#include <string>

namespace tp
{

class IPost
{
public:
    virtual ~IPost() = default;

    virtual std::string name() const = 0;
    virtual std::string generate(const tp::Toolpath& toolpath,
                                 common::UnitSystem units,
                                 const tp::UserParams& params) = 0;
};

} // namespace tp
