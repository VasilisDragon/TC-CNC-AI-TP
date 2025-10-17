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
    virtual std::string emit(const tp::Toolpath& toolpath,
                             common::Unit units,
                             const tp::UserParams& params) = 0;
};

} // namespace tp
