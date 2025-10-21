#pragma once

#include "tp/IPost.h"
#include "tp/Toolpath.h"

#include <QtCore/QString>

namespace tp
{

class GCodeExporter
{
public:
    static bool exportToFile(const tp::Toolpath& toolpath,
                             const QString& path,
                             IPost& post,
                             common::UnitSystem units,
                             const tp::UserParams& params,
                             QString* error = nullptr);
};

} // namespace tp

