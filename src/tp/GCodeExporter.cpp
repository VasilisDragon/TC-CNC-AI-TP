#include "tp/GCodeExporter.h"

#include <QtCore/QFile>

#include <string>

namespace tp
{

bool GCodeExporter::exportToFile(const tp::Toolpath& toolpath,
                                 const QString& path,
                                 IPost& post,
                                 common::Unit units,
                                 const tp::UserParams& params,
                                 QString* error)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
    {
        if (error)
        {
            *error = QStringLiteral("Unable to open %1 for writing.").arg(path);
        }
        return false;
    }

    const std::string data = post.generate(toolpath, units, params);
    if (file.write(data.c_str(), static_cast<qint64>(data.size())) == -1)
    {
        if (error)
        {
            *error = QStringLiteral("Failed to write to %1.").arg(path);
        }
        return false;
    }

    return true;
}

} // namespace tp
