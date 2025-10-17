#include "io/ImportWorker.h"

#include "io/ModelImporter.h"
#include "render/Model.h"

#include <QtCore/QMetaType>

#include <filesystem>

namespace io
{

ImportWorker::ImportWorker(QString filePath, QObject* parent)
    : QThread(parent)
    , m_filePath(std::move(filePath))
{
    qRegisterMetaType<std::shared_ptr<render::Model>>("std::shared_ptr<render::Model>");
}

void ImportWorker::requestCancel()
{
    m_cancelled.store(true, std::memory_order_relaxed);
}

void ImportWorker::run()
{
    emit progress(0);

    if (m_cancelled.load(std::memory_order_relaxed))
    {
        emit error(tr("Import cancelled."));
        return;
    }

    io::ModelImporter importer;
    std::shared_ptr<render::Model> model = std::make_shared<render::Model>();
    std::string errorMessage;

#if defined(_WIN32)
    const std::filesystem::path path = std::filesystem::path(m_filePath.toStdWString());
#else
    const std::filesystem::path path = std::filesystem::path(m_filePath.toStdString());
#endif

    if (m_cancelled.load(std::memory_order_relaxed))
    {
        emit error(tr("Import cancelled."));
        return;
    }

    const bool ok = importer.load(path, *model, errorMessage);

    if (m_cancelled.load(std::memory_order_relaxed))
    {
        emit error(tr("Import cancelled."));
        return;
    }

    if (!ok)
    {
        emit error(QString::fromStdString(errorMessage));
        return;
    }

    emit progress(100);
    emit finished(model);
}

} // namespace io
