// ModelManager.cpp coordinates model discovery and lifecycle, making sure the UI and planners always see
// a coherent set of AI backends. Centralising the directory walking here keeps the rest of the ai module
// agnostic to on-disk layout changes.
#include "ai/ModelManager.h"

#include "ai/TorchAI.h"
#ifdef AI_WITH_ONNXRUNTIME
#include "ai/OnnxAI.h"
#endif

#include <QtCore/QDebug>
#include <QtCore/QCoreApplication>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>

#include <algorithm>
#include <filesystem>

namespace ai
{

namespace
{

QString defaultModelsDirectory()
{
    return QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("models"));
}

} // namespace

ModelManager::ModelManager(QString modelsDirectory)
    : m_modelsDirectory(modelsDirectory.isEmpty() ? defaultModelsDirectory() : std::move(modelsDirectory))
{
    refresh();
}

void ModelManager::refresh()
{
    m_models.clear();

    QDir dir(m_modelsDirectory);
    if (!dir.exists())
    {
        dir.mkpath(QStringLiteral("."));
        return;
    }

    const QFileInfoList entries = dir.entryInfoList(QDir::Files | QDir::Readable | QDir::NoDotAndDotDot);
    for (const QFileInfo& info : entries)
    {
        ModelDescriptor descriptor;
        descriptor.fileName = info.fileName();
        descriptor.absolutePath = info.absoluteFilePath();
        descriptor.modified = info.lastModified();
        descriptor.sizeBytes = info.size();

        const QString suffix = info.suffix().toLower();
        if (suffix == QStringLiteral("pt"))
        {
            descriptor.backend = ModelDescriptor::Backend::Torch;
        }
        else if (suffix == QStringLiteral("onnx"))
        {
            descriptor.backend = ModelDescriptor::Backend::Onnx;
        }
        else
        {
            continue;
        }

        m_models.push_back(std::move(descriptor));
    }

    std::sort(m_models.begin(), m_models.end(), [](const ModelDescriptor& a, const ModelDescriptor& b) {
        return a.fileName.compare(b.fileName, Qt::CaseInsensitive) < 0;
    });
}

std::unique_ptr<IPathAI> ModelManager::createModel(const QString& absolutePath) const
{
    std::filesystem::path path;
    if (!absolutePath.isEmpty())
    {
#ifdef _WIN32
        path = std::filesystem::path(absolutePath.toStdWString());
#else
        path = std::filesystem::path(absolutePath.toStdString());
#endif
    }

    QString suffix = QFileInfo(absolutePath).suffix().toLower();
    if (suffix == QStringLiteral("onnx"))
    {
#ifdef AI_WITH_ONNXRUNTIME
        return std::make_unique<OnnxAI>(std::move(path));
#else
        qWarning().noquote() << "OnnxAI requested but built without ONNX Runtime support.";
        return nullptr;
#endif
    }

    return std::make_unique<TorchAI>(std::move(path));
}

} // namespace ai
