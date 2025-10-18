#pragma once

#include <QtCore/QDateTime>
#include <QtCore/QVector>

#include <memory>
#include <QString>

namespace ai
{

class IPathAI;

struct ModelDescriptor
{
    QString fileName;
    QString absolutePath;
    QDateTime modified;
    qint64 sizeBytes{0};

    enum class Backend
    {
        Torch,
        Onnx
    };

    Backend backend{Backend::Torch};
};

class ModelManager
{
public:
    explicit ModelManager(QString modelsDirectory = QString());

    void refresh();

    [[nodiscard]] const QVector<ModelDescriptor>& models() const noexcept { return m_models; }
    [[nodiscard]] QString modelsDirectory() const noexcept { return m_modelsDirectory; }

    std::unique_ptr<IPathAI> createModel(const QString& absolutePath) const;

private:
    QString m_modelsDirectory;
    QVector<ModelDescriptor> m_models;
};

} // namespace ai
