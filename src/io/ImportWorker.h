#pragma once

#include <QtCore/QThread>
#include <QtCore/QMetaType>
#include <QtCore/QString>

#include <atomic>
#include <memory>

namespace render
{
class Model;
}

namespace io
{

class ImportWorker : public QThread
{
    Q_OBJECT

public:
    explicit ImportWorker(QString filePath, QObject* parent = nullptr);

    void requestCancel();

signals:
    void progress(int value);
    void finished(std::shared_ptr<render::Model> model);
    void error(const QString& message);

protected:
    void run() override;

private:
    QString m_filePath;
    std::atomic<bool> m_cancelled{false};
};

} // namespace io

Q_DECLARE_METATYPE(std::shared_ptr<render::Model>)
