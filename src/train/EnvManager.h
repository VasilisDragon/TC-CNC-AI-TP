#pragma once

#include <QObject>
#include <QMutex>
#include <QString>
#include <atomic>

class QProcess;

namespace train
{

class EnvManager : public QObject
{
    Q_OBJECT

public:
    explicit EnvManager(QObject* parent = nullptr);
    ~EnvManager() override;

    void prepareEnvironment(bool forceRepair = false);
    void cancel();
    bool isBusy() const noexcept { return m_busy; }

    void setCpuOnly(bool cpuOnly);
    [[nodiscard]] bool cpuOnly() const noexcept { return m_cpuOnly; }

    [[nodiscard]] QString gpuSummary() const;
    void refreshGpuInfo();

    Q_SIGNALS:
    void progress(int value);
    void log(const QString& message);
    void finished(bool success);
    void error(const QString& message);
    void gpuInfoChanged(const QString& summary);

private:
    void runPreparation(bool forceRepair);
    void finalizeRun(bool success);

    bool ensureRuntimeDirectories(QString& errorMessage);
    bool detectPython(QString& errorMessage);
    bool downloadEmbeddedPython(QString& errorMessage);
    bool extractEmbeddedPython(const QString& archivePath, QString& errorMessage);
    bool ensureVenv(QString& errorMessage);
    bool installPackages(QString& errorMessage);
    bool verifyEmbeddedPythonArchive(const QString& archivePath, QString& errorMessage);
    bool runProcess(const QString& program,
                    const QStringList& arguments,
                    const QString& workingDirectory,
                    QString& errorMessage,
                    int progressAfterStep = -1,
                    bool silent = false);

    bool checkPythonVersion(const QString& executable, QString& versionString) const;
    QString pythonCandidateFromSettings() const;
    void persistPythonPath(const QString& path);
    void persistStatus(bool ready);
    QString timestamped(const QString& message) const;
    void emitProgress(int value);
    void appendLog(const QString& message);
    bool ensureSiteEnabled(QString& errorMessage) const;

    QString detectGpuViaNvidiaSmi() const;

    QString m_runtimeRoot;
    QString m_pythonHome;
    QString m_pythonExecutable;
    QString m_venvPath;
    QString m_venvPython;

    QString m_gpuSummary;
    bool m_cpuOnly{false};
    std::atomic_bool m_cancelRequested{false};
    std::atomic_bool m_busy{false};

    QProcess* m_activeProcess{nullptr};
    mutable QMutex m_processMutex;
};

} // namespace train

