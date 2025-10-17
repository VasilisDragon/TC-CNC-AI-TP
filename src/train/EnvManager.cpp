#include "train/EnvManager.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QDir>
#include <QtCore/QEventLoop>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QProcess>
#include <QtCore/QRegularExpression>
#include <QtCore/QSettings>
#include <QtCore/QDateTime>
#include <QtCore/QStandardPaths>
#include <QtCore/QTextStream>
#include <QtCore/QThread>
#include <QtCore/QTimer>
#include <QtNetwork/QNetworkAccessManager>
#include <QtNetwork/QNetworkReply>
#include <QtNetwork/QNetworkRequest>
#include <QtCore/QMutexLocker>

#include <QVersionNumber>

#include <algorithm>
#ifdef Q_OS_WIN
#    include <windows.h>
#endif

#include <array>

namespace train
{

namespace
{
constexpr auto kPythonVersion = "3.11.5";
constexpr auto kPythonEmbedUrl = "https://www.python.org/ftp/python/3.11.5/python-3.11.5-embed-amd64.zip";

QString runtimeRootDefault()
{
    QDir baseDir(QCoreApplication::applicationDirPath());
    baseDir.mkpath(QStringLiteral("runtime"));
    return baseDir.filePath(QStringLiteral("runtime"));
}

QString nativePath(const QString& path)
{
    return QDir::toNativeSeparators(path);
}

} // namespace

EnvManager::EnvManager(QObject* parent)
    : QObject(parent)
    , m_runtimeRoot(runtimeRootDefault())
{
    QSettings settings;
    m_cpuOnly = settings.value(QStringLiteral("training/cpuOnly"), false).toBool();
    m_pythonExecutable = pythonCandidateFromSettings();
    if (!m_pythonExecutable.isEmpty())
    {
        QFileInfo info(m_pythonExecutable);
        if (info.exists() && info.isExecutable())
        {
            m_pythonHome = info.absolutePath();
        }
    }
    m_venvPath = QDir(m_runtimeRoot).filePath(QStringLiteral("venv"));
    m_venvPython = QDir(m_venvPath).filePath(QStringLiteral("Scripts/python.exe"));
    refreshGpuInfo();
}

EnvManager::~EnvManager()
{
    cancel();
}

void EnvManager::prepareEnvironment(bool forceRepair)
{
    if (m_busy)
    {
        appendLog(timestamped(tr("Environment preparation already in progress.")));
        return;
    }

    m_busy = true;
    m_cancelRequested.store(false);
    emitProgress(0);
    appendLog(timestamped(tr("Starting environment preparation...")));

    QThread* worker = QThread::create([this, forceRepair]() {
        runPreparation(forceRepair);
    });
    connect(worker, &QThread::finished, worker, &QObject::deleteLater);
    worker->start();
}

void EnvManager::cancel()
{
    if (!m_busy)
    {
        return;
    }
    m_cancelRequested.store(true);
    QMutexLocker locker(&m_processMutex);
    if (m_activeProcess)
    {
        appendLog(timestamped(tr("Cancelling current operation...")));
        m_activeProcess->terminate();
        if (!m_activeProcess->waitForFinished(3000))
        {
            m_activeProcess->kill();
        }
    }
}

void EnvManager::setCpuOnly(bool cpuOnly)
{
    if (m_cpuOnly == cpuOnly)
    {
        return;
    }
    m_cpuOnly = cpuOnly;
    QSettings settings;
    settings.setValue(QStringLiteral("training/cpuOnly"), m_cpuOnly);
    settings.sync();
}

QString EnvManager::gpuSummary() const
{
    return m_gpuSummary;
}

void EnvManager::refreshGpuInfo()
{
    QString summary = detectGpuViaNvidiaSmi();
    if (summary.isEmpty())
    {
        summary = tr("GPU: none detected");
    }
    if (summary != m_gpuSummary)
    {
        m_gpuSummary = summary;
        emit gpuInfoChanged(m_gpuSummary);
    }
}

void EnvManager::runPreparation(bool forceRepair)
{
    QString errorMessage;
    bool success = ensureRuntimeDirectories(errorMessage);
    if (success && !m_cancelRequested.load())
    {
        if (forceRepair)
        {
            appendLog(timestamped(tr("Forcing environment repair.")));
        }
        success = detectPython(errorMessage);
    }
    if (success && !m_cancelRequested.load())
    {
        success = ensureVenv(errorMessage);
    }
    if (success && !m_cancelRequested.load())
    {
        success = installPackages(errorMessage);
    }

    if (!success && !errorMessage.isEmpty())
    {
        emit error(errorMessage);
        appendLog(timestamped(QStringLiteral("Error: %1").arg(errorMessage)));
    }
    else if (success && !m_cancelRequested.load())
    {
        appendLog(timestamped(tr("Environment is ready.")));
    }
    finalizeRun(success && !m_cancelRequested.load());
}

void EnvManager::finalizeRun(bool success)
{
    if (m_cancelRequested.load())
    {
        appendLog(timestamped(tr("Environment preparation cancelled.")));
        success = false;
    }
    persistStatus(success);
    emitProgress(success ? 100 : 0);
    emit finished(success);
    m_busy = false;
}

bool EnvManager::ensureRuntimeDirectories(QString& errorMessage)
{
    QDir dir(m_runtimeRoot);
    if (!dir.exists() && !dir.mkpath(QStringLiteral(".")))
    {
        errorMessage = tr("Failed to create runtime directory: %1").arg(m_runtimeRoot);
        return false;
    }
    QDir pythonDir(dir.filePath(QStringLiteral("python")));
    if (!pythonDir.exists())
    {
        dir.mkpath(QStringLiteral("python"));
    }
    return true;
}

bool EnvManager::detectPython(QString& errorMessage)
{
    if (m_cancelRequested.load())
    {
        return false;
    }

    QString versionString;
    if (!m_pythonExecutable.isEmpty() && checkPythonVersion(m_pythonExecutable, versionString))
    {
        appendLog(timestamped(tr("Using cached Python: %1 (%2)").arg(nativePath(m_pythonExecutable), versionString)));
        emitProgress(15);
        return true;
    }

    const QStringList systemCandidates = {
        QStandardPaths::findExecutable(QStringLiteral("python3")),
        QStandardPaths::findExecutable(QStringLiteral("python"))
    };

    for (const QString& candidate : systemCandidates)
    {
        if (candidate.isEmpty())
        {
            continue;
        }
        if (checkPythonVersion(candidate, versionString))
        {
            m_pythonExecutable = candidate;
            m_pythonHome = QFileInfo(candidate).absolutePath();
            appendLog(timestamped(tr("Found system Python: %1 (%2)").arg(nativePath(candidate), versionString)));
            persistPythonPath(m_pythonExecutable);
            emitProgress(15);
            return true;
        }
    }

#ifdef Q_OS_WIN
    // Try Python launcher
    if (checkPythonVersion(QStringLiteral("py"), versionString))
    {
        m_pythonExecutable = QStringLiteral("py -3.11");
        m_pythonHome.clear();
        appendLog(timestamped(tr("Using Python launcher: %1").arg(m_pythonExecutable)));
        emitProgress(15);
        return true;
    }
#endif

    appendLog(timestamped(tr("No suitable Python found. Downloading embedded distribution...")));
    if (!downloadEmbeddedPython(errorMessage))
    {
        return false;
    }

    if (!m_pythonExecutable.isEmpty() && checkPythonVersion(m_pythonExecutable, versionString))
    {
        appendLog(timestamped(tr("Embedded Python ready: %1").arg(versionString)));
        emitProgress(25);
        persistPythonPath(m_pythonExecutable);
        return true;
    }

    errorMessage = tr("Failed to validate embedded Python executable.");
    return false;
}

bool EnvManager::downloadEmbeddedPython(QString& errorMessage)
{
    if (m_cancelRequested.load())
    {
        return false;
    }

    QNetworkAccessManager manager;
    QNetworkRequest request(QUrl(QString::fromLatin1(kPythonEmbedUrl)));
    const QString archivePath = QDir(m_runtimeRoot).filePath(QStringLiteral("python/python-embed.zip"));
    QFile archive(archivePath);
    if (!archive.open(QIODevice::WriteOnly | QIODevice::Truncate))
    {
        errorMessage = tr("Unable to write Python archive: %1").arg(archivePath);
        return false;
    }

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    request.setTransferTimeout(120000);
#endif
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);

    QNetworkReply* reply = manager.get(request);
    QEventLoop loop;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);

    bool writeError = false;
    QObject::connect(reply,
                     &QIODevice::readyRead,
                     &manager,
                     [reply, &archive, &writeError]() {
                         if (writeError)
                         {
                             reply->readAll();
                             return;
                         }
                         const QByteArray chunk = reply->readAll();
                         if (chunk.isEmpty())
                         {
                             return;
                         }
                         if (archive.write(chunk) == -1)
                         {
                             writeError = true;
                             reply->abort();
                         }
                     });

    QObject::connect(reply,
                     &QNetworkReply::downloadProgress,
                     &manager,
                     [this, reply](qint64 received, qint64 total) {
                         if (total > 0)
                         {
                             const int percent = static_cast<int>((received * 30) / total);
                             emitProgress(15 + percent / 2);
                         }
                         if (m_cancelRequested.load(std::memory_order_relaxed))
                         {
                             reply->abort();
                         }
                     });

    loop.exec();

    if (!writeError)
    {
        const QByteArray tail = reply->readAll();
        if (!tail.isEmpty() && archive.write(tail) == -1)
        {
            writeError = true;
        }
    }

    archive.flush();
    archive.close();

    const auto cleanupPartial = [&]() {
        QFile::remove(archivePath);
    };

    if (writeError)
    {
        errorMessage = tr("Unable to stream Python archive to disk.");
        reply->deleteLater();
        cleanupPartial();
        return false;
    }

    if (reply->error() != QNetworkReply::NoError)
    {
        if (reply->error() == QNetworkReply::OperationCanceledError && m_cancelRequested.load())
        {
            reply->deleteLater();
            cleanupPartial();
            return false;
        }
        errorMessage = tr("Download failed: %1").arg(reply->errorString());
        reply->deleteLater();
        cleanupPartial();
        return false;
    }

    reply->deleteLater();

    if (m_cancelRequested.load())
    {
        cleanupPartial();
        return false;
    }

    emitProgress(40);

    if (!extractEmbeddedPython(archivePath, errorMessage))
    {
        return false;
    }

    emitProgress(45);
    if (!ensureSiteEnabled(errorMessage))
    {
        return false;
    }

    QDir pythonDir(QDir(m_runtimeRoot).filePath(QStringLiteral("python")));
    m_pythonExecutable = pythonDir.filePath(QStringLiteral("python.exe"));
    m_pythonHome = pythonDir.absolutePath();
    return true;
}

bool EnvManager::extractEmbeddedPython(const QString& archivePath, QString& errorMessage)
{
    if (m_cancelRequested.load())
    {
        return false;
    }

    QDir pythonDir(QDir(m_runtimeRoot).filePath(QStringLiteral("python")));
    if (pythonDir.exists())
    {
        pythonDir.removeRecursively();
    }
    pythonDir.mkpath(QStringLiteral("."));

    QProcess process;
    process.setProgram(QStringLiteral("powershell"));
    process.setArguments(QStringList{
        QStringLiteral("-NoProfile"),
        QStringLiteral("-Command"),
        QStringLiteral("Expand-Archive -Path \"%1\" -DestinationPath \"%2\" -Force")
            .arg(nativePath(archivePath), nativePath(pythonDir.absolutePath()))
    });
#ifdef Q_OS_WIN
    process.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* args) {
        args->flags |= CREATE_NO_WINDOW;
    });
#endif

    {
        QMutexLocker locker(&m_processMutex);
        m_activeProcess = &process;
    }

    process.start();
    if (!process.waitForStarted())
    {
        errorMessage = tr("Failed to start PowerShell for extraction.");
        return false;
    }
    while (process.state() == QProcess::Running)
    {
        if (m_cancelRequested.load())
        {
            process.terminate();
            process.waitForFinished(2000);
            return false;
        }
        process.waitForFinished(200);
    }

    {
        QMutexLocker locker(&m_processMutex);
        m_activeProcess = nullptr;
    }

    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
    {
        errorMessage = tr("PowerShell Expand-Archive failed (exit %1).").arg(process.exitCode());
        return false;
    }

    QFile::remove(archivePath);
    return true;
}

bool EnvManager::ensureSiteEnabled(QString& errorMessage) const
{
    const QString pthPath = QDir(QDir(m_runtimeRoot).filePath(QStringLiteral("python"))).filePath(QStringLiteral("python311._pth"));
    QFile pthFile(pthPath);
    if (!pthFile.exists())
    {
        return true;
    }
    if (!pthFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        errorMessage = tr("Unable to patch embedded Python path file.");
        return false;
    }
    QString content = QString::fromUtf8(pthFile.readAll());
    pthFile.close();
    if (!content.contains(QStringLiteral("import site"), Qt::CaseInsensitive))
    {
        content.append(QStringLiteral("\nimport site\n"));
    }
    else
    {
        content.replace(QStringLiteral("#import site"), QStringLiteral("import site"));
    }
    if (!pthFile.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate))
    {
        errorMessage = tr("Unable to update embedded Python path file.");
        return false;
    }
    QTextStream stream(&pthFile);
    stream << content;
    return true;
}

bool EnvManager::ensureVenv(QString& errorMessage)
{
    if (m_cancelRequested.load())
    {
        return false;
    }

    emitProgress(50);

    QDir venvDir(m_venvPath);
    if (venvDir.exists())
    {
        appendLog(timestamped(tr("Repairing existing virtual environment...")));
    }
    else
    {
        appendLog(timestamped(tr("Creating virtual environment...")));
    }

    QString python = m_pythonExecutable;
    QStringList args;
#ifdef Q_OS_WIN
    if (python == QStringLiteral("py -3.11"))
    {
        python = QStringLiteral("py");
        args << QStringLiteral("-3.11");
    }
#endif
    args << QStringLiteral("-m") << QStringLiteral("venv") << nativePath(m_venvPath);

    if (!runProcess(python, args, m_runtimeRoot, errorMessage, 60))
    {
        return false;
    }

    m_venvPython = QDir(m_venvPath).filePath(QStringLiteral("Scripts/python.exe"));
    if (!QFileInfo::exists(m_venvPython))
    {
        errorMessage = tr("Virtual environment Python executable missing at %1").arg(nativePath(m_venvPython));
        return false;
    }

    emitProgress(65);
    return true;
}

bool EnvManager::installPackages(QString& errorMessage)
{
    if (m_cancelRequested.load())
    {
        return false;
    }
    appendLog(timestamped(tr("Installing training dependencies...")));

    QStringList upgradeArgs = {QStringLiteral("-m"), QStringLiteral("pip"), QStringLiteral("install"), QStringLiteral("--upgrade"), QStringLiteral("pip")};
    if (!runProcess(m_venvPython, upgradeArgs, m_runtimeRoot, errorMessage, 70))
    {
        return false;
    }

    const bool hasGpu = m_gpuSummary.contains(QStringLiteral("NVIDIA"), Qt::CaseInsensitive) && !m_cpuOnly;
    QStringList torchArgs = {QStringLiteral("-m"),
                             QStringLiteral("pip"),
                             QStringLiteral("install"),
                             QStringLiteral("--upgrade"),
                             QStringLiteral("torch"),
                             QStringLiteral("torchvision"),
                             QStringLiteral("torchaudio")};
    if (hasGpu)
    {
        torchArgs << QStringLiteral("--index-url") << QStringLiteral("https://download.pytorch.org/whl/cu118");
    }
    if (!runProcess(m_venvPython, torchArgs, m_runtimeRoot, errorMessage, 80))
    {
        return false;
    }

    QStringList runtimeArgs = {QStringLiteral("-m"),
                               QStringLiteral("pip"),
                               QStringLiteral("install"),
                               hasGpu ? QStringLiteral("onnxruntime-gpu") : QStringLiteral("onnxruntime")};
    if (!runProcess(m_venvPython, runtimeArgs, m_runtimeRoot, errorMessage, 85))
    {
        return false;
    }

    QStringList extras = {QStringLiteral("-m"),
                          QStringLiteral("pip"),
                          QStringLiteral("install"),
                          QStringLiteral("cadquery"),
                          QStringLiteral("build123d"),
                          QStringLiteral("trimesh"),
                          QStringLiteral("numpy"),
                          QStringLiteral("matplotlib")};
    if (!runProcess(m_venvPython, extras, m_runtimeRoot, errorMessage, 95))
    {
        return false;
    }

    emitProgress(100);
    return true;
}

bool EnvManager::runProcess(const QString& program,
                            const QStringList& arguments,
                            const QString& workingDirectory,
                            QString& errorMessage,
                            int progressAfterStep,
                            bool silent)
{
    if (m_cancelRequested.load())
    {
        return false;
    }

    QProcess process;
    process.setProgram(program);
    process.setArguments(arguments);
    if (!workingDirectory.isEmpty())
    {
        process.setWorkingDirectory(workingDirectory);
    }
    process.setProcessChannelMode(QProcess::MergedChannels);

#ifdef Q_OS_WIN
    process.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* args) {
        args->flags |= CREATE_NO_WINDOW;
    });
#endif

    {
        QMutexLocker locker(&m_processMutex);
        m_activeProcess = &process;
    }

    process.start();
    if (!process.waitForStarted())
    {
        errorMessage = tr("Failed to start process: %1").arg(program);
        QMutexLocker locker(&m_processMutex);
        m_activeProcess = nullptr;
        return false;
    }

    auto flushOutput = [&]() {
        const QByteArray data = process.readAll();
        if (!data.isEmpty() && !silent)
        {
            appendLog(QString::fromLocal8Bit(data));
        }
    };

    while (process.state() == QProcess::Running)
    {
        if (process.waitForReadyRead(200))
        {
            flushOutput();
        }
        if (m_cancelRequested.load())
        {
            process.terminate();
            process.waitForFinished(3000);
            break;
        }
    }
    flushOutput();

    const int exitCode = process.exitCode();
    const QProcess::ExitStatus exitStatus = process.exitStatus();

    {
        QMutexLocker locker(&m_processMutex);
        m_activeProcess = nullptr;
    }

    if (m_cancelRequested.load())
    {
        return false;
    }

    if (exitStatus != QProcess::NormalExit || exitCode != 0)
    {
        errorMessage = tr("Process failed (%1): %2").arg(program, QString::number(exitCode));
        return false;
    }

    if (progressAfterStep >= 0)
    {
        emitProgress(progressAfterStep);
    }
    return true;
}

bool EnvManager::checkPythonVersion(const QString& executable, QString& versionString) const
{
    QString program = executable;
    QStringList args;
#ifdef Q_OS_WIN
    if (executable == QStringLiteral("py"))
    {
        program = QStringLiteral("py");
        args << QStringLiteral("-3.11") << QStringLiteral("--version");
    }
    else if (executable == QStringLiteral("py -3.11"))
    {
        program = QStringLiteral("py");
        args << QStringLiteral("-3.11") << QStringLiteral("--version");
    }
    else
#endif
    {
        args << QStringLiteral("--version");
    }

    QProcess process;
    process.setProgram(program);
    process.setArguments(args);
#ifdef Q_OS_WIN
    process.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* arguments) {
        arguments->flags |= CREATE_NO_WINDOW;
    });
#endif
    process.start();
    if (!process.waitForFinished(5000))
    {
        process.kill();
        return false;
    }

    const QString output = QString::fromLocal8Bit(process.readAllStandardOutput() + process.readAllStandardError()).trimmed();
    QRegularExpression regex(QStringLiteral(R"(Python\s+(\d+\.\d+\.\d+))"));
    const QRegularExpressionMatch match = regex.match(output);
    if (!match.hasMatch())
    {
        return false;
    }
    versionString = match.captured(1);
    const QVersionNumber version = QVersionNumber::fromString(versionString);
    return version >= QVersionNumber(3, 11, 0);
}

QString EnvManager::pythonCandidateFromSettings() const
{
    QSettings settings;
    return settings.value(QStringLiteral("training/pythonExecutable")).toString();
}

void EnvManager::persistPythonPath(const QString& path)
{
    QSettings settings;
    settings.setValue(QStringLiteral("training/pythonExecutable"), path);
    settings.sync();
}

void EnvManager::persistStatus(bool ready)
{
    QSettings settings;
    settings.setValue(QStringLiteral("training/envReady"), ready);
    settings.setValue(QStringLiteral("training/envReadyTimestamp"), QDateTime::currentDateTime());
    settings.sync();
}

QString EnvManager::timestamped(const QString& message) const
{
    return QStringLiteral("[%1] %2").arg(QDateTime::currentDateTime().toString(Qt::ISODateWithMs), message);
}

void EnvManager::emitProgress(int value)
{
    emit progress(std::clamp(value, 0, 100));
}

void EnvManager::appendLog(const QString& message)
{
    emit log(message);
}

QString EnvManager::detectGpuViaNvidiaSmi() const
{
    QProcess process;
    process.setProgram(QStringLiteral("nvidia-smi"));
    process.setArguments(QStringList{QStringLiteral("--query-gpu=name,compute_cap"), QStringLiteral("--format=csv,noheader")});
#ifdef Q_OS_WIN
    process.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* args) {
        args->flags |= CREATE_NO_WINDOW;
    });
#endif
    process.start();
    if (!process.waitForFinished(3000))
    {
        process.kill();
        return QString();
    }
    if (process.exitStatus() != QProcess::NormalExit || process.exitCode() != 0)
    {
        return QString();
    }
    QString output = QString::fromLocal8Bit(process.readAllStandardOutput()).trimmed();
    if (output.isEmpty())
    {
        return QString();
    }
    const QStringList lines = output.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    if (lines.isEmpty())
    {
        return QString();
    }
    const QStringList parts = lines.first().split(QLatin1Char(','), Qt::SkipEmptyParts);
    if (parts.size() < 2)
    {
        return QStringLiteral("GPU: %1").arg(lines.first().trimmed());
    }
    return QStringLiteral("GPU: %1 (CC %2)").arg(parts.at(0).trimmed(), parts.at(1).trimmed());
}

} // namespace train

