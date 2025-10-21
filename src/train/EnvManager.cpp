#include "train/EnvManager.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QDir>
#include <QtCore/QEventLoop>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QCryptographicHash>
#include <QtCore/QByteArray>
#include <QtCore/QProcess>
#include <QtCore/QRegularExpression>
#include <QtCore/QSettings>
#include <QtCore/QDateTime>
#include <QtCore/QStandardPaths>
#include <QtCore/QTextStream>
#include <QtCore/QThread>
#include <QtCore/QTimer>
#include <QtCore/QStringList>
#include <QtCore/QTemporaryFile>
#include <QtCore/QLatin1String>
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
constexpr auto kPythonEmbedSha256 = "d82391a2e51c3684987c61f6b7cedbff3ce9fbe2e39cd948d32b0da866544b17";
constexpr auto kCudaExtraIndex = "https://download.pytorch.org/whl/cu118";
constexpr auto kOnnxRuntimeVersion = "1.17.1";
constexpr auto kTorchVersion = "2.2.2";
constexpr auto kTorchVisionVersion = "0.17.2";
constexpr auto kTorchAudioVersion = "2.2.2";
const QStringList kTorchCu118FallbackHashes = {
    QStringLiteral("sha256:3a624d02d874f110056e4c00c0e5cbac990884e91210a0cf610d408d52530e54"),
    QStringLiteral("sha256:f9ef0a648310435511e76905f9b89612e45ef2c8b023bee294f5e6f7e73a3e7c"),
    QStringLiteral("sha256:021a8cb75cc80ec86b2fecd72090face26d9752cb9fa3a2f12c5df8b470a2334"),
    QStringLiteral("sha256:2e32a36a5363c7a9acf058e24442e4033a3fb128de4a90cb0f16baf6681c89f7"),
    QStringLiteral("sha256:472538c602abb9ea316bc91b3a6d775553ee481338b5559b68812a06833ea92e"),
    QStringLiteral("sha256:593bea28e420118f60055787d4916209aa1f07371f3cbaf56c5b932f3a3d7335"),
    QStringLiteral("sha256:676b99efd763abd76cc4b3f70711ee2a0b85bdeafb656d4365740c807abbba69"),
    QStringLiteral("sha256:8c026047c6a920f0aae2a0bdf70dbc96f3574825d509579f5131f4cf2ae90084"),
    QStringLiteral("sha256:c0fa31b79d2c06012422e4ed4ed08a86179615463647ac5c44c8f6abef1d4aec"),
    QStringLiteral("sha256:cff5ddd8da79d44894a0dec709d6deb393f376924d50ee824da50e537c6ee08a"),
    QStringLiteral("sha256:ec7c837979bf974f1a2162c779f419402ebe88d5922c921e0a90730d1ab9cb32"),
};
const QStringList kTorchVisionCu118FallbackHashes = {
    QStringLiteral("sha256:0c9bb5dd2a8e4d28e79f29bee66a8a6edff7d607643549fffbe5c1348297e496"),
    QStringLiteral("sha256:19a53484c500f65d8f78aa89093e7474dea1fa3345d379971f24fdbf8716f432"),
    QStringLiteral("sha256:400e43934aa078b8e9b0da06e7c173a0facbf7d5ebd1550cada1829bbd940345"),
    QStringLiteral("sha256:69ea7cba762b958a74e8accbb1f92d822b59497b36d3aecae759bdbab75f85d2"),
    QStringLiteral("sha256:6e19d89e1caba4bd358cf5e5208e230583db55b348011c83bb607dd292f97af6"),
    QStringLiteral("sha256:7e7ef9024a15cc4292b3da9fbd42707d59ba1eb5e098f1df9fdb17a520f52f5f"),
    QStringLiteral("sha256:961d9ca8364d6bfb4063902f1d73d84b446fa51a2cf0d1fb4cd643212e4f8c07"),
    QStringLiteral("sha256:c9b9b80f5cd48d74b34bff1f0e5cc758a53f628e0cd71c97f1e80acf5d2d9b4f"),
    QStringLiteral("sha256:f6cc9ff70c973ad9b6fad8aac5d7a740ebcba962245b65d21018f832a65e07cf"),
    QStringLiteral("sha256:f8b7a96cc061d7d4d981da2851c075d8bca0f8877d544991b230ae2646a4bb45"),
};
const QStringList kTorchAudioCu118FallbackHashes = {
    QStringLiteral("sha256:0b8ee953ef68f90974d1f7741ac2301c39d7c574aebc51081cfda64f5dad0ad4"),
    QStringLiteral("sha256:10c1a4aa08054532b40f805c7775ea9879c46b21bf5e6992f67f71f4f35c55e2"),
    QStringLiteral("sha256:2a443a1f3afcee6f1fa471d2e01afb9889a60d96151af464cef10861224243da"),
    QStringLiteral("sha256:5748b4788791e6c4e5721a5cea42bca24b32c45df95ed5081d51d10140cdebc0"),
    QStringLiteral("sha256:7468fd1ccf7701814c453127c0da0f285997c7e9bf01382c3d3f15822bc9f986"),
    QStringLiteral("sha256:7fbd81465e98248e4e1225052202dcba9b9048132cb6e2312d1e5cae24240511"),
    QStringLiteral("sha256:938d6851974fea9f1734de56dbc8e1d41873d24164003dd1e2f9b50c9809b358"),
    QStringLiteral("sha256:9f03eb87dfb418f83fb2a0d633b01c5fca8cd676da7a9b71c8b88b4dbb5d638e"),
    QStringLiteral("sha256:d03dacfd6ae18447acde8dcf16dd334977b584e4ec06a6cf83d31aa8a24cb5b3"),
    QStringLiteral("sha256:d9525cb49a98a219dd6e5e6895da98cba5143f9be31fff4af3dbe545693e03e7"),
};

QStringList mergeWithFallback(const QStringList& dynamicHashes, const QStringList& fallbackHashes)
{
    QStringList merged = dynamicHashes;
    if (merged.isEmpty())
    {
        merged = fallbackHashes;
    }
    else
    {
        for (const QString& fallback : fallbackHashes)
        {
            if (!merged.contains(fallback))
            {
                merged.append(fallback);
            }
        }
    }
    merged.removeDuplicates();
    return merged;
}

QStringList fetchPytorchPackageHashes(const QString& packageName,
                                      const QString& versionSuffix,
                                      QString& errorOut)
{
    static QHash<QString, QStringList> cache;
    const QString cacheKey = packageName + QLatin1Char('|') + versionSuffix;
    if (cache.contains(cacheKey))
    {
        errorOut.clear();
        return cache.value(cacheKey);
    }

    const QUrl url(QStringLiteral("https://download.pytorch.org/whl/cu118/%1/").arg(packageName));
    QNetworkAccessManager manager;
    QNetworkRequest request(url);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    request.setTransferTimeout(10000);
#endif
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);

    QNetworkReply* reply = manager.get(request);
    QEventLoop loop;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
    QObject::connect(reply, &QNetworkReply::errorOccurred, &loop, &QEventLoop::quit);
#else
    QObject::connect(reply,
                     qOverload<QNetworkReply::NetworkError>(&QNetworkReply::error),
                     &loop,
                     &QEventLoop::quit);
#endif
    loop.exec();

    QStringList hashes;
    if (reply->error() != QNetworkReply::NoError)
    {
        errorOut = QStringLiteral("Unable to query PyTorch index for %1: %2.")
                       .arg(packageName, reply->errorString());
    }
    else if (reply->url().host() != QStringLiteral("download.pytorch.org"))
    {
        errorOut = QStringLiteral("Unexpected redirect while querying PyTorch index for %1.").arg(packageName);
    }
    else
    {
        const QString html = QString::fromUtf8(reply->readAll());
        const QString marker = QStringLiteral("%1-%2").arg(packageName, versionSuffix);
        const QString shaMarker = QStringLiteral("sha256=");
        const QStringList lines = html.split(QRegularExpression(QStringLiteral("[\\r\\n]+")), Qt::SkipEmptyParts);
        static const QRegularExpression kHashRegex(QStringLiteral("^[0-9a-f]{64}$"));
        for (const QString& line : lines)
        {
            if (!line.contains(marker, Qt::CaseInsensitive))
            {
                continue;
            }
            int idx = line.indexOf(shaMarker, 0, Qt::CaseInsensitive);
            if (idx == -1)
            {
                continue;
            }
            const int start = idx + shaMarker.length();
            if (start + 64 > line.length())
            {
                continue;
            }
            const QString candidate = line.mid(start, 64).toLower();
            if (!kHashRegex.match(candidate).hasMatch())
            {
                continue;
            }
            const QString normalized = QStringLiteral("sha256:%1").arg(candidate);
            if (!hashes.contains(normalized))
            {
                hashes.append(normalized);
            }
        }
        if (hashes.isEmpty())
        {
            errorOut = QStringLiteral("No hashes found for %1 %2 on download.pytorch.org.")
                           .arg(packageName, versionSuffix);
        }
        else
        {
            errorOut.clear();
        }
    }

    reply->deleteLater();
    cache.insert(cacheKey, hashes);
    return hashes;
}

constexpr auto kOnnxRuntimeGpuHash = "sha256:bdffcced8a5f6275c0df202220e9232138b336f868cd671c9d2c571e834d2a80";

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
#ifndef Q_OS_WIN
    appendLog(timestamped(tr("Embedded Python bootstrap is only available on Windows.")));
    errorMessage = tr("Install Python 3.11 manually and point CNCTC at that interpreter.");
    return false;
#else
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

    if (!verifyEmbeddedPythonArchive(archivePath, errorMessage))
    {
        cleanupPartial();
        return false;
    }

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
#endif
}

bool EnvManager::verifyEmbeddedPythonArchive(const QString& archivePath, QString& errorMessage)
{
    QFile file(archivePath);
    if (!file.exists())
    {
        errorMessage = tr("Downloaded Python archive missing: %1").arg(archivePath);
        return false;
    }
    if (!file.open(QIODevice::ReadOnly))
    {
        errorMessage = tr("Unable to open Python archive for hashing: %1").arg(file.errorString());
        return false;
    }

    QCryptographicHash hash(QCryptographicHash::Sha256);
    while (!file.atEnd())
    {
        const QByteArray chunk = file.read(1 << 15);
        if (chunk.isEmpty() && file.error() != QFile::NoError)
        {
            errorMessage = tr("Failed while reading Python archive for hashing.");
            return false;
        }
        hash.addData(chunk);
    }

    const QByteArray actual = hash.result().toHex();
    const QByteArray expected = QByteArrayLiteral(kPythonEmbedSha256);

    if (expected.isEmpty())
    {
        return true;
    }

    if (!actual.isEmpty() && actual.toLower() == expected.toLower())
    {
        appendLog(timestamped(tr("Embedded Python archive hash verified.")));
        return true;
    }

    errorMessage = tr("Embedded Python hash mismatch. Expected %1 but received %2.")
                       .arg(QString::fromLatin1(expected))
                       .arg(QString::fromLatin1(actual));
    return false;
}

bool EnvManager::extractEmbeddedPython(const QString& archivePath, QString& errorMessage)
{
#ifndef Q_OS_WIN
    Q_UNUSED(archivePath);
    errorMessage = tr("Embedded Python extraction is unsupported on this platform.");
    return false;
#else
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
#endif
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
    appendLog(timestamped(tr("Upgrading pip...")));

    QStringList upgradeArgs = {QStringLiteral("-m"), QStringLiteral("pip"), QStringLiteral("install"), QStringLiteral("--upgrade"), QStringLiteral("pip")};
    if (!runProcess(m_venvPython, upgradeArgs, m_runtimeRoot, errorMessage, 65))
    {
        return false;
    }

    appendLog(timestamped(tr("Installing training dependencies from lock file...")));

    const QStringList lockCandidates = {
        QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("train/requirements.lock")),
        QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("../train/requirements.lock")),
        QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("../../train/requirements.lock")),
        QDir(m_runtimeRoot).filePath(QStringLiteral("requirements.lock"))
    };

    QString requirementsLock;
    for (const QString& candidate : lockCandidates)
    {
        QFileInfo info(candidate);
        if (info.exists() && info.isFile())
        {
            requirementsLock = info.canonicalFilePath();
            break;
        }
    }

    if (requirementsLock.isEmpty())
    {
        errorMessage = tr("Unable to locate train/requirements.lock.");
        return false;
    }

    appendLog(timestamped(tr("Using %1 for hash-locked install.")).arg(nativePath(requirementsLock)));

    QStringList lockInstallArgs = {QStringLiteral("-m"),
                                   QStringLiteral("pip"),
                                   QStringLiteral("install"),
                                   QStringLiteral("--require-hashes"),
                                   QStringLiteral("-r"),
                                   nativePath(requirementsLock)};
    if (!runProcess(m_venvPython, lockInstallArgs, m_runtimeRoot, errorMessage, 90))
    {
        return false;
    }

    const bool hasGpu = m_gpuSummary.contains(QStringLiteral("NVIDIA"), Qt::CaseInsensitive) && !m_cpuOnly;
    if (hasGpu)
    {
        appendLog(timestamped(tr("NVIDIA GPU detected; switching to CUDA-enabled wheels...")));

        const auto makeRequirementBlock = [](const QString& spec, const QStringList& hashes) {
            QStringList block;
            if (hashes.isEmpty())
            {
                block << spec;
            }
            else
            {
                block << QStringLiteral("%1 \\").arg(spec);
                for (int i = 0; i < hashes.size(); ++i)
                {
                    const bool isLast = (i == hashes.size() - 1);
                    if (isLast)
                    {
                        block << QStringLiteral("    --hash=%1").arg(hashes.at(i));
                    }
                    else
                    {
                        block << QStringLiteral("    --hash=%1 \\").arg(hashes.at(i));
                    }
                }
            }
            block << QString();
            return block;
        };

        auto writeRequirementsFile = [&](const QString& prefix, const QStringList& lines, QString& outputPath) -> bool {
            QTemporaryFile temp(QDir(m_runtimeRoot).filePath(QStringLiteral("%1-req-XXXXXX.txt").arg(prefix)));
            temp.setAutoRemove(false);
            if (!temp.open())
            {
                errorMessage = tr("Unable to create temporary requirements file: %1").arg(temp.errorString());
                return false;
            }

            QTextStream stream(&temp);
            for (const QString& line : lines)
            {
                stream << line << '\n';
            }
            stream.flush();
            if (stream.status() != QTextStream::Ok)
            {
                errorMessage = tr("Unable to write temporary requirements file.");
                temp.close();
                QFile::remove(temp.fileName());
                return false;
            }

            temp.close();
            outputPath = temp.fileName();
            return true;
        };

        QString torchFetchError;
        const QStringList torchDynamicHashes =
            fetchPytorchPackageHashes(QStringLiteral("torch"),
                                      QStringLiteral("%1+cu118").arg(QLatin1String(kTorchVersion)),
                                      torchFetchError);
        const QStringList torchHashes = mergeWithFallback(torchDynamicHashes, kTorchCu118FallbackHashes);
        if (!torchFetchError.isEmpty())
        {
            appendLog(timestamped(tr("Warning: %1 Falling back to bundled torch hashes.")
                                      .arg(torchFetchError)));
        }
        else if (!torchDynamicHashes.isEmpty())
        {
            appendLog(timestamped(tr("Using %1 torch hashes from download.pytorch.org."))
                          .arg(torchDynamicHashes.size()));
        }

        QString visionFetchError;
        const QStringList visionDynamicHashes =
            fetchPytorchPackageHashes(QStringLiteral("torchvision"),
                                      QStringLiteral("%1+cu118").arg(QLatin1String(kTorchVisionVersion)),
                                      visionFetchError);
        const QStringList visionHashes = mergeWithFallback(visionDynamicHashes, kTorchVisionCu118FallbackHashes);
        if (!visionFetchError.isEmpty())
        {
            appendLog(timestamped(tr("Warning: %1 Falling back to bundled torchvision hashes.")
                                      .arg(visionFetchError)));
        }
        else if (!visionDynamicHashes.isEmpty())
        {
            appendLog(timestamped(tr("Using %1 torchvision hashes from download.pytorch.org."))
                          .arg(visionDynamicHashes.size()));
        }

        QString audioFetchError;
        const QStringList audioDynamicHashes =
            fetchPytorchPackageHashes(QStringLiteral("torchaudio"),
                                      QStringLiteral("%1+cu118").arg(QLatin1String(kTorchAudioVersion)),
                                      audioFetchError);
        const QStringList audioHashes = mergeWithFallback(audioDynamicHashes, kTorchAudioCu118FallbackHashes);
        if (!audioFetchError.isEmpty())
        {
            appendLog(timestamped(tr("Warning: %1 Falling back to bundled torchaudio hashes.")
                                      .arg(audioFetchError)));
        }
        else if (!audioDynamicHashes.isEmpty())
        {
            appendLog(timestamped(tr("Using %1 torchaudio hashes from download.pytorch.org."))
                          .arg(audioDynamicHashes.size()));
        }

        QStringList cudaLines;
        cudaLines << QStringLiteral("--extra-index-url %1").arg(QString::fromLatin1(kCudaExtraIndex));
        cudaLines << QString();
        cudaLines += makeRequirementBlock(QStringLiteral("torch==%1+cu118").arg(QLatin1String(kTorchVersion)), torchHashes);
        cudaLines += makeRequirementBlock(QStringLiteral("torchvision==%1+cu118").arg(QLatin1String(kTorchVisionVersion)),
                                          visionHashes);
        cudaLines += makeRequirementBlock(QStringLiteral("torchaudio==%1+cu118").arg(QLatin1String(kTorchAudioVersion)),
                                          audioHashes);

        QString cudaRequirementsPath;
        if (!writeRequirementsFile(QStringLiteral("cuda"), cudaLines, cudaRequirementsPath))
        {
            return false;
        }

        const QString cudaRequirementsNative = nativePath(cudaRequirementsPath);
        QStringList cudaArgs = {QStringLiteral("-m"),
                                QStringLiteral("pip"),
                                QStringLiteral("install"),
                                QStringLiteral("--require-hashes"),
                                QStringLiteral("-r"),
                                cudaRequirementsNative};
        if (!runProcess(m_venvPython, cudaArgs, m_runtimeRoot, errorMessage, 95))
        {
            QFile::remove(cudaRequirementsPath);
            return false;
        }
        QFile::remove(cudaRequirementsPath);

        QStringList onnxLines = makeRequirementBlock(QStringLiteral("onnxruntime-gpu==%1").arg(QLatin1String(kOnnxRuntimeVersion)),
                                                     {QString::fromLatin1(kOnnxRuntimeGpuHash)});
        QString onnxRequirementsPath;
        if (!writeRequirementsFile(QStringLiteral("onnx"), onnxLines, onnxRequirementsPath))
        {
            return false;
        }

        const QString onnxRequirementsNative = nativePath(onnxRequirementsPath);
        QStringList onnxArgs = {QStringLiteral("-m"),
                                QStringLiteral("pip"),
                                QStringLiteral("install"),
                                QStringLiteral("--require-hashes"),
                                QStringLiteral("-r"),
                                onnxRequirementsNative};
        if (!runProcess(m_venvPython, onnxArgs, m_runtimeRoot, errorMessage, 97))
        {
            QFile::remove(onnxRequirementsPath);
            return false;
        }
        QFile::remove(onnxRequirementsPath);
    }
    else
    {
        appendLog(timestamped(tr("CPU-only mode enabled; GPU wheels will be skipped.")));
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

    QString combinedOutput;
    auto flushOutput = [&]() {
        const QByteArray data = process.readAll();
        if (data.isEmpty())
        {
            return;
        }
        const QString text = QString::fromLocal8Bit(data);
        if (!silent)
        {
            appendLog(text);
        }
        combinedOutput.append(text);
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
        const QString lowerOutput = combinedOutput.toLower();
        if (lowerOutput.contains(QStringLiteral("do not match the hashes")) ||
            lowerOutput.contains(QStringLiteral("hashes are required in --require-hashes")) ||
            lowerOutput.contains(QStringLiteral("hash mismatch")))
        {
            errorMessage = tr("Python package hash verification failed. Regenerate train/requirements.lock and retry.");
        }
        else
        {
            errorMessage = tr("Process failed (%1): %2").arg(program, QString::number(exitCode));
        }
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

