#include "train/TrainingManager.h"

#include "ai/ModelManager.h"
#include "train/EnvManager.h"

#include <QtCore/QCoreApplication>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonValue>
#include <QtCore/QJsonObject>
#include <QtCore/QProcess>
#include <QtCore/QProcessEnvironment>
#include <QtCore/QRegularExpression>
#include <QtCore/QSettings>
#include <QtCore/QStandardPaths>
#include <QtCore/QTextStream>
#include <QtCore/QTimer>

#include <algorithm>

#ifdef Q_OS_WIN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace train
{

namespace
{
constexpr auto kSyntheticDiversityEnv = "CNCTC_DATASET_DIVERSITY";
constexpr auto kSyntheticSlopeEnv = "CNCTC_DATASET_SLOPE_MIX";
constexpr auto kDatasetPathEnv = "CNCTC_DATASET_PATH";
constexpr auto kBaseModelEnv = "CNCTC_BASE_MODEL";
constexpr auto kFineTuneEnv = "CNCTC_FINE_TUNE";

int countSamplesOnDisk(const QString& datasetPath)
{
    QDir dir(datasetPath);
    if (!dir.exists())
    {
        return 0;
    }

    const QFileInfoList entries = dir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);
    int count = 0;
    for (const QFileInfo& info : entries)
    {
        if (info.fileName().compare(QStringLiteral("train_manifest.json"), Qt::CaseInsensitive) == 0 ||
            info.fileName().compare(QStringLiteral("val_manifest.json"), Qt::CaseInsensitive) == 0)
        {
            continue;
        }
        const QFileInfo meta(info.absoluteFilePath() + QStringLiteral("/meta.json"));
        if (meta.exists())
        {
            ++count;
        }
    }
    return count;
}

QStringList readManifestEntries(const QString& manifestPath)
{
    QFile file(manifestPath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return {};
    }
    const auto doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isArray())
    {
        return {};
    }
    QStringList result;
    for (const QJsonValue& value : doc.array())
    {
        if (value.isString())
        {
            result.push_back(value.toString());
        }
    }
    return result;
}

int estimateSamplesFromManifest(const QString& datasetPath)
{
    const QString manifest = QDir(datasetPath).filePath(QStringLiteral("train_manifest.json"));
    const QString valManifest = QDir(datasetPath).filePath(QStringLiteral("val_manifest.json"));
    int total = 0;
    const QStringList trainEntries = readManifestEntries(manifest);
    const QStringList valEntries = readManifestEntries(valManifest);
    total += trainEntries.size();
    total += valEntries.size();
    if (total > 0)
    {
        return total;
    }
    const int diskCount = countSamplesOnDisk(datasetPath);
    return diskCount > 0 ? diskCount : 2000;
}

QString utcTimestamp()
{
    return QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmmss"));
}

} // namespace

TrainingManager::TrainingManager(QObject* parent)
    : QObject(parent)
{
    qRegisterMetaType<JobStatus>("train::TrainingManager::JobStatus");
    qRegisterMetaType<JobType>("train::TrainingManager::JobType");
    qRegisterMetaType<JobState>("train::TrainingManager::JobState");

    ensureDirectory(datasetsRoot());
}

void TrainingManager::setEnvManager(EnvManager* manager)
{
    m_envManager = manager;
}

void TrainingManager::setModelManager(ai::ModelManager* manager)
{
    m_modelManager = manager;
}

QList<TrainingManager::JobStatus> TrainingManager::jobs() const
{
    QList<JobStatus> result;
    result.reserve(m_jobs.size());
    for (const auto& jobPtr : m_jobs)
    {
        result.push_back(toStatus(*jobPtr));
    }
    return result;
}

QString TrainingManager::datasetsRoot() const
{
    QDir base(QCoreApplication::applicationDirPath());
    const QString path = base.filePath(QStringLiteral("datasets"));
    QDir().mkpath(path);
    return path;
}

QString TrainingManager::modelsRoot() const
{
    if (m_modelManager)
    {
        const QString dir = m_modelManager->modelsDirectory();
        if (!dir.isEmpty())
        {
            QDir().mkpath(dir);
            return dir;
        }
    }
    QDir base(QCoreApplication::applicationDirPath());
    const QString fallback = base.filePath(QStringLiteral("models"));
    QDir().mkpath(fallback);
    return fallback;
}

void TrainingManager::enqueueSyntheticDataset(const SyntheticJobRequest& request)
{
    if (!ensureEnvironmentReady(tr("generate synthetic data")))
    {
        return;
    }

    SyntheticJobRequest normalized = request;
    normalized.sampleCount = std::max(1, normalized.sampleCount);
    normalized.diversity = std::clamp(normalized.diversity, 0.0, 1.0);
    normalized.slopeMix = std::clamp(normalized.slopeMix, 0.0, 1.0);

    QString datasetName = sanitizeName(normalized.label);
    if (datasetName.isEmpty())
    {
        datasetName = QStringLiteral("dataset_%1").arg(utcTimestamp());
    }

    QString targetDir = normalized.outputDir;
    if (targetDir.isEmpty())
    {
        targetDir = QDir(datasetsRoot()).filePath(datasetName);
    }

    QFileInfo dirInfo(targetDir);
    if (dirInfo.exists() && !normalized.overwrite)
    {
        emit toastRequested(tr("Dataset folder already exists: %1").arg(QDir::toNativeSeparators(targetDir)));
        return;
    }
    ensureDirectory(targetDir);

    auto job = std::make_unique<Job>();
    job->id = QUuid::createUuid();
    job->type = JobType::SyntheticDataset;
    job->label = datasetName;
    job->detail = QDir::toNativeSeparators(targetDir);
    job->enqueuedAt = QDateTime::currentDateTimeUtc();
    job->payload = SyntheticPayload{targetDir, normalized.sampleCount, normalized.diversity, normalized.slopeMix, normalized.overwrite};

    queueJob(std::move(job));
}

void TrainingManager::enqueueTraining(const TrainJobRequest& request)
{
    if (!ensureEnvironmentReady(request.fineTune ? tr("fine-tune the model") : tr("train the model")))
    {
        return;
    }

    TrainJobRequest normalized = request;
    normalized.modelName = sanitizeName(normalized.modelName);
    if (normalized.modelName.isEmpty())
    {
        normalized.modelName = QStringLiteral("model_%1").arg(utcTimestamp());
    }
    normalized.epochs = std::max(1, normalized.epochs);
    if (normalized.learningRate <= 0.0)
    {
        normalized.learningRate = 2e-3;
    }
    if (normalized.device.trimmed().isEmpty())
    {
        normalized.device = QStringLiteral("cpu");
    }

    if (!normalized.datasetPath.isEmpty())
    {
        QFileInfo datasetInfo(normalized.datasetPath);
        if (!datasetInfo.exists())
        {
            emit toastRequested(tr("Dataset path does not exist: %1").arg(QDir::toNativeSeparators(normalized.datasetPath)));
            return;
        }
    }

    QString workDir = QDir(modelsRoot()).filePath(QStringLiteral(".tmp_%1").arg(QUuid::createUuid().toString(QUuid::WithoutBraces)));
    QDir work(workDir);
    if (work.exists())
    {
        work.removeRecursively();
    }
    ensureDirectory(workDir);

    const QString modelBase = normalized.modelName;
    const QString torchFile = QDir(modelsRoot()).filePath(modelBase + QStringLiteral(".pt"));
    const QString onnxFile = QDir(modelsRoot()).filePath(modelBase + QStringLiteral(".onnx"));
    const QString schemaFile = QDir(modelsRoot()).filePath(modelBase + QStringLiteral(".onnx.json"));
    const QString cardFile = QDir(modelsRoot()).filePath(modelBase + QStringLiteral(".card.json"));

    auto job = std::make_unique<Job>();
    job->id = QUuid::createUuid();
    job->type = normalized.fineTune ? JobType::FineTune : JobType::Train;
    job->label = modelBase;
    job->detail = normalized.datasetPath.isEmpty() ? tr("Synthetic dataset") : QDir::toNativeSeparators(normalized.datasetPath);
    job->enqueuedAt = QDateTime::currentDateTimeUtc();
    job->payload = TrainPayload{
        modelBase,
        normalized.datasetPath,
        normalized.baseModelPath,
        normalized.device,
        normalized.epochs,
        normalized.learningRate,
        normalized.useV2Features,
        normalized.fineTune,
        workDir,
        torchFile,
        onnxFile,
        schemaFile,
        cardFile,
    };

    queueJob(std::move(job));
}

void TrainingManager::cancelJob(const QUuid& id)
{
    Job* job = findJob(id);
    if (!job)
    {
        return;
    }

    if (job->state == JobState::Queued)
    {
        job->state = JobState::Cancelled;
        job->finishedAt = QDateTime::currentDateTimeUtc();
        emitStatus(*job);
        removeFromPending(job->id);
        return;
    }

    if (job->state == JobState::Running && job->process)
    {
        job->cancelRequested = true;
        job->process->terminate();
        if (!job->process->waitForFinished(3000))
        {
            job->process->kill();
        }
    }
}

void TrainingManager::queueJob(std::unique_ptr<Job> job)
{
    Job* raw = job.get();
    m_jobs.push_back(std::move(job));
    m_pendingOrder.enqueue(raw->id);
    emit jobAdded(toStatus(*raw));
    startNext();
}

void TrainingManager::startNext()
{
    if (!m_activeJob.isNull())
    {
        return;
    }

    while (!m_pendingOrder.isEmpty())
    {
        const QUuid next = m_pendingOrder.dequeue();
        Job* job = findJob(next);
        if (!job || job->state != JobState::Queued)
        {
            continue;
        }
        startJob(*job);
        break;
    }
}

void TrainingManager::startJob(Job& job)
{
    const QString python = pythonExecutable();
    if (python.isEmpty())
    {
        emit toastRequested(tr("Python runtime not found. Please prepare the training environment again."));
        job.state = JobState::Failed;
        job.finishedAt = QDateTime::currentDateTimeUtc();
        emitStatus(job);
        startNext();
        return;
    }

    job.process = std::make_unique<QProcess>(this);
    job.process->setProcessChannelMode(QProcess::SeparateChannels);
    job.process->setProcessEnvironment(QProcessEnvironment::systemEnvironment());

#ifdef Q_OS_WIN
    job.process->setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* args) {
        args->flags |= CREATE_NO_WINDOW;
    });
#endif

    QProcessEnvironment env = job.process->processEnvironment();
    env.insert(QStringLiteral("PYTHONIOENCODING"), QStringLiteral("utf-8"));
    env.insert(QStringLiteral("PYTHONUNBUFFERED"), QStringLiteral("1"));

    QStringList arguments;
    QString workingDir;

    if (job.type == JobType::SyntheticDataset)
    {
        const auto& payload = std::get<SyntheticPayload>(job.payload);
        arguments << scriptPath(QStringLiteral("generate_synthetic.py"));
        arguments << QStringLiteral("--out") << payload.datasetDir;
        arguments << QStringLiteral("--n") << QString::number(payload.sampleCount);
        if (payload.overwrite)
        {
            arguments << QStringLiteral("--force");
        }
        workingDir = payload.datasetDir;
        env.insert(QString::fromLatin1(kSyntheticDiversityEnv), QString::number(payload.diversity, 'f', 3));
        env.insert(QString::fromLatin1(kSyntheticSlopeEnv), QString::number(payload.slopeMix, 'f', 3));
        job.total = payload.sampleCount;
    }
    else
    {
        auto& payload = std::get<TrainPayload>(job.payload);
        arguments << scriptPath(QStringLiteral("train_strategy.py"));
        arguments << QStringLiteral("--epochs") << QString::number(payload.epochs);
        arguments << QStringLiteral("--learning-rate") << QString::number(payload.learningRate, 'g', 6);
        arguments << QStringLiteral("--device") << payload.device;
        arguments << QStringLiteral("--output-dir") << payload.workDir;
        arguments << QStringLiteral("--torchscript-name") << QFileInfo(payload.torchFile).fileName();
        arguments << QStringLiteral("--onnx-name") << QFileInfo(payload.onnxFile).fileName();
        arguments << QStringLiteral("--onnx-json-name") << QFileInfo(payload.schemaFile).fileName();
        arguments << QStringLiteral("--model-card-name") << QFileInfo(payload.cardFile).fileName();
        arguments << QStringLiteral("--samples");

        int sampleCount = 0;
        if (!payload.datasetPath.isEmpty())
        {
            sampleCount = estimateSamplesFromManifest(payload.datasetPath);
            env.insert(QString::fromLatin1(kDatasetPathEnv), payload.datasetPath);
        }
        if (sampleCount <= 0)
        {
            sampleCount = 2000;
        }
        arguments << QString::number(sampleCount);

        if (payload.useV2Features)
        {
            arguments << QStringLiteral("--v2-features");
        }

        if (!payload.baseModelPath.isEmpty())
        {
            env.insert(QString::fromLatin1(kBaseModelEnv), payload.baseModelPath);
        }
        if (payload.fineTune)
        {
            env.insert(QString::fromLatin1(kFineTuneEnv), QStringLiteral("1"));
        }
        env.insert(QStringLiteral("CNCTC_TRAINING_MODE"), job.type == JobType::FineTune ? QStringLiteral("fine_tune")
                                                                                          : QStringLiteral("train"));

        workingDir = payload.workDir;
        job.total = payload.epochs;
    }

    job.process->setProgram(python);
    job.process->setArguments(arguments);
    if (!workingDir.isEmpty())
    {
        job.process->setWorkingDirectory(workingDir);
    }
    job.process->setProcessEnvironment(env);

    Job* jobPtr = &job;
    connect(job.process.get(), &QProcess::readyReadStandardOutput, this, [this, jobPtr]() {
        handleStdOut(*jobPtr);
    });
    connect(job.process.get(), &QProcess::readyReadStandardError, this, [this, jobPtr]() {
        handleStdErr(*jobPtr);
    });

    connect(
        job.process.get(),
        &QProcess::errorOccurred,
        this,
        [this, jobPtr](QProcess::ProcessError) {
            if (jobPtr->state == JobState::Running)
            {
                completeJob(*jobPtr, false, jobPtr->process ? jobPtr->process->exitCode() : -1);
            }
        });

    connect(
        job.process.get(),
        qOverload<int, QProcess::ExitStatus>(&QProcess::finished),
        this,
        [this, jobPtr](int exitCode, QProcess::ExitStatus status) {
            const bool success = (status == QProcess::NormalExit) && (exitCode == 0) && !jobPtr->cancelRequested;
            completeJob(*jobPtr, success, exitCode);
        });

    job.state = JobState::Running;
    job.startedAt = QDateTime::currentDateTimeUtc();
    job.timer.restart();
    job.progress = job.total > 0 ? 0 : -1;
    emitStatus(job);

    job.process->start();
    m_activeJob = job.id;
}

void TrainingManager::handleStdOut(Job& job)
{
    if (!job.process)
    {
        return;
    }
    const QString text = QString::fromUtf8(job.process->readAllStandardOutput());
    appendLog(job, text);
    parseOutput(job, text);
}

void TrainingManager::handleStdErr(Job& job)
{
    if (!job.process)
    {
        return;
    }
    const QString text = QString::fromUtf8(job.process->readAllStandardError());
    appendLog(job, text);
}

void TrainingManager::parseOutput(Job& job, const QString& chunk)
{
    if (chunk.isEmpty())
    {
        return;
    }
    const QStringList lines = chunk.split(QRegularExpression(QStringLiteral("[\\r\\n]+")), Qt::SkipEmptyParts);
    if (lines.isEmpty())
    {
        return;
    }

    static const QRegularExpression kSampleRegex(QStringLiteral(R"(\[(\d+)\s*/\s*(\d+)\])"));
    static const QRegularExpression kEpochRegex(QStringLiteral(R"(\[Epoch\s+(\d+))"));
    static const QRegularExpression kMetricsRegex(
        QStringLiteral(R"(val_loss=([0-9.]+)\s+val_acc=([0-9.]+))"), QRegularExpression::CaseInsensitiveOption);

    for (const QString& line : lines)
    {
        if (job.type == JobType::SyntheticDataset)
        {
            const auto match = kSampleRegex.match(line);
            if (match.hasMatch())
            {
                bool okCurrent = false;
                bool okTotal = false;
                const int current = match.captured(1).toInt(&okCurrent);
                const int total = match.captured(2).toInt(&okTotal);
                if (okCurrent && okTotal && total > 0)
                {
                    updateProgress(job, current, total);
                }
            }
        }
        else
        {
            const auto epochMatch = kEpochRegex.match(line);
            if (epochMatch.hasMatch())
            {
                bool okEpoch = false;
                const int epoch = epochMatch.captured(1).toInt(&okEpoch);
                if (okEpoch)
                {
                    updateProgress(job, epoch, std::max(job.total, 1));
                }
            }

            const auto metricsMatch = kMetricsRegex.match(line);
            if (metricsMatch.hasMatch())
            {
                job.detail = tr("val_loss=%1 val_acc=%2").arg(metricsMatch.captured(1), metricsMatch.captured(2));
                emitStatus(job);
            }
        }
    }
}

void TrainingManager::completeJob(Job& job, bool success, int)
{
    if (job.state != JobState::Running)
    {
        return;
    }

    job.finishedAt = QDateTime::currentDateTimeUtc();
    job.timer.invalidate();

    if (job.cancelRequested)
    {
        job.state = JobState::Cancelled;
    }
    else
    {
        job.state = success ? JobState::Succeeded : JobState::Failed;
    }

    if (success)
    {
        if (job.type == JobType::SyntheticDataset)
        {
            finalizeSyntheticJob(job, success);
        }
        else
        {
            finalizeTrainingJob(job, success);
        }
        job.progress = 100;
    }
    else if (job.type != JobType::SyntheticDataset)
    {
        auto& payload = std::get<TrainPayload>(job.payload);
        QDir(payload.workDir).removeRecursively();
    }

    emitStatus(job);

    if (job.process)
    {
        job.process->deleteLater();
        job.process.release();
    }

    if (m_activeJob == job.id)
    {
        m_activeJob = QUuid();
        QTimer::singleShot(0, this, [this]() {
            startNext();
        });
    }
}

void TrainingManager::finalizeSyntheticJob(Job& job, bool success)
{
    if (!success)
    {
        return;
    }

    const auto& payload = std::get<SyntheticPayload>(job.payload);
    const QString manifestPath = QDir(payload.datasetDir).filePath(QStringLiteral("train_manifest.json"));
    if (QFileInfo::exists(manifestPath))
    {
        const QStringList entries = readManifestEntries(manifestPath);
        if (!entries.isEmpty())
        {
            job.detail = tr("%1 samples ready at %2")
                             .arg(entries.size())
                             .arg(QDir::toNativeSeparators(payload.datasetDir));
        }
        else
        {
            job.detail = QDir::toNativeSeparators(payload.datasetDir);
        }
    }
    else
    {
        job.detail = QDir::toNativeSeparators(payload.datasetDir);
    }
}

void TrainingManager::finalizeTrainingJob(Job& job, bool success)
{
    if (!success)
    {
        return;
    }

    auto& payload = std::get<TrainPayload>(job.payload);

    const auto moveWithOverwrite = [](const QString& src, const QString& dst) {
        if (src.isEmpty() || dst.isEmpty())
        {
            return false;
        }
        if (!QFileInfo::exists(src))
        {
            return false;
        }
        if (QFileInfo::exists(dst))
        {
            QFile::remove(dst);
        }
        QDir().mkpath(QFileInfo(dst).absolutePath());
        return QFile::rename(src, dst);
    };

    const QString workTorch = QDir(payload.workDir).filePath(QFileInfo(payload.torchFile).fileName());
    const QString workOnnx = QDir(payload.workDir).filePath(QFileInfo(payload.onnxFile).fileName());
    const QString workSchema = QDir(payload.workDir).filePath(QFileInfo(payload.schemaFile).fileName());
    const QString workCard = QDir(payload.workDir).filePath(QFileInfo(payload.cardFile).fileName());

    moveWithOverwrite(workTorch, payload.torchFile);
    moveWithOverwrite(workOnnx, payload.onnxFile);
    moveWithOverwrite(workSchema, payload.schemaFile);
    moveWithOverwrite(workCard, payload.cardFile);

    QDir(payload.workDir).removeRecursively();

    if (m_modelManager)
    {
        m_modelManager->refresh();
    }

    emit modelRegistered(payload.torchFile);

    QFile card(payload.cardFile);
    if (card.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        const QJsonDocument doc = QJsonDocument::fromJson(card.readAll());
        if (doc.isObject())
        {
            const QJsonObject root = doc.object();
            const QJsonObject metrics = root.value(QStringLiteral("validation")).toObject();
            if (!metrics.isEmpty())
            {
                const double loss = metrics.value(QStringLiteral("loss")).toDouble(-1.0);
                const double acc = metrics.value(QStringLiteral("accuracy")).toDouble(-1.0);
                if (loss >= 0.0 && acc >= 0.0)
                {
                    job.detail = tr("val_loss=%1 val_acc=%2")
                                     .arg(QString::number(loss, 'f', 4), QString::number(acc, 'f', 3));
                }
            }
        }
    }
}

void TrainingManager::updateProgress(Job& job, int current, int total)
{
    if (total <= 0)
    {
        return;
    }
    job.current = std::clamp(current, 0, total);
    job.total = total;
    const int percent = static_cast<int>(std::clamp((job.current * 100) / std::max(1, job.total), 0, 100));
    job.progress = percent;
    refreshEta(job);
    emitStatus(job);
}

void TrainingManager::refreshEta(Job& job)
{
    if (!job.timer.isValid() || job.current <= 0 || job.total <= 0)
    {
        job.etaMs = -1;
        return;
    }
    const qint64 elapsed = job.timer.elapsed();
    const qint64 remainingUnits = job.total - job.current;
    if (remainingUnits <= 0)
    {
        job.etaMs = 0;
        return;
    }
    const double perUnit = static_cast<double>(elapsed) / static_cast<double>(job.current);
    job.etaMs = static_cast<qint64>(perUnit * static_cast<double>(remainingUnits));
}

TrainingManager::Job* TrainingManager::findJob(const QUuid& id)
{
    for (auto& job : m_jobs)
    {
        if (job->id == id)
        {
            return job.get();
        }
    }
    return nullptr;
}

const TrainingManager::Job* TrainingManager::findJob(const QUuid& id) const
{
    for (const auto& job : m_jobs)
    {
        if (job->id == id)
        {
            return job.get();
        }
    }
    return nullptr;
}

void TrainingManager::removeFromPending(const QUuid& id)
{
    if (m_pendingOrder.isEmpty())
    {
        return;
    }
    QQueue<QUuid> updated;
    while (!m_pendingOrder.isEmpty())
    {
        QUuid next = m_pendingOrder.dequeue();
        if (next != id)
        {
            updated.enqueue(next);
        }
    }
    m_pendingOrder = std::move(updated);
}

bool TrainingManager::ensureEnvironmentReady(const QString& actionLabel)
{
    if (m_envManager && m_envManager->isBusy())
    {
        emit toastRequested(tr("Please wait for environment preparation to finish before %1.").arg(actionLabel));
        return false;
    }

    QSettings settings;
    if (!settings.value(QStringLiteral("training/envReady"), false).toBool())
    {
        emit toastRequested(
            tr("Training environment is not ready. Run “Prepare Environment” before %1.").arg(actionLabel));
        return false;
    }

    const QString python = pythonExecutable();
    if (python.isEmpty() || !QFileInfo::exists(python))
    {
        emit toastRequested(tr("Python runtime missing. Repair the training environment."));
        return false;
    }
    return true;
}

QString TrainingManager::pythonExecutable() const
{
    const QString root = runtimeRoot();
    QDir venv(root);
    venv.cd(QStringLiteral("venv"));
#ifdef Q_OS_WIN
    const QString candidate = venv.filePath(QStringLiteral("Scripts/python.exe"));
#else
    const QString candidate = venv.filePath(QStringLiteral("bin/python"));
#endif
    if (QFileInfo::exists(candidate))
    {
        return candidate;
    }

    QSettings settings;
    const QString configured = settings.value(QStringLiteral("training/pythonExecutable")).toString();
    if (QFileInfo::exists(configured))
    {
        return configured;
    }
    return QString();
}

QString TrainingManager::runtimeRoot() const
{
    QDir base(QCoreApplication::applicationDirPath());
    const QString runtime = base.filePath(QStringLiteral("runtime"));
    QDir().mkpath(runtime);
    return runtime;
}

QString TrainingManager::scriptPath(const QString& fileName) const
{
    return QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("train/%1").arg(fileName));
}

QString TrainingManager::ensureDirectory(const QString& path) const
{
    if (path.isEmpty())
    {
        return path;
    }
    QDir dir(path);
    if (!dir.exists())
    {
        QDir().mkpath(path);
    }
    return path;
}

QString TrainingManager::sanitizeName(const QString& name) const
{
    QString clean = name.trimmed();
    if (clean.isEmpty())
    {
        return {};
    }
    clean.replace(QRegularExpression(QStringLiteral(R"([^A-Za-z0-9_\-]+)")), QStringLiteral("_"));
    while (clean.startsWith(QLatin1Char('_')))
    {
        clean.remove(0, 1);
    }
    while (clean.endsWith(QLatin1Char('_')))
    {
        clean.chop(1);
    }
    return clean;
}

TrainingManager::JobStatus TrainingManager::toStatus(const Job& job) const
{
    JobStatus status;
    status.id = job.id;
    status.type = job.type;
    status.state = job.state;
    status.label = job.label;
    status.detail = job.detail;
    status.progress = job.progress;
    status.current = job.current;
    status.total = job.total;
    status.enqueuedAt = job.enqueuedAt;
    status.startedAt = job.startedAt;
    status.finishedAt = job.finishedAt;

    if (job.timer.isValid())
    {
        status.elapsedMs = job.timer.elapsed();
        status.etaMs = job.etaMs;
    }
    else if (job.startedAt.isValid() && job.finishedAt.isValid())
    {
        status.elapsedMs = job.startedAt.msecsTo(job.finishedAt);
        status.etaMs = 0;
    }
    else
    {
        status.elapsedMs = 0;
        status.etaMs = -1;
    }
    return status;
}

void TrainingManager::emitStatus(const Job& job)
{
    emit jobUpdated(toStatus(job));
}

void TrainingManager::appendLog(Job& job, const QString& text)
{
    if (text.isEmpty())
    {
        return;
    }
    emit jobLog(job.id, text);
}

} // namespace train

