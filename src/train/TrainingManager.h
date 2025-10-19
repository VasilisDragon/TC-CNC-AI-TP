#pragma once

#include <QtCore/QDateTime>
#include <QtCore/QElapsedTimer>
#include <QtCore/QList>
#include <QtCore/QObject>
#include <QtCore/QPointer>
#include <QtCore/QProcess>
#include <QtCore/QQueue>
#include <QtCore/QString>
#include <QtCore/QUuid>

#include <memory>
#include <variant>
#include <vector>

namespace ai
{
class ModelManager;
}

namespace train
{

class EnvManager;

class TrainingManager : public QObject
{
    Q_OBJECT

public:
    enum class JobType
    {
        SyntheticDataset,
        Train,
        FineTune
    };
    Q_ENUM(JobType)

    enum class JobState
    {
        Queued,
        Running,
        Succeeded,
        Failed,
        Cancelled
    };
    Q_ENUM(JobState)

    struct JobStatus
    {
        QUuid id;
        JobType type{JobType::SyntheticDataset};
        JobState state{JobState::Queued};
        QString label;
        QString detail;
        int progress{-1};
        int current{0};
        int total{0};
        qint64 elapsedMs{0};
        qint64 etaMs{-1};
        QDateTime enqueuedAt;
        QDateTime startedAt;
        QDateTime finishedAt;
    };

    struct SyntheticJobRequest
    {
        QString label;
        int sampleCount{0};
        double diversity{0.5};
        double slopeMix{0.5};
        bool overwrite{false};
        QString outputDir;
    };

    struct TrainJobRequest
    {
        QString modelName;
        QString datasetPath;
        QString baseModelPath;
        QString device{QStringLiteral("cpu")};
        int epochs{10};
        double learningRate{2e-3};
        bool useV2Features{true};
        bool fineTune{false};
    };

    explicit TrainingManager(QObject* parent = nullptr);

    void setEnvManager(EnvManager* manager);
    void setModelManager(ai::ModelManager* manager);

    [[nodiscard]] QList<JobStatus> jobs() const;
    [[nodiscard]] QString datasetsRoot() const;
    [[nodiscard]] QString modelsRoot() const;

    Q_SIGNALS:
    void jobAdded(const train::TrainingManager::JobStatus& status);
    void jobUpdated(const train::TrainingManager::JobStatus& status);
    void jobRemoved(const QUuid& id);
    void jobLog(const QUuid& id, const QString& text);
    void toastRequested(const QString& message);
    void modelRegistered(const QString& absolutePath);

public slots:
    void enqueueSyntheticDataset(const SyntheticJobRequest& request);
    void enqueueTraining(const TrainJobRequest& request);
    void cancelJob(const QUuid& id);

private:
    struct SyntheticPayload
    {
        QString datasetDir;
        int sampleCount{0};
        double diversity{0.5};
        double slopeMix{0.5};
        bool overwrite{false};
    };

    struct TrainPayload
    {
        QString modelName;
        QString datasetPath;
        QString baseModelPath;
        QString device;
        int epochs{0};
        double learningRate{0.0};
        bool useV2Features{true};
        bool fineTune{false};
        QString workDir;
        QString torchFile;
        QString onnxFile;
        QString schemaFile;
        QString torchCardFile;
        QString onnxCardFile;
    };

    struct Job
    {
        QUuid id;
        JobType type{JobType::SyntheticDataset};
        JobState state{JobState::Queued};
        QString label;
        QString detail;
        int progress{-1};
        int current{0};
        int total{0};
        QDateTime enqueuedAt;
        QDateTime startedAt;
        QDateTime finishedAt;
        qint64 etaMs{-1};
        bool cancelRequested{false};
        QElapsedTimer timer;
        std::unique_ptr<QProcess> process;
        std::variant<SyntheticPayload, TrainPayload> payload;
    };

    [[nodiscard]] JobStatus toStatus(const Job& job) const;
    void emitStatus(const Job& job);
    void appendLog(Job& job, const QString& text);
    void queueJob(std::unique_ptr<Job> job);
    void startNext();
    void startJob(Job& job);
    void handleStdOut(Job& job);
    void handleStdErr(Job& job);
    void parseOutput(Job& job, const QString& chunk);
    void completeJob(Job& job, bool success, int exitCode);
    void finalizeSyntheticJob(Job& job, bool success);
    void finalizeTrainingJob(Job& job, bool success);
    void updateProgress(Job& job, int current, int total);
    void refreshEta(Job& job);

    [[nodiscard]] Job* findJob(const QUuid& id);
    [[nodiscard]] const Job* findJob(const QUuid& id) const;
    void removeFromPending(const QUuid& id);

    [[nodiscard]] bool ensureEnvironmentReady(const QString& actionLabel);
    [[nodiscard]] QString pythonExecutable() const;
    [[nodiscard]] QString runtimeRoot() const;
    [[nodiscard]] QString scriptPath(const QString& fileName) const;
    [[nodiscard]] QString ensureDirectory(const QString& path) const;
    [[nodiscard]] QString sanitizeName(const QString& name) const;

    std::vector<std::unique_ptr<Job>> m_jobs;
    QQueue<QUuid> m_pendingOrder;
    QUuid m_activeJob;
    QPointer<EnvManager> m_envManager;
    ai::ModelManager* m_modelManager{nullptr};
};

} // namespace train

Q_DECLARE_METATYPE(train::TrainingManager::JobStatus)
Q_DECLARE_METATYPE(train::TrainingManager::JobType)
Q_DECLARE_METATYPE(train::TrainingManager::JobState)

