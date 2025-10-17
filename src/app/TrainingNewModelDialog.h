#pragma once

#include <QDialog>
#include <QVector>
#include <memory>

namespace Ui
{
class TrainingNewModelDialog;
}

namespace ai
{
struct ModelDescriptor;
}

class TrainingNewModelDialog : public QDialog
{
    Q_OBJECT

public:
    explicit TrainingNewModelDialog(const QVector<ai::ModelDescriptor>& models,
                                    bool gpuAvailable,
                                    QWidget* parent = nullptr);
    ~TrainingNewModelDialog() override;

    [[nodiscard]] QString modelName() const;
    [[nodiscard]] QString baseModelPath() const;
    [[nodiscard]] QString device() const;
    [[nodiscard]] int epochs() const;
    [[nodiscard]] double learningRate() const;
    [[nodiscard]] QString datasetPath() const;
    [[nodiscard]] bool useV2Features() const;

    void setSuggestedName(const QString& name);
    void setSuggestedDataset(const QString& path);
    void lockBaseModel(const QString& basePath);

private slots:
    void onNameChanged(const QString& text);
    void browseDataset();

private:
    void populateBaseModels(const QVector<ai::ModelDescriptor>& models);
    void populateDevices(bool gpuAvailable);
    void updateAcceptState();

    std::unique_ptr<Ui::TrainingNewModelDialog> ui;
    QString m_lockedBasePath;
};
