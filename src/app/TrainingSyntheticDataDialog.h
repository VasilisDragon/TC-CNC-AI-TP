#pragma once

#include <QDialog>
#include <memory>

namespace Ui
{
class TrainingSyntheticDataDialog;
}

class TrainingSyntheticDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit TrainingSyntheticDataDialog(const QString& defaultRoot, QWidget* parent = nullptr);
    ~TrainingSyntheticDataDialog() override;

    [[nodiscard]] QString datasetLabel() const;
    [[nodiscard]] QString outputDirectory() const;
    [[nodiscard]] int sampleCount() const;
    [[nodiscard]] double diversity() const;
    [[nodiscard]] double slopeMix() const;
    [[nodiscard]] bool overwriteExisting() const;

    void setSuggestedLabel(const QString& label);
    void setSuggestedDirectory(const QString& directory);

private slots:
    void syncSamplesFromSlider(int value);
    void syncSamplesFromSpin(int value);
    void syncDiversityFromSlider(int value);
    void syncDiversityFromSpin(double value);
    void syncSlopeFromSlider(int value);
    void syncSlopeFromSpin(double value);
    void browseDirectory();

private:
    void updateDirectoryPlaceholder(const QString& root);

    std::unique_ptr<Ui::TrainingSyntheticDataDialog> ui;
    QString m_defaultRoot;
};
