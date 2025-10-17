#pragma once

#include <QDialog>
#include <memory>

namespace Ui
{
class AiPreferencesDialog;
}

class AiPreferencesDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AiPreferencesDialog(QWidget* parent = nullptr);
    ~AiPreferencesDialog() override;

    void setForceCpu(bool checked);
    bool forceCpu() const;

    void setModelInfo(const QString& name,
                      const QString& path,
                      const QString& device,
                      const QString& modified);

    void setStatus(const QString& text);
    void setTestEnabled(bool enabled);
    void setLastTestResult(const QString& text);

signals:
    void testRequested();
    void forceCpuChanged(bool checked);

private slots:
    void onForceCpuToggled(bool checked);
    void onTestClicked();

private:
    std::unique_ptr<Ui::AiPreferencesDialog> ui;
};
