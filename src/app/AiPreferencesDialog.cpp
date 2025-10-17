#include "app/AiPreferencesDialog.h"
#include "ui_AiPreferencesDialog.h"

#include <QtWidgets/QCheckBox>
#include <QtWidgets/QPushButton>

AiPreferencesDialog::AiPreferencesDialog(QWidget* parent)
    : QDialog(parent)
    , ui(std::make_unique<Ui::AiPreferencesDialog>())
{
    ui->setupUi(this);

    connect(ui->forceCpuCheckBox, &QCheckBox::toggled, this, &AiPreferencesDialog::onForceCpuToggled);
    connect(ui->testButton, &QPushButton::clicked, this, &AiPreferencesDialog::onTestClicked);
}

AiPreferencesDialog::~AiPreferencesDialog() = default;

void AiPreferencesDialog::setForceCpu(bool checked)
{
    ui->forceCpuCheckBox->setChecked(checked);
}

bool AiPreferencesDialog::forceCpu() const
{
    return ui->forceCpuCheckBox->isChecked();
}

void AiPreferencesDialog::setModelInfo(const QString& name,
                                       const QString& path,
                                       const QString& device,
                                       const QString& modified)
{
    ui->modelNameValue->setText(name);
    ui->modelPathValue->setText(path);
    ui->deviceValue->setText(device);
    ui->modifiedValue->setText(modified);
}

void AiPreferencesDialog::setStatus(const QString& text)
{
    ui->statusLabel->setText(text);
}

void AiPreferencesDialog::setTestEnabled(bool enabled)
{
    ui->testButton->setEnabled(enabled);
    if (!enabled)
    {
        ui->statusLabel->setText(tr("Load a model to enable testing."));
        ui->lastTestValue->setText(QStringLiteral("-"));
    }
}

void AiPreferencesDialog::setLastTestResult(const QString& text)
{
    ui->lastTestValue->setText(text);
}

void AiPreferencesDialog::onForceCpuToggled(bool checked)
{
    ui->statusLabel->clear();
    emit forceCpuChanged(checked);
}

void AiPreferencesDialog::onTestClicked()
{
    emit testRequested();
}

