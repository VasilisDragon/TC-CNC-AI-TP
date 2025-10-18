#include "app/AiPreferencesDialog.h"
#include "ui_AiPreferencesDialog.h"

#include <QtCore/QSignalBlocker>
#include <QtCore/Qt>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QPushButton>

AiPreferencesDialog::AiPreferencesDialog(QWidget* parent)
    : QDialog(parent)
    , ui(std::make_unique<Ui::AiPreferencesDialog>())
{
    ui->setupUi(this);

    connect(ui->devicePreferenceCombo,
            qOverload<int>(&QComboBox::currentIndexChanged),
            this,
            &AiPreferencesDialog::onDevicePreferenceChanged);
    connect(ui->testButton, &QPushButton::clicked, this, &AiPreferencesDialog::onTestClicked);
}

AiPreferencesDialog::~AiPreferencesDialog() = default;

void AiPreferencesDialog::setForceCpu(bool checked)
{
    QSignalBlocker blocker(ui->devicePreferenceCombo);
    ui->devicePreferenceCombo->setCurrentIndex(checked ? 1 : 0);
}

bool AiPreferencesDialog::forceCpu() const
{
    return ui->devicePreferenceCombo->currentIndex() == 1;
}

void AiPreferencesDialog::setGpuAvailable(bool available)
{
    const int flags = available ? static_cast<int>(Qt::ItemIsEnabled | Qt::ItemIsSelectable)
                                : static_cast<int>(Qt::NoItemFlags);
    ui->devicePreferenceCombo->setItemData(0, flags, Qt::UserRole - 1);
    if (!available && ui->devicePreferenceCombo->currentIndex() == 0)
    {
        QSignalBlocker blocker(ui->devicePreferenceCombo);
        ui->devicePreferenceCombo->setCurrentIndex(1);
    }
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

void AiPreferencesDialog::onDevicePreferenceChanged(int index)
{
    ui->statusLabel->clear();
    emit forceCpuChanged(index == 1);
}

void AiPreferencesDialog::onTestClicked()
{
    emit testRequested();
}

