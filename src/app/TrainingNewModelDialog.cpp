#include "app/TrainingNewModelDialog.h"

#include "ai/ModelManager.h"
#include "ui_TrainingNewModelDialog.h"

#include <QDialogButtonBox>
#include <QFileDialog>
#include <QLineEdit>
#include <QToolButton>
#include <QPushButton>

#include <QtCore/QDateTime>
#include <QtCore/QFileInfo>
#include <QtCore/QRegularExpression>

namespace
{

QString defaultModelName()
{
    return QStringLiteral("strategy_%1").arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd_HHmm")));
}

QString displayForModel(const ai::ModelDescriptor& descriptor)
{
    const QString backend = descriptor.backend == ai::ModelDescriptor::Backend::Onnx ? QStringLiteral("ONNX")
                                                                                    : QStringLiteral("Torch");
    return QStringLiteral("%1 (%2, %3)")
        .arg(descriptor.fileName,
             backend,
             descriptor.modified.toLocalTime().toString(QStringLiteral("yyyy-MM-dd HH:mm")));
}

} // namespace

TrainingNewModelDialog::TrainingNewModelDialog(const QVector<ai::ModelDescriptor>& models,
                                               bool gpuAvailable,
                                               QWidget* parent)
    : QDialog(parent)
    , ui(std::make_unique<Ui::TrainingNewModelDialog>())
{
    ui->setupUi(this);

    populateBaseModels(models);
    populateDevices(gpuAvailable);

    ui->nameLineEdit->setText(defaultModelName());
    ui->datasetLineEdit->setPlaceholderText(tr("Use prepared dataset folder"));
    ui->buttonBox->button(QDialogButtonBox::Ok)->setEnabled(false);

    connect(ui->nameLineEdit, &QLineEdit::textChanged, this, &TrainingNewModelDialog::onNameChanged);
    connect(ui->browseDatasetButton, &QToolButton::clicked, this, &TrainingNewModelDialog::browseDataset);

    updateAcceptState();
}

TrainingNewModelDialog::~TrainingNewModelDialog() = default;

QString TrainingNewModelDialog::modelName() const
{
    return ui->nameLineEdit->text().trimmed();
}

QString TrainingNewModelDialog::baseModelPath() const
{
    if (!m_lockedBasePath.isEmpty())
    {
        return m_lockedBasePath;
    }
    return ui->baseComboBox->currentData().toString();
}

QString TrainingNewModelDialog::device() const
{
    return ui->deviceComboBox->currentData().toString();
}

int TrainingNewModelDialog::epochs() const
{
    return ui->epochsSpinBox->value();
}

double TrainingNewModelDialog::learningRate() const
{
    return ui->learningRateSpinBox->value();
}

QString TrainingNewModelDialog::datasetPath() const
{
    return ui->datasetLineEdit->text().trimmed();
}

bool TrainingNewModelDialog::useV2Features() const
{
    return ui->v2FeaturesCheckBox->isChecked();
}

void TrainingNewModelDialog::setSuggestedName(const QString& name)
{
    if (!name.isEmpty())
    {
        ui->nameLineEdit->setText(name);
    }
    updateAcceptState();
}

void TrainingNewModelDialog::setSuggestedDataset(const QString& path)
{
    ui->datasetLineEdit->setText(path);
}

void TrainingNewModelDialog::lockBaseModel(const QString& basePath)
{
    m_lockedBasePath = basePath;
    ui->baseComboBox->setEnabled(basePath.isEmpty());
    if (!basePath.isEmpty())
    {
        const QString nativeBase = QFileInfo(basePath).fileName();
        ui->baseComboBox->clear();
        ui->baseComboBox->addItem(nativeBase, basePath);
        ui->baseComboBox->setCurrentIndex(0);
    }
}

void TrainingNewModelDialog::onNameChanged(const QString&)
{
    updateAcceptState();
}

void TrainingNewModelDialog::browseDataset()
{
    const QString current = ui->datasetLineEdit->text();
    const QString selected = QFileDialog::getExistingDirectory(this, tr("Select Dataset Folder"), current);
    if (!selected.isEmpty())
    {
        ui->datasetLineEdit->setText(selected);
    }
}

void TrainingNewModelDialog::populateBaseModels(const QVector<ai::ModelDescriptor>& models)
{
    ui->baseComboBox->clear();
    ui->baseComboBox->addItem(tr("Train from scratch"), QString());
    for (const auto& descriptor : models)
    {
        ui->baseComboBox->addItem(displayForModel(descriptor), descriptor.absolutePath);
    }
}

void TrainingNewModelDialog::populateDevices(bool gpuAvailable)
{
    ui->deviceComboBox->clear();
    ui->deviceComboBox->addItem(tr("CPU"), QStringLiteral("cpu"));
    if (gpuAvailable)
    {
        ui->deviceComboBox->addItem(tr("GPU"), QStringLiteral("cuda"));
    }
}

void TrainingNewModelDialog::updateAcceptState()
{
    const QString name = ui->nameLineEdit->text().trimmed();
    bool enable = !name.isEmpty();
    if (enable)
    {
        static const QRegularExpression allowed(QStringLiteral("^[A-Za-z0-9_\\-]+$"));
        enable = allowed.match(name).hasMatch();
    }
    if (auto* okButton = ui->buttonBox->button(QDialogButtonBox::Ok))
    {
        okButton->setEnabled(enable);
    }
}
