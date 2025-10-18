#include "app/TrainingSyntheticDataDialog.h"

#include "ui_TrainingSyntheticDataDialog.h"

#include <QtCore/QDateTime>
#include <QtCore/QDir>
#include <QtCore/QtGlobal>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QLineEdit>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QPushButton>

#include <cmath>

namespace
{

QString defaultLabel()
{
    return QStringLiteral("synthetic_%1").arg(QDateTime::currentDateTimeUtc().toString(QStringLiteral("yyyyMMdd")));
}

QString joinPath(const QString& root, const QString& label)
{
    QDir dir(root);
    return dir.filePath(label);
}

} // namespace

TrainingSyntheticDataDialog::TrainingSyntheticDataDialog(const QString& defaultRoot, QWidget* parent)
    : QDialog(parent)
    , ui(std::make_unique<Ui::TrainingSyntheticDataDialog>())
    , m_defaultRoot(defaultRoot)
{
    ui->setupUi(this);
    ui->labelLineEdit->setText(defaultLabel());
    ui->samplesSpinBox->setValue(ui->samplesSlider->value());
    ui->diversitySpinBox->setValue(ui->diversitySlider->value() / 100.0);
    ui->slopeSpinBox->setValue(ui->slopeSlider->value() / 100.0);

    updateDirectoryPlaceholder(defaultRoot);
    ui->directoryLineEdit->setText(joinPath(defaultRoot, ui->labelLineEdit->text()));

    connect(ui->samplesSlider, &QSlider::valueChanged, this, &TrainingSyntheticDataDialog::syncSamplesFromSlider);
    connect(
        ui->samplesSpinBox, qOverload<int>(&QSpinBox::valueChanged), this, &TrainingSyntheticDataDialog::syncSamplesFromSpin);

    connect(ui->diversitySlider, &QSlider::valueChanged, this, &TrainingSyntheticDataDialog::syncDiversityFromSlider);
    connect(ui->diversitySpinBox,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &TrainingSyntheticDataDialog::syncDiversityFromSpin);

    connect(ui->slopeSlider, &QSlider::valueChanged, this, &TrainingSyntheticDataDialog::syncSlopeFromSlider);
    connect(ui->slopeSpinBox,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            &TrainingSyntheticDataDialog::syncSlopeFromSpin);

    connect(ui->labelLineEdit, &QLineEdit::textChanged, this, [this](const QString& text) {
        if (ui->directoryLineEdit->text().trimmed().isEmpty() || ui->directoryLineEdit->text() == joinPath(m_defaultRoot, defaultLabel()))
        {
            ui->directoryLineEdit->setText(joinPath(m_defaultRoot, text.trimmed()));
        }
    });

    connect(ui->browseDirectoryButton, &QToolButton::clicked, this, &TrainingSyntheticDataDialog::browseDirectory);

    if (auto* okButton = ui->buttonBox->button(QDialogButtonBox::Ok))
    {
        okButton->setEnabled(true);
    }
}

TrainingSyntheticDataDialog::~TrainingSyntheticDataDialog() = default;

QString TrainingSyntheticDataDialog::datasetLabel() const
{
    return ui->labelLineEdit->text().trimmed();
}

QString TrainingSyntheticDataDialog::outputDirectory() const
{
    const QString explicitDir = ui->directoryLineEdit->text().trimmed();
    if (!explicitDir.isEmpty())
    {
        return explicitDir;
    }
    return joinPath(m_defaultRoot, datasetLabel());
}

int TrainingSyntheticDataDialog::sampleCount() const
{
    return ui->samplesSpinBox->value();
}

double TrainingSyntheticDataDialog::diversity() const
{
    return ui->diversitySpinBox->value();
}

double TrainingSyntheticDataDialog::slopeMix() const
{
    return ui->slopeSpinBox->value();
}

bool TrainingSyntheticDataDialog::overwriteExisting() const
{
    return ui->overwriteCheckBox->isChecked();
}

void TrainingSyntheticDataDialog::setSuggestedLabel(const QString& label)
{
    if (!label.isEmpty())
    {
        ui->labelLineEdit->setText(label);
    }
}

void TrainingSyntheticDataDialog::setSuggestedDirectory(const QString& directory)
{
    ui->directoryLineEdit->setText(directory);
}

void TrainingSyntheticDataDialog::syncSamplesFromSlider(int value)
{
    if (ui->samplesSpinBox->value() != value)
    {
        ui->samplesSpinBox->setValue(value);
    }
}

void TrainingSyntheticDataDialog::syncSamplesFromSpin(int value)
{
    if (ui->samplesSlider->value() != value)
    {
        ui->samplesSlider->setValue(value);
    }
}

void TrainingSyntheticDataDialog::syncDiversityFromSlider(int value)
{
    const double ratio = static_cast<double>(value) / 100.0;
    if (!qFuzzyCompare(ui->diversitySpinBox->value(), ratio))
    {
        ui->diversitySpinBox->setValue(ratio);
    }
}

void TrainingSyntheticDataDialog::syncDiversityFromSpin(double value)
{
    const int slider = static_cast<int>(std::round(value * 100.0));
    if (ui->diversitySlider->value() != slider)
    {
        ui->diversitySlider->setValue(slider);
    }
}

void TrainingSyntheticDataDialog::syncSlopeFromSlider(int value)
{
    const double ratio = static_cast<double>(value) / 100.0;
    if (!qFuzzyCompare(ui->slopeSpinBox->value(), ratio))
    {
        ui->slopeSpinBox->setValue(ratio);
    }
}

void TrainingSyntheticDataDialog::syncSlopeFromSpin(double value)
{
    const int slider = static_cast<int>(std::round(value * 100.0));
    if (ui->slopeSlider->value() != slider)
    {
        ui->slopeSlider->setValue(slider);
    }
}

void TrainingSyntheticDataDialog::browseDirectory()
{
    const QString selected = QFileDialog::getExistingDirectory(this, tr("Select Output Directory"), outputDirectory());
    if (!selected.isEmpty())
    {
        ui->directoryLineEdit->setText(selected);
    }
}

void TrainingSyntheticDataDialog::updateDirectoryPlaceholder(const QString& root)
{
    ui->directoryLineEdit->setPlaceholderText(joinPath(root, defaultLabel()));
}
