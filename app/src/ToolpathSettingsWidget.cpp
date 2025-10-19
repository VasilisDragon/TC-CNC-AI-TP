#include "app/ToolpathSettingsWidget.h"

#include <QtCore/QSignalBlocker>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QVBoxLayout>

#include <algorithm>

namespace app
{

namespace
{
constexpr double kMinDiameterMm = 0.01;
constexpr double kMaxDiameterMm = 100.0;
constexpr double kMinStepMm = 0.001;
constexpr double kMaxStepMm = 200.0;
constexpr double kMinDepthMm = 0.001;
constexpr double kMaxDepthMm = 100.0;
constexpr double kMinFeedMm = 1.0;
constexpr double kMaxFeedMm = 80'000.0;
constexpr double kMinSpindle = 100.0;
constexpr double kMaxSpindle = 80'000.0;

tp::UserParams::CutterType cutterTypeFromTool(const common::Tool& tool)
{
    const QString typeLower = tool.type.toLower();
    if (typeLower.contains(QStringLiteral("ball")))
    {
        return tp::UserParams::CutterType::BallNose;
    }
    return tp::UserParams::CutterType::FlatEndmill;
}
}

ToolpathSettingsWidget::ToolpathSettingsWidget(QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(6, 6, 6, 6);
    layout->setSpacing(6);

    auto* toolLabel = new QLabel(tr("Tool"), this);
    toolLabel->setContentsMargins(0, 4, 0, 0);
    layout->addWidget(toolLabel);

    m_toolCombo = new QComboBox(this);
    m_toolCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    layout->addWidget(m_toolCombo);
    toolLabel->setBuddy(m_toolCombo);

    auto* form = new QFormLayout();
    form->setContentsMargins(0, 0, 0, 0);
    form->setSpacing(4);

    m_toolDiameter = new QDoubleSpinBox(this);
    m_toolDiameter->setDecimals(3);
    m_toolDiameter->setToolTip(tr("Cutter diameter."));

    m_stepOver = new QDoubleSpinBox(this);
   m_stepOver->setDecimals(3);
   m_stepOver->setToolTip(tr("Raster spacing between passes (suggested ~40% of diameter)."));

    m_leaveStock = new QDoubleSpinBox(this);
    m_leaveStock->setDecimals(3);
    m_leaveStock->setToolTip(tr("Minimum stock to leave after waterline adjustment."));

    m_maxDepth = new QDoubleSpinBox(this);
    m_maxDepth->setDecimals(3);
    m_maxDepth->setToolTip(tr("Maximum depth per pass."));

    m_feedRate = new QDoubleSpinBox(this);
    m_feedRate->setDecimals(2);
    m_feedRate->setToolTip(tr("Cutting feed rate."));

    m_spindle = new QDoubleSpinBox(this);
    m_spindle->setDecimals(0);
    m_spindle->setToolTip(tr("Spindle speed."));
    m_spindle->setRange(kMinSpindle, kMaxSpindle);
    m_spindle->setValue(12'000.0);
    m_spindle->setSuffix(QStringLiteral(" RPM"));

    form->addRow(tr("Tool Diameter"), m_toolDiameter);
    form->addRow(tr("Step-over"), m_stepOver);
    form->addRow(tr("Leave Stock"), m_leaveStock);
    form->addRow(tr("Max Depth/Pass"), m_maxDepth);
    form->addRow(tr("Feed Rate"), m_feedRate);
    form->addRow(tr("Spindle RPM"), m_spindle);

    m_rasterAngle = new QDoubleSpinBox(this);
    m_rasterAngle->setDecimals(1);
    m_rasterAngle->setRange(-180.0, 180.0);
    m_rasterAngle->setSingleStep(5.0);
    m_rasterAngle->setToolTip(tr("Raster scan angle in degrees."));
    form->addRow(tr("Raster Angle"), m_rasterAngle);

    m_useHeightField = new QCheckBox(tr("Use HeightField (software raster)"), this);
    m_useHeightField->setChecked(true);
    m_useHeightField->setToolTip(tr("Build a sampled height field when OpenCL acceleration is unavailable."));
    form->addRow(tr("HeightField"), m_useHeightField);

    m_paramsMm.leaveStock_mm = m_paramsMm.stockAllowance_mm;

    layout->addLayout(form);
    layout->addStretch(1);

    m_generateButton = new QPushButton(tr("Generate Toolpath"), this);
    layout->addWidget(m_generateButton);

    connect(m_toolCombo, &QComboBox::currentIndexChanged, this, [this](int index) {
        if (index < 0 || index >= m_tools.size())
        {
            return;
        }
        if (m_blockToolDefaults)
        {
            emit toolChanged(m_tools.at(index).id);
            return;
        }

        const common::Tool& tool = m_tools.at(index);
        if (!tool.isValid())
        {
            emit warningGenerated(tr("Tool entry \"%1\" is invalid.").arg(tool.name.isEmpty() ? tool.id : tool.name));
            return;
        }

        applyToolDefaults(tool);
        emit toolChanged(tool.id);
    });

    const auto boxes = {m_toolDiameter, m_stepOver, m_maxDepth, m_feedRate, m_spindle};
    for (QDoubleSpinBox* box : boxes)
    {
        connect(box, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
            syncParamsFromWidgets();
            validateInputs();
        });
        connect(box, &QDoubleSpinBox::editingFinished, this, [this]() {
            syncParamsFromWidgets();
            validateInputs();
        });
    }

    connect(m_rasterAngle, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
        syncParamsFromWidgets();
        validateInputs();
    });

    connect(m_useHeightField, &QCheckBox::toggled, this, [this](bool) {
        syncParamsFromWidgets();
        validateInputs();
    });

    connect(m_generateButton, &QPushButton::clicked, this, &ToolpathSettingsWidget::emitGenerate);

    m_paramsMm.toolDiameter = 6.0;
    m_paramsMm.stepOver = 3.0;
    m_paramsMm.maxDepthPerPass = 1.0;
    m_paramsMm.feed = 800.0;
    m_paramsMm.spindle = 12'000.0;
    m_paramsMm.rasterAngleDeg = 0.0;
    m_paramsMm.useHeightField = true;
    m_paramsMm.cutterType = tp::UserParams::CutterType::FlatEndmill;

    updateRanges();
    applyUnitsToWidgets();
    syncWidgetsFromParams();
    validateInputs();
}

void ToolpathSettingsWidget::setUnits(common::Unit unit)
{
    if (unit == m_unit)
    {
        return;
    }

    syncParamsFromWidgets();
    m_unit = unit;
    applyUnitsToWidgets();
    updateRanges();
    refreshToolComboLabels();
    syncWidgetsFromParams();
}

void ToolpathSettingsWidget::setToolLibrary(const QVector<common::Tool>& tools, common::Unit unit)
{
    m_tools = tools;
    m_unit = unit;

    QSignalBlocker blocker(m_toolCombo);
    m_toolCombo->clear();
    for (const common::Tool& tool : m_tools)
    {
        m_toolCombo->addItem(tool.displayLabel(m_unit), tool.id);
    }

    m_blockToolDefaults = true;
    if (!m_tools.isEmpty())
    {
        m_toolCombo->setCurrentIndex(0);
    }
    m_blockToolDefaults = false;

    refreshToolComboLabels();

    if (!m_tools.isEmpty())
    {
        applyToolDefaults(m_tools.first());
    }
}

void ToolpathSettingsWidget::setSelectedToolId(const QString& id, bool applyDefaults)
{
    const int index = m_toolCombo->findData(id);
    if (index < 0)
    {
        return;
    }

    m_blockToolDefaults = !applyDefaults;
    {
        QSignalBlocker blocker(m_toolCombo);
        m_toolCombo->setCurrentIndex(index);
    }
    m_blockToolDefaults = false;

    if (applyDefaults && index < m_tools.size())
    {
        applyToolDefaults(m_tools.at(index));
    }

    if (index < m_tools.size())
    {
        m_paramsMm.cutterType = cutterTypeFromTool(m_tools.at(index));
        emit toolChanged(m_tools.at(index).id);
    }
}

QString ToolpathSettingsWidget::currentToolId() const
{
    return m_toolCombo->currentData().toString();
}

void ToolpathSettingsWidget::setParameters(const tp::UserParams& paramsMm)
{
    m_paramsMm = paramsMm;
    syncWidgetsFromParams();
    validateInputs();
}

tp::UserParams ToolpathSettingsWidget::currentParameters() const
{
    return m_paramsMm;
}

tp::UserParams ToolpathSettingsWidget::gatherSettings() const
{
    return m_paramsMm;
}

void ToolpathSettingsWidget::emitGenerate()
{
    syncParamsFromWidgets();
    if (!validateInputs())
    {
        return;
    }

    emit generateRequested(m_paramsMm);
}

bool ToolpathSettingsWidget::validateInputs()
{
    bool allValid = true;

    const auto check = [this, &allValid](QDoubleSpinBox* box) {
        const bool valid = box && box->value() > 0.0;
        applyValidity(box, valid);
        allValid = allValid && valid;
    };

    check(m_toolDiameter);
    check(m_stepOver);
    check(m_leaveStock);
    check(m_maxDepth);
    check(m_feedRate);
    check(m_spindle);

    if (m_generateButton)
    {
        m_generateButton->setEnabled(allValid);
    }

    return allValid;
}

void ToolpathSettingsWidget::applyValidity(QDoubleSpinBox* box, bool valid)
{
    if (!box)
    {
        return;
    }

    if (valid)
    {
        box->setStyleSheet(QString());
    }
    else
    {
        box->setStyleSheet(QStringLiteral("QDoubleSpinBox { border: 1px solid #C62828; }"));
    }
}

void ToolpathSettingsWidget::applyUnitsToWidgets()
{
    const QString lengthSuffix = QStringLiteral(" ") + common::unitSuffix(m_unit);
    const QString feedSuffix = QStringLiteral(" ") + common::feedSuffix(m_unit);

    m_toolDiameter->setSuffix(lengthSuffix);
    m_stepOver->setSuffix(lengthSuffix);
    m_leaveStock->setSuffix(lengthSuffix);
    m_maxDepth->setSuffix(lengthSuffix);
    m_feedRate->setSuffix(feedSuffix);
    if (m_rasterAngle)
    {
        m_rasterAngle->setSuffix(QStringLiteral(" °"));
    }
}

void ToolpathSettingsWidget::updateRanges()
{
    m_toolDiameter->setRange(displayFromMm(kMinDiameterMm), displayFromMm(kMaxDiameterMm));
    m_toolDiameter->setSingleStep(displayFromMm(0.1));

    m_stepOver->setRange(displayFromMm(kMinStepMm), displayFromMm(kMaxStepMm));
    m_stepOver->setSingleStep(displayFromMm(0.1));

    m_leaveStock->setRange(0.0, displayFromMm(20.0));
    m_leaveStock->setSingleStep(displayFromMm(0.05));

    m_maxDepth->setRange(displayFromMm(kMinDepthMm), displayFromMm(kMaxDepthMm));
    m_maxDepth->setSingleStep(displayFromMm(0.1));

    m_feedRate->setRange(displayFromMm(kMinFeedMm), displayFromMm(kMaxFeedMm));
    m_feedRate->setSingleStep(displayFromMm(10.0));
    if (m_rasterAngle)
    {
        m_rasterAngle->setRange(-180.0, 180.0);
        m_rasterAngle->setSingleStep(5.0);
    }
}

void ToolpathSettingsWidget::syncWidgetsFromParams()
{
    QSignalBlocker blocker1(m_toolDiameter);
    QSignalBlocker blocker2(m_stepOver);
    QSignalBlocker blocker3(m_maxDepth);
    QSignalBlocker blocker4(m_feedRate);
    QSignalBlocker blocker5(m_spindle);
    QSignalBlocker blocker6(m_rasterAngle);
    QSignalBlocker blocker7(m_useHeightField);
    QSignalBlocker blocker8(m_leaveStock);

    m_toolDiameter->setValue(displayFromMm(m_paramsMm.toolDiameter));
    m_stepOver->setValue(displayFromMm(m_paramsMm.stepOver));
    m_leaveStock->setValue(displayFromMm(m_paramsMm.leaveStock_mm));
    m_maxDepth->setValue(displayFromMm(m_paramsMm.maxDepthPerPass));
    m_feedRate->setValue(displayFromMm(m_paramsMm.feed));
    m_spindle->setValue(m_paramsMm.spindle);
    if (m_rasterAngle)
    {
        m_rasterAngle->setValue(m_paramsMm.rasterAngleDeg);
    }
    if (m_useHeightField)
    {
        m_useHeightField->setChecked(m_paramsMm.useHeightField);
    }
}

void ToolpathSettingsWidget::syncParamsFromWidgets()
{
    m_paramsMm.toolDiameter = mmFromDisplay(m_toolDiameter->value());
    m_paramsMm.stepOver = mmFromDisplay(m_stepOver->value());
    m_paramsMm.leaveStock_mm = mmFromDisplay(m_leaveStock->value());
    m_paramsMm.maxDepthPerPass = mmFromDisplay(m_maxDepth->value());
    m_paramsMm.feed = mmFromDisplay(m_feedRate->value());
    m_paramsMm.spindle = m_spindle->value();
    m_paramsMm.stockAllowance_mm = m_paramsMm.leaveStock_mm;
    if (m_rasterAngle)
    {
        m_paramsMm.rasterAngleDeg = m_rasterAngle->value();
    }
    if (m_useHeightField)
    {
        m_paramsMm.useHeightField = m_useHeightField->isChecked();
    }
}

void ToolpathSettingsWidget::applyToolDefaults(const common::Tool& tool)
{
    m_paramsMm.toolDiameter = tool.diameterMm;
    m_paramsMm.stepOver = tool.recommendedStepOverMm();
    m_paramsMm.maxDepthPerPass = tool.recommendedMaxDepthMm();
    m_paramsMm.leaveStock_mm = std::clamp(tool.recommendedStepOverMm() * 0.25, 0.0, tool.diameterMm * 0.5);
    m_paramsMm.stockAllowance_mm = m_paramsMm.leaveStock_mm;
    m_paramsMm.cutterType = cutterTypeFromTool(tool);
    syncWidgetsFromParams();
    validateInputs();
}

void ToolpathSettingsWidget::refreshToolComboLabels()
{
    QSignalBlocker blocker(m_toolCombo);
    for (int i = 0; i < m_tools.size(); ++i)
    {
        m_toolCombo->setItemText(i, m_tools.at(i).displayLabel(m_unit));
    }
}

double ToolpathSettingsWidget::displayFromMm(double valueMm) const
{
    return common::fromMillimeters(valueMm, m_unit);
}

double ToolpathSettingsWidget::mmFromDisplay(double value) const
{
    return common::toMillimeters(value, m_unit);
}

} // namespace app
