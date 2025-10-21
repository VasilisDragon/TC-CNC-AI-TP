#include "app/ToolpathSettingsWidget.h"

#include <QtCore/QSignalBlocker>
#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QVBoxLayout>

#include <algorithm>
#include <vector>

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

std::vector<ai::StrategyStep> defaultStrategySteps()
{
    ai::StrategyStep rough;
    rough.type = ai::StrategyStep::Type::Raster;
    rough.stepover = 3.0;
    rough.stepdown = 1.0;
    rough.angle_deg = 45.0;
    rough.finish_pass = false;

    ai::StrategyStep finish = rough;
    finish.finish_pass = true;
    finish.stepdown = 0.5;
    return {rough, finish};
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

    m_enableRamp = new QCheckBox(tr("Ramp entries"), this);
    m_enableRamp->setToolTip(tr("When enabled, replace plunges with linear ramps using the angle below."));
    form->addRow(tr("Ramp Entry"), m_enableRamp);

    m_rampAngle = new QDoubleSpinBox(this);
    m_rampAngle->setDecimals(1);
    m_rampAngle->setRange(0.5, 45.0);
    m_rampAngle->setSingleStep(0.5);
    m_rampAngle->setToolTip(tr("Ramp angle in degrees (smaller angles ramp further away but reduce chip load spikes)."));
    form->addRow(tr("Ramp Angle"), m_rampAngle);

    m_enableHelical = new QCheckBox(tr("Helical entries"), this);
    m_enableHelical->setToolTip(tr("Attempt to generate a segmented helix when stepdown is required."));
    form->addRow(tr("Helical Entry"), m_enableHelical);

    m_rampRadius = new QDoubleSpinBox(this);
    m_rampRadius->setDecimals(2);
    m_rampRadius->setToolTip(tr("Approximate helix radius measured in the cutting plane."));
    form->addRow(tr("Helix Radius"), m_rampRadius);

    m_leadIn = new QDoubleSpinBox(this);
    m_leadIn->setDecimals(2);
    m_leadIn->setToolTip(tr("Length of the tangent lead-in segment before each cutting move."));
    form->addRow(tr("Lead-in Length"), m_leadIn);

    m_leadOut = new QDoubleSpinBox(this);
    m_leadOut->setDecimals(2);
    m_leadOut->setToolTip(tr("Length of the tangent lead-out segment after each cutting move."));
    form->addRow(tr("Lead-out Length"), m_leadOut);

    m_cutDirection = new QComboBox(this);
    m_cutDirection->addItem(tr("Climb"), QVariant::fromValue(static_cast<int>(tp::UserParams::CutDirection::Climb)));
    m_cutDirection->addItem(tr("Conventional"), QVariant::fromValue(static_cast<int>(tp::UserParams::CutDirection::Conventional)));
    m_cutDirection->setToolTip(tr("Desired chip load orientation. Climb is generally preferred for CNC routers."));
    form->addRow(tr("Cut Direction"), m_cutDirection);

    m_paramsMm.leaveStock_mm = m_paramsMm.stockAllowance_mm;

    layout->addLayout(form);

    auto* strategyLabel = new QLabel(tr("Strategy Steps"), this);
    strategyLabel->setContentsMargins(0, 8, 0, 0);
    layout->addWidget(strategyLabel);

    m_strategyOverrideCheck = new QCheckBox(tr("Override AI plan"), this);
    layout->addWidget(m_strategyOverrideCheck);

    m_strategyTable = new QTableWidget(this);
    m_strategyTable->setColumnCount(5);
    m_strategyTable->setHorizontalHeaderLabels(
        {tr("Type"), tr("Step-over (mm)"), tr("Step-down (mm)"), tr("Angle (deg)"), tr("Finish")});
    m_strategyTable->horizontalHeader()->setStretchLastSection(true);
    m_strategyTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_strategyTable->setSelectionMode(QAbstractItemView::SingleSelection);
    m_strategyTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_strategyTable->setEditTriggers(QAbstractItemView::NoEditTriggers);
    layout->addWidget(m_strategyTable);

    auto* strategyButtons = new QHBoxLayout();
    strategyButtons->setContentsMargins(0, 0, 0, 0);
    strategyButtons->setSpacing(6);
    m_addStrategyButton = new QPushButton(tr("Add Step"), this);
    m_removeStrategyButton = new QPushButton(tr("Remove Step"), this);
    strategyButtons->addWidget(m_addStrategyButton);
    strategyButtons->addWidget(m_removeStrategyButton);
    strategyButtons->addStretch(1);
    layout->addLayout(strategyButtons);

    layout->addSpacing(6);
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

    const auto boxes = {m_toolDiameter,
                        m_stepOver,
                        m_leaveStock,
                        m_maxDepth,
                        m_feedRate,
                        m_spindle,
                        m_rampAngle,
                        m_rampRadius,
                        m_leadIn,
                        m_leadOut};
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

    connect(m_enableRamp, &QCheckBox::toggled, this, [this](bool checked) {
        if (m_rampAngle)
        {
            m_rampAngle->setEnabled(checked);
        }
        syncParamsFromWidgets();
        validateInputs();
    });

    connect(m_enableHelical, &QCheckBox::toggled, this, [this](bool checked) {
        if (m_rampRadius)
        {
            m_rampRadius->setEnabled(checked);
        }
        syncParamsFromWidgets();
        validateInputs();
    });

    connect(m_cutDirection, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
        syncParamsFromWidgets();
    });

    connect(m_strategyOverrideCheck, &QCheckBox::toggled, this, [this](bool) {
        if (m_blockStrategySignals)
        {
            return;
        }
        refreshStrategyTable();
        syncStrategyParams();
    });

    connect(m_addStrategyButton, &QPushButton::clicked, this, &ToolpathSettingsWidget::appendStrategyStep);
    connect(m_removeStrategyButton, &QPushButton::clicked, this, &ToolpathSettingsWidget::removeStrategyStep);
    connect(m_strategyTable->selectionModel(),
            &QItemSelectionModel::selectionChanged,
            this,
            [this](const QItemSelection&, const QItemSelection&) {
                const bool overrideEnabled = m_strategyOverrideCheck && m_strategyOverrideCheck->isChecked();
                if (m_removeStrategyButton)
                {
                    m_removeStrategyButton->setEnabled(overrideEnabled && m_strategyTable->currentRow() >= 0);
                }
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
    m_paramsMm.enableRamp = true;
    m_paramsMm.rampAngleDeg = 3.0;
    m_paramsMm.enableHelical = false;
    m_paramsMm.rampRadius = 3.0;
    m_paramsMm.leadInLength = 0.0;
    m_paramsMm.leadOutLength = 0.0;
    m_paramsMm.cutDirection = tp::UserParams::CutDirection::Climb;

    m_aiPreviewSteps = defaultStrategySteps();
    m_strategySteps = m_aiPreviewSteps;
    if (m_strategyOverrideCheck)
    {
        m_strategyOverrideCheck->setChecked(false);
    }
    refreshStrategyTable();
    syncStrategyParams();

    updateRanges();
    applyUnitsToWidgets();
    syncWidgetsFromParams();
    validateInputs();
}

void ToolpathSettingsWidget::setUnits(common::UnitSystem unit)
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

void ToolpathSettingsWidget::setToolLibrary(const QVector<common::Tool>& tools, common::UnitSystem unit)
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

    const auto validateBox = [this, &allValid](QDoubleSpinBox* box, double minValue, bool inclusive) {
        if (!box)
        {
            return;
        }
        const double value = box->value();
        const bool valid = inclusive ? (value >= minValue) : (value > minValue);
        applyValidity(box, valid);
        allValid = allValid && valid;
    };

    validateBox(m_toolDiameter, 0.0, false);
    validateBox(m_stepOver, 0.0, false);
    validateBox(m_leaveStock, 0.0, false);
    validateBox(m_maxDepth, 0.0, false);
    validateBox(m_feedRate, 0.0, false);
    validateBox(m_spindle, 0.0, false);

    if (m_enableRamp && m_enableRamp->isChecked())
    {
        validateBox(m_rampAngle, 0.0, false);
    }
    else
    {
        applyValidity(m_rampAngle, true);
    }

    if (m_enableHelical && m_enableHelical->isChecked())
    {
        validateBox(m_rampRadius, 0.0, false);
    }
    else
    {
        applyValidity(m_rampRadius, true);
    }

    validateBox(m_leadIn, 0.0, true);
    validateBox(m_leadOut, 0.0, true);

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

    const auto setLengthSuffix = [&](QDoubleSpinBox* box) {
        if (box)
        {
            box->setSuffix(lengthSuffix);
        }
    };

    setLengthSuffix(m_toolDiameter);
    setLengthSuffix(m_stepOver);
    setLengthSuffix(m_leaveStock);
    setLengthSuffix(m_maxDepth);
    if (m_feedRate)
    {
        m_feedRate->setSuffix(feedSuffix);
    }
    setLengthSuffix(m_rampRadius);
    setLengthSuffix(m_leadIn);
    setLengthSuffix(m_leadOut);

    if (m_rampAngle)
    {
        m_rampAngle->setSuffix(QStringLiteral(" \u00B0"));
    }
    if (m_rasterAngle)
    {
        m_rasterAngle->setSuffix(QStringLiteral(" \u00B0"));
    }
}

void ToolpathSettingsWidget::updateRanges()
{
    if (m_toolDiameter)
    {
        m_toolDiameter->setRange(displayFromMm(kMinDiameterMm), displayFromMm(kMaxDiameterMm));
        m_toolDiameter->setSingleStep(displayFromMm(0.1));
    }

    if (m_stepOver)
    {
        m_stepOver->setRange(displayFromMm(kMinStepMm), displayFromMm(kMaxStepMm));
        m_stepOver->setSingleStep(displayFromMm(0.1));
    }

    if (m_leaveStock)
    {
        m_leaveStock->setRange(0.0, displayFromMm(20.0));
        m_leaveStock->setSingleStep(displayFromMm(0.05));
    }

    if (m_maxDepth)
    {
        m_maxDepth->setRange(displayFromMm(kMinDepthMm), displayFromMm(kMaxDepthMm));
        m_maxDepth->setSingleStep(displayFromMm(0.1));
    }

    if (m_feedRate)
    {
        m_feedRate->setRange(displayFromMm(kMinFeedMm), displayFromMm(kMaxFeedMm));
        m_feedRate->setSingleStep(displayFromMm(10.0));
    }

    if (m_rasterAngle)
    {
        m_rasterAngle->setRange(-180.0, 180.0);
        m_rasterAngle->setSingleStep(5.0);
    }

    if (m_rampAngle)
    {
        m_rampAngle->setRange(0.5, 45.0);
        m_rampAngle->setSingleStep(0.5);
    }

    if (m_rampRadius)
    {
        m_rampRadius->setRange(displayFromMm(0.1), displayFromMm(200.0));
        m_rampRadius->setSingleStep(displayFromMm(1.0));
    }

    if (m_leadIn)
    {
        m_leadIn->setRange(0.0, displayFromMm(200.0));
        m_leadIn->setSingleStep(displayFromMm(0.5));
    }

    if (m_leadOut)
    {
        m_leadOut->setRange(0.0, displayFromMm(200.0));
        m_leadOut->setSingleStep(displayFromMm(0.5));
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
    QSignalBlocker blocker9(m_enableRamp);
    QSignalBlocker blocker10(m_rampAngle);
    QSignalBlocker blocker11(m_enableHelical);
    QSignalBlocker blocker12(m_rampRadius);
    QSignalBlocker blocker13(m_leadIn);
    QSignalBlocker blocker14(m_leadOut);
    QSignalBlocker blocker15(m_cutDirection);

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
    if (m_enableRamp)
    {
        m_enableRamp->setChecked(m_paramsMm.enableRamp);
    }
    if (m_rampAngle)
    {
        m_rampAngle->setValue(m_paramsMm.rampAngleDeg);
        m_rampAngle->setEnabled(m_paramsMm.enableRamp);
    }
    if (m_enableHelical)
    {
        m_enableHelical->setChecked(m_paramsMm.enableHelical);
    }
    if (m_rampRadius)
    {
        m_rampRadius->setValue(displayFromMm(m_paramsMm.rampRadius));
        m_rampRadius->setEnabled(m_paramsMm.enableHelical);
    }
    if (m_leadIn)
    {
        m_leadIn->setValue(displayFromMm(m_paramsMm.leadInLength));
    }
    if (m_leadOut)
    {
        m_leadOut->setValue(displayFromMm(m_paramsMm.leadOutLength));
    }
    if (m_cutDirection)
    {
        const int value = static_cast<int>(m_paramsMm.cutDirection);
        const int index = m_cutDirection->findData(value);
        if (index >= 0)
        {
            m_cutDirection->setCurrentIndex(index);
        }
    }
    setStrategyOverride(m_paramsMm.strategyOverride, m_paramsMm.useStrategyOverride);
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
    if (m_enableRamp)
    {
        m_paramsMm.enableRamp = m_enableRamp->isChecked();
    }
    if (m_rampAngle)
    {
        m_paramsMm.rampAngleDeg = m_rampAngle->value();
    }
    if (m_enableHelical)
    {
        m_paramsMm.enableHelical = m_enableHelical->isChecked();
    }
    if (m_rampRadius)
    {
        m_paramsMm.rampRadius = mmFromDisplay(m_rampRadius->value());
    }
    if (m_leadIn)
    {
        m_paramsMm.leadInLength = mmFromDisplay(m_leadIn->value());
    }
    if (m_leadOut)
    {
        m_paramsMm.leadOutLength = mmFromDisplay(m_leadOut->value());
    }
    if (m_cutDirection)
    {
        const int value = m_cutDirection->currentData().toInt();
        m_paramsMm.cutDirection = static_cast<tp::UserParams::CutDirection>(value);
    }
    syncStrategyParams();
}

void ToolpathSettingsWidget::setStrategyOverride(const std::vector<ai::StrategyStep>& steps, bool enabled)
{
    m_strategySteps = steps.empty() ? defaultStrategySteps() : steps;
    if (m_strategyOverrideCheck)
    {
        QSignalBlocker blocker(m_strategyOverrideCheck);
        m_strategyOverrideCheck->setChecked(enabled);
    }
    refreshStrategyTable();
}

void ToolpathSettingsWidget::setAiStrategyPreview(const std::vector<ai::StrategyStep>& steps)
{
    m_aiPreviewSteps = steps.empty() ? defaultStrategySteps() : steps;
    if (!m_strategyOverrideCheck || !m_strategyOverrideCheck->isChecked())
    {
        refreshStrategyTable();
    }
}

void ToolpathSettingsWidget::refreshStrategyTable()
{
    if (!m_strategyTable)
    {
        return;
    }

    const bool overrideEnabled = m_strategyOverrideCheck && m_strategyOverrideCheck->isChecked();
    if (overrideEnabled && m_strategySteps.empty())
    {
        m_strategySteps = defaultStrategySteps();
    }
    if (!overrideEnabled && m_aiPreviewSteps.empty())
    {
        m_aiPreviewSteps = defaultStrategySteps();
    }

    const auto& sourceSteps = overrideEnabled ? m_strategySteps : m_aiPreviewSteps;

    m_blockStrategySignals = true;
    m_strategyTable->setRowCount(0);
    m_strategyTable->setRowCount(static_cast<int>(sourceSteps.size()));

    for (int row = 0; row < static_cast<int>(sourceSteps.size()); ++row)
    {
        const ai::StrategyStep& step = sourceSteps[row];

        auto* angleSpin = new QDoubleSpinBox(m_strategyTable);
        angleSpin->setDecimals(1);
        angleSpin->setRange(-180.0, 180.0);
        angleSpin->setSingleStep(5.0);
        angleSpin->setSuffix(QStringLiteral(" deg"));
        angleSpin->setValue(step.angle_deg);

        auto* typeBox = new QComboBox(m_strategyTable);
        typeBox->addItem(tr("Raster"));
        typeBox->addItem(tr("Waterline"));
        typeBox->setCurrentIndex(step.type == ai::StrategyStep::Type::Raster ? 0 : 1);
        typeBox->setEnabled(overrideEnabled);

        auto* stepoverSpin = new QDoubleSpinBox(m_strategyTable);
        stepoverSpin->setDecimals(3);
        stepoverSpin->setRange(0.0, 500.0);
        stepoverSpin->setSingleStep(0.1);
        stepoverSpin->setSuffix(QStringLiteral(" mm"));
        stepoverSpin->setValue(step.stepover);
        stepoverSpin->setEnabled(overrideEnabled);

        auto* stepdownSpin = new QDoubleSpinBox(m_strategyTable);
        stepdownSpin->setDecimals(3);
        stepdownSpin->setRange(0.0, 500.0);
        stepdownSpin->setSingleStep(0.1);
        stepdownSpin->setSuffix(QStringLiteral(" mm"));
        stepdownSpin->setValue(step.stepdown);
        stepdownSpin->setEnabled(overrideEnabled);

        QWidget* finishContainer = new QWidget(m_strategyTable);
        auto* finishLayout = new QHBoxLayout(finishContainer);
        finishLayout->setContentsMargins(0, 0, 0, 0);
        finishLayout->setAlignment(Qt::AlignCenter);
        auto* finishCheck = new QCheckBox(finishContainer);
        finishCheck->setChecked(step.finish_pass);
        finishCheck->setEnabled(overrideEnabled);
        finishLayout->addWidget(finishCheck);

        m_strategyTable->setCellWidget(row, 0, typeBox);
        m_strategyTable->setCellWidget(row, 1, stepoverSpin);
        m_strategyTable->setCellWidget(row, 2, stepdownSpin);
        m_strategyTable->setCellWidget(row, 3, angleSpin);
        m_strategyTable->setCellWidget(row, 4, finishContainer);

        angleSpin->setEnabled(overrideEnabled && step.type == ai::StrategyStep::Type::Raster);

        if (overrideEnabled)
        {
            connect(typeBox, qOverload<int>(&QComboBox::currentIndexChanged), this, [this, row, angleSpin](int index) {
                if (row < 0 || row >= static_cast<int>(m_strategySteps.size()))
                {
                    return;
                }
                m_strategySteps[row].type = (index == 0) ? ai::StrategyStep::Type::Raster
                                                         : ai::StrategyStep::Type::Waterline;
                if (angleSpin)
                {
                    const bool enableAngle = (index == 0);
                    angleSpin->setEnabled(enableAngle);
                    if (!enableAngle)
                    {
                        angleSpin->setValue(0.0);
                        m_strategySteps[row].angle_deg = 0.0;
                    }
                }
                syncStrategyParams();
            });
            connect(stepoverSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, row](double value) {
                if (row < 0 || row >= static_cast<int>(m_strategySteps.size()))
                {
                    return;
                }
                m_strategySteps[row].stepover = value;
                syncStrategyParams();
            });
            connect(stepdownSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, row](double value) {
                if (row < 0 || row >= static_cast<int>(m_strategySteps.size()))
                {
                    return;
                }
                m_strategySteps[row].stepdown = value;
                syncStrategyParams();
            });
            connect(angleSpin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this, row](double value) {
                if (row < 0 || row >= static_cast<int>(m_strategySteps.size()))
                {
                    return;
                }
                m_strategySteps[row].angle_deg = value;
                syncStrategyParams();
            });
            connect(finishCheck, &QCheckBox::toggled, this, [this, row](bool checked) {
                if (row < 0 || row >= static_cast<int>(m_strategySteps.size()))
                {
                    return;
                }
                m_strategySteps[row].finish_pass = checked;
                syncStrategyParams();
            });
        }
    }

    if (m_addStrategyButton)
    {
        m_addStrategyButton->setEnabled(overrideEnabled);
    }
    if (m_removeStrategyButton)
    {
        m_removeStrategyButton->setEnabled(overrideEnabled && m_strategyTable->currentRow() >= 0);
    }

    m_blockStrategySignals = false;
}

void ToolpathSettingsWidget::appendStrategyStep()
{
    if (!m_strategyOverrideCheck)
    {
        return;
    }
    if (!m_strategyOverrideCheck->isChecked())
    {
        m_strategyOverrideCheck->setChecked(true);
    }

    ai::StrategyStep step;
    if (!m_strategySteps.empty())
    {
        step = m_strategySteps.back();
    }
    else
    {
        const auto defaults = defaultStrategySteps();
        step = defaults.front();
    }
    step.finish_pass = false;
    m_strategySteps.push_back(step);
    refreshStrategyTable();
    syncStrategyParams();
}

void ToolpathSettingsWidget::removeStrategyStep()
{
    if (!m_strategyOverrideCheck || !m_strategyOverrideCheck->isChecked())
    {
        return;
    }

    const int row = m_strategyTable ? m_strategyTable->currentRow() : -1;
    if (row < 0 || row >= static_cast<int>(m_strategySteps.size()))
    {
        return;
    }

    m_strategySteps.erase(m_strategySteps.begin() + row);
    refreshStrategyTable();
    syncStrategyParams();
}

void ToolpathSettingsWidget::syncStrategyParams()
{
    if (m_blockStrategySignals || !m_strategyOverrideCheck)
    {
        return;
    }

    m_paramsMm.useStrategyOverride = m_strategyOverrideCheck->isChecked();
    if (m_paramsMm.useStrategyOverride)
    {
        m_paramsMm.strategyOverride = m_strategySteps;
    }
    else
    {
        m_paramsMm.strategyOverride.clear();
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
    m_paramsMm.enableRamp = true;
    m_paramsMm.rampAngleDeg = std::clamp(m_paramsMm.rampAngleDeg, 0.5, 45.0);
    m_paramsMm.enableHelical = false;
    m_paramsMm.rampRadius = std::max(tool.diameterMm * 0.5, 1.0);
    m_paramsMm.leadInLength = 0.0;
    m_paramsMm.leadOutLength = 0.0;
    m_paramsMm.cutDirection = tp::UserParams::CutDirection::Climb;
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



