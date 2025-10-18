#pragma once

#include "common/Tool.h"
#include "common/Units.h"
#include "tp/ToolpathGenerator.h"

#include <QtWidgets/QWidget>

#include <QVector>

class QComboBox;
class QDoubleSpinBox;
class QCheckBox;
class QPushButton;

namespace app
{

class ToolpathSettingsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ToolpathSettingsWidget(QWidget* parent = nullptr);

    void setUnits(common::Unit unit);
    void setToolLibrary(const QVector<common::Tool>& tools, common::Unit unit);
    void setSelectedToolId(const QString& id, bool applyDefaults);
    QString currentToolId() const;

    void setParameters(const tp::UserParams& paramsMm);
    tp::UserParams currentParameters() const;

    Q_SIGNALS:
    void generateRequested(const tp::UserParams& settings);
    void toolChanged(const QString& toolId);
    void warningGenerated(const QString& message);

private:
    tp::UserParams gatherSettings() const;
    void emitGenerate();
    bool validateInputs();
    void applyValidity(QDoubleSpinBox* box, bool valid);
    void applyUnitsToWidgets();
    void updateRanges();
    void syncWidgetsFromParams();
    void syncParamsFromWidgets();
    void applyToolDefaults(const common::Tool& tool);
    void refreshToolComboLabels();
    double displayFromMm(double valueMm) const;
    double mmFromDisplay(double value) const;

    QComboBox* m_toolCombo{nullptr};
    QDoubleSpinBox* m_toolDiameter{nullptr};
    QDoubleSpinBox* m_stepOver{nullptr};
    QDoubleSpinBox* m_maxDepth{nullptr};
    QDoubleSpinBox* m_feedRate{nullptr};
    QDoubleSpinBox* m_spindle{nullptr};
    QDoubleSpinBox* m_rasterAngle{nullptr};
    QCheckBox* m_useHeightField{nullptr};
    QPushButton* m_generateButton{nullptr};

    QVector<common::Tool> m_tools;
    common::Unit m_unit{common::Unit::Millimeters};
    tp::UserParams m_paramsMm{};
    bool m_blockToolDefaults{false};
};

} // namespace app
