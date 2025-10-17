#pragma once

#include "common/ToolLibrary.h"
#include "common/Units.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"
#include "render/SimulationController.h"

#include <QtCore/QElapsedTimer>

#include <memory>

class QProgressDialog;
class QTabWidget;

class QAction;
class QActionGroup;
class QComboBox;
class QListWidget;
class QPlainTextEdit;
class QDockWidget;
class QToolBar;
class QSlider;
class QLabel;
class QStatusBar;
class QGroupBox;
class QDoubleSpinBox;
class QLineEdit;
class QVBoxLayout;

class AiPreferencesDialog;

namespace ai
{
class IPathAI;
class ModelManager;
struct StrategyDecision;
}

namespace io
{
class ImportWorker;
}

namespace tp
{
class GenerateWorker;
}

#include <QtCore/QString>
#include <QtWidgets/QMainWindow>

namespace render
{
class ModelViewerWidget;
class Model;
class SimulationController;
}

namespace app
{

class ToolpathSettingsWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private:
    void createMenus();
    void createDockWidgets();
    void createUnitMenu(QMenu* viewMenu);
    void createMachineMenu(QMenu* menuBar);
    void createSimulationToolbar();
    void loadToolLibrary();
    void refreshAiModels();
    bool setActiveAiModel(const QString& path, bool quiet = false);
    void openAiModelDialog();
    void onAiComboChanged(int index);
    void createCameraToolbar();
    void createStatusBar();
    void applyDarkTheme();
    void updateStatusBarTheme();
    void startImportWorker(const QString& path);
    void startGenerateWorker(const tp::UserParams& settings);
    void cleanupImport();
    void cleanupGeneration();
    void loadSettings();
    void saveSettings() const;
    void openAiPreferences();
    void applyUnits(common::Unit unit, bool fromSettings = false);
    void updateActiveAiSummary();
    void onSimulationProgressChanged(double normalized);
    void onSimulationStateChanged(render::SimulationController::State state);
    void applySimulationSlider(double normalized);
    void setupStockMachineControls(QWidget* container, QVBoxLayout* layout);
    void syncStockUiFromData();
    void syncMachineUiFromData();
    void syncStockDataFromUi();
    void syncMachineDataFromUi();
    void updateStockDerivedFromModel();
    void updateStockUiEnabledState();
    void applyMachinePreset(const tp::Machine& machine);
    double lengthDisplayFromMm(double valueMm) const;
    double lengthMmFromDisplay(double value) const;
    double feedDisplayFromMmPerMin(double valueMmPerMin) const;
    double feedMmPerMinFromDisplay(double value) const;

    void openModelFromFile();
    void saveToolpathToFile();
    void resetCamera();
    void showAboutDialog();
    void selectModelWithAI();

    void onToolpathRequested(const tp::UserParams& settings);
    void onToolSelected(const QString& toolId);
    void updateModelBrowser();
    void logMessage(const QString& text);
    void displayToolpathMessage(const QString& text);
    void logWarning(const QString& text);
    void updateAiPreferencesDialog(AiPreferencesDialog& dialog);
    void handleAiTestRequest(AiPreferencesDialog& dialog);
    void applyAiOverrides(ai::IPathAI* ai) const;
    bool ensureActiveAiPrototype();
    void maybeRunFirstRunTour();
    void updateStatusAiLabel() const;
    void onRendererInfoChanged(const QString& vendor, const QString& renderer, const QString& version);
    void onFrameStatsUpdated(float fps);

    render::ModelViewerWidget* m_viewer{nullptr};
    QListWidget* m_modelBrowser{nullptr};
    QPlainTextEdit* m_console{nullptr};
    QTabWidget* m_consoleTabs{nullptr};
    QPlainTextEdit* m_gcodePreview{nullptr};
    ToolpathSettingsWidget* m_toolpathSettings{nullptr};
    QLabel* m_aiDeviceLabel{nullptr};
    QToolBar* m_simulationToolbar{nullptr};
    QAction* m_simPlayAction{nullptr};
    QAction* m_simPauseAction{nullptr};
    QAction* m_simStopAction{nullptr};
    QSlider* m_simProgressSlider{nullptr};
    QSlider* m_simSpeedSlider{nullptr};
    QLabel* m_simSpeedLabel{nullptr};
    QToolBar* m_cameraToolbar{nullptr};
    bool m_simSliderPressed{false};
    bool m_updatingSimSlider{false};

    enum class StockOriginMode
    {
        ModelMin,
        ModelCenter,
        Custom
    };

    tp::Stock m_stock{};
    tp::Machine m_machine{};
    StockOriginMode m_stockOriginMode{StockOriginMode::ModelMin};
    bool m_blockStockSignals{false};
    bool m_blockMachineSignals{false};

    QGroupBox* m_stockMachineGroup{nullptr};
    QDoubleSpinBox* m_stockWidth{nullptr};
    QDoubleSpinBox* m_stockLength{nullptr};
    QDoubleSpinBox* m_stockHeight{nullptr};
    QDoubleSpinBox* m_stockMargin{nullptr};
    QComboBox* m_stockOriginCombo{nullptr};
    QDoubleSpinBox* m_stockOriginX{nullptr};
    QDoubleSpinBox* m_stockOriginY{nullptr};
    QDoubleSpinBox* m_stockOriginZ{nullptr};
    QLineEdit* m_machineNameEdit{nullptr};
    QDoubleSpinBox* m_machineRapidFeed{nullptr};
    QDoubleSpinBox* m_machineMaxFeed{nullptr};
    QDoubleSpinBox* m_machineMaxSpindle{nullptr};
    QDoubleSpinBox* m_machineClearanceZ{nullptr};
    QDoubleSpinBox* m_machineSafeZ{nullptr};

    tp::ToolpathGenerator m_generator;
    common::ToolLibrary m_toolLibrary;
    std::shared_ptr<render::Model> m_currentModel;
    std::shared_ptr<tp::Toolpath> m_currentToolpath;
    QString m_currentModelPath;
    QString m_lastModelDirectory;
    QString m_aiModelPath;
    QString m_aiModelLabel;
    common::Unit m_units{common::Unit::Millimeters};
    QActionGroup* m_unitsGroup{nullptr};
    QAction* m_unitsMmAction{nullptr};
    QAction* m_unitsInAction{nullptr};
    QComboBox* m_aiModelCombo{nullptr};
    bool m_loadingSettings{false};
    std::unique_ptr<ai::ModelManager> m_modelManager;
    io::ImportWorker* m_importWorker{nullptr};
    tp::GenerateWorker* m_generateWorker{nullptr};
    QProgressDialog* m_importProgress{nullptr};
    QProgressDialog* m_generateProgress{nullptr};
    QElapsedTimer m_importTimer;
    QElapsedTimer m_generateTimer;
    std::unique_ptr<ai::IPathAI> m_activeAiPrototype;
    bool m_forceCpuInference{false};
    std::unique_ptr<render::SimulationController> m_simulation;
    tp::UserParams m_lastUserParams{};
    QLabel* m_statusGpuLabel{nullptr};
    QLabel* m_statusAiLabel{nullptr};
    QLabel* m_statusFpsLabel{nullptr};
    QString m_rendererVendor;
    QString m_rendererName;
    QString m_rendererVersion;
    float m_lastFps{0.0f};

protected:
    void closeEvent(QCloseEvent* event) override;
};

} // namespace app
