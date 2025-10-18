#pragma once

#include "common/ToolLibrary.h"
#include "common/Units.h"
#include "tp/Toolpath.h"
#include "tp/ToolpathGenerator.h"
#include "render/SimulationController.h"
#include "train/TrainingManager.h"

#include <QtCore/QElapsedTimer>
#include <QtCore/QHash>
#include <QtCore/QUuid>

#include <memory>

class QProgressDialog;
class QTabWidget;

class QAction;
class QActionGroup;
class QComboBox;
class QListWidget;
class QListWidgetItem;
class QPlainTextEdit;
class QDockWidget;
class QToolBar;
class QSlider;
class QLabel;
class QStatusBar;
class QGroupBox;
class QDoubleSpinBox;
class QCheckBox;
class QLineEdit;
class QVBoxLayout;
class QProgressBar;
class QPushButton;
class QMenu;
class QWidget;
class QMenu;

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
}

namespace train
{
class EnvManager;
class TrainingManager;
}

namespace app
{

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private:
    struct JobWidgets
    {
        QListWidgetItem* item{nullptr};
        QWidget* container{nullptr};
        QLabel* title{nullptr};
        QLabel* subtitle{nullptr};
        QProgressBar* progress{nullptr};
        QLabel* status{nullptr};
        QLabel* eta{nullptr};
        QPushButton* cancel{nullptr};
        train::TrainingManager::JobStatus snapshot;
    };

    void createMenus();
    void createDockWidgets();
    void createJobsDock(QDockWidget* envDock);
    void createUnitMenu(QMenu* viewMenu);
    void createMachineMenu(QMenu* menuBar);
    void createSimulationToolbar();
    void createTrainingMenu();
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
    void syncPassControlsFromData();
    void updatePassControlUnits();
    void applyPassPreferences(tp::UserParams& params) const;
    void appendEnvLog(const QString& text);
    void updateEnvironmentControls(bool busy);
    void startEnvironmentPreparation();
    void cancelEnvironmentPreparation();
    void onEnvProgress(int value);
    void onEnvFinished(bool success);
    void onEnvError(const QString& message);
    void onGpuInfoChanged(const QString& info);

    void openModelFromFile();
    void openSampleModel();
    void generateDemoToolpath();
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
    void ensureTrainingManager();
    bool isGpuAvailableForTraining() const;
    void updateTrainingActions();
    void openTrainingNewModelDialog();
    void openSyntheticDataDialog();
    void fineTuneCurrentModel();
    void openModelsFolder();
    void openDatasetsFolder();
    void onTrainingJobAdded(const train::TrainingManager::JobStatus& status);
    void onTrainingJobUpdated(const train::TrainingManager::JobStatus& status);
    void onTrainingJobRemoved(const QUuid& id);
    void onTrainingJobLog(const QUuid& id, const QString& text);
    void onTrainingToast(const QString& message);
    void onTrainingModelRegistered(const QString& path);
    void onJobSelectionChanged();
    void updateJobLogView();
    void requestJobCancellation(const QUuid& id);
    QString summarizeJob(const train::TrainingManager::JobStatus& status) const;
    QString jobStateLabel(train::TrainingManager::JobState state) const;
    QString jobEtaLabel(qint64 etaMs) const;
    QString jobTypeLabel(train::TrainingManager::JobType type) const;
    void applyJobStatus(JobWidgets& widgets, const train::TrainingManager::JobStatus& status);

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
    QAction* m_openSampleAction{nullptr};
    QAction* m_generateSampleAction{nullptr};
    bool m_simSliderPressed{false};
    bool m_updatingSimSlider{false};
    bool m_generateDemoPending{false};

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
    QGroupBox* m_passGroup{nullptr};
    QCheckBox* m_enableRoughPassCheck{nullptr};
    QCheckBox* m_enableFinishPassCheck{nullptr};
    QDoubleSpinBox* m_stockAllowanceSpin{nullptr};
    QDoubleSpinBox* m_rampAngleSpin{nullptr};
    double m_stockAllowanceMm{0.3};
    double m_rampAngleDeg{3.0};
    bool m_enableRoughPassUser{true};
    bool m_enableFinishPassUser{true};

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
    train::EnvManager* m_envManager{nullptr};
    train::TrainingManager* m_trainingManager{nullptr};
    QPlainTextEdit* m_envLog{nullptr};
    QPushButton* m_envPrepareButton{nullptr};
    QPushButton* m_envCancelButton{nullptr};
    QLabel* m_envGpuLabel{nullptr};
    QProgressBar* m_envProgress{nullptr};
    QCheckBox* m_envCpuOnlyCheck{nullptr};
    bool m_envReady{false};
    QAction* m_trainingNewModelAction{nullptr};
    QAction* m_trainingSyntheticAction{nullptr};
    QAction* m_trainingFineTuneAction{nullptr};
    QAction* m_trainingOpenModelsAction{nullptr};
    QAction* m_trainingOpenDatasetsAction{nullptr};
#if WITH_EMBEDDED_TESTS
    QAction* m_diagnosticsAction{nullptr};
#endif
    QDockWidget* m_jobsDock{nullptr};
    QListWidget* m_jobsList{nullptr};
    QPlainTextEdit* m_jobLog{nullptr};
    QLabel* m_jobSelectionLabel{nullptr};
    QHash<QUuid, JobWidgets> m_jobWidgets;
    QHash<QUuid, QStringList> m_jobLogs;
    QUuid m_selectedJob;

protected:
    void closeEvent(QCloseEvent* event) override;
};

} // namespace app
