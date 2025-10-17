#include "app/MainWindow.h"



#include "app/AiPreferencesDialog.h"
#include "app/BuildInfo.h"
#include "app/ToolpathSettingsWidget.h"
#include "app/TrainingNewModelDialog.h"
#include "app/TrainingSyntheticDataDialog.h"
#include "ai/ModelManager.h"
#include "ai/IPathAI.h"
#include "ai/TorchAI.h"
#include "ai/OnnxAI.h"
#include "io/ImportWorker.h"
#include "tp/GenerateWorker.h"
#include "tp/GCodeExporter.h"
#include "tp/GRBLPost.h"
#include "common/Units.h"
#include "render/Model.h"
#include "render/ModelViewerWidget.h"
#include "render/SimulationController.h"
#include "train/EnvManager.h"
#include "train/TrainingManager.h"

#include <glm/glm.hpp>

#include <QtCore/QCoreApplication>
#include <QtCore/QDateTime>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QJsonArray>
#include <QtCore/QJsonDocument>
#include <QtCore/QJsonObject>
#include <QtCore/QLocale>
#include <QtCore/QSettings>
#include <QtCore/QSize>
#include <QtCore/QSignalBlocker>
#include <QtCore/QStringList>
#include <QtCore/QRegularExpression>
#include <QtCore/Qt>
#include <QtCore/QTimer>
#include <QtCore/QUrl>
#include <QtWidgets/QProgressDialog>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QSlider>
#include <QtWidgets/QLabel>
#include <QtWidgets/QStyle>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QFrame>

#include <algorithm>
#include <QtGui/QCloseEvent>
#include <QtWidgets/QAction>
#include <QtWidgets/QActionGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QSizePolicy>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QAbstractItemView>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDockWidget>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QLabel>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QProgressBar>

#include <QtCore/QTextStream>
#include <QtGui/QDesktopServices>
#include <QtGui/QPalette>
#include <QtGui/QColor>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace app
{

namespace
{

QString formatTimestamped(const QString& text)
{
    return QStringLiteral("[%1] %2").arg(QDateTime::currentDateTime().toString(Qt::ISODateWithMs), text);
}

std::unique_ptr<QAction> makeAction(QObject* parent, const QString& text, const QKeySequence& shortcut = {})
{
    auto action = std::make_unique<QAction>(text, parent);
    if (!shortcut.isEmpty())
    {
        action->setShortcut(shortcut);
    }
    return action;
}

QString backendBadge(ai::ModelDescriptor::Backend backend)
{
    switch (backend)
    {
    case ai::ModelDescriptor::Backend::Torch:
        return QStringLiteral("[Torch]");
    case ai::ModelDescriptor::Backend::Onnx:
        return QStringLiteral("[ONNX]");
    }
    return QStringLiteral("[Torch]");
}

QString runtimeBadge(const ai::IPathAI* ai)
{
    if (dynamic_cast<const ai::OnnxAI*>(ai))
    {
        return QStringLiteral("[ONNX]");
    }
    return QStringLiteral("[Torch]");
}

QString runtimeDevice(const ai::IPathAI* ai)
{
    if (const auto* torchAi = dynamic_cast<const ai::TorchAI*>(ai))
    {
        return QString::fromStdString(torchAi->device());
    }
    if (const auto* onnxAi = dynamic_cast<const ai::OnnxAI*>(ai))
    {
        return QString::fromStdString(onnxAi->device());
    }
    return QStringLiteral("CPU");
}

bool runtimeLoaded(const ai::IPathAI* ai)
{
    if (const auto* torchAi = dynamic_cast<const ai::TorchAI*>(ai))
    {
        return torchAi->isLoaded();
    }
    if (const auto* onnxAi = dynamic_cast<const ai::OnnxAI*>(ai))
    {
        return onnxAi->isLoaded();
    }
    return false;
}

QString runtimeLastError(const ai::IPathAI* ai)
{
    if (const auto* torchAi = dynamic_cast<const ai::TorchAI*>(ai))
    {
        return QString::fromStdString(torchAi->lastError());
    }
    if (const auto* onnxAi = dynamic_cast<const ai::OnnxAI*>(ai))
    {
        return QString::fromStdString(onnxAi->lastError());
    }
    return QString();
}

double runtimeLatencyMs(const ai::IPathAI* ai)
{
    if (const auto* torchAi = dynamic_cast<const ai::TorchAI*>(ai))
    {
        return torchAi->lastLatencyMs();
    }
    if (const auto* onnxAi = dynamic_cast<const ai::OnnxAI*>(ai))
    {
        return onnxAi->lastLatencyMs();
    }
    return 0.0;
}

void setRuntimeForceCpu(ai::IPathAI* ai, bool forceCpu)
{
    if (auto* torchAi = dynamic_cast<ai::TorchAI*>(ai))
    {
        torchAi->setForceCpu(forceCpu);
    }
    else if (auto* onnxAi = dynamic_cast<ai::OnnxAI*>(ai))
    {
        onnxAi->setForceCpu(forceCpu);
    }
}

} // namespace

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    qRegisterMetaType<ai::StrategyDecision>("ai::StrategyDecision");
    qRegisterMetaType<std::shared_ptr<render::Model>>("std::shared_ptr<render::Model>");
    qRegisterMetaType<std::shared_ptr<tp::Toolpath>>("std::shared_ptr<tp::Toolpath>");
    qRegisterMetaType<render::SimulationController::State>("render::SimulationController::State");

    setWindowTitle(tr("AIToolpathGenerator"));
    resize(1280, 800);
    m_envManager = new train::EnvManager(this);

    applyDarkTheme();

    m_modelManager = std::make_unique<ai::ModelManager>();
    ensureTrainingManager();
    m_aiModelLabel = tr("Default");
    m_stock = tp::makeDefaultStock();
    m_machine = tp::makeDefaultMachine();

    m_viewer = new render::ModelViewerWidget(this);
    setCentralWidget(m_viewer);
    connect(m_viewer, &render::ModelViewerWidget::rendererInfoChanged, this, &MainWindow::onRendererInfoChanged);
    connect(m_viewer, &render::ModelViewerWidget::frameStatsUpdated, this, &MainWindow::onFrameStatsUpdated);

    m_simulation = std::make_unique<render::SimulationController>(this);
    m_viewer->setSimulationController(m_simulation.get());
    connect(m_simulation.get(), &render::SimulationController::progressChanged, this, &MainWindow::onSimulationProgressChanged);
    connect(m_simulation.get(), &render::SimulationController::stateChanged, this, &MainWindow::onSimulationStateChanged);

    createMenus();
    createStatusBar();
    createDockWidgets();
    if (m_envManager) {
        connect(m_envManager, &train::EnvManager::log, this, &MainWindow::appendEnvLog);
        connect(m_envManager, &train::EnvManager::progress, this, &MainWindow::onEnvProgress);
        connect(m_envManager, &train::EnvManager::finished, this, &MainWindow::onEnvFinished);
        connect(m_envManager, &train::EnvManager::error, this, &MainWindow::onEnvError);
        connect(m_envManager, &train::EnvManager::gpuInfoChanged, this, &MainWindow::onGpuInfoChanged);
        m_envManager->refreshGpuInfo();
        onGpuInfoChanged(m_envManager->gpuSummary());
        updateEnvironmentControls(false);
    }
    createCameraToolbar();
    createSimulationToolbar();
    loadToolLibrary();
    refreshAiModels();
    loadSettings();
    if (!setActiveAiModel(m_aiModelPath, true))
    {
        setActiveAiModel(QString(), true);
    }
    updateModelBrowser();
    logMessage(tr("Ready."));
    maybeRunFirstRunTour();
}

MainWindow::~MainWindow() = default;

void MainWindow::applyDarkTheme()
{
    QApplication::setStyle(QStringLiteral("Fusion"));

    QPalette palette;
    palette.setColor(QPalette::Window, QColor(31, 34, 46));
    palette.setColor(QPalette::WindowText, QColor(229, 234, 244));
    palette.setColor(QPalette::Base, QColor(21, 25, 33));
    palette.setColor(QPalette::AlternateBase, QColor(28, 32, 42));
    palette.setColor(QPalette::ToolTipBase, QColor(45, 49, 61));
    palette.setColor(QPalette::ToolTipText, QColor(236, 240, 248));
    palette.setColor(QPalette::Text, QColor(224, 229, 238));
    palette.setColor(QPalette::Button, QColor(38, 42, 54));
    palette.setColor(QPalette::ButtonText, QColor(229, 234, 244));
    palette.setColor(QPalette::BrightText, QColor(248, 94, 94));
    palette.setColor(QPalette::Highlight, QColor(86, 145, 255));
    palette.setColor(QPalette::HighlightedText, QColor(244, 248, 255));
    palette.setColor(QPalette::Link, QColor(102, 153, 255));

    qApp->setPalette(palette);

    qApp->setStyleSheet(QStringLiteral(R"(
        QWidget { color: #E4E8F2; font: 10pt "Segoe UI"; }
        QMainWindow, QDockWidget { background-color: #1F2230; }
        QToolBar { background-color: #292D3C; border: none; padding: 6px 10px; spacing: 6px; }
        QDockWidget::title { background-color: #24283A; padding: 6px 10px; font-weight: 600; }
        QMenuBar { background-color: #1F2230; padding: 4px 12px; }
        QMenu { background-color: #262A39; border: 1px solid #2E3344; }
        QMenu::item:selected { background-color: #303649; }
        QPlainTextEdit, QListWidget, QTabWidget::pane, QTableView, QTreeView {
            background-color: #151923;
            border: 1px solid #2C3241;
            border-radius: 6px;
        }
        QStatusBar { background-color: #181B26; }
        QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            background-color: #232736;
            border: 1px solid #3A4154;
            border-radius: 4px;
            padding: 4px 8px;
        }
        QPushButton:hover { background-color: #2C3144; }
        QPushButton:pressed { background-color: #1A1D29; }
        QTabBar::tab { padding: 6px 12px; margin-right: 2px; border-radius: 4px; }
        QTabBar::tab:selected { background-color: #2C3144; }
        QScrollBar:vertical, QScrollBar:horizontal { background: #1A1D29; }
    )"));
}

void MainWindow::createStatusBar()
{
    auto* bar = new QStatusBar(this);
    bar->setObjectName(QStringLiteral("MainStatusBar"));
    bar->setSizeGripEnabled(false);
    setStatusBar(bar);
    updateStatusBarTheme();

    auto makeLabel = [bar](const QString& text) {
        auto* label = new QLabel(text, bar);
        label->setObjectName(QStringLiteral("StatusValue"));
        label->setMinimumWidth(150);
        return label;
    };

    m_statusGpuLabel = makeLabel(tr("GPU: detecting..."));
    m_statusAiLabel = makeLabel(tr("AI: --"));
    m_statusFpsLabel = makeLabel(tr("FPS: --"));

    m_statusGpuLabel->setToolTip(tr("Renderer information"));
    m_statusAiLabel->setToolTip(tr("Active AI backend"));
    m_statusFpsLabel->setToolTip(tr("Rendering frame rate"));

    bar->addWidget(m_statusGpuLabel, 1);

    auto makeSeparator = [bar]() {
        auto* line = new QFrame(bar);
        line->setObjectName(QStringLiteral("StatusSeparator"));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Plain);
        line->setLineWidth(1);
        line->setFixedHeight(18);
        line->setFixedWidth(1);
        return line;
    };

    bar->addPermanentWidget(makeSeparator());
    bar->addPermanentWidget(m_statusAiLabel);
    bar->addPermanentWidget(makeSeparator());
    bar->addPermanentWidget(m_statusFpsLabel);

    updateStatusAiLabel();
}

void MainWindow::updateStatusBarTheme()
{
    if (auto* bar = statusBar())
    {
        bar->setContentsMargins(12, 0, 12, 0);
        bar->setStyleSheet(QStringLiteral(
            "QStatusBar { background-color: #181B26; border-top: 1px solid #2A2E3D; color: #CDD2E4; }"
            "QLabel#StatusValue { padding: 0 12px; font-weight: 500; }"
            "QFrame#StatusSeparator { background-color: #2F3446; min-width: 1px; max-width: 1px; }"));
    }
}

void MainWindow::createMenus()
{
    auto* fileMenu = menuBar()->addMenu(tr("&File"));
    {
        auto openAction = makeAction(this, tr("&Open..."), QKeySequence::Open);
        connect(openAction.get(), &QAction::triggered, this, &MainWindow::openModelFromFile);
        fileMenu->addAction(openAction.release());

        auto saveAction = makeAction(this, tr("&Save Toolpath..."), QKeySequence::Save);
        connect(saveAction.get(), &QAction::triggered, this, &MainWindow::saveToolpathToFile);
        fileMenu->addAction(saveAction.release());

        fileMenu->addSeparator();

        auto exitAction = makeAction(this, tr("E&xit"), QKeySequence::Quit);
        connect(exitAction.get(), &QAction::triggered, qApp, &QApplication::quit);
        fileMenu->addAction(exitAction.release());
    }

    auto* viewMenu = menuBar()->addMenu(tr("&View"));
    {
        auto resetAction = makeAction(this, tr("&Reset Camera"), QKeySequence(Qt::CTRL | Qt::Key_R));
        connect(resetAction.get(), &QAction::triggered, this, &MainWindow::resetCamera);
        viewMenu->addAction(resetAction.release());

        createUnitMenu(viewMenu);
    }

    auto* machineMenu = menuBar()->addMenu(tr("&Machine"));
    createMachineMenu(machineMenu);

    auto* aiMenu = menuBar()->addMenu(tr("&AI"));
    {
        auto selectAction = makeAction(this, tr("Select Model..."));
        connect(selectAction.get(), &QAction::triggered, this, &MainWindow::selectModelWithAI);
        aiMenu->addAction(selectAction.release());

        aiMenu->addSeparator();
        auto preferencesAction = makeAction(this, tr("Preferences..."));
        connect(preferencesAction.get(), &QAction::triggered, this, &MainWindow::openAiPreferences);
        aiMenu->addAction(preferencesAction.release());
    }

    createTrainingMenu();

    auto* helpMenu = menuBar()->addMenu(tr("&Help"));
    {
        auto aboutAction = makeAction(this, tr("&About"));
        connect(aboutAction.get(), &QAction::triggered, this, &MainWindow::showAboutDialog);
        helpMenu->addAction(aboutAction.release());
    }
}

void MainWindow::createTrainingMenu()
{
    ensureTrainingManager();

    auto* trainingMenu = menuBar()->addMenu(tr("&Training"));

    m_trainingNewModelAction = trainingMenu->addAction(tr("New Model..."));
    m_trainingNewModelAction->setToolTip(tr("Launch a training run with optional base weights."));
    connect(m_trainingNewModelAction, &QAction::triggered, this, &MainWindow::openTrainingNewModelDialog);

    m_trainingSyntheticAction = trainingMenu->addAction(tr("Generate Synthetic Data..."));
    m_trainingSyntheticAction->setToolTip(tr("Synthesize geometry samples for training and validation."));
    connect(m_trainingSyntheticAction, &QAction::triggered, this, &MainWindow::openSyntheticDataDialog);

    m_trainingFineTuneAction = trainingMenu->addAction(tr("Fine-Tune Current Model..."));
    m_trainingFineTuneAction->setToolTip(tr("Continue training using the active model as a starting point."));
    connect(m_trainingFineTuneAction, &QAction::triggered, this, &MainWindow::fineTuneCurrentModel);

    trainingMenu->addSeparator();

    m_trainingOpenModelsAction = trainingMenu->addAction(tr("Open Models Folder"));
    connect(m_trainingOpenModelsAction, &QAction::triggered, this, &MainWindow::openModelsFolder);

    m_trainingOpenDatasetsAction = trainingMenu->addAction(tr("Open Datasets Folder"));
    connect(m_trainingOpenDatasetsAction, &QAction::triggered, this, &MainWindow::openDatasetsFolder);

    updateTrainingActions();
}

void MainWindow::createDockWidgets()
{
    auto* modelDock = new QDockWidget(tr("Model Browser"), this);
    modelDock->setObjectName(QStringLiteral("ModelBrowserDock"));
    m_modelBrowser = new QListWidget(modelDock);
    modelDock->setWidget(m_modelBrowser);
    addDockWidget(Qt::LeftDockWidgetArea, modelDock);

    auto* toolpathDock = new QDockWidget(tr("Toolpath Settings"), this);
    toolpathDock->setObjectName(QStringLiteral("ToolpathSettingsDock"));

    auto* dockContainer = new QWidget(toolpathDock);
    auto* dockLayout = new QVBoxLayout(dockContainer);
    dockLayout->setContentsMargins(12, 12, 12, 12);
    dockLayout->setSpacing(12);

    m_toolpathSettings = new ToolpathSettingsWidget(dockContainer);
    m_toolpathSettings->setUnits(m_units);
    dockLayout->addWidget(m_toolpathSettings);

    setupStockMachineControls(dockContainer, dockLayout);

    m_passGroup = new QGroupBox(tr("Pass Planning"), dockContainer);
    auto* passLayout = new QVBoxLayout(m_passGroup);
    passLayout->setContentsMargins(8, 8, 8, 8);
    passLayout->setSpacing(6);

    m_enableRoughPassCheck = new QCheckBox(tr("Enable roughing pass"), m_passGroup);
    m_enableFinishPassCheck = new QCheckBox(tr("Enable finishing pass"), m_passGroup);
    passLayout->addWidget(m_enableRoughPassCheck);
    passLayout->addWidget(m_enableFinishPassCheck);

    auto* passForm = new QFormLayout();
    passForm->setContentsMargins(0, 0, 0, 0);
    passForm->setSpacing(4);

    m_stockAllowanceSpin = new QDoubleSpinBox(m_passGroup);
    m_stockAllowanceSpin->setDecimals(3);
    m_stockAllowanceSpin->setToolTip(tr("Material to leave after roughing."));
    passForm->addRow(tr("Stock allowance"), m_stockAllowanceSpin);

    m_rampAngleSpin = new QDoubleSpinBox(m_passGroup);
    m_rampAngleSpin->setDecimals(1);
    m_rampAngleSpin->setRange(0.5, 30.0);
    m_rampAngleSpin->setSingleStep(0.5);
    m_rampAngleSpin->setSuffix(tr(" deg"));
    m_rampAngleSpin->setToolTip(tr("Lead-in ramp angle in degrees."));
    passForm->addRow(tr("Ramp angle"), m_rampAngleSpin);

    passLayout->addLayout(passForm);
    dockLayout->addWidget(m_passGroup);

    auto ensureOnePassEnabled = [this]() {
        if (!m_enableRoughPassCheck || !m_enableFinishPassCheck)
        {
            return;
        }
        if (!m_enableRoughPassCheck->isChecked() && !m_enableFinishPassCheck->isChecked())
        {
            QSignalBlocker finishBlocker(m_enableFinishPassCheck);
            m_enableFinishPassCheck->setChecked(true);
            m_enableFinishPassUser = true;
        }
    };

    connect(m_enableRoughPassCheck, &QCheckBox::toggled, this, [this, ensureOnePassEnabled](bool checked) {
        m_enableRoughPassUser = checked;
        ensureOnePassEnabled();
        if (!m_enableFinishPassCheck)
        {
            return;
        }
        m_enableFinishPassUser = m_enableFinishPassCheck->isChecked();
        if (!m_loadingSettings)
        {
            saveSettings();
        }
    });
    connect(m_enableFinishPassCheck, &QCheckBox::toggled, this, [this, ensureOnePassEnabled](bool checked) {
        m_enableFinishPassUser = checked;
        ensureOnePassEnabled();
        if (!m_enableRoughPassCheck)
        {
            return;
        }
        m_enableRoughPassUser = m_enableRoughPassCheck->isChecked();
        if (!m_loadingSettings)
        {
            saveSettings();
        }
    });
    connect(m_stockAllowanceSpin,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            [this](double value) {
                m_stockAllowanceMm = std::max(0.0, lengthMmFromDisplay(value));
                if (!m_loadingSettings)
                {
                    saveSettings();
                }
            });
    connect(m_rampAngleSpin,
            qOverload<double>(&QDoubleSpinBox::valueChanged),
            this,
            [this](double value) {
                m_rampAngleDeg = value;
                if (!m_loadingSettings)
                {
                    saveSettings();
                }
            });

    syncPassControlsFromData();
    updatePassControlUnits();

    auto* aiLabel = new QLabel(tr("AI Model"), dockContainer);
    dockLayout->addWidget(aiLabel);

    m_aiModelCombo = new QComboBox(dockContainer);
    m_aiModelCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_aiModelCombo->setToolTip(tr("Select the path-planning AI model."));
    m_aiModelCombo->setMinimumContentsLength(1);
    m_aiModelCombo->setMaxVisibleItems(12);
    dockLayout->addWidget(m_aiModelCombo);

    m_aiDeviceLabel = new QLabel(tr("Device: CPU"), dockContainer);
    m_aiDeviceLabel->setObjectName(QStringLiteral("AiDeviceLabel"));
    dockLayout->addWidget(m_aiDeviceLabel);

    dockLayout->addStretch(1);

    toolpathDock->setWidget(dockContainer);
    addDockWidget(Qt::RightDockWidgetArea, toolpathDock);

    auto* consoleDock = new QDockWidget(tr("Console"), this);
    consoleDock->setObjectName(QStringLiteral("ConsoleDock"));
    m_consoleTabs = new QTabWidget(consoleDock);
    m_consoleTabs->setTabPosition(QTabWidget::South);
    m_consoleTabs->setDocumentMode(true);

    m_console = new QPlainTextEdit(m_consoleTabs);
    m_console->setReadOnly(true);
    m_console->setLineWrapMode(QPlainTextEdit::NoWrap);
    m_console->setMaximumBlockCount(2'000);
    m_console->setPlaceholderText(tr("Status messages, warnings, and toolpath generation logs will appear here."));
    m_consoleTabs->addTab(m_console, tr("Log"));

    m_gcodePreview = new QPlainTextEdit(m_consoleTabs);
    m_gcodePreview->setReadOnly(true);
    m_gcodePreview->setPlainText(tr("G-code preview will appear here after export."));
    m_consoleTabs->addTab(m_gcodePreview, tr("G-code Preview"));
    m_consoleTabs->setTabToolTip(1, tr("Preview of the last exported G-code."));

    consoleDock->setWidget(m_consoleTabs);
    addDockWidget(Qt::BottomDockWidgetArea, consoleDock);

    auto* envDock = new QDockWidget(tr("Training Environment"), this);
    envDock->setObjectName(QStringLiteral("TrainingEnvironmentDock"));
    auto* envWidget = new QWidget(envDock);
    auto* envLayout = new QVBoxLayout(envWidget);
    envLayout->setContentsMargins(12, 12, 12, 12);
    envLayout->setSpacing(8);

    m_envGpuLabel = new QLabel(tr("GPU: detecting..."), envWidget);
    envLayout->addWidget(m_envGpuLabel);

    m_envCpuOnlyCheck = new QCheckBox(tr("CPU-only mode"), envWidget);
    envLayout->addWidget(m_envCpuOnlyCheck);

    auto* buttonRow = new QHBoxLayout();
    buttonRow->setContentsMargins(0, 0, 0, 0);
    buttonRow->setSpacing(8);

    m_envPrepareButton = new QPushButton(tr("Prepare Environment"), envWidget);
    m_envCancelButton = new QPushButton(tr("Cancel"), envWidget);
    m_envCancelButton->setEnabled(false);

    buttonRow->addWidget(m_envPrepareButton);
    buttonRow->addWidget(m_envCancelButton);
    envLayout->addLayout(buttonRow);

    m_envProgress = new QProgressBar(envWidget);
    m_envProgress->setRange(0, 100);
    m_envProgress->setValue(0);
    m_envProgress->setFormat(QStringLiteral("%p%"));
    envLayout->addWidget(m_envProgress);

    m_envLog = new QPlainTextEdit(envWidget);
    m_envLog->setReadOnly(true);
    m_envLog->setMaximumBlockCount(2'000);
    m_envLog->setPlaceholderText(tr("Environment preparation logs will appear here."));
    envLayout->addWidget(m_envLog, 1);

    envWidget->setLayout(envLayout);
    envDock->setWidget(envWidget);
    addDockWidget(Qt::RightDockWidgetArea, envDock);
    createJobsDock(envDock);

    connect(m_toolpathSettings, &ToolpathSettingsWidget::generateRequested, this, &MainWindow::onToolpathRequested);
    connect(m_toolpathSettings, &ToolpathSettingsWidget::toolChanged, this, &MainWindow::onToolSelected);
    connect(m_toolpathSettings, &ToolpathSettingsWidget::warningGenerated, this, &MainWindow::logWarning);
    connect(m_aiModelCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, &MainWindow::onAiComboChanged);
    connect(m_envPrepareButton, &QPushButton::clicked, this, &MainWindow::startEnvironmentPreparation);
    connect(m_envCancelButton, &QPushButton::clicked, this, &MainWindow::cancelEnvironmentPreparation);
    connect(m_envCpuOnlyCheck, &QCheckBox::toggled, this, [this](bool checked) {
        if (m_envManager)
        {
            m_envManager->setCpuOnly(checked);
        }
        if (!m_loadingSettings)
        {
            saveSettings();
        }
    });
}

void MainWindow::createJobsDock(QDockWidget* envDock)
{
    if (m_jobsDock)
    {
        return;
    }

    m_jobsDock = new QDockWidget(tr("Training Jobs"), this);
    m_jobsDock->setObjectName(QStringLiteral("TrainingJobsDock"));
    m_jobsDock->setMinimumWidth(320);

    auto* container = new QWidget(m_jobsDock);
    auto* layout = new QVBoxLayout(container);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(8);

    m_jobsList = new QListWidget(container);
    m_jobsList->setSelectionMode(QAbstractItemView::SingleSelection);
    m_jobsList->setUniformItemSizes(false);
    m_jobsList->setSpacing(6);
    layout->addWidget(m_jobsList);

    m_jobSelectionLabel = new QLabel(tr("Select a job to view logs."), container);
    layout->addWidget(m_jobSelectionLabel);

    m_jobLog = new QPlainTextEdit(container);
    m_jobLog->setReadOnly(true);
    m_jobLog->setMaximumBlockCount(4000);
    m_jobLog->setPlaceholderText(tr("Job output will appear here."));
    layout->addWidget(m_jobLog, 1);

    container->setLayout(layout);
    m_jobsDock->setWidget(container);

    addDockWidget(Qt::RightDockWidgetArea, m_jobsDock);
    if (envDock)
    {
        tabifyDockWidget(envDock, m_jobsDock);
        envDock->raise();
    }

    connect(
        m_jobsList,
        &QListWidget::currentItemChanged,
        this,
        [this](QListWidgetItem*, QListWidgetItem*) {
            onJobSelectionChanged();
        });
}

void MainWindow::createCameraToolbar()
{
    if (m_cameraToolbar)
    {
        return;
    }

    m_cameraToolbar = new QToolBar(tr("Camera"), this);
    m_cameraToolbar->setObjectName(QStringLiteral("CameraToolbar"));
    m_cameraToolbar->setMovable(false);
    m_cameraToolbar->setToolButtonStyle(Qt::ToolButtonTextOnly);
    m_cameraToolbar->setIconSize(QSize(20, 20));
    m_cameraToolbar->setStyleSheet(QStringLiteral("QToolBar#CameraToolbar { padding: 6px 12px; spacing: 10px; }"));
    addToolBar(Qt::TopToolBarArea, m_cameraToolbar);

    struct PresetDefinition
    {
        QString label;
        QString tooltip;
        render::ModelViewerWidget::ViewPreset preset;
    };

    const PresetDefinition presets[] = {
        {tr("Top"), tr("Look down from the Z+ axis"), render::ModelViewerWidget::ViewPreset::Top},
        {tr("Front"), tr("Look from the Y+ axis"), render::ModelViewerWidget::ViewPreset::Front},
        {tr("Right"), tr("Look from the X+ axis"), render::ModelViewerWidget::ViewPreset::Right},
        {tr("Iso"), tr("Three-quarter isometric view"), render::ModelViewerWidget::ViewPreset::Iso},
    };

    for (const auto& preset : presets)
    {
        auto* action = m_cameraToolbar->addAction(preset.label);
        action->setToolTip(preset.tooltip);
        connect(action, &QAction::triggered, this, [this, preset]() {
            if (m_viewer)
            {
                m_viewer->setViewPreset(preset.preset);
            }
        });
    }
}

void MainWindow::createSimulationToolbar()
{
    m_simulationToolbar = new QToolBar(tr("Simulation"), this);
    m_simulationToolbar->setObjectName(QStringLiteral("SimulationToolbar"));
    m_simulationToolbar->setMovable(false);
    m_simulationToolbar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_simulationToolbar->setIconSize(QSize(20, 20));
    m_simulationToolbar->setStyleSheet(QStringLiteral("QToolBar#SimulationToolbar { padding: 6px 12px; spacing: 10px; }"));
    addToolBar(Qt::BottomToolBarArea, m_simulationToolbar);

    m_simPlayAction = m_simulationToolbar->addAction(style()->standardIcon(QStyle::SP_MediaPlay), tr("Play"));
    m_simPauseAction = m_simulationToolbar->addAction(style()->standardIcon(QStyle::SP_MediaPause), tr("Pause"));
    m_simStopAction = m_simulationToolbar->addAction(style()->standardIcon(QStyle::SP_MediaStop), tr("Stop"));

    connect(m_simPlayAction, &QAction::triggered, this, [this]() {
        if (m_simulation)
        {
            m_simulation->play();
        }
    });
    connect(m_simPauseAction, &QAction::triggered, this, [this]() {
        if (m_simulation)
        {
            m_simulation->pause();
        }
    });
    connect(m_simStopAction, &QAction::triggered, this, [this]() {
        if (m_simulation)
        {
            m_simulation->stop();
        }
    });

    m_simulationToolbar->addSeparator();
    m_simulationToolbar->addWidget(new QLabel(tr("Progress"), this));

    m_simProgressSlider = new QSlider(Qt::Horizontal, this);
    m_simProgressSlider->setRange(0, 1000);
    m_simProgressSlider->setPageStep(25);
    m_simProgressSlider->setValue(0);
    m_simProgressSlider->setFixedWidth(240);
    m_simulationToolbar->addWidget(m_simProgressSlider);

    connect(m_simProgressSlider, &QSlider::sliderPressed, this, [this]() {
        m_simSliderPressed = true;
    });
    connect(m_simProgressSlider, &QSlider::sliderReleased, this, [this]() {
        m_simSliderPressed = false;
        const int max = m_simProgressSlider->maximum();
        if (max <= 0)
        {
            return;
        }
        const double normalized = static_cast<double>(m_simProgressSlider->value()) / static_cast<double>(max);
        applySimulationSlider(normalized);
    });
    connect(m_simProgressSlider, &QSlider::sliderMoved, this, [this](int value) {
        const int max = m_simProgressSlider->maximum();
        if (max <= 0)
        {
            return;
        }
        const double normalized = static_cast<double>(value) / static_cast<double>(max);
        applySimulationSlider(normalized);
    });

    m_simulationToolbar->addSeparator();
    m_simulationToolbar->addWidget(new QLabel(tr("Speed"), this));

    m_simSpeedSlider = new QSlider(Qt::Horizontal, this);
    m_simSpeedSlider->setRange(25, 400); // 0.25x to 4x
    m_simSpeedSlider->setSingleStep(5);
    m_simSpeedSlider->setValue(100);
    m_simSpeedSlider->setFixedWidth(140);
    m_simulationToolbar->addWidget(m_simSpeedSlider);

    m_simSpeedLabel = new QLabel(tr("1.00x"), this);
    m_simulationToolbar->addWidget(m_simSpeedLabel);

    connect(m_simSpeedSlider, &QSlider::valueChanged, this, [this](int value) {
        const double multiplier = static_cast<double>(value) / 100.0;
        if (m_simulation)
        {
            m_simulation->setSpeedMultiplier(multiplier);
        }
        if (m_simSpeedLabel)
        {
            m_simSpeedLabel->setText(QString::number(multiplier, 'f', 2) + QStringLiteral("x"));
        }
    });

    if (m_simulation)
    {
        m_simulation->setSpeedMultiplier(1.0);
    }

    onSimulationStateChanged(render::SimulationController::State::Stopped);
    onSimulationProgressChanged(0.0);
}

void MainWindow::createUnitMenu(QMenu* viewMenu)
{
    auto* unitsMenu = viewMenu->addMenu(tr("Units"));

    m_unitsGroup = new QActionGroup(this);
    m_unitsGroup->setExclusive(true);

    m_unitsMmAction = unitsMenu->addAction(tr("Millimeters"));
    m_unitsMmAction->setCheckable(true);
    m_unitsMmAction->setData(static_cast<int>(common::Unit::Millimeters));
    m_unitsGroup->addAction(m_unitsMmAction);

    m_unitsInAction = unitsMenu->addAction(tr("Inches"));
    m_unitsInAction->setCheckable(true);
    m_unitsInAction->setData(static_cast<int>(common::Unit::Inches));
    m_unitsGroup->addAction(m_unitsInAction);

    connect(m_unitsGroup, &QActionGroup::triggered, this, [this](QAction* action) {
        const auto unit = static_cast<common::Unit>(action->data().toInt());
        applyUnits(unit);
    });

    applyUnits(m_units, true);
}

void MainWindow::createMachineMenu(QMenu* machineMenu)
{
    if (!machineMenu)
    {
        return;
    }

    auto* presetsMenu = machineMenu->addMenu(tr("Presets"));

    QAction* grblAction = presetsMenu->addAction(tr("GRBL Router"));
    connect(grblAction, &QAction::triggered, this, [this]() {
        tp::Machine machine = tp::makeDefaultMachine();
        machine.name = "GRBL Router";
        machine.rapidFeed_mm_min = 9'000.0;
        machine.maxFeed_mm_min = 3'000.0;
        machine.maxSpindleRPM = 24'000.0;
        machine.clearanceZ_mm = 5.0;
        machine.safeZ_mm = 15.0;
        applyMachinePreset(machine);
    });

    QAction* mach3Action = presetsMenu->addAction(tr("Mach3 Router"));
    connect(mach3Action, &QAction::triggered, this, [this]() {
        tp::Machine machine = tp::makeDefaultMachine();
        machine.name = "Mach3 Router";
        machine.rapidFeed_mm_min = 6'000.0;
        machine.maxFeed_mm_min = 2'500.0;
        machine.maxSpindleRPM = 18'000.0;
        machine.clearanceZ_mm = 6.0;
        machine.safeZ_mm = 20.0;
        applyMachinePreset(machine);
    });

    machineMenu->addSeparator();
    auto* recomputeStock = machineMenu->addAction(tr("Recompute Stock From Model"));
    connect(recomputeStock, &QAction::triggered, this, [this]() {
        if (m_stockOriginMode != StockOriginMode::Custom)
        {
            updateStockDerivedFromModel();
            syncStockUiFromData();
        }
    });
}

void MainWindow::setupStockMachineControls(QWidget* container, QVBoxLayout* layout)
{
    if (!layout)
    {
        return;
    }

    m_stockMachineGroup = new QGroupBox(tr("Stock & Machine"), container);
    m_stockMachineGroup->setObjectName(QStringLiteral("StockMachineGroup"));

    auto* groupLayout = new QVBoxLayout(m_stockMachineGroup);
    groupLayout->setContentsMargins(12, 8, 12, 12);
    groupLayout->setSpacing(10);

    auto createDimensionSpin = [this]() {
        auto* spin = new QDoubleSpinBox(m_stockMachineGroup);
        spin->setDecimals(3);
        spin->setMinimum(0.0);
        spin->setMaximum(50'000.0);
        spin->setSingleStep(0.1);
        spin->setKeyboardTracking(false);
        return spin;
    };

    auto createCoordinateSpin = [this]() {
        auto* spin = new QDoubleSpinBox(m_stockMachineGroup);
        spin->setDecimals(3);
        spin->setMinimum(-50'000.0);
        spin->setMaximum(50'000.0);
        spin->setSingleStep(0.1);
        spin->setKeyboardTracking(false);
        return spin;
    };

    auto createFeedSpin = [this]() {
        auto* spin = new QDoubleSpinBox(m_stockMachineGroup);
        spin->setDecimals(1);
        spin->setMinimum(0.0);
        spin->setMaximum(120'000.0);
        spin->setSingleStep(100.0);
        spin->setKeyboardTracking(false);
        return spin;
    };

    auto* stockForm = new QFormLayout();
    stockForm->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
    stockForm->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    stockForm->setHorizontalSpacing(10);
    stockForm->setVerticalSpacing(6);

    m_stockWidth = createDimensionSpin();
    m_stockLength = createDimensionSpin();
    m_stockHeight = createDimensionSpin();
    m_stockMargin = createDimensionSpin();
    m_stockMargin->setMaximum(5'000.0);

    m_stockOriginCombo = new QComboBox(m_stockMachineGroup);
    m_stockOriginCombo->addItem(tr("Model Minimum"));
    m_stockOriginCombo->addItem(tr("Model Center"));
    m_stockOriginCombo->addItem(tr("Custom"));

    stockForm->addRow(tr("Block Width"), m_stockWidth);
    stockForm->addRow(tr("Block Length"), m_stockLength);
    stockForm->addRow(tr("Block Height"), m_stockHeight);
    stockForm->addRow(tr("Margin"), m_stockMargin);
    stockForm->addRow(tr("Origin Mode"), m_stockOriginCombo);

    auto* originWidget = new QWidget(m_stockMachineGroup);
    auto* originLayout = new QHBoxLayout(originWidget);
    originLayout->setContentsMargins(0, 0, 0, 0);
    originLayout->setSpacing(6);

    auto addOriginField = [originLayout, this](const QString& labelText, QDoubleSpinBox*& targetSpin) {
        auto* label = new QLabel(labelText, m_stockMachineGroup);
        label->setMinimumWidth(16);
        targetSpin = createCoordinateSpin();
        originLayout->addWidget(label);
        originLayout->addWidget(targetSpin, 1);
    };

    addOriginField(QStringLiteral("X"), m_stockOriginX);
    addOriginField(QStringLiteral("Y"), m_stockOriginY);
    addOriginField(QStringLiteral("Z"), m_stockOriginZ);

    stockForm->addRow(tr("Origin (XYZ)"), originWidget);
    groupLayout->addLayout(stockForm);

    auto* machineForm = new QFormLayout();
    machineForm->setFormAlignment(Qt::AlignLeft | Qt::AlignTop);
    machineForm->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
    machineForm->setHorizontalSpacing(10);
    machineForm->setVerticalSpacing(6);

    m_machineNameEdit = new QLineEdit(m_stockMachineGroup);
    m_machineNameEdit->setPlaceholderText(tr("Machine name"));

    m_machineRapidFeed = createFeedSpin();
    m_machineMaxFeed = createFeedSpin();
    m_machineMaxSpindle = createFeedSpin();
    m_machineMaxSpindle->setMaximum(200'000.0);
    m_machineMaxSpindle->setDecimals(0);
    m_machineMaxSpindle->setSingleStep(100.0);

    m_machineClearanceZ = createDimensionSpin();
    m_machineSafeZ = createDimensionSpin();

    machineForm->addRow(tr("Machine Name"), m_machineNameEdit);
    machineForm->addRow(tr("Rapid Feed"), m_machineRapidFeed);
    machineForm->addRow(tr("Max Feed"), m_machineMaxFeed);
    machineForm->addRow(tr("Max Spindle RPM"), m_machineMaxSpindle);
    machineForm->addRow(tr("Clearance Z"), m_machineClearanceZ);
    machineForm->addRow(tr("Safe Z"), m_machineSafeZ);

    groupLayout->addLayout(machineForm);

    layout->addWidget(m_stockMachineGroup);

    connect(m_stockMargin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, [this](double) {
        syncStockDataFromUi();
        if (m_stockOriginMode != StockOriginMode::Custom)
        {
            updateStockDerivedFromModel();
            syncStockUiFromData();
        }
    });

    auto stockDimensionChanged = [this](double) {
        syncStockDataFromUi();
    };
    connect(m_stockWidth, qOverload<double>(&QDoubleSpinBox::valueChanged), this, stockDimensionChanged);
    connect(m_stockLength, qOverload<double>(&QDoubleSpinBox::valueChanged), this, stockDimensionChanged);
    connect(m_stockHeight, qOverload<double>(&QDoubleSpinBox::valueChanged), this, stockDimensionChanged);

    connect(m_stockOriginCombo, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int index) {
        if (m_blockStockSignals)
        {
            return;
        }
        StockOriginMode newMode = StockOriginMode::ModelMin;
        switch (index)
        {
        case 1:
            newMode = StockOriginMode::ModelCenter;
            break;
        case 2:
            newMode = StockOriginMode::Custom;
            break;
        default:
            break;
        }
        if (newMode == m_stockOriginMode)
        {
            return;
        }
        m_stockOriginMode = newMode;
        updateStockUiEnabledState();
        if (m_stockOriginMode != StockOriginMode::Custom)
        {
            updateStockDerivedFromModel();
        }
        syncStockUiFromData();
    });

    auto originChanged = [this](double) {
        syncStockDataFromUi();
    };
    connect(m_stockOriginX, qOverload<double>(&QDoubleSpinBox::valueChanged), this, originChanged);
    connect(m_stockOriginY, qOverload<double>(&QDoubleSpinBox::valueChanged), this, originChanged);
    connect(m_stockOriginZ, qOverload<double>(&QDoubleSpinBox::valueChanged), this, originChanged);

    connect(m_machineNameEdit, &QLineEdit::textEdited, this, [this](const QString&) {
        syncMachineDataFromUi();
    });

    auto machineValueChanged = [this](double) {
        syncMachineDataFromUi();
    };
    connect(m_machineRapidFeed, qOverload<double>(&QDoubleSpinBox::valueChanged), this, machineValueChanged);
    connect(m_machineMaxFeed, qOverload<double>(&QDoubleSpinBox::valueChanged), this, machineValueChanged);
    connect(m_machineMaxSpindle, qOverload<double>(&QDoubleSpinBox::valueChanged), this, machineValueChanged);
    connect(m_machineClearanceZ, qOverload<double>(&QDoubleSpinBox::valueChanged), this, machineValueChanged);
    connect(m_machineSafeZ, qOverload<double>(&QDoubleSpinBox::valueChanged), this, machineValueChanged);

    syncStockUiFromData();
    syncMachineUiFromData();
    updateStockUiEnabledState();
}

void MainWindow::syncStockUiFromData()
{
    if (!m_stockWidth)
    {
        return;
    }

    const QString lengthSuffix = QStringLiteral(" %1").arg(common::unitSuffix(m_units));

    const QSignalBlocker blockWidth(m_stockWidth);
    const QSignalBlocker blockLength(m_stockLength);
    const QSignalBlocker blockHeight(m_stockHeight);
    const QSignalBlocker blockMargin(m_stockMargin);
    const QSignalBlocker blockOriginX(m_stockOriginX);
    const QSignalBlocker blockOriginY(m_stockOriginY);
    const QSignalBlocker blockOriginZ(m_stockOriginZ);
    const QSignalBlocker blockOriginCombo(m_stockOriginCombo);

    m_blockStockSignals = true;

    m_stockWidth->setSuffix(lengthSuffix);
    m_stockLength->setSuffix(lengthSuffix);
    m_stockHeight->setSuffix(lengthSuffix);
    m_stockMargin->setSuffix(lengthSuffix);
    m_stockOriginX->setSuffix(lengthSuffix);
    m_stockOriginY->setSuffix(lengthSuffix);
    m_stockOriginZ->setSuffix(lengthSuffix);

    m_stockWidth->setValue(lengthDisplayFromMm(m_stock.sizeXYZ_mm.x));
    m_stockLength->setValue(lengthDisplayFromMm(m_stock.sizeXYZ_mm.y));
    m_stockHeight->setValue(lengthDisplayFromMm(m_stock.sizeXYZ_mm.z));
    m_stockMargin->setValue(lengthDisplayFromMm(m_stock.margin_mm));
    m_stockOriginX->setValue(lengthDisplayFromMm(m_stock.originXYZ_mm.x));
    m_stockOriginY->setValue(lengthDisplayFromMm(m_stock.originXYZ_mm.y));
    m_stockOriginZ->setValue(lengthDisplayFromMm(m_stock.originXYZ_mm.z));

    int originIndex = 0;
    switch (m_stockOriginMode)
    {
    case StockOriginMode::ModelMin:
        originIndex = 0;
        break;
    case StockOriginMode::ModelCenter:
        originIndex = 1;
        break;
    case StockOriginMode::Custom:
        originIndex = 2;
        break;
    }
    m_stockOriginCombo->setCurrentIndex(originIndex);

    m_blockStockSignals = false;
    updateStockUiEnabledState();
}

void MainWindow::syncMachineUiFromData()
{
    if (!m_machineNameEdit)
    {
        return;
    }

    const QString lengthSuffix = QStringLiteral(" %1").arg(common::unitSuffix(m_units));
    const QString feedSuffix = QStringLiteral(" %1").arg(common::feedSuffix(m_units));

    const QSignalBlocker blockName(m_machineNameEdit);
    const QSignalBlocker blockRapid(m_machineRapidFeed);
    const QSignalBlocker blockMaxFeed(m_machineMaxFeed);
    const QSignalBlocker blockMaxSpindle(m_machineMaxSpindle);
    const QSignalBlocker blockClearance(m_machineClearanceZ);
    const QSignalBlocker blockSafe(m_machineSafeZ);

    m_blockMachineSignals = true;

    m_machineNameEdit->setText(QString::fromStdString(m_machine.name));

    m_machineRapidFeed->setSuffix(feedSuffix);
    m_machineMaxFeed->setSuffix(feedSuffix);
    m_machineMaxSpindle->setSuffix(QStringLiteral(" RPM"));
    m_machineClearanceZ->setSuffix(lengthSuffix);
    m_machineSafeZ->setSuffix(lengthSuffix);

    m_machineRapidFeed->setValue(feedDisplayFromMmPerMin(m_machine.rapidFeed_mm_min));
    m_machineMaxFeed->setValue(feedDisplayFromMmPerMin(m_machine.maxFeed_mm_min));
    m_machineMaxSpindle->setValue(m_machine.maxSpindleRPM);
    m_machineClearanceZ->setValue(lengthDisplayFromMm(m_machine.clearanceZ_mm));
    m_machineSafeZ->setValue(lengthDisplayFromMm(m_machine.safeZ_mm));

    m_blockMachineSignals = false;
}

void MainWindow::syncStockDataFromUi()
{
    if (m_blockStockSignals)
    {
        return;
    }

    m_stock.margin_mm = lengthMmFromDisplay(m_stockMargin->value());

    if (m_stockOriginMode == StockOriginMode::Custom)
    {
        m_stock.sizeXYZ_mm.x = std::max(0.0, lengthMmFromDisplay(m_stockWidth->value()));
        m_stock.sizeXYZ_mm.y = std::max(0.0, lengthMmFromDisplay(m_stockLength->value()));
        m_stock.sizeXYZ_mm.z = std::max(0.0, lengthMmFromDisplay(m_stockHeight->value()));
        m_stock.originXYZ_mm.x = lengthMmFromDisplay(m_stockOriginX->value());
        m_stock.originXYZ_mm.y = lengthMmFromDisplay(m_stockOriginY->value());
        m_stock.originXYZ_mm.z = lengthMmFromDisplay(m_stockOriginZ->value());
        m_stock.topZ_mm = m_stock.originXYZ_mm.z + m_stock.sizeXYZ_mm.z;
        m_stock.ensureValid();
    }
}

void MainWindow::syncMachineDataFromUi()
{
    if (m_blockMachineSignals)
    {
        return;
    }

    m_machine.name = m_machineNameEdit->text().trimmed().toStdString();
    m_machine.rapidFeed_mm_min = feedMmPerMinFromDisplay(m_machineRapidFeed->value());
    m_machine.maxFeed_mm_min = feedMmPerMinFromDisplay(m_machineMaxFeed->value());
    m_machine.maxSpindleRPM = std::max(0.0, m_machineMaxSpindle->value());
    m_machine.clearanceZ_mm = lengthMmFromDisplay(m_machineClearanceZ->value());
    m_machine.safeZ_mm = lengthMmFromDisplay(m_machineSafeZ->value());
    m_machine.ensureValid();

    m_blockMachineSignals = true;
    syncMachineUiFromData();
    m_blockMachineSignals = false;
}

void MainWindow::syncPassControlsFromData()
{
    if (!m_stockAllowanceSpin || !m_rampAngleSpin || !m_enableRoughPassCheck || !m_enableFinishPassCheck)
    {
        return;
    }

    const QSignalBlocker allowanceBlocker(m_stockAllowanceSpin);
    const QSignalBlocker rampBlocker(m_rampAngleSpin);
    const QSignalBlocker roughBlocker(m_enableRoughPassCheck);
    const QSignalBlocker finishBlocker(m_enableFinishPassCheck);

    m_stockAllowanceSpin->setValue(lengthDisplayFromMm(m_stockAllowanceMm));
    m_rampAngleSpin->setValue(m_rampAngleDeg);
    m_enableRoughPassCheck->setChecked(m_enableRoughPassUser);
    m_enableFinishPassCheck->setChecked(m_enableFinishPassUser);
}

void MainWindow::updatePassControlUnits()
{
    if (!m_stockAllowanceSpin)
    {
        return;
    }

    QSignalBlocker allowanceBlocker(m_stockAllowanceSpin);
    m_stockAllowanceSpin->setSuffix(QStringLiteral(" %1").arg(common::unitSuffix(m_units)));
    m_stockAllowanceSpin->setSingleStep(lengthDisplayFromMm(0.05));
    m_stockAllowanceSpin->setRange(lengthDisplayFromMm(0.0), lengthDisplayFromMm(5.0));
    m_stockAllowanceSpin->setValue(lengthDisplayFromMm(m_stockAllowanceMm));

    if (m_rampAngleSpin)
    {
        QSignalBlocker rampBlocker(m_rampAngleSpin);
        m_rampAngleSpin->setSuffix(tr(" deg"));
        m_rampAngleSpin->setValue(m_rampAngleDeg);
    }
}

void MainWindow::applyPassPreferences(tp::UserParams& params) const
{
    const bool roughEnabled = m_enableRoughPassCheck ? m_enableRoughPassCheck->isChecked() : m_enableRoughPassUser;
    const bool finishEnabled = m_enableFinishPassCheck ? m_enableFinishPassCheck->isChecked() : m_enableFinishPassUser;

    params.enableRoughPass = roughEnabled;
void MainWindow::appendEnvLog(const QString& text)
{
    if (!m_envLog)
    {
        return;
    }
    m_envLog->appendPlainText(text);
}

void MainWindow::updateEnvironmentControls(bool busy)
{
    if (m_envPrepareButton)
    {
        m_envPrepareButton->setEnabled(!busy);
    }
    if (m_envCancelButton)
    {
        m_envCancelButton->setEnabled(busy);
    }
    updateTrainingActions();
}

void MainWindow::startEnvironmentPreparation()
{
    if (!m_envManager)
    {
        return;
    }
    if (m_envManager->isBusy())
    {
        return;
    }
    m_envReady = false;
    updateEnvironmentControls(true);
    if (m_envProgress)
    {
        m_envProgress->setValue(0);
    }
    if (m_envCpuOnlyCheck)
    {
        m_envManager->setCpuOnly(m_envCpuOnlyCheck->isChecked());
    }
    m_envManager->prepareEnvironment(false);
}
void MainWindow::cancelEnvironmentPreparation()
{
    if (!m_envManager)
    {
        return;
    }
    m_envManager->cancel();
    updateEnvironmentControls(false);
}

void MainWindow::onEnvProgress(int value)
{
    if (m_envProgress)
    {
        m_envProgress->setValue(value);
    }
}

void MainWindow::onEnvFinished(bool success)
{
    m_envReady = success;
    updateEnvironmentControls(false);
    if (m_envProgress && success)
    {
        m_envProgress->setValue(100);
    }
    if (statusBar())
    {
        statusBar()->showMessage(success ? tr("Training environment ready.") : tr("Environment not prepared."), 5000);
    }
}

void MainWindow::onEnvError(const QString& message)
{
    m_envReady = false;
    updateEnvironmentControls(false);
    if (m_envLog)
    {
        m_envLog->appendPlainText(message);
    }
    if (statusBar())
    {
        statusBar()->showMessage(message, 8000);
    }
}

void MainWindow::onGpuInfoChanged(const QString& info)
{
    if (m_envGpuLabel)
    {
        m_envGpuLabel->setText(info);
        m_envGpuLabel->setToolTip(info);
    }
}
void MainWindow::ensureTrainingManager()
{
    if (!m_trainingManager)
    {
        m_trainingManager = new train::TrainingManager(this);
        if (m_envManager)
        {
            m_trainingManager->setEnvManager(m_envManager);
        }

        connect(m_trainingManager,
                &train::TrainingManager::jobAdded,
                this,
                &MainWindow::onTrainingJobAdded);
        connect(m_trainingManager,
                &train::TrainingManager::jobUpdated,
                this,
                &MainWindow::onTrainingJobUpdated);
        connect(m_trainingManager,
                &train::TrainingManager::jobRemoved,
                this,
                &MainWindow::onTrainingJobRemoved);
        connect(m_trainingManager,
                &train::TrainingManager::jobLog,
                this,
                &MainWindow::onTrainingJobLog);
        connect(m_trainingManager,
                &train::TrainingManager::toastRequested,
                this,
                &MainWindow::onTrainingToast);
        connect(m_trainingManager,
                &train::TrainingManager::modelRegistered,
                this,
                &MainWindow::onTrainingModelRegistered);
    }
    else if (m_envManager)
    {
        m_trainingManager->setEnvManager(m_envManager);
    }

    if (m_trainingManager && m_modelManager)
    {
        m_trainingManager->setModelManager(m_modelManager.get());
    }
}

bool MainWindow::isGpuAvailableForTraining() const
{
    if (!m_envManager)
    {
        return false;
    }
    const QString summary = m_envManager->gpuSummary();
    if (summary.isEmpty())
    {
        return false;
    }
    return !summary.contains(QStringLiteral("none"), Qt::CaseInsensitive);
}

void MainWindow::updateTrainingActions()
{
    const bool hasManager = m_trainingManager != nullptr;
    const bool envBusy = m_envManager && m_envManager->isBusy();
    const bool ready = hasManager && m_envReady && !envBusy;

    if (m_trainingNewModelAction)
    {
        m_trainingNewModelAction->setEnabled(ready);
    }
    if (m_trainingSyntheticAction)
    {
        m_trainingSyntheticAction->setEnabled(ready);
    }
    if (m_trainingFineTuneAction)
    {
        const bool hasActiveModel = !m_aiModelPath.isEmpty();
        m_trainingFineTuneAction->setEnabled(ready && hasActiveModel);
    }
    if (m_trainingOpenModelsAction)
    {
        m_trainingOpenModelsAction->setEnabled(hasManager);
    }
    if (m_trainingOpenDatasetsAction)
    {
        m_trainingOpenDatasetsAction->setEnabled(hasManager);
    }
}

void MainWindow::openTrainingNewModelDialog()
{
    ensureTrainingManager();
    if (!m_trainingManager)
    {
        return;
    }

    const QVector<ai::ModelDescriptor>& models = m_modelManager ? m_modelManager->models() : QVector<ai::ModelDescriptor>{};
    TrainingNewModelDialog dialog(models, isGpuAvailableForTraining(), this);
    dialog.setWindowTitle(tr("New Model"));
    if (dialog.exec() != QDialog::Accepted)
    {
        return;
    }

    train::TrainingManager::TrainJobRequest request;
    request.modelName = dialog.modelName();
    request.datasetPath = dialog.datasetPath();
    request.baseModelPath = dialog.baseModelPath();
    request.device = dialog.device();
    request.epochs = dialog.epochs();
    request.learningRate = dialog.learningRate();
    request.useV2Features = dialog.useV2Features();
    request.fineTune = false;

    if (!request.datasetPath.isEmpty())
    {
        request.datasetPath = QDir::toNativeSeparators(request.datasetPath);
    }
    if (!request.baseModelPath.isEmpty())
    {
        request.baseModelPath = QDir::toNativeSeparators(request.baseModelPath);
    }

    m_trainingManager->enqueueTraining(request);
    if (statusBar())
    {
        statusBar()->showMessage(tr("Queued training job: %1").arg(request.modelName), 5000);
    }
}

void MainWindow::openSyntheticDataDialog()
{
    ensureTrainingManager();
    if (!m_trainingManager)
    {
        return;
    }

    TrainingSyntheticDataDialog dialog(m_trainingManager->datasetsRoot(), this);
    if (dialog.exec() != QDialog::Accepted)
    {
        return;
    }

    train::TrainingManager::SyntheticJobRequest request;
    request.label = dialog.datasetLabel();
    request.sampleCount = dialog.sampleCount();
    request.diversity = dialog.diversity();
    request.slopeMix = dialog.slopeMix();
    request.overwrite = dialog.overwriteExisting();
    request.outputDir = dialog.outputDirectory();

    m_trainingManager->enqueueSyntheticDataset(request);
    if (statusBar())
    {
        statusBar()->showMessage(tr("Queued dataset generation: %1").arg(request.label), 5000);
    }
}

void MainWindow::fineTuneCurrentModel()
{
    ensureTrainingManager();
    if (!m_trainingManager)
    {
        return;
    }
    if (m_aiModelPath.isEmpty())
    {
        onTrainingToast(tr("Select an AI model before fine-tuning."));
        return;
    }

    const QString datasetRoot = m_trainingManager->datasetsRoot();
    const QString datasetSelection =
        QFileDialog::getExistingDirectory(this, tr("Select Dataset Folder"), datasetRoot);
    if (datasetSelection.isEmpty())
    {
        return;
    }

    const QVector<ai::ModelDescriptor>& models = m_modelManager ? m_modelManager->models() : QVector<ai::ModelDescriptor>{};
    TrainingNewModelDialog dialog(models, isGpuAvailableForTraining(), this);
    dialog.setWindowTitle(tr("Fine-Tune Current Model"));
    dialog.lockBaseModel(m_aiModelPath);
    dialog.setSuggestedName(QFileInfo(m_aiModelPath).completeBaseName() + QStringLiteral("_ft"));
    dialog.setSuggestedDataset(datasetSelection);
    if (dialog.exec() != QDialog::Accepted)
    {
        return;
    }

    train::TrainingManager::TrainJobRequest request;
    request.modelName = dialog.modelName();
    request.datasetPath = dialog.datasetPath();
    if (request.datasetPath.isEmpty())
    {
        request.datasetPath = datasetSelection;
    }
    if (request.datasetPath.isEmpty())
    {
        onTrainingToast(tr("Select a dataset folder before fine-tuning."));
        return;
    }
    request.datasetPath = QDir::toNativeSeparators(request.datasetPath);
    request.baseModelPath = QDir::toNativeSeparators(dialog.baseModelPath());
    request.device = dialog.device();
    request.epochs = dialog.epochs();
    request.learningRate = dialog.learningRate();
    request.useV2Features = dialog.useV2Features();
    request.fineTune = true;

    m_trainingManager->enqueueTraining(request);
    if (statusBar())
    {
        statusBar()->showMessage(tr("Queued fine-tune job: %1").arg(request.modelName), 5000);
    }
}\r\n
void MainWindow::openModelsFolder()
{
    ensureTrainingManager();
    if (!m_trainingManager)
    {
        return;
    }
    const QString path = m_trainingManager->modelsRoot();
    if (!QFileInfo::exists(path))
    {
        QDir().mkpath(path);
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

void MainWindow::openDatasetsFolder()
{
    ensureTrainingManager();
    if (!m_trainingManager)
    {
        return;
    }
    const QString path = m_trainingManager->datasetsRoot();
    if (!QFileInfo::exists(path))
    {
        QDir().mkpath(path);
    }
    QDesktopServices::openUrl(QUrl::fromLocalFile(path));
}

void MainWindow::onTrainingJobAdded(const train::TrainingManager::JobStatus& status)
{
    ensureTrainingManager();
    if (!m_jobsList)
    {
        return;
    }

    JobWidgets widgets;
    widgets.item = new QListWidgetItem(m_jobsList);
    widgets.container = new QWidget(m_jobsList);

    auto* layout = new QVBoxLayout(widgets.container);
    layout->setContentsMargins(8, 6, 8, 6);
    layout->setSpacing(4);

    widgets.title = new QLabel(summarizeJob(status), widgets.container);
    widgets.title->setObjectName(QStringLiteral("JobTitle"));
    layout->addWidget(widgets.title);

    widgets.subtitle = new QLabel(status.detail, widgets.container);
    widgets.subtitle->setObjectName(QStringLiteral("JobSubtitle"));
    widgets.subtitle->setWordWrap(true);
    layout->addWidget(widgets.subtitle);

    widgets.progress = new QProgressBar(widgets.container);
    widgets.progress->setRange(0, 100);
    widgets.progress->setTextVisible(true);
    layout->addWidget(widgets.progress);

    auto* footer = new QHBoxLayout();
    footer->setContentsMargins(0, 0, 0, 0);
    footer->setSpacing(8);

    widgets.status = new QLabel(jobStateLabel(status.state), widgets.container);
    footer->addWidget(widgets.status);

    widgets.eta = new QLabel(jobEtaLabel(status.etaMs), widgets.container);
    footer->addWidget(widgets.eta, 1);

    widgets.cancel = new QPushButton(tr("Cancel"), widgets.container);
    footer->addWidget(widgets.cancel);
    layout->addLayout(footer);

    widgets.container->setLayout(layout);

    widgets.item->setSizeHint(widgets.container->sizeHint());
    widgets.item->setData(Qt::UserRole, status.id.toString(QUuid::WithoutBraces));
    m_jobsList->setItemWidget(widgets.item, widgets.container);

    widgets.snapshot = status;
    m_jobWidgets.insert(status.id, widgets);
    m_jobLogs[status.id];

    connect(widgets.cancel, &QPushButton::clicked, this, [this, jobId = status.id]() {
        requestJobCancellation(jobId);
    });

    onTrainingJobUpdated(status);

    if (m_jobsList->count() == 1)
    {
        m_jobsList->setCurrentItem(widgets.item);
    }
}

void MainWindow::onTrainingJobUpdated(const train::TrainingManager::JobStatus& status)
{
    auto it = m_jobWidgets.find(status.id);
    if (it == m_jobWidgets.end())
    {
        onTrainingJobAdded(status);
        return;
    }

    JobWidgets& widgets = it.value();
    widgets.snapshot = status;

    if (widgets.title)
    {
        widgets.title->setText(summarizeJob(status));
    }
    if (widgets.subtitle)
    {
        widgets.subtitle->setText(status.detail);
    }
    if (widgets.progress)
    {
        if (status.progress >= 0)
        {
            widgets.progress->setFormat(QStringLiteral("%p%"));
            widgets.progress->setValue(std::clamp(status.progress, 0, 100));
        }
        else
        {
            widgets.progress->setFormat(jobStateLabel(status.state));
            widgets.progress->setValue(status.state == train::TrainingManager::JobState::Succeeded ? 100 : 0);
        }
    }
    if (widgets.status)
    {
        widgets.status->setText(jobStateLabel(status.state));
    }
    if (widgets.eta)
    {
        widgets.eta->setText(jobEtaLabel(status.etaMs));
    }
    if (widgets.cancel)
    {
        const bool cancellable = status.state == train::TrainingManager::JobState::Queued
                                 || status.state == train::TrainingManager::JobState::Running;
        widgets.cancel->setEnabled(cancellable);
    }

    if (m_selectedJob == status.id)
    {
        updateJobLogView();
    }
}

void MainWindow::onTrainingJobRemoved(const QUuid& id)
{
    auto it = m_jobWidgets.find(id);
    if (it == m_jobWidgets.end())
    {
        return;
    }

    JobWidgets widgets = it.value();
    m_jobWidgets.remove(id);
    m_jobLogs.remove(id);

    if (widgets.item)
    {
        delete widgets.item;
    }

    if (m_selectedJob == id)
    {
        m_selectedJob = {};
        updateJobLogView();
    }
}

void MainWindow::onTrainingJobLog(const QUuid& id, const QString& text)
{
    if (text.isEmpty())
    {
        return;
    }

    QStringList lines = text.split(QRegularExpression(QStringLiteral("[\r\n]+")), Qt::SkipEmptyParts);
    if (lines.isEmpty())
    {
        return;
    }

    QStringList& buffer = m_jobLogs[id];
    for (const QString& line : lines)
    {
        const QString trimmed = line.trimmed();
        if (trimmed.isEmpty())
        {
            continue;
        }
        const QString entry = formatTimestamped(trimmed);
        buffer.append(entry);
        if (buffer.size() > 4000)
        {
            buffer.removeFirst();
        }
        if (m_selectedJob == id && m_jobLog)
        {
            m_jobLog->appendPlainText(entry);
        }
    }
}

void MainWindow::onTrainingToast(const QString& message)
{
    if (statusBar())
    {
        statusBar()->showMessage(message, 6000);
    }
    logMessage(message);
}

void MainWindow::onTrainingModelRegistered(const QString& path)
{
    refreshAiModels();
    if (!path.isEmpty())
    {
        setActiveAiModel(QDir::toNativeSeparators(path), true);
        updateTrainingActions();
        if (statusBar())
        {
            statusBar()->showMessage(tr("Loaded new model: %1").arg(QFileInfo(path).fileName()), 6000);
        }
    }
}

void MainWindow::onJobSelectionChanged()
{
    if (!m_jobsList)
    {
        return;
    }

    const QListWidgetItem* item = m_jobsList->currentItem();
    if (!item)
    {
        m_selectedJob = {};
        if (m_jobSelectionLabel)
        {
            m_jobSelectionLabel->setText(tr("Select a job to view logs."));
        }
        updateJobLogView();
        return;
    }

    const QUuid id(item->data(Qt::UserRole).toString());
    m_selectedJob = id;

    if (m_jobSelectionLabel)
    {
        auto it = m_jobWidgets.find(id);
        const QString title = (it != m_jobWidgets.end()) ? summarizeJob(it.value().snapshot)
                                                          : tr("Job");
        m_jobSelectionLabel->setText(tr("Logs � %1").arg(title));
    }

    updateJobLogView();
}

void MainWindow::updateJobLogView()
{
    if (!m_jobLog)
    {
        return;
    }

    m_jobLog->clear();
    if (m_selectedJob.isNull())
    {
        return;
    }

    const QStringList logs = m_jobLogs.value(m_selectedJob);
    if (logs.isEmpty())
    {
        return;
    }

    for (const QString& line : logs)
    {
        m_jobLog->appendPlainText(line);
    }
}

void MainWindow::requestJobCancellation(const QUuid& id)
{
    if (!m_trainingManager)
    {
        return;
    }
    m_trainingManager->cancelJob(id);
}

QString MainWindow::summarizeJob(const train::TrainingManager::JobStatus& status) const
{
    return QStringLiteral("%1 � %2").arg(jobTypeLabel(status.type), status.label);
}

QString MainWindow::jobStateLabel(train::TrainingManager::JobState state) const
{
    switch (state)
    {
    case train::TrainingManager::JobState::Queued:
        return tr("Queued");
    case train::TrainingManager::JobState::Running:
        return tr("Running");
    case train::TrainingManager::JobState::Succeeded:
        return tr("Done");
    case train::TrainingManager::JobState::Failed:
        return tr("Failed");
    case train::TrainingManager::JobState::Cancelled:
        return tr("Cancelled");
    }
    return tr("Unknown");
}

QString MainWindow::jobEtaLabel(qint64 etaMs) const
{
    if (etaMs < 0)
    {
        return tr("ETA --");
    }
    if (etaMs == 0)
    {
        return tr("ETA 00:00");
    }

    const qint64 totalSeconds = (etaMs + 500) / 1000;
    const qint64 minutes = totalSeconds / 60;
    const qint64 seconds = totalSeconds % 60;
    if (minutes >= 60)
    {
        const qint64 hours = minutes / 60;
        const qint64 remMinutes = minutes % 60;
        return tr("ETA %1h %2m").arg(hours).arg(remMinutes, 2, 10, QLatin1Char('0'));
    }
    return tr("ETA %1:%2").arg(minutes, 2, 10, QLatin1Char('0')).arg(seconds, 2, 10, QLatin1Char('0'));
}

QString MainWindow::jobTypeLabel(train::TrainingManager::JobType type) const
{
    switch (type)
    {
    case train::TrainingManager::JobType::SyntheticDataset:
        return tr("Synthetic Data");
    case train::TrainingManager::JobType::Train:
        return tr("Training");
    case train::TrainingManager::JobType::FineTune:
        return tr("Fine-Tune");
    }
    return tr("Job");
}    params.enableFinishPass = finishEnabled;
    params.stockAllowance_mm = std::max(0.0, m_stockAllowanceMm);
    params.rampAngleDeg = std::clamp(m_rampAngleDeg, 0.5, 45.0);
}

void MainWindow::updateStockDerivedFromModel()
{
    if (m_stockOriginMode == StockOriginMode::Custom)
    {
        return;
    }
    if (!m_currentModel || !m_currentModel->isValid())
    {
        return;
    }

    const auto bounds = m_currentModel->bounds();
    const double margin = std::max(0.0, m_stock.margin_mm);

    const glm::dvec3 min(bounds.min.x(), bounds.min.y(), bounds.min.z());
    const glm::dvec3 max(bounds.max.x(), bounds.max.y(), bounds.max.z());
    glm::dvec3 size = glm::dvec3(max - min);
    size.x = std::max(size.x, 0.0);
    size.y = std::max(size.y, 0.0);
    size.z = std::max(size.z, 0.0);

    const glm::dvec3 marginVec(margin, margin, margin);
    size += marginVec * 2.0;

    glm::dvec3 origin{0.0, 0.0, 0.0};
    switch (m_stockOriginMode)
    {
    case StockOriginMode::ModelMin:
        origin = min - marginVec;
        break;
    case StockOriginMode::ModelCenter:
    {
        const glm::dvec3 center(bounds.center().x(), bounds.center().y(), bounds.center().z());
        origin = center - size * 0.5;
        break;
    }
    case StockOriginMode::Custom:
        return;
    }

    m_stock.sizeXYZ_mm = size;
    m_stock.originXYZ_mm = origin;
    m_stock.topZ_mm = origin.z + size.z;
    m_stock.ensureValid();
}

void MainWindow::updateStockUiEnabledState()
{
    if (!m_stockWidth)
    {
        return;
    }
    const bool custom = (m_stockOriginMode == StockOriginMode::Custom);

    m_stockWidth->setEnabled(custom);
    m_stockLength->setEnabled(custom);
    m_stockHeight->setEnabled(custom);
    m_stockOriginX->setEnabled(custom);
    m_stockOriginY->setEnabled(custom);
    m_stockOriginZ->setEnabled(custom);
}

void MainWindow::applyMachinePreset(const tp::Machine& machine)
{
    m_machine = machine;
    m_machine.ensureValid();
    syncMachineUiFromData();
}

double MainWindow::lengthDisplayFromMm(double valueMm) const
{
    return common::fromMillimeters(valueMm, m_units);
}

double MainWindow::lengthMmFromDisplay(double value) const
{
    return (m_units == common::Unit::Millimeters)
               ? value
               : common::toMillimeters(value, m_units);
}

double MainWindow::feedDisplayFromMmPerMin(double valueMmPerMin) const
{
    return common::fromMillimeters(valueMmPerMin, m_units);
}

double MainWindow::feedMmPerMinFromDisplay(double value) const
{
    return (m_units == common::Unit::Millimeters)
               ? value
               : common::toMillimeters(value, m_units);
}

void MainWindow::loadToolLibrary()
{
    if (!m_toolpathSettings)
    {
        return;
    }

    QStringList warnings;
    QByteArray data;
    bool loaded = false;

    const QString userPath = QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("tools.json"));
    QFile userFile(userPath);
    if (userFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        data = userFile.readAll();
        loaded = m_toolLibrary.loadFromJson(data, warnings);
    }

    if (!loaded)
    {
        for (const QString& warning : warnings)
        {
            logWarning(warning);
        }
        warnings.clear();

        QFile resourceFile(QStringLiteral(":/app/tools.json"));
        if (resourceFile.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            data = resourceFile.readAll();
            loaded = m_toolLibrary.loadFromJson(data, warnings);
        }
    }

    if (!loaded)
    {
        for (const QString& warning : warnings)
        {
            logWarning(warning);
        }
        logWarning(tr("Tool library not found; using manual parameters."));
        return;
    }

    for (const QString& warning : warnings)
    {
        logWarning(warning);
    }

    m_toolpathSettings->setToolLibrary(m_toolLibrary.tools(), m_units);
}

void MainWindow::refreshAiModels()
{
    if (!m_modelManager)
    {
        m_modelManager = std::make_unique<ai::ModelManager>();
    }

    ensureTrainingManager();

    if (!m_aiModelCombo)
    {
        return;
    }

    m_modelManager->refresh();

    QSignalBlocker blocker(m_aiModelCombo);
    m_aiModelCombo->clear();
    const QString defaultLabel = tr("Default (built-in) [Torch]");
    m_aiModelCombo->addItem(defaultLabel, QString());
    m_aiModelCombo->setItemData(0, tr("Built-in TorchAI fallback"), Qt::ToolTipRole);

    int indexToSelect = 0;
    const auto& models = m_modelManager->models();
    for (int i = 0; i < models.size(); ++i)
    {
        const auto& model = models.at(i);
        const QString badge = backendBadge(model.backend);
        const QString display = QStringLiteral("%1 %2").arg(model.fileName, badge);
        m_aiModelCombo->addItem(display, model.absolutePath);
        const int comboIndex = i + 1;
        m_aiModelCombo->setItemData(comboIndex, model.absolutePath, Qt::ToolTipRole);
        if (!m_aiModelPath.isEmpty() && model.absolutePath.compare(m_aiModelPath, Qt::CaseInsensitive) == 0)
        {
            indexToSelect = i + 1;
        }
    }

    m_aiModelCombo->setCurrentIndex(indexToSelect);

    const QString currentPath = m_aiModelCombo->currentData().toString();
    setActiveAiModel(currentPath, true);
}

void MainWindow::startImportWorker(const QString& path)
{
    if (m_importWorker)
    {
        logWarning(tr("Import already in progress."));
        return;
    }

    m_importWorker = new io::ImportWorker(path, this);

    m_importProgress = new QProgressDialog(tr("Importing %1...").arg(QFileInfo(path).fileName()),
                                           tr("Cancel"),
                                           0,
                                           100,
                                           this);
    m_importProgress->setWindowModality(Qt::ApplicationModal);
    m_importProgress->setAutoClose(false);
    m_importProgress->setAutoReset(false);
    m_importProgress->setMinimumDuration(0);
    m_importProgress->setValue(0);

    connect(m_importProgress, &QProgressDialog::canceled, m_importWorker, &io::ImportWorker::requestCancel);
    connect(m_importWorker, &io::ImportWorker::progress, m_importProgress, &QProgressDialog::setValue);

    m_importTimer.start();
    logMessage(tr("Import started: %1").arg(path));

    m_importProgress->show();

    connect(m_importWorker,
            &io::ImportWorker::finished,
            this,
            [this, path](std::shared_ptr<render::Model> model) {
                const qint64 elapsed = m_importTimer.elapsed();
                if (m_importProgress)
                {
                    m_importProgress->setValue(100);
                }
                if (!model || !model->isValid())
                {
                    logWarning(tr("Imported model is invalid."));
                    QMessageBox::warning(this, tr("Import Failed"), tr("Model is empty or invalid."));
                    cleanupImport();
                    return;
                }

                m_currentModel = std::move(model);
                m_currentModelPath = path;
                m_lastModelDirectory = QFileInfo(path).absolutePath();
                  m_viewer->setModel(m_currentModel);
                  m_viewer->resetCamera();

                  if (m_stockOriginMode != StockOriginMode::Custom)
                  {
                      updateStockDerivedFromModel();
                  }
                  syncStockUiFromData();

                  m_currentToolpath.reset();
                m_viewer->setToolpath(nullptr);
                if (m_simulation)
                {
                    m_simulation->stop();
                    m_simulation->setToolpath(nullptr);
                    onSimulationStateChanged(m_simulation->state());
                    onSimulationProgressChanged(0.0);
                }

                updateModelBrowser();

                logMessage(tr("Import finished in %1 ms.").arg(elapsed));
                displayToolpathMessage(tr("Loaded model: %1").arg(QFileInfo(path).fileName()));

                cleanupImport();
                saveSettings();
            });

    connect(m_importWorker,
            &io::ImportWorker::error,
            this,
            [this](const QString& message) {
                const QString cancelledText = tr("Import cancelled.");
                if (message.compare(cancelledText, Qt::CaseInsensitive) == 0)
                {
                    logMessage(cancelledText);
                }
                else
                {
                    logWarning(message);
                    QMessageBox::warning(this, tr("Import Failed"), message);
                    displayToolpathMessage(message);
                }
                cleanupImport();
            });

    connect(m_importWorker, &QThread::finished, this, [this]() {
        if (m_importWorker)
        {
            m_importWorker->deleteLater();
            m_importWorker = nullptr;
        }
    });

    m_importWorker->start();
}

void MainWindow::startGenerateWorker(const tp::UserParams& settings)
{
    if (m_generateWorker)
    {
        logWarning(tr("Toolpath generation already in progress."));
        return;
    }

    m_lastUserParams = settings;
    if (m_simulation)
    {
        m_simulation->stop();
        m_simulation->setToolDiameter(settings.toolDiameter);
        m_simulation->setToolpath(nullptr);
        onSimulationStateChanged(render::SimulationController::State::Stopped);
    }

    std::unique_ptr<ai::IPathAI> aiInstance;
    if (m_modelManager)
    {
        aiInstance = m_modelManager->createModel(m_aiModelPath);
    }
    if (!aiInstance)
    {
        aiInstance = std::make_unique<ai::TorchAI>(std::filesystem::path());
    }

    applyAiOverrides(aiInstance.get());

    m_generateWorker = new tp::GenerateWorker(m_currentModel, settings, std::move(aiInstance), this);

    m_generateProgress = new QProgressDialog(tr("Generating toolpath..."), tr("Cancel"), 0, 100, this);
    m_generateProgress->setWindowModality(Qt::ApplicationModal);
    m_generateProgress->setAutoClose(false);
    m_generateProgress->setAutoReset(false);
    m_generateProgress->setMinimumDuration(0);
    m_generateProgress->setValue(0);

    connect(m_generateProgress, &QProgressDialog::canceled, m_generateWorker, &tp::GenerateWorker::requestCancel);
    connect(m_generateWorker, &tp::GenerateWorker::progress, m_generateProgress, &QProgressDialog::setValue);
    connect(m_generateWorker,
            &tp::GenerateWorker::banner,
            this,
            [this](const QString& message) {
                if (!message.isEmpty())
                {
                    logMessage(message);
                }
            });

    m_generateTimer.start();
    logMessage(tr("Toolpath generation started."));

    m_generateProgress->show();

    connect(m_generateWorker,
            &tp::GenerateWorker::finished,
            this,
            [this](std::shared_ptr<tp::Toolpath> toolpath, ai::StrategyDecision decision) {
                const qint64 elapsed = m_generateTimer.elapsed();

                if (m_generateProgress)
                {
                    m_generateProgress->setValue(100);
                }

                cleanupGeneration();

                if (!toolpath || toolpath->passes.empty())
                {
                    if (m_simulation)
                    {
                        m_simulation->setToolpath(nullptr);
                        onSimulationStateChanged(render::SimulationController::State::Stopped);
                        onSimulationProgressChanged(0.0);
                    }
                    displayToolpathMessage(tr("Toolpath generation returned no moves."));
                    return;
                }

                const bool hadToolpath = m_currentToolpath && !m_currentToolpath->empty();
                m_currentToolpath = std::move(toolpath);
                m_viewer->setToolpath(m_currentToolpath);

                if (!hadToolpath)
                {
                    m_viewer->resetCamera();
                }

                updateModelBrowser();

                const int segmentCount = static_cast<int>(m_currentToolpath->passes.size());
                const double feedDisplay = common::fromMillimeters(m_currentToolpath->feed, m_units);
                const QString feedText = QLocale().toString(feedDisplay, 'f', 1);
                const QString rpmText = QLocale().toString(m_currentToolpath->spindle, 'f', 0);
                displayToolpathMessage(tr("Generated toolpath with %1 segments at %2 %3, %4 RPM.")
                                           .arg(segmentCount)
                                           .arg(feedText)
                                           .arg(common::feedSuffix(m_units))
                                           .arg(rpmText));

                const QString strategyName = (decision.strat == ai::StrategyDecision::Strategy::Raster)
                                                 ? tr("Raster")
                                                 : tr("Waterline");
                const QString angleText = QLocale().toString(decision.rasterAngleDeg, 'f', 1);
                const QString stepText = QLocale().toString(decision.stepOverMM, 'f', 3);
                logMessage(tr("Toolpath generated in %1 ms.").arg(elapsed));
                logMessage(tr("AI: Using %1, decision: %2 angle=%3 deg, step-over=%4 mm")
                               .arg(m_aiModelLabel)
                               .arg(strategyName)
                               .arg(angleText)
                               .arg(stepText));

                if (m_simulation)
                {
                    m_simulation->setToolDiameter(m_lastUserParams.toolDiameter);
                    m_simulation->setToolpath(m_currentToolpath);
                    onSimulationStateChanged(m_simulation->state());
                    onSimulationProgressChanged(m_simulation->progress());
                }

                saveSettings();
            });

    connect(m_generateWorker,
            &tp::GenerateWorker::error,
            this,
            [this](const QString& message) {
                const QString cancelledText = tr("Toolpath generation cancelled.");
                if (message.compare(cancelledText, Qt::CaseInsensitive) == 0)
                {
                    logMessage(cancelledText);
                }
                else
                {
                    logWarning(message);
                }
                displayToolpathMessage(message);
                if (m_simulation)
                {
                    if (m_currentToolpath && !m_currentToolpath->empty())
                    {
                        m_simulation->setToolpath(m_currentToolpath);
                    }
                    else
                    {
                        m_simulation->setToolpath(nullptr);
                        onSimulationProgressChanged(0.0);
                    }
                    onSimulationStateChanged(m_simulation->state());
                }
                cleanupGeneration();
            });

    connect(m_generateWorker, &QThread::finished, this, [this]() {
        if (m_generateWorker)
        {
            m_generateWorker->deleteLater();
            m_generateWorker = nullptr;
        }
    });

    m_generateWorker->start();
}

void MainWindow::cleanupImport()
{
    if (m_importProgress)
    {
        m_importProgress->hide();
        m_importProgress->deleteLater();
        m_importProgress = nullptr;
    }

    if (m_importWorker)
    {
        m_importWorker->deleteLater();
        m_importWorker = nullptr;
    }
}

void MainWindow::cleanupGeneration()
{
    if (m_generateProgress)
    {
        m_generateProgress->hide();
        m_generateProgress->deleteLater();
        m_generateProgress = nullptr;
    }

    if (m_generateWorker)
    {
        m_generateWorker->deleteLater();
        m_generateWorker = nullptr;
    }
}

bool MainWindow::setActiveAiModel(const QString& path, bool quiet)
{
    if (!m_modelManager)
    {
        m_modelManager = std::make_unique<ai::ModelManager>();
    }

    QString normalizedPath = path;
    std::unique_ptr<ai::IPathAI> prototype;
    const bool selectingDefault = normalizedPath.isEmpty();

    if (selectingDefault)
    {
        prototype = m_modelManager->createModel(QString());
    }
    else
    {
        prototype = m_modelManager->createModel(normalizedPath);
        if (!prototype)
        {
            if (!quiet)
            {
                logWarning(tr("Unable to create AI instance for %1").arg(normalizedPath));
            }
            return setActiveAiModel(QString(), quiet);
        }
    }

    if (m_aiModelCombo)
    {
        int targetIndex = 0;
        if (!selectingDefault)
        {
            targetIndex = m_aiModelCombo->findData(normalizedPath);
            if (targetIndex < 0)
            {
                if (!quiet)
                {
                    logWarning(tr("AI model not found in list: %1").arg(normalizedPath));
                }
                return setActiveAiModel(QString(), true);
            }
        }
        QSignalBlocker blocker(m_aiModelCombo);
        m_aiModelCombo->setCurrentIndex(targetIndex);
    }

    m_aiModelPath = selectingDefault ? QString() : normalizedPath;

    if (!prototype)
    {
        prototype = m_modelManager->createModel(QString());
        m_aiModelPath.clear();
    }

    applyAiOverrides(prototype.get());
    m_activeAiPrototype = std::move(prototype);

    const QString error = runtimeLastError(m_activeAiPrototype.get());
    if (!error.isEmpty())
    {
        logWarning(tr("%1 runtime error: %2").arg(runtimeBadge(m_activeAiPrototype.get()), error));
    }

    updateActiveAiSummary();
    updateTrainingActions();

    if (!quiet)
    {
        logMessage(tr("AI model set to %1").arg(m_aiModelLabel));
    }

    return true;
}

void MainWindow::applyAiOverrides(ai::IPathAI* ai) const
{
    setRuntimeForceCpu(ai, m_forceCpuInference);
}

void MainWindow::updateActiveAiSummary()
{
    const QString baseName = m_aiModelPath.isEmpty() ? tr("Default") : QFileInfo(m_aiModelPath).fileName();
    const ai::IPathAI* runtime = m_activeAiPrototype.get();
    const QString badge = runtimeBadge(runtime);
    const QString deviceText = runtime ? runtimeDevice(runtime) : tr("CPU");
    m_aiModelLabel = QStringLiteral("%1 %2 (%3)").arg(baseName, badge, deviceText);
    if (m_aiDeviceLabel)
    {
        m_aiDeviceLabel->setText(tr("Device: %1").arg(deviceText));
        m_aiDeviceLabel->setToolTip(tr("Runtime device in use: %1").arg(deviceText));
    }
    updateStatusAiLabel();
}

void MainWindow::onSimulationProgressChanged(double normalized)
{
    if (!m_simProgressSlider)
    {
        return;
    }

    const int max = m_simProgressSlider->maximum();
    if (max <= 0)
    {
        return;
    }

    const int value = std::clamp(static_cast<int>(normalized * max), 0, max);
    if (!m_simSliderPressed)
    {
        m_updatingSimSlider = true;
        m_simProgressSlider->setValue(value);
        m_updatingSimSlider = false;
    }
}

void MainWindow::onSimulationStateChanged(render::SimulationController::State state)
{
    if (!m_simulationToolbar)
    {
        return;
    }

    const bool hasPath = m_simulation && m_simulation->hasPath();

    if (m_simPlayAction)
    {
        m_simPlayAction->setEnabled(hasPath && state != render::SimulationController::State::Playing);
    }
    if (m_simPauseAction)
    {
        m_simPauseAction->setEnabled(hasPath && state == render::SimulationController::State::Playing);
    }
    if (m_simStopAction)
    {
        m_simStopAction->setEnabled(hasPath && state != render::SimulationController::State::Stopped);
    }
    if (m_simProgressSlider)
    {
        m_simProgressSlider->setEnabled(hasPath);
    }
    if (m_simSpeedSlider)
    {
        m_simSpeedSlider->setEnabled(hasPath);
    }
}

void MainWindow::applySimulationSlider(double normalized)
{
    if (!m_simulation || m_updatingSimSlider || !m_simulation->hasPath())
    {
        return;
    }
    m_simulation->setProgress(std::clamp(normalized, 0.0, 1.0));
}

bool MainWindow::ensureActiveAiPrototype()
{
    if (m_activeAiPrototype)
    {
        applyAiOverrides(m_activeAiPrototype.get());
        updateActiveAiSummary();
        return true;
    }

    if (!m_modelManager)
    {
        m_modelManager = std::make_unique<ai::ModelManager>();
    }

    m_activeAiPrototype = m_modelManager->createModel(m_aiModelPath);
    if (!m_activeAiPrototype)
    {
        m_activeAiPrototype = m_modelManager->createModel(QString());
        m_aiModelPath.clear();
    }

    applyAiOverrides(m_activeAiPrototype.get());
    updateActiveAiSummary();
    return static_cast<bool>(m_activeAiPrototype);
}

void MainWindow::openAiPreferences()
{
    if (!ensureActiveAiPrototype())
    {
        logWarning(tr("AI runtime is unavailable in this build."));
        return;
    }

    AiPreferencesDialog dialog(this);
    dialog.setForceCpu(m_forceCpuInference);

    const bool previousForceCpu = m_forceCpuInference;
    setRuntimeForceCpu(m_activeAiPrototype.get(), dialog.forceCpu());

    updateAiPreferencesDialog(dialog);

    connect(&dialog,
            &AiPreferencesDialog::forceCpuChanged,
            this,
            [this, &dialog](bool checked) {
                setRuntimeForceCpu(m_activeAiPrototype.get(), checked);
                updateAiPreferencesDialog(dialog);
            });

    connect(&dialog, &AiPreferencesDialog::testRequested, this, [this, &dialog]() {
        handleAiTestRequest(dialog);
    });

    if (dialog.exec() == QDialog::Accepted)
    {
        const bool newForceCpu = dialog.forceCpu();
        if (newForceCpu != m_forceCpuInference)
        {
            m_forceCpuInference = newForceCpu;
            setRuntimeForceCpu(m_activeAiPrototype.get(), newForceCpu);
            const QString message = newForceCpu ? tr("AI inference forced to CPU.")
                                                : tr("AI inference will use CUDA when available.");
            logMessage(message);
        }
        applyAiOverrides(m_activeAiPrototype.get());
        updateActiveAiSummary();
        saveSettings();
    }
    else
    {
        setRuntimeForceCpu(m_activeAiPrototype.get(), previousForceCpu);
        applyAiOverrides(m_activeAiPrototype.get());
        updateActiveAiSummary();
    }
}

void MainWindow::updateAiPreferencesDialog(AiPreferencesDialog& dialog)
{
    QString pathDisplay = m_aiModelPath;
    QString modifiedText;
    if (m_aiModelPath.isEmpty())
    {
        pathDisplay = tr("(default)");
        modifiedText = tr("Built-in");
    }
    else
    {
        QFileInfo info(m_aiModelPath);
        modifiedText = info.exists() ? QLocale().toString(info.lastModified(), QLocale::ShortFormat)
                                     : tr("Not found");
    }

    const ai::IPathAI* runtime = m_activeAiPrototype.get();
    const QString deviceText = runtime ? runtimeDevice(runtime) : tr("Unavailable");
    QString statusText;
    const QString lastError = runtimeLastError(runtime);
    const bool loaded = runtime ? runtimeLoaded(runtime) : false;

    if (!runtime)
    {
        statusText = tr("AI runtime not available.");
    }
    else if (!loaded)
    {
        statusText = lastError.isEmpty()
                         ? tr("Model not loaded. Using fallback strategy.")
                         : lastError;
    }
    else if (!lastError.isEmpty())
    {
        statusText = lastError;
    }

    dialog.setModelInfo(m_aiModelLabel,
                        pathDisplay,
                        deviceText,
                        modifiedText);
    dialog.setLastTestResult(QStringLiteral("-"));

    const bool canTest = m_currentModel && runtime && loaded;
    dialog.setTestEnabled(canTest);
    if (!statusText.isEmpty())
    {
        dialog.setStatus(statusText);
    }
    else if (!canTest)
    {
        dialog.setStatus(tr("Load a model to enable inference testing."));
    }
    else
    {
        dialog.setStatus(QString());
    }
}

void MainWindow::handleAiTestRequest(AiPreferencesDialog& dialog)
{
    if (!m_currentModel)
    {
        dialog.setStatus(tr("Load a model before running inference."));
        return;
    }

    if (!m_modelManager)
    {
        m_modelManager = std::make_unique<ai::ModelManager>();
    }

    std::unique_ptr<ai::IPathAI> aiInstance = m_modelManager->createModel(m_aiModelPath);
    if (!aiInstance)
    {
        aiInstance = m_modelManager->createModel(QString());
    }

    if (!aiInstance)
    {
        dialog.setStatus(tr("Unable to construct AI instance."));
        return;
    }

    setRuntimeForceCpu(aiInstance.get(), dialog.forceCpu());

    if (!runtimeLoaded(aiInstance.get()))
    {
        const QString error = runtimeLastError(aiInstance.get());
        const QString message = error.isEmpty()
                                    ? tr("Model failed to load. Using fallback strategy.")
                                    : error;
        dialog.setStatus(message);
        dialog.setLastTestResult(tr("-"));
        return;
    }

    const tp::UserParams params = m_toolpathSettings ? m_toolpathSettings->currentParameters() : tp::UserParams{};
    const ai::StrategyDecision decision = aiInstance->predict(*m_currentModel, params);
    const double latency = runtimeLatencyMs(aiInstance.get());
    const QString deviceText = runtimeDevice(aiInstance.get());
    const QString stratText = (decision.strat == ai::StrategyDecision::Strategy::Raster)
                                  ? tr("Raster")
                                  : tr("Waterline");
    const QString badge = runtimeBadge(aiInstance.get());

    const QString summary = tr("%1 %2 ms via %3, %4 @ %5 deg, step %6 mm")
                                .arg(badge)
                                .arg(QString::number(latency, 'f', 2))
                                .arg(deviceText)
                                .arg(stratText)
                                .arg(QString::number(decision.rasterAngleDeg, 'f', 1))
                                .arg(QString::number(decision.stepOverMM, 'f', 3));

    dialog.setLastTestResult(summary);
    const QString error = runtimeLastError(aiInstance.get());
    if (!error.isEmpty())
    {
        dialog.setStatus(error);
    }
    else
    {
        dialog.setStatus(tr("Inference succeeded."));
    }

    logMessage(tr("AI test inference: %1").arg(summary));
}

void MainWindow::openAiModelDialog()
{
    refreshAiModels();

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Select AI Model"));
    dialog.resize(520, 320);

    auto* layout = new QVBoxLayout(&dialog);

    auto* table = new QTableWidget(&dialog);
    table->setColumnCount(4);
    table->setHorizontalHeaderLabels({tr("Name"), tr("Backend"), tr("Modified"), tr("Size")});
    table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    table->verticalHeader()->setVisible(false);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);

    const auto& models = m_modelManager->models();
    table->setRowCount(models.size() + 1);

    auto setRow = [&](int row, const QString& name, const QString& backend, const QString& path, const QString& modified, const QString& size) {
        auto* nameItem = new QTableWidgetItem(name);
        nameItem->setData(Qt::UserRole, path);
        table->setItem(row, 0, nameItem);
        table->setItem(row, 1, new QTableWidgetItem(backend));
        table->setItem(row, 2, new QTableWidgetItem(modified));
        table->setItem(row, 3, new QTableWidgetItem(size));
    };

    setRow(0,
           tr("Default (built-in)"),
           QStringLiteral("[Torch]"),
           QString(),
           QStringLiteral("-"),
           QStringLiteral("-"));

    int selectedRow = m_aiModelPath.isEmpty() ? 0 : -1;
    for (int i = 0; i < models.size(); ++i)
    {
        const auto& model = models.at(i);
        const QString modified = QLocale().toString(model.modified, QLocale::ShortFormat);
        const QString sizeText = QLocale().formattedDataSize(model.sizeBytes);
        const QString badge = backendBadge(model.backend);
        setRow(i + 1, model.fileName, badge, model.absolutePath, modified, sizeText);

        if (!m_aiModelPath.isEmpty() && model.absolutePath.compare(m_aiModelPath, Qt::CaseInsensitive) == 0)
        {
            selectedRow = i + 1;
        }
    }

    if (selectedRow < 0)
    {
        selectedRow = 0;
    }
    table->selectRow(selectedRow);

    layout->addWidget(table);

    auto* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttonBox);

    connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(table, &QTableWidget::itemDoubleClicked, &dialog, &QDialog::accept);

    if (dialog.exec() == QDialog::Accepted)
    {
        const int row = table->currentRow();
        if (row >= 0)
        {
            const QString path = table->item(row, 0)->data(Qt::UserRole).toString();
            if (setActiveAiModel(path))
            {
                saveSettings();
            }
        }
    }
}

void MainWindow::onAiComboChanged(int index)
{
    if (m_loadingSettings || !m_aiModelCombo || index < 0)
    {
        return;
    }

    const QString path = m_aiModelCombo->itemData(index).toString();
    if (path == m_aiModelPath)
    {
        return;
    }
    if (setActiveAiModel(path))
    {
        saveSettings();
    }
}

void MainWindow::applyUnits(common::Unit unit, bool fromSettings)
{
    if (m_unitsMmAction && m_unitsInAction)
    {
        QSignalBlocker blockMm(m_unitsMmAction);
        QSignalBlocker blockIn(m_unitsInAction);
        m_unitsMmAction->setChecked(unit == common::Unit::Millimeters);
        m_unitsInAction->setChecked(unit == common::Unit::Inches);
    }

    const bool changed = (m_units != unit);
    m_units = unit;

    if (m_toolpathSettings)
    {
        m_toolpathSettings->setUnits(unit);
    }

    if (changed)
    {
        updateModelBrowser();
        if (!fromSettings)
        {
            logMessage(tr("Units set to %1").arg(common::unitName(unit)));
        }
    }

    syncStockUiFromData();
    syncMachineUiFromData();
    updatePassControlUnits();

    if (!fromSettings)
    {
        saveSettings();
    }
}

void MainWindow::loadSettings()
{
    if (!m_toolpathSettings)
    {
        return;
    }

    QSettings settings;

    const QString unitKey = settings.value(QStringLiteral("ui/unit"), common::unitKey(m_units)).toString();
    applyUnits(common::unitFromString(unitKey, m_units), true);

    const QString toolId = settings.value(QStringLiteral("tool/id")).toString();
    m_aiModelPath = settings.value(QStringLiteral("ai/modelPath"), m_aiModelPath).toString();
    m_forceCpuInference = settings.value(QStringLiteral("ai/forceCpu"), m_forceCpuInference).toBool();

    tp::UserParams params = m_toolpathSettings->currentParameters();
    params.toolDiameter = settings.value(QStringLiteral("params/toolDiameter"), params.toolDiameter).toDouble();
    params.stepOver = settings.value(QStringLiteral("params/stepOver"), params.stepOver).toDouble();
    params.maxDepthPerPass = settings.value(QStringLiteral("params/maxDepthPerPass"), params.maxDepthPerPass).toDouble();
    params.feed = settings.value(QStringLiteral("params/feed"), params.feed).toDouble();
    params.spindle = settings.value(QStringLiteral("params/spindle"), params.spindle).toDouble();
    params.rasterAngleDeg = settings.value(QStringLiteral("params/rasterAngle"), params.rasterAngleDeg).toDouble();
    params.useHeightField = settings.value(QStringLiteral("params/useHeightField"), params.useHeightField).toBool();
    {
        const int cutterValue = settings.value(QStringLiteral("params/cutterType"),
                                               static_cast<int>(params.cutterType)).toInt();
        if (cutterValue == static_cast<int>(tp::UserParams::CutterType::BallNose))
        {
            params.cutterType = tp::UserParams::CutterType::BallNose;
        }
        else
        {
            params.cutterType = tp::UserParams::CutterType::FlatEndmill;
        }
    }

    m_enableRoughPassUser = settings.value(QStringLiteral("params/roughPassEnabled"), params.enableRoughPass).toBool();
    m_enableFinishPassUser = settings.value(QStringLiteral("params/finishPassEnabled"), params.enableFinishPass).toBool();
    m_stockAllowanceMm = settings.value(QStringLiteral("params/stockAllowanceMm"), params.stockAllowance_mm).toDouble();
    m_rampAngleDeg = settings.value(QStringLiteral("params/rampAngleDeg"), params.rampAngleDeg).toDouble();
    m_stockAllowanceMm = std::max(0.0, m_stockAllowanceMm);
    m_rampAngleDeg = std::clamp(m_rampAngleDeg, 0.5, 45.0);
    params.enableRoughPass = m_enableRoughPassUser;
    params.enableFinishPass = m_enableFinishPassUser;
    params.stockAllowance_mm = m_stockAllowanceMm;
    params.rampAngleDeg = m_rampAngleDeg;

    const int originModeValue = settings.value(QStringLiteral("stock/originMode"), static_cast<int>(m_stockOriginMode)).toInt();
    m_stockOriginMode = static_cast<StockOriginMode>(std::clamp(originModeValue, 0, 2));
    m_stock.margin_mm = settings.value(QStringLiteral("stock/marginMm"), m_stock.margin_mm).toDouble();
    m_stock.sizeXYZ_mm.x = settings.value(QStringLiteral("stock/widthMm"), m_stock.sizeXYZ_mm.x).toDouble();
    m_stock.sizeXYZ_mm.y = settings.value(QStringLiteral("stock/lengthMm"), m_stock.sizeXYZ_mm.y).toDouble();
    m_stock.sizeXYZ_mm.z = settings.value(QStringLiteral("stock/heightMm"), m_stock.sizeXYZ_mm.z).toDouble();
    m_stock.originXYZ_mm.x = settings.value(QStringLiteral("stock/originX"), m_stock.originXYZ_mm.x).toDouble();
    m_stock.originXYZ_mm.y = settings.value(QStringLiteral("stock/originY"), m_stock.originXYZ_mm.y).toDouble();
    m_stock.originXYZ_mm.z = settings.value(QStringLiteral("stock/originZ"), m_stock.originXYZ_mm.z).toDouble();
    m_stock.topZ_mm = settings.value(QStringLiteral("stock/topZ"), m_stock.originXYZ_mm.z + m_stock.sizeXYZ_mm.z).toDouble();
    m_stock.ensureValid();

    m_machine.name = settings.value(QStringLiteral("machine/name"), QString::fromStdString(m_machine.name)).toString().toStdString();
    m_machine.rapidFeed_mm_min = settings.value(QStringLiteral("machine/rapidFeed"), m_machine.rapidFeed_mm_min).toDouble();
    m_machine.maxFeed_mm_min = settings.value(QStringLiteral("machine/maxFeed"), m_machine.maxFeed_mm_min).toDouble();
    m_machine.maxSpindleRPM = settings.value(QStringLiteral("machine/maxSpindle"), m_machine.maxSpindleRPM).toDouble();
    m_machine.clearanceZ_mm = settings.value(QStringLiteral("machine/clearanceZ"), m_machine.clearanceZ_mm).toDouble();
    m_machine.safeZ_mm = settings.value(QStringLiteral("machine/safeZ"), m_machine.safeZ_mm).toDouble();
    m_machine.ensureValid();

    m_loadingSettings = true;
    if (!toolId.isEmpty())
    {
        m_toolpathSettings->setSelectedToolId(toolId, false);
    }
    m_toolpathSettings->setParameters(params);
    syncPassControlsFromData();
    updatePassControlUnits();
    m_loadingSettings = false;

    if (m_stockOriginMode != StockOriginMode::Custom)
    {
        updateStockDerivedFromModel();
    }
    syncStockUiFromData();
    syncMachineUiFromData();

    m_lastModelDirectory = settings.value(QStringLiteral("paths/lastModelDir")).toString();
    m_currentModelPath = settings.value(QStringLiteral("paths/lastModel")).toString();


    const bool envReady = settings.value(QStringLiteral("training/envReady"), false).toBool();
    m_envReady = envReady;
    const bool cpuOnly = settings.value(QStringLiteral("training/cpuOnly"),
                                        m_envManager ? m_envManager->cpuOnly() : false)
                             .toBool();
    if (m_envProgress)
    {
        m_envProgress->setValue(envReady ? 100 : 0);
    }
    if (m_envCpuOnlyCheck)
    {
        QSignalBlocker blocker(m_envCpuOnlyCheck);
        m_envCpuOnlyCheck->setChecked(cpuOnly);
    }
    if (m_envManager)
    {
        m_envManager->setCpuOnly(cpuOnly);
        onGpuInfoChanged(m_envManager->gpuSummary());
    }
    updateTrainingActions();

void MainWindow::saveSettings() const
{
    QSettings settings;
    settings.setValue(QStringLiteral("ui/unit"), common::unitKey(m_units));

    auto* self = const_cast<MainWindow*>(this);
    if (self)
    {
        self->syncStockDataFromUi();
        if (self->m_stockOriginMode != StockOriginMode::Custom)
        {
            self->updateStockDerivedFromModel();
        }
        self->syncMachineDataFromUi();
    }

    if (m_toolpathSettings)
    {
        settings.setValue(QStringLiteral("tool/id"), m_toolpathSettings->currentToolId());
        const tp::UserParams params = m_toolpathSettings->currentParameters();
        settings.setValue(QStringLiteral("params/toolDiameter"), params.toolDiameter);
        settings.setValue(QStringLiteral("params/stepOver"), params.stepOver);
        settings.setValue(QStringLiteral("params/maxDepthPerPass"), params.maxDepthPerPass);
        settings.setValue(QStringLiteral("params/feed"), params.feed);
        settings.setValue(QStringLiteral("params/spindle"), params.spindle);
        settings.setValue(QStringLiteral("params/rasterAngle"), params.rasterAngleDeg);
        settings.setValue(QStringLiteral("params/useHeightField"), params.useHeightField);
        settings.setValue(QStringLiteral("params/cutterType"), static_cast<int>(params.cutterType));
    }

    settings.setValue(QStringLiteral("stock/originMode"), static_cast<int>(m_stockOriginMode));
    settings.setValue(QStringLiteral("stock/marginMm"), m_stock.margin_mm);
    settings.setValue(QStringLiteral("stock/widthMm"), m_stock.sizeXYZ_mm.x);
    settings.setValue(QStringLiteral("stock/lengthMm"), m_stock.sizeXYZ_mm.y);
    settings.setValue(QStringLiteral("stock/heightMm"), m_stock.sizeXYZ_mm.z);
    settings.setValue(QStringLiteral("stock/originX"), m_stock.originXYZ_mm.x);
    settings.setValue(QStringLiteral("stock/originY"), m_stock.originXYZ_mm.y);
    settings.setValue(QStringLiteral("stock/originZ"), m_stock.originXYZ_mm.z);
    settings.setValue(QStringLiteral("stock/topZ"), m_stock.topZ_mm);

    settings.setValue(QStringLiteral("machine/name"), QString::fromStdString(m_machine.name));
    settings.setValue(QStringLiteral("machine/rapidFeed"), m_machine.rapidFeed_mm_min);
    settings.setValue(QStringLiteral("machine/maxFeed"), m_machine.maxFeed_mm_min);
    settings.setValue(QStringLiteral("machine/maxSpindle"), m_machine.maxSpindleRPM);
    settings.setValue(QStringLiteral("machine/clearanceZ"), m_machine.clearanceZ_mm);
    settings.setValue(QStringLiteral("machine/safeZ"), m_machine.safeZ_mm);

    settings.setValue(QStringLiteral("paths/lastModelDir"), m_lastModelDirectory);
    settings.setValue(QStringLiteral("paths/lastModel"), m_currentModelPath);
    settings.setValue(QStringLiteral("ai/modelPath"), m_aiModelPath);
    settings.setValue(QStringLiteral("ai/forceCpu"), m_forceCpuInference);
    if (m_envCpuOnlyCheck)
    {
        settings.setValue(QStringLiteral("training/cpuOnly"), m_envCpuOnlyCheck->isChecked());
    }
}

void MainWindow::onToolSelected(const QString& toolId)
{
    if (m_loadingSettings)
    {
        return;
    }

    if (const common::Tool* tool = m_toolLibrary.toolById(toolId))
    {
        logMessage(tr("Tool selected: %1").arg(tool->name));
    }
    else if (!toolId.isEmpty())
    {
        logWarning(tr("Tool id \"%1\" not found in library.").arg(toolId));
    }
    saveSettings();
}

void MainWindow::logWarning(const QString& text)
{
    logMessage(tr("Warning: %1").arg(text));
}

void MainWindow::closeEvent(QCloseEvent* event)
{
    saveSettings();
    QMainWindow::closeEvent(event);
}

void MainWindow::openModelFromFile()
{
    const QString filter = tr("Meshes (*.obj *.stl *.step *.stp *.iges *.igs);;All Files (*.*)");
    const QString initialDir = m_lastModelDirectory.isEmpty() ? QString() : m_lastModelDirectory;
    const QString selected = QFileDialog::getOpenFileName(this, tr("Open Model"), initialDir, filter);
    if (selected.isEmpty())
    {
        return;
    }
    startImportWorker(selected);
}

void MainWindow::saveToolpathToFile()
{
    if (!m_currentToolpath || m_currentToolpath->empty())
    {
        QMessageBox::information(this, tr("No Toolpath"), tr("Generate a toolpath before saving."));
        return;
    }

    struct PostOption
    {
        QString label;
        std::function<std::unique_ptr<tp::IPost>()> factory;
    };

    const std::vector<PostOption> options = {
        {QStringLiteral("GRBL"), []() { return std::make_unique<tp::GRBLPost>(); }}
    };

    QDialog dialog(this);
    dialog.setWindowTitle(tr("Export Toolpath"));
    auto* layout = new QVBoxLayout(&dialog);
    layout->setContentsMargins(12, 12, 12, 12);
    layout->setSpacing(8);

    auto* form = new QFormLayout();
    form->setContentsMargins(0, 0, 0, 0);
    form->setSpacing(6);

    auto* postCombo = new QComboBox(&dialog);
    for (const auto& option : options)
    {
        postCombo->addItem(option.label);
    }
    form->addRow(tr("Post"), postCombo);

    auto* pathLayout = new QHBoxLayout();
    pathLayout->setSpacing(6);

    auto* pathEdit = new QLineEdit(&dialog);
    const QString defaultName = QStringLiteral("toolpath.gcode");
    const QString initialPath = m_lastModelDirectory.isEmpty()
                                    ? defaultName
                                    : QDir(m_lastModelDirectory).filePath(defaultName);
    pathEdit->setText(initialPath);
    pathEdit->setMinimumWidth(320);

    auto* browseButton = new QPushButton(tr("Browse..."), &dialog);
    pathLayout->addWidget(pathEdit);
    pathLayout->addWidget(browseButton);
    form->addRow(tr("File"), pathLayout);

    layout->addLayout(form);

    auto* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttonBox);

    connect(buttonBox, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    connect(browseButton, &QPushButton::clicked, &dialog, [this, pathEdit]() {
        const QString filter = tr("G-code (*.gcode *.nc);;All Files (*.*)");
        const QString dir = QFileInfo(pathEdit->text()).absoluteDir().absolutePath();
        const QString selected = QFileDialog::getSaveFileName(this, tr("Save Toolpath"), dir, filter);
        if (!selected.isEmpty())
        {
            pathEdit->setText(selected);
        }
    });

    if (dialog.exec() != QDialog::Accepted)
    {
        return;
    }

    QString filePath = pathEdit->text().trimmed();
    if (filePath.isEmpty())
    {
        QMessageBox::warning(this, tr("Export"), tr("Choose a destination file."));
        return;
    }

    if (QFileInfo(filePath).suffix().isEmpty())
    {
        filePath += QStringLiteral(".gcode");
    }

    const int postIndex = postCombo->currentIndex();
    if (postIndex < 0 || postIndex >= static_cast<int>(options.size()))
    {
        QMessageBox::warning(this, tr("Export"), tr("Select a valid post."));
        return;
    }

    auto post = options[postIndex].factory();
    QString error;
    const tp::UserParams paramsMm = m_toolpathSettings ? m_toolpathSettings->currentParameters() : tp::UserParams{};
    if (!tp::GCodeExporter::exportToFile(*m_currentToolpath, filePath, *post, m_units, paramsMm, &error))
    {
        QMessageBox::critical(this, tr("Export Failed"), error);
        return;
    }

    m_lastModelDirectory = QFileInfo(filePath).absolutePath();
    saveSettings();

    QFile previewFile(filePath);
    if (previewFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        QTextStream stream(&previewFile);
        QStringList lines;
        lines.reserve(101);
        lines.append(tr("; Preview of %1").arg(QFileInfo(filePath).fileName()));
        for (int i = 0; i < 100 && !stream.atEnd(); ++i)
        {
            lines.append(stream.readLine());
        }
        previewFile.close();

        if (m_gcodePreview)
        {
            m_gcodePreview->setPlainText(lines.join(QLatin1Char('\n')));
        }
        if (m_consoleTabs && m_gcodePreview)
        {
            m_consoleTabs->setCurrentWidget(m_gcodePreview);
        }
    }

    logMessage(tr("Saved toolpath to %1 using %2").arg(filePath, QString::fromStdString(post->name())));
}

void MainWindow::resetCamera()
{
    if (m_viewer)
    {
        m_viewer->resetCamera();
        logMessage(tr("Camera reset."));
    }
}

void MainWindow::showAboutDialog()
{
    QString buildConfig = QString::fromLatin1(APP_BUILD_CONFIG);
    if (buildConfig.trimmed().isEmpty())
    {
        buildConfig = QString::fromLatin1(app::buildinfo::kConfigFallback);
    }
    if (buildConfig.trimmed().isEmpty())
    {
        buildConfig = tr("Unknown");
    }

    const QString version = QString::fromLatin1(app::buildinfo::kVersion);
    const QString commit = QString::fromLatin1(app::buildinfo::kCommit);
    const QString backends = QString::fromLatin1(app::buildinfo::kBackends);
    const QString builtOn = QString::fromLatin1(app::buildinfo::kBuildDate);

    QString gpuSummary;
    if (!m_rendererName.isEmpty() || !m_rendererVendor.isEmpty())
    {
        const QString namePart = m_rendererName.isEmpty()
                                     ? m_rendererVendor
                                     : (m_rendererVendor.isEmpty()
                                            ? m_rendererName
                                            : QStringLiteral("%1 • %2").arg(m_rendererVendor, m_rendererName));
        gpuSummary = tr("<br/>GPU: %1").arg(namePart);
        if (!m_rendererVersion.isEmpty())
        {
            gpuSummary.append(tr(" (%1)").arg(m_rendererVersion));
        }
    }

    const QString aboutText = tr(
        "<b>AIToolpathGenerator %1</b><br/>"
        "Commit: <code>%2</code><br/>"
        "Build type: %3<br/>"
        "Enabled backends: %4<br/>"
        "Built: %5 UTC%6")
                                  .arg(version,
                                       commit.isEmpty() ? tr("unknown") : commit,
                                       buildConfig,
                                       backends.isEmpty() ? tr("None") : backends,
                                       builtOn.isEmpty() ? tr("unknown") : builtOn,
                                       gpuSummary);

    QMessageBox::about(this, tr("About AIToolpathGenerator"), aboutText);
}

void MainWindow::selectModelWithAI()
{
    openAiModelDialog();
}

void MainWindow::onToolpathRequested(const tp::UserParams& settings)
{
    if (!m_currentModel || !m_currentModel->isValid())
    {
        QMessageBox::information(this, tr("No Model Loaded"), tr("Load a model before generating a toolpath."));
        return;
    }
    syncStockDataFromUi();
    if (m_stockOriginMode != StockOriginMode::Custom)
    {
        updateStockDerivedFromModel();
        syncStockUiFromData();
    }
    syncMachineDataFromUi();

    tp::UserParams enriched = settings;
    enriched.stock = m_stock;
    enriched.machine = m_machine;
    m_lastUserParams = enriched;

    startGenerateWorker(enriched);
}

void MainWindow::updateModelBrowser()
{
    if (!m_modelBrowser)
    {
        return;
    }

    m_modelBrowser->clear();
    if (m_currentModel)
    {
        const QString fileName = m_currentModelPath.isEmpty()
                                     ? m_currentModel->name()
                                     : QFileInfo(m_currentModelPath).fileName();
        const auto triangleCount = m_currentModel->indices().size() / 3;
        const auto bounds = m_currentModel->bounds();

        m_modelBrowser->addItem(tr("File: %1").arg(fileName));
        m_modelBrowser->addItem(tr("Triangles: %1").arg(QString::number(static_cast<qulonglong>(triangleCount))));
        m_modelBrowser->addItem(tr("AABB: [%1, %2, %3] -> [%4, %5, %6]")
                                    .arg(bounds.min.x(), 0, 'f', 3)
                                    .arg(bounds.min.y(), 0, 'f', 3)
                                    .arg(bounds.min.z(), 0, 'f', 3)
                                    .arg(bounds.max.x(), 0, 'f', 3)
                                    .arg(bounds.max.y(), 0, 'f', 3)
                                    .arg(bounds.max.z(), 0, 'f', 3));

        if (m_currentToolpath && !m_currentToolpath->empty())
        {
            m_modelBrowser->addItem(tr("Toolpath segments: %1").arg(m_currentToolpath->passes.size()));
            const double feedDisplay = common::fromMillimeters(m_currentToolpath->feed, m_units);
            const QString feedText = QLocale().toString(feedDisplay, 'f', 1);
            const QString rpmText = QLocale().toString(m_currentToolpath->spindle, 'f', 0);
            m_modelBrowser->addItem(tr("Feed %1 %2, Spindle %3 RPM")
                                        .arg(feedText)
                                        .arg(common::feedSuffix(m_units))
                                        .arg(rpmText));
        }
        else
        {
            m_modelBrowser->addItem(tr("Toolpath: none"));
        }
    }
    else
    {
        m_modelBrowser->addItem(tr("No model loaded."));
    }
}

void MainWindow::logMessage(const QString& text)
{
    if (m_console)
    {
        m_console->appendPlainText(formatTimestamped(text));
    }
}

void MainWindow::updateStatusAiLabel() const
{
    if (!m_statusAiLabel)
    {
        return;
    }

    if (m_aiModelLabel.isEmpty())
    {
        m_statusAiLabel->setText(tr("AI: --"));
        m_statusAiLabel->setToolTip(tr("AI runtime inactive"));
        return;
    }

    m_statusAiLabel->setText(tr("AI: %1").arg(m_aiModelLabel));
    QString tooltip = tr("Active AI: %1").arg(m_aiModelLabel);
    if (!m_aiModelPath.isEmpty())
    {
        tooltip.append(tr("\nModel path: %1").arg(QDir::toNativeSeparators(m_aiModelPath)));
    }
    m_statusAiLabel->setToolTip(tooltip);
}

void MainWindow::onRendererInfoChanged(const QString& vendor, const QString& renderer, const QString& version)
{
    m_rendererVendor = vendor.trimmed();
    m_rendererName = renderer.trimmed();
    m_rendererVersion = version.trimmed();

    if (!m_statusGpuLabel)
    {
        return;
    }

    QString text;
    const bool mergeNames = !m_rendererVendor.isEmpty()
                            && !m_rendererName.isEmpty()
                            && !m_rendererName.contains(m_rendererVendor, Qt::CaseInsensitive);
    if (mergeNames)
    {
        text = tr("GPU: %1 %2").arg(m_rendererVendor, m_rendererName);
    }
    else if (!m_rendererName.isEmpty())
    {
        text = tr("GPU: %1").arg(m_rendererName);
    }
    else if (!m_rendererVendor.isEmpty())
    {
        text = tr("GPU: %1").arg(m_rendererVendor);
    }
    else
    {
        text = tr("GPU: Unknown");
    }

    if (!m_rendererVersion.isEmpty())
    {
        text.append(tr(" (%1)").arg(m_rendererVersion));
    }

    m_statusGpuLabel->setText(text);
    m_statusGpuLabel->setToolTip(tr("Vendor: %1\nRenderer: %2\nVersion: %3")
                                     .arg(m_rendererVendor.isEmpty() ? tr("Unknown") : m_rendererVendor)
                                     .arg(m_rendererName.isEmpty() ? tr("Unknown") : m_rendererName)
                                     .arg(m_rendererVersion.isEmpty() ? tr("Unknown") : m_rendererVersion));
}

void MainWindow::onFrameStatsUpdated(float fps)
{
    m_lastFps = fps;
    if (!m_statusFpsLabel)
    {
        return;
    }

    if (fps <= 0.01f)
    {
        m_statusFpsLabel->setText(tr("FPS: --"));
        m_statusFpsLabel->setToolTip({});
        return;
    }

    m_statusFpsLabel->setText(tr("FPS: %1").arg(QString::number(fps, 'f', 1)));
    m_statusFpsLabel->setToolTip(tr("Average frames per second over the last second."));
}

void MainWindow::maybeRunFirstRunTour()
{
    QSettings settings;
    const bool completed = settings.value(QStringLiteral("ui/firstRunCompleted"), false).toBool();
    if (completed)
    {
        return;
    }

    settings.setValue(QStringLiteral("ui/firstRunCompleted"), true);

    const QString samplePath = QDir(QCoreApplication::applicationDirPath())
                                   .filePath(QStringLiteral("samples/sample_part.stl"));

    QTimer::singleShot(250, this, [this, samplePath]() {
        QString message;
        if (QFileInfo::exists(samplePath))
        {
            startImportWorker(samplePath);
            message = tr("A sample project has been opened from:\n%1\n\n"
                         "Use the Toolpath Settings panel to generate paths, "
                         "or open your own STL/OBJ models from File -> Open.")
                          .arg(QDir::toNativeSeparators(samplePath));
        }
        else
        {
            message = tr("Welcome to AIToolpathGenerator!\n\n"
                         "The bundled sample part is missing:\n%1\n"
                         "You can still open your own models via File -> Open.")
                          .arg(QDir::toNativeSeparators(samplePath));
        }

        QMessageBox::information(this,
                                 tr("Welcome to AIToolpathGenerator"),
                                 message);
    });
}

void MainWindow::displayToolpathMessage(const QString& text)
{
    logMessage(text);
    if (m_console)
    {
        m_console->appendPlainText(QString());
    }
}

} // namespace app




























