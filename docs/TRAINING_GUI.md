# Training GUI Workflow

> _Screenshots marked `TODO` are placeholders. Capture them from the running application after building the feature._

![TODO - Training Menu Overview](images/training_menu.png)

## Overview

The CNC Toolpath Companion introduces a full graphical workflow for model lifecycle management. The new flow spans synthetic dataset creation, initial training, fine-tuning, and hot swapping the active inference model -- all without leaving the desktop shell.

Key components:

- `src/train/TrainingManager.{h,cpp}` orchestrates long-running tasks via `QProcess`, wraps logging, and coordinates with `EnvManager`.
- `src/app/TrainingNewModelDialog.*` and `src/app/TrainingSyntheticDataDialog.*` gather wizard input.
- `src/app/MainWindow.{h,cpp}` integrates the Training menu, Jobs dock, and dispatches requests to the manager.
- Python backends in `train/generate_synthetic.py` and `train/train_strategy.py` perform data generation and learning.

## Prerequisites

1. Prepare the embedded runtime through **Training > Prepare Environment** (re-checks via `train::EnvManager`).
2. Confirm Python packages in `train/requirements.txt` are installed -- the Training Manager will verify before running jobs.
3. Models are stored under `<install-root>/models`; datasets default to `<install-root>/datasets`.

![TODO - Environment Dock](images/training_environment.png)

## Training Menu

`MainWindow::createTrainingMenu()` installs a top-level **Training** entry with five actions:

| Action                           | Behaviour                                                                 | Source Path                              |
|----------------------------------|---------------------------------------------------------------------------|------------------------------------------|
| **New Model...**                   | Launches the wizard (`TrainingNewModelDialog`) for scratch/fork training  | `src/app/TrainingNewModelDialog.*`       |
| **Generate Synthetic Data...**     | Opens the sampling dialog (`TrainingSyntheticDataDialog`)                 | `src/app/TrainingSyntheticDataDialog.*`  |
| **Fine-Tune Current Model...**     | Locks the active AI model and prompts for dataset + hyperparameters       | `MainWindow::fineTuneCurrentModel`       |
| **Open Models Folder**           | Opens `TrainingManager::modelsRoot()` in the OS explorer                  | `MainWindow::openModelsFolder`           |
| **Open Datasets Folder**         | Opens `TrainingManager::datasetsRoot()`                                   | `MainWindow::openDatasetsFolder`         |

Actions stay disabled until the environment is marked ready (`training/envReady` via `EnvManager::persistStatus`), preventing accidental execution.

![TODO - Training Menu Callouts](images/training_menu_callouts.png)

## New Model Wizard

`TrainingNewModelDialog` supplies:

- Model name (auto-sanitised to `[A-Za-z0-9_-]`).
- Base selection: **Train from scratch** or existing `.pt/.onnx` discovered by `ai::ModelManager`.
- Device drop-down obeying runtime GPU detection.
- Epoch count, learning rate, and optional dataset override.
- Feature toggle for the v2 geometry-aware descriptors.

Submitting produces a `TrainingManager::TrainJobRequest`. When no dataset is supplied the Python script fabricates synthetic samples internally.

## Synthetic Dataset Dialog

`TrainingSyntheticDataDialog` offers three slider-driven fields and the output directory:

- Sample count (`100 ... 5000`).
- Shape diversity (`0.0 ... 1.0`) and slope mix ratios.
- Overwrite confirmation.

The Training Manager forwards these parameters to `train/generate_synthetic.py` and records progress based on log output.

## Jobs Dock

![TODO - Jobs Dock](images/training_jobs.png)

`MainWindow::createJobsDock()` installs a tabbed dock alongside the environment panel:

- Every job row shows title (`jobTypeLabel + label`), detail/tooltip, live progress, state, ETA and a **Cancel** button.
- The dock subscribes to:
  - `TrainingManager::jobAdded` / `jobUpdated` for UI state.
  - `TrainingManager::jobLog` to stream stdout/stderr into a persistent log buffer.
  - `TrainingManager::modelRegistered` to refresh the AI roster and optionally hot-swap the active model.
- Selecting a job reveals timestamped logs in the lower pane (`m_jobLog`).

Progress and ETA are computed in `TrainingManager` (see `parseOutput` + `refreshEta`) and mirrored in UI via `MainWindow::jobEtaLabel`.

## End-to-End Flow

1. **Prepare** the environment (downloads Python, seeds venv).
2. **Generate** synthetic data if required (`Training > Generate Synthetic Data...`).
3. **Train** a new model (`Training > New Model...`) or fork from an existing checkpoint.
4. **Fine-tune** the active model on curated datasets (`Training > Fine-Tune Current Model...`).
5. Monitor progress in the **Training Jobs** dock; cancel jobs if needed.
6. When `TrainingManager` registers new weights, the UI refreshes `ai::ModelManager`, hot-swaps the active AI (`MainWindow::setActiveAiModel`), and updates UI badges via `updateActiveAiSummary`.

## Interactions with the Rest of the Codebase

- **Toolpath Generation:** `MainWindow::onToolpathRequested` continues to rely on `tp::ToolpathGenerator`; the active AI prototype is swapped seamlessly once training completes.
- **Logging:** Job logs are appended to the main console (`MainWindow::logMessage`) in parallel to per-job buffers.
- **Settings:** Environment readiness, CPU-only preference, and last-used models persist through `QSettings` (see `MainWindow::loadSettings`).
- **Scripting:** The Python layer reads environment variables set in `TrainingManager::startJob` for dataset metadata, base model hints, and fine-tune indicators.

## Next Steps

- Capture final screenshots for the placeholders in this document.
- Extend `TrainingManager` if additional scripts (evaluation, export) are introduced.
- Consider persisting job history between sessions for auditability.


