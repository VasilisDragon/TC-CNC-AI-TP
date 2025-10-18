# Training GUI Workflow

> _Screenshots marked `TODO` are placeholders. Capture them from the running application after building the feature._

![TODO - Training Menu Overview](images/training_menu.png)

## Overview

The CNC Toolpath Companion introduces a full graphical workflow for model lifecycle management. The new flow spans synthetic dataset creation, initial training, fine-tuning, environment provisioning, and hot swapping the active inference model-no command-line gymnastics required (unless you're into that aesthetic).

Key components:

- `src/train/TrainingManager.{h,cpp}` orchestrates long-running tasks via `QProcess`, wraps logging, and coordinates with `EnvManager`.
- `src/train/EnvManager.{h,cpp}` bootstraps the embedded Python runtime, tracks GPU availability, and persists readiness flags.
- `src/app/TrainingNewModelDialog.*` and `src/app/TrainingSyntheticDataDialog.*` gather wizard input.
- `src/app/MainWindow.{h,cpp}` integrates the Training menu, Environment dock, Jobs dock, and dispatches requests to the manager.
- Python backends in `train/generate_synthetic.py` and `train/train_strategy.py` perform data generation and learning.

## Prerequisites

1. Prepare the embedded runtime through **Training -> Prepare Environment**. This drives `train::EnvManager`, downloads embedded Python, verifies the checksum, and hydrates a virtual environment. Cancel actually cancels-because forcing a broken download to finish is peak chaos.
2. Confirm the packages in `train/requirements.txt` are installed. The Training Manager will double-check and complain loudly if anything is missing.
3. Models live under `<install-root>/models`; datasets default to `<install-root>/datasets`. The Jobs dock and AI model combo refresh automatically when new artefacts appear, so keep those paths writable.

![TODO - Environment Dock](images/training_environment.png)

## Environment Dock

The **Training Environment** dock (created in `MainWindow::createDockWidgets`) is home base for `train::EnvManager`:

- **Prepare Environment** kicks off the embedded Python bootstrap, progress bar, and log stream. Expect timestamps, SHA-256 validation, and the occasional spicy warning if your disk permissions are sus.
- **Cancel** stops the active operation and terminates any `QProcess` child. No more waiting out a hung unzip like it's character building.
- **CPU-only** toggle persists via `training/cpuOnly`. When enabled, `EnvManager` will skip GPU providers and the New Model wizard hides the "Auto (GPU when available)" option.
- **GPU summary** reports `nvidia-smi` output (when available). If the label says "GPU: Not detected", the training dialogs will nudge users toward CPU training so nobody hallucinated CUDA support.
- **Log panel** mirrors the console output and keeps the newest lines in view. It's the fastest way to catch pip output or download failures without digging through temp folders.

All environment state (ready flag, GPU summary, CPU-only preference) is cached via `QSettings` and reapplied on the next launch, so the UI doesn't make you relive setup every morning.

## Training Menu

`MainWindow::createTrainingMenu()` installs a top-level **Training** entry with five actions:

| Action                           | Behaviour                                                                 | Source Path                              |
|----------------------------------|---------------------------------------------------------------------------|------------------------------------------|
| **New Model...**                   | Launches the wizard (`TrainingNewModelDialog`) for scratch/fork training with device preference  | `src/app/TrainingNewModelDialog.*`       |
| **Generate Synthetic Data...**     | Opens the sampling dialog (`TrainingSyntheticDataDialog`)                 | `src/app/TrainingSyntheticDataDialog.*`  |
| **Fine-Tune Current Model...**     | Locks the active AI model and prompts for dataset + hyperparameters       | `MainWindow::fineTuneCurrentModel`       |
| **Open Models Folder**           | Opens `TrainingManager::modelsRoot()` in the OS explorer                  | `MainWindow::openModelsFolder`           |
| **Open Datasets Folder**         | Opens `TrainingManager::datasetsRoot()`                                   | `MainWindow::openDatasetsFolder`         |

Actions stay disabled until the environment is marked ready (`training/envReady` via `EnvManager::persistStatus`), preventing accidental execution. If the user toggles "CPU-only" in the environment dock, the New Model wizard will automatically hide the GPU option-no more pretending your laptop 3050 is a data center card.

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

`MainWindow::createJobsDock()` installs a dock tabbed next to the environment panel:

- Every job row shows title (`jobTypeLabel + label`), detail/tooltip, live progress, state, ETA, and a **Cancel** button wired to `TrainingManager::cancelJob`.
- The dock subscribes to:
  - `TrainingManager::jobAdded` / `TrainingManager::jobUpdated` for UI state.
  - `TrainingManager::jobLog` to stream stdout/stderr into a persistent log buffer.
  - `TrainingManager::modelRegistered` to refresh the AI roster and hot-swap the active model when new artefacts land in `models/`.
- Selecting a job refreshes the lower log pane with the accumulated output and updates the summary banner. It's like doomscrolling, but for gradient descent.

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


