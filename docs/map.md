# Project Map

Legend  
- [Verified] Observed directly in the referenced source.  
- [Assumed] Based on naming or structure; confirm separately.

## Target Inventory
- [Verified] `common` static library wires shared logging, math helpers, and tool definitions for reuse across the app (`common/CMakeLists.txt:1`, `src/common/ToolLibrary.cpp:1`).
- [Verified] `render` static library hosts the OpenGL viewer, overlays, and simulation plumbing (`render/CMakeLists.txt:1`, `src/render/ModelViewerWidget.cpp:461`, `src/render/ToolpathOverlay.cpp:5`).
- [Verified] `io` static library encapsulates mesh ingestion and worker threads for background imports (`src/io/CMakeLists.txt:1`, `src/io/ModelImporter.cpp:12`).
- [Verified] `ai` static library provides feature extraction plus Torch/ONNX backends selected at build time (`src/ai/CMakeLists.txt:1`, `src/ai/TorchAI.cpp:62`, `src/ai/OnnxAI.cpp:61`, `src/ai/ModelManager.cpp:27`).
- [Verified] `tp` static library implements toolpath generation, exporters, posts, and optional OpenCAMLib hooks (`src/tp/CMakeLists.txt:1`, `src/tp/ToolpathGenerator.cpp:741`, `src/tp/GenerateWorker.cpp:34`).
- [Verified] `app` Qt executable stitches every module together, layers in training dialogs, and owns runtime wiring (`app/CMakeLists.txt:1`, `src/app/MainWindow.cpp:2099`, `src/app/MainWindow.cpp:3027`).

## Toolpath Generation Flow
- [Verified] **UI trigger** `ToolpathSettingsWidget::generateRequested` feeds `MainWindow::onToolpathRequested`, which enriches stock/machine data before delegating to `startGenerateWorker` (`app/include/app/ToolpathSettingsWidget.h:27`, `src/app/MainWindow.cpp:508`, `src/app/MainWindow.cpp:3219`).
- [Verified] **Worker setup** `startGenerateWorker` materializes an AI instance through `ModelManager::createModel`, applies overrides, and spawns a `tp::GenerateWorker` to keep the UI responsive (`src/app/MainWindow.cpp:2099`, `src/ai/ModelManager.cpp:77`).
- [Verified] **Generator entry point** `tp::GenerateWorker::run` calls `ToolpathGenerator::generate`, relaying progress, cancellation, and banner messages back to the main thread (`src/tp/GenerateWorker.cpp:34`, `src/tp/ToolpathGenerator.cpp:741`).
- [Verified] **Pass planning** `ToolpathGenerator::generate` asks the AI for a `StrategyDecision` before assembling rough/finish pass profiles via `buildPassPlan` (`src/tp/ToolpathGenerator.cpp:767`).
- [Verified] **Heightfield & slicing** Raster passes sample the mesh with `heightfield::HeightField::build`, while waterline passes route through `generateWaterlineSlicer` and optional OpenCAMLib support in `OclAdapter::waterline` (`src/tp/heightfield/HeightField.cpp:32`, `src/tp/ToolpathGenerator.cpp:1199`, `src/tp/ocl/OclAdapter.cpp:125`).
- [Verified] **Finalization** Completed toolpaths capture feed, machine, and stock metadata before the viewer and simulator pick them up (`src/app/MainWindow.cpp:2181`, `src/render/SimulationController.cpp:30`).

## AI Integration
- [Verified] **Build toggles** `WITH_TORCH`, `WITH_ONNXRUNTIME`, and `WITH_OCL` options gate backend discovery and propagate compile definitions (`CMakeLists.txt:17`).
- [Verified] **Model catalog** `ai::ModelManager` scans the packaged `models/` directory, records metadata, and instantiates backend-specific AI objects (`src/ai/ModelManager.cpp:27`, `src/ai/ModelManager.cpp:77`).
- [Verified] **Torch runtime** `ai::TorchAI` loads TorchScript modules when LibTorch is present, tracks latency, and falls back to heuristics otherwise (`src/ai/TorchAI.cpp:62`, `src/ai/TorchAI.cpp:157`, `src/ai/TorchAI.cpp:310`).
- [Verified] **ONNX runtime** `ai::OnnxAI` configures the ONNX Runtime session, resolves provider capabilities, and mirrors the Torch feature-alignment safeguards (`src/ai/OnnxAI.cpp:61`, `src/ai/OnnxAI.cpp:99`, `src/ai/OnnxAI.cpp:199`).
- [Verified] **Feature engineering** `FeatureExtractor::computeGlobalFeatures` generates normalized descriptors that both backends pad or truncate to match their learned schema (`src/ai/FeatureExtractor.cpp:1`, `src/ai/TorchAI.cpp:222`, `src/ai/OnnxAI.cpp:219`).

## Post Processing & Export
- [Verified] **Post interface** `tp::IPost` defines the contract that exporters rely on for controller-specific output (`src/tp/IPost.h:12`).
- [Verified] **Template DSL & base** `tp::TemplateEngine` resolves `{{tokens}}`, conditional blocks, and `{{else}}` branches while `tp::GCodePostBase` centralizes motion formatting, arc fitting, and header/footer templating (`src/tp/TemplateEngine.cpp:1`, `src/tp/GCodePostBase.cpp:1`).
- [Verified] **GRBL post** `tp::GRBLPost` reuses the shared base to emit GRBL headers, spindle control, and arc-enabled motion planning (`src/tp/GRBLPost.cpp:1`).
- [Verified] **Fanuc post** `tp::FanucPost` layers a work-offset centric header (`G54`, `G17`, `G90`, `G94`) and terminates programs with `M30` (`src/tp/FanucPost.cpp:1`).
- [Verified] **Marlin post** `tp::MarlinPost` disables spindle output, reports optional arc usage in comments, and wraps up with `M84` motor release commands (`src/tp/MarlinPost.cpp:1`).
- [Verified] **Heidenhain post** `tp::HeidenhainPost` outputs plain-language `BEGIN/END PGM` blocks with `L` motions that inline feed rates and fall back to linearized arcs (`src/tp/HeidenhainPost.cpp:1`).
- [Verified] **Exporter** `tp::GCodeExporter::exportToFile` streams the selected post, embeds a tolerance fingerprint, and reports errors back to the caller (`src/tp/GCodeExporter.cpp:80`).
- [Verified] **UI wiring** `MainWindow::saveToolpathToFile` presents post options, persists the file, and surfaces a preview in the console dock (`src/app/MainWindow.cpp:3027`).

## Viewer & Simulation
- [Verified] **Overlay rebuild** `ModelViewerWidget::updateToolpathOverlay` classifies toolpath segments into cut/rapid/link/unsafe buckets for rendering (`src/render/ModelViewerWidget.cpp:461`).
- [Verified] **OpenGL overlay** `ToolpathOverlay` owns the VBO/VAO lifecycle and draws colored line batches with dash hints (`src/render/ToolpathOverlay.cpp:5`).
- [Verified] **Simulation playback** `SimulationController` rebuilds motion segments, exposes play/pause/seek, and emits the live tooltip position (`src/render/SimulationController.cpp:30`).
- [Verified] **Viewer hookup** The widget listens to `SimulationController::positionChanged` and toggles the overlay/simulation actor as toolpaths arrive (`src/render/ModelViewerWidget.cpp:88`, `src/app/MainWindow.cpp:2181`).

## Training & Test Utilities
- [Verified] **Embedded training env** `train::EnvManager` provisions the Python runtime and validates downloads before enabling training workflows (`src/train/EnvManager.cpp:24`).
- [Verified] **Training orchestration** `train::TrainingManager` drives dataset generation and fine-tuning via external processes, honoring environment controls (`src/train/TrainingManager.cpp:29`).
- [Verified] **Embedded diagnostics** Optional `tests_core` library exposes doctest-based checks when `WITH_EMBEDDED_TESTS` is enabled (`src/tests_core/CMakeLists.txt:1`, `src/tests_core/TestsCore.cpp:1`).
- [Verified] **Headless smoke tests** Standalone executables such as `headless_pipeline.cpp` validate the importer → generator → post chain without the GUI (`tests/headless_pipeline.cpp:1`, `CMakeLists.txt:141`).
- [Added] **Stock simulator** `sim::StockGrid` voxelizes model bounds, subtracts tool sweeps, and drives the viewer heatmap overlay with residual metrics (`sim/src/StockGrid.cpp:1`, `src/app/MainWindow.cpp:runStockSimulation`).
