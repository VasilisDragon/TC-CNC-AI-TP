# CNCTC

CNCTC is an experimental CNC toolpath playground that pairs a Qt 6 desktop client with a modular toolpath and AI stack. The application ingests triangle meshes, extracts geometric descriptors, and blends heuristics with TorchScript/ONNX models to recommend machining strategies in real time.

## Usage & Licensing

- (c) 2025 CNCTC contributors. Redistribution, modification, and commercial use require a valid CNCTC license agreement. By using the binaries or source you agree not to strip or tamper with embedded provenance markers.
- Toolpaths exported by the desktop application include a machine-safe provenance footer. It preserves machining behaviour while allowing a neutral party to attest to a file's origin if that ever becomes a contractual dispute.
- Verification tooling ships only to licensed users upon request and is not part of the public bundle. Keep any forensic utilities in a private workspace alongside your legal paperwork.
- If you need a bespoke license (e.g., OEM redistribution), reach out via the contacts listed in `LICENSE.txt`. Getting ghosted mid-contract doesn't magically erase the provenance tag.

## Architecture At-A-Glance

- **Desktop front-end** - `app/`, `src/app/`, and `resources/` provide the Qt widgets, persistent preferences, build metadata, and icons that make up the user interface.
- **Shared C++ libraries** - `common/`, `io/`, `render/`, and `src/{common,io,render,tp}` expose geometry primitives, importers, GPU viewers, and toolpath kernels (waterline, raster, OpenCAMLib integration).
- **AI integration** - `src/ai/`, `models/`, and `src/train/` orchestrate feature extraction, Torch/ONNX inference (with optional CUDA acceleration), and long running training jobs managed by the application. Every model artefact ships with a sibling `*.model.json` card.
- **Training toolchain** - `train/`, `new_generate_synthetic.py`, and `old_generate_synthetic.py` generate synthetic CAD datasets, train baseline models, and export reproducible artefacts.
- **Docs, tests, and automation** - `docs/`, `tests/`, `samples/`, and `scripts/` provide reference material, doctest-based coverage, sample geometry, and environment diagnostics.

## Toolpath Entry Controls

- `enableRamp` pairs with `rampAngleDeg` to replace plunges with linear entries, while `enableHelical` and `rampRadius` approximate helical drops when the cut plane sits below clearance.
- `leadInLength` and `leadOutLength` add tangent lead segments to raster and waterline passes so the cutter reaches engagement smoothly.
- `cutDirection` flips raster rows and waterline contours between climb and conventional before linking so downstream motion honours the requested chip load.

## Repository Layout

- `app/` - CMake entry point for the GUI. `src/` contains `main.cpp` and widgets such as `ToolpathSettingsWidget`. `include/app/BuildInfo.h.in` is configured at generate time.
- `common/` - Stand-alone Qt utility library (logging, math, tool library) that is linked by the main executable.
- `io/` - Public headers like `io/EmbeddedModel.h` plus the Assimp-backed importer defined in `src/io/`.
- `render/` - Scene graph core, OpenGL shaders (`render/shaders/`), and camera/polyline rendering logic.
- `resources/` - Qt resource collection (`app.qrc`), application icon, and the default tool library (`tools.json`).
- `src/` - Primary module tree:
  - `src/app/` - Main window, dialogs, training workflow panels, and application glue.
  - `src/ai/` - `FeatureExtractor`, `TorchAI`, `OnnxAI`, and `ModelManager` bridging between models and UI.
  - `src/common/` - Runtime units, tool metadata, and JSON-backed tool libraries. The shared `UnitSystem` keeps millimeters as the canonical internal representation while formatting inches for the UI and post processors.
  - `src/io/` - Background worker threads and Assimp importer plumbing.
  - `src/render/` - OpenGL viewer widgets and simulation controller for time-based playback.
  - `src/tp/` - Toolpath generation core, height-field sampling, G-code export, and optional OpenCAMLib bindings.
  - `src/train/` - `EnvManager` and `TrainingManager` that supervise Python jobs from the GUI.
- `train/` - Python utilities:
  - `generate_synthetic.py` (current generator),
  - `common.py` (shared helpers),
  - `train_strategy.py` (model training + export),
  - `make_fixed_model.py` (TorchScript smoke-test stub),
  - `requirements.txt` (pip dependencies).
- `models/` - Example TorchScript/ONNX artefacts and metadata (e.g., `strategy_v2.pt.model.json`).
- `samples/` - STL sample geometry for quick imports.
- `testdata/` - Curated smoke datasets for evaluation (e.g., `testdata/eval/smoke` for the offline metrics script, and `testdata/cad/tiny_block.step` for the OCCT importer sanity test).
- `tests/` - Doctest-powered C++ unit and smoke tests (`basic_sanity.cpp`, `onnx_ai_smoke.cpp`, etc.).
- `docs/` - Architecture notes, build instructions, QA checklists, and deployment playbooks. See `docs/map.md` for a verified module map tying targets to their responsibilities.
- `scripts/verify_env.ps1` - PowerShell helper to validate a Windows build environment.
- Root configuration:
  - `CMakeLists.txt`, `CMakePresets.json`, `BUILD.md`, `vcpkg.json`, and `LICENSE.txt` control the build and dependency graph.

## Build, Run, and Ship Without Drama

1. **Install prerequisites - no vibes without deps**
   - CMake >= 3.22, Ninja or an MSVC/Clang generator, and Qt 6.5+ with Widgets/OpenGL modules.
   - Optional backends: LibTorch (TorchScript inference), ONNX Runtime (ONNX inference), OpenCAMLib (OpenCL drop-cutter). The app auto-detects GPU support and exposes it in AI Preferences so you can flip between "Auto (GPU when available)" and "CPU only" without guessing.
   - vcpkg is already wired through vcpkg.json. Set VCPKG_ROOT or let the preset toolchain take it from there.

2. **Configure**
   ```bash
   cmake --preset vs2022-x64
   ```
   Swap to `build-vs-debug`, `build-vs-release`, or author your own preset if you prefer Ninja/Clang. Enable optional stacks with `-DWITH_TORCH=ON`, `-DWITH_ONNXRUNTIME=ON`, `-DWITH_OCL=ON`, or `-DWITH_OCCT=ON` (requires OpenCASCADE). Double-check `CMAKE_PREFIX_PATH`; mis-pointed SDKs are a jump-scare no one deserves.

3. **Build**
   ```bash
   cmake --build --preset build-vs-debug
   ```

4. **Run**
   Launch the produced executable (AIToolpathGenerator.exe on Windows). First-time checklist:
   - File -> **Open Sample Part** to load samples/sample_part.stl instantly.
   - File -> **Generate Demo Toolpath** to spawn a ready-to-simulate cut using the current settings.
   - **AI -> Preferences...** to confirm the active model, device, and run a latency sanity-check. The dialog now offers "Auto (GPU when available)" so CUDA can flex when it's around and chill when it's not.

5. **Package (one-liner installer)**
   ```bash
   cmake --build --preset build-vs-release --target package
   ```
   With NSIS on the PATH (makensis), CPack emits AIToolpathGenerator-<version>-win64.exe. Hand that single file to teammates and everyone installs the exact same stack. Low effort, high serotonin.

## STEP/IGES CAD Import (optional)

- Configure with `-DWITH_OCCT=ON` to enable the OpenCASCADE-backed CAD importer. On vcpkg-based setups run `vcpkg install cnctc[occt-importer]` first so the OpenCASCADE targets are discoverable.
- The desktop app continues to accept OBJ/STL without OCCT; attempting to open STEP/IGES while the feature is disabled surfaces a dialog with the exact flag/package recipe above.
- `tests/io_step_import.cpp` exercises the tessellation path against `testdata/cad/tiny_block.step`, checking triangle counts and axis-aligned bounds so regressions surface immediately in CI.

## Training & Data Generation

- The **Training Environment** dock fronts train::EnvManager: download embedded Python, verify SHA-256, hydrate a venv, install requirements.txt, and stream logs. Cancel actually cancels, so rage-quitting hotel Wi-Fi is officially supported.
- The **Training** menu now includes:
  - **Prepare Environment** - bootstrap or repair the runtime.
  - **Generate Synthetic Data...** - slider-driven wizard feeding TrainingSyntheticDataDialog.
  - **New Model...** - launches TrainingNewModelDialog with device preference (Auto GPU or CPU-only), dataset path, epochs, learning rate, and the v2 feature toggle.
  - **Fine-Tune Current Model...** - locks to the active inference model, prompts for dataset + hyperparameters, and queues a fine-tune job.
  - **Open Models/Datasets Folder** - jump directly to runtime directories.
- The **Training Jobs** dock lists every job with title, status, ETA, progress bar, and Cancel button. Selecting a job streams raw stdout/stderr into the lower pane, so you can read Python logs without spelunking %LOCALAPPDATA% at 2 a.m.
- Command-line automation stays identical:
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate  # or source .venv/bin/activate
  pip install -r train/requirements.txt
  python train/generate_synthetic.py --out datasets/demo --n 256
  python train/train_strategy.py --output-dir models --epochs 40 --device cpu --v2-features
  python train/make_fixed_model.py --output models/fixed_test.pt
  ```
  Both new_generate_synthetic.py and old_generate_synthetic.py remain for reproducibility and regression testing.

## Model Cards

- Every TorchScript (`*.pt`) and ONNX (`*.onnx`) artefact must ship with a sibling `*.model.json` file in the same directory.
- The schema lives at `ai/model_card.schema.json`; offline validation matches the loader's runtime checks.
- `features.count` must equal `FeatureExtractor::featureCount() + 2` (currently 17) and both `normalize.mean` and `normalize.std` need the same number of entries.
- `training.framework` is compared against the active backend (`PyTorch` for TorchAI, `ONNXRuntime` for OnnxAI). A mismatch disables inference and surfaces the card error in the UI.
- Sample cards can be found under `models/strategy_v2.*.model.json` and make for a good template when exporting new models.

## Models & Runtime AI

- Drop TorchScript (.pt) or ONNX (.onnx) artefacts into models/. ModelManager refreshes the combo, and the Training Jobs dock auto-registers new models once a job succeeds.
- FeatureExtractor computes bounding boxes, slope histograms, curvature, pocket depth, and appends tool parameters before inference. Feature vectors auto pad/truncate-manual tensor carpentry is cancelled.
- TorchAI and OnnxAI expose latency, last-error messages, and GPU availability. They gracefully downgrade to CPU when CUDA providers are absent and keep the UI honest.
- Feature previews log once per session; redact console output if geometry-derived numbers are regulated where you deploy.

## Offline Evaluation Toolkit

- `tools/eval/run_eval.py` mirrors the C++ feature extractor, loads local meshes/metadata, and scores TorchScript or ONNX models. Select the execution target with `--device {auto,cpu,cuda}`.
- Outputs include `metrics.csv` (per-sample predictions, logits, absolute errors) and `report.md` (accuracy, macro-F1, confusion matrix, time proxy).
- Kick the tyres with the bundled dataset:
  ```bash
  python tools/eval/run_eval.py \
      --dataset testdata/eval/smoke \
      --model models/test_run/strategy_v0.onnx \
      --output build/eval/smoke \
      --device auto
  ```
- CUDA runs require the GPU-enabled LibTorch/ONNX packages plus the redistributable CUDA libraries (e.g., `libcublasLt`). Missing pieces trigger a logged warning and a safe fallback to CPU. See `docs/gpu_setup.md` for dependency wiring and troubleshooting.

## Testing

1. Configure a debug build (linking is faster).
2. Run:
   ```bash
   cmake --build --preset build-vs-debug --target tests
   ctest --preset test-vs-debug --output-on-failure
   ```
   Coverage spans importer sanity checks, Torch/ONNX inference, toolpath kernels, and Python orchestration. Skipping tests is basically tweeting "I enjoy shipping gremlins."

## Documentation & Utilities

- BUILD.md, docs/BUILD_DOCTOR.md, docs/OCL_WINDOWS.md - expanded build instructions and troubleshooting.
- docs/QA_CHECKLIST.md, docs/TESTING.md, docs/SMOKE_P*.md - manual QA flows.
- docs/TRAINING_GUI.md - updated walkthrough of the Training menu, environment dock, and jobs panel (now with GPU detection call-outs).
- docs/gpu_setup.md - end-to-end instructions for enabling CUDA-backed Torch/ONNX inference, including runtime diagnostics.
- docs/offline_eval.md - guide for `tools/eval/run_eval.py`, which mirrors the in-app feature extractor and generates CSV/Markdown reports with optional GPU inference.
- resources/tools.json - default tool library for the Toolpath Settings widget.
- samples/sample_part.stl - fast-track demo mesh.
- scripts/verify_env.ps1 - PowerShell helper for MSVC, Qt, CMake, Ninja, LibTorch/ONNX sanity checks.

## Security & Data Handling Notes

- Treat everything in models/ as executable. Verify checksums, control provenance, and prefer signed artefacts.
- Embedded Python downloads stream to disk; verify SHA-256 before extraction:
  ```powershell
  certutil -hashfile python-3.11.5-embed-amd64.zip SHA256
  ```
  ```bash
  shasum -a 256 python-3.11.5-embed-amd64.zip
  ```
- io::ModelImporter trusts incoming meshes-screen untrusted CAD for size and intent.
- Keep third-party dependencies (Qt, LibTorch, ONNX Runtime, CadQuery, build123d, trimesh, etc.) patched. Zero-day roulette is not a personality.
- Disable feature previews or redact logs if they could leak geometry-derived metrics.

## Roadmap Ideas

- Replay recorded spindle/axis telemetry to evaluate AI decisions against real machine sessions.
- Extend feature vectors with curvature histograms, material codes, or tool families once pipelines can surface them reliably.
- Add CI coverage that loads TorchScript/ONNX artefacts, validates schema JSON, and runs representative toolpaths through the simulator.
- Experiment with spatial acceleration structures in OclAdapter::rasterDropCutter to accelerate dense models without sacrificing accuracy.

## Special Note

Thank you for pushing CNCTC forward. Here's to keeping the project extraordinary, unapologetically performant, and just the right amount of chaotic good.

