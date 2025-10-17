# CNCTC

CNCTC is an experimental CNC toolpath playground that pairs a Qt 6 desktop client with a modular toolpath and AI stack. The application ingests triangle meshes, extracts geometric descriptors, and blends heuristics with TorchScript/ONNX models to recommend machining strategies in real time.

## Architecture At-A-Glance

- **Desktop front-end** – `app/`, `src/app/`, and `resources/` provide the Qt widgets, persistent preferences, build metadata, and icons that make up the user interface.
- **Shared C++ libraries** – `common/`, `io/`, `render/`, and `src/{common,io,render,tp}` expose geometry primitives, importers, GPU viewers, and toolpath kernels (waterline, raster, OpenCAMLib integration).
- **AI integration** – `src/ai/`, `models/`, and `src/train/` orchestrate feature extraction, Torch/ONNX inference, and long running training jobs managed by the application.
- **Training toolchain** – `train/`, `new_generate_synthetic.py`, and `old_generate_synthetic.py` generate synthetic CAD datasets, train baseline models, and export reproducible artefacts.
- **Docs, tests, and automation** – `docs/`, `tests/`, `samples/`, and `scripts/` provide reference material, doctest-based coverage, sample geometry, and environment diagnostics.

## Repository Layout

- `app/` – CMake entry point for the GUI. `src/` contains `main.cpp` and widgets such as `ToolpathSettingsWidget`. `include/app/BuildInfo.h.in` is configured at generate time.
- `common/` – Stand-alone Qt utility library (logging, math, tool library) that is linked by the main executable.
- `io/` – Public headers like `io/EmbeddedModel.h` plus the Assimp-backed importer defined in `src/io/`.
- `render/` – Scene graph core, OpenGL shaders (`render/shaders/`), and camera/polyline rendering logic.
- `resources/` – Qt resource collection (`app.qrc`), application icon, and the default tool library (`tools.json`).
- `src/` – Primary module tree:
  - `src/app/` – Main window, dialogs, training workflow panels, and application glue.
  - `src/ai/` – `FeatureExtractor`, `TorchAI`, `OnnxAI`, and `ModelManager` bridging between models and UI.
  - `src/common/` – Runtime units, tool metadata, and JSON-backed tool libraries.
  - `src/io/` – Background worker threads and Assimp importer plumbing.
  - `src/render/` – OpenGL viewer widgets and simulation controller for time-based playback.
  - `src/tp/` – Toolpath generation core, height-field sampling, G-code export, and optional OpenCAMLib bindings.
  - `src/train/` – `EnvManager` and `TrainingManager` that supervise Python jobs from the GUI.
- `train/` – Python utilities:
  - `generate_synthetic.py` (current generator),
  - `common.py` (shared helpers),
  - `train_strategy.py` (model training + export),
  - `make_fixed_model.py` (TorchScript smoke-test stub),
  - `requirements.txt` (pip dependencies).
- `models/` – Example TorchScript/ONNX artefacts and metadata (e.g., `strategy_v2.card.json`).
- `samples/` – STL sample geometry for quick imports.
- `tests/` – Doctest-powered C++ unit and smoke tests (`basic_sanity.cpp`, `onnx_ai_smoke.cpp`, etc.).
- `docs/` – Architecture notes, build instructions, QA checklists, and deployment playbooks.
- `scripts/verify_env.ps1` – PowerShell helper to validate a Windows build environment.
- Root configuration:
  - `CMakeLists.txt`, `CMakePresets.json`, `BUILD.md`, `vcpkg.json`, and `LICENSE.txt` control the build and dependency graph.

## Build & Run (Desktop App)

1. **Install prerequisites**
   - CMake ≥ 3.22, Ninja or an MSVC/Clang generator, and Qt 6.5+ with Widgets/OpenGL modules.
   - Optional backends: LibTorch (for TorchScript inference), ONNX Runtime (for ONNX inference), and OpenCAMLib if you plan to enable OpenCL drop-cutter.
   - The repository ships with a `vcpkg.json`; configure `VCPKG_ROOT` or integrate via your toolchain file if you prefer vcpkg.

2. **Configure**
   ```bash
   cmake --preset ninja-release
   ```
   Adjust the preset or generator (e.g., `ninja-debug`, `visual-studio`) and pass `-DWITH_TORCH=ON`, `-DWITH_ONNXRUNTIME=ON`, or `-DWITH_OCL=ON` as needed. The optional modules expect their SDKs on `PATH`/`CMAKE_PREFIX_PATH`.

3. **Build**
   ```bash
   cmake --build --preset ninja-release
   ```

4. **Run**
   Launch the produced `cnctc` executable. The app will prompt for model directories, tool libraries (`resources/tools.json`), and provides dialogs for training, synthetic data generation, and AI runtime preferences.

## Training & Data Generation

- `src/train/EnvManager` prepares an isolated Python runtime. Embedded Python downloads are now streamed directly to disk, abort cleanly if you cancel, and can be hash-checked before extraction.
- Inside the GUI you can trigger environment preparation, synthetic dataset generation, or model training jobs; `TrainingManager` manages background `QProcess` instances and surfaces stdout/stderr in the console dock.
- Command-line usage mirrors the GUI:
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate  # or source .venv/bin/activate
  pip install -r train/requirements.txt
  python train/generate_synthetic.py --out datasets/demo --n 256
  python train/train_strategy.py --output-dir models --epochs 40 --device cpu --v2-features
  python train/make_fixed_model.py --output models/fixed_test.pt
  ```
- `new_generate_synthetic.py` and `old_generate_synthetic.py` are retained as reference revisions when iterating on dataset heuristics.

## Models & Runtime AI

- `models/` hosts TorchScript (`.pt`) and ONNX (`.onnx`) artefacts plus companion schema/model-card JSON. Load your own models by dropping them here; `ModelManager` indexes the directory at startup.
- `FeatureExtractor` (`src/ai/FeatureExtractor.*`) produces global metrics: bounding boxes, slope histograms, curvature, pocket depth, and appends user tool parameters before inference. Feature vectors are padded/truncated automatically to match model expectations.
- `TorchAI` and `OnnxAI` share a common fallback strategy, gracefully downgrade to CPU, capture latency metrics, and emit warning banners when artefacts misbehave.
- `TorchAI` logs a single feature preview per session to aid debugging; disable or redact application logs if those values are sensitive for your deployment.

## Testing

1. Configure a build as above (debug presets speed up unit testing).
2. Build and run the tests:
   ```bash
   cmake --build --preset ninja-debug --target tests
   ctest --preset ninja-debug --output-on-failure
   ```
   The suite covers importer sanity checks, Torch/ONNX inference, toolpath geometry, and regression fixtures under `tests/`.

## Documentation & Utilities

- `BUILD.md`, `docs/BUILD_DOCTOR.md`, and `docs/OCL_WINDOWS.md` expand on build generator specifics, environment troubleshooting, and optional OpenCL integration.
- `docs/QA_CHECKLIST.md`, `docs/TESTING.md`, and the `docs/SMOKE_P*.md` series capture manual validation flows.
- `resources/tools.json` seeds the `ToolpathSettingsWidget` list; edit it or point the UI at your own JSON.
- `samples/sample_part.stl` gives the viewer an immediate mesh to explore.
- `scripts/verify_env.ps1` validates MSVC, CMake, Ninja, and Qt hints on Windows shells.

## Security & Data Handling Notes

- Treat everything in `models/` as executable code: verify hashes, control provenance, and prefer signed artefacts for distribution.
- The embedded Python bootstrap now writes the archive incrementally, but it still downloads from the public Python CDN. Capture the expected SHA-256 out of band and compare before trusting the environment in high-assurance contexts.
- `io::ModelImporter` (OBJ/STL via Assimp) trusts client-supplied meshes. Vet untrusted CAD files for size and malicious constructs to avoid memory exhaustion or processing spikes.
- Feature previews sent to the log (`TorchAI::logFeaturePreview`) contain the first few normalized values. Disable logging or filter stdout/stderr if product environments must avoid leaking geometry metadata.
- Keep third-party dependencies (Qt, LibTorch, ONNX Runtime, CadQuery, build123d, trimesh, etc.) patched to pick up CVEs.
- Verify large artefacts (models, embedded Python zip) before trusting them. For example:
  ```powershell
  certutil -hashfile python-3.11.5-embed-amd64.zip SHA256
  ```
  or on POSIX shells:
  ```bash
  shasum -a 256 python-3.11.5-embed-amd64.zip
  ```
  Compare the result with the published checksum before proceeding.

## Roadmap Ideas

- Replay recorded spindle/axis telemetry to evaluate AI decisions against real machine sessions.
- Extend the feature vector with curvature histograms, material codes, or tool families once the data pipeline can surface them reliably.
- Add CI coverage that loads TorchScript/ONNX artefacts, validates schema JSON, and runs a representative toolpath through the simulator.
- Experiment with spatial acceleration structures in `OclAdapter::rasterDropCutter` to accelerate dense models without sacrificing accuracy.

## Special Note

Thank you for pushing CNCTC forward. Here’s to keeping the project extraordinary, inorbinantly performant, and joyfully out of the box.
