# CNCTC

CNCTC is an experimental CNC toolpath playground that combines a Qt front-end, a lightweight geometry stack, and TorchScript models to explore AI-assisted machining strategies. The project ships with a synthetic data generator, baseline learning utilities, and LibTorch bindings so the desktop application can evaluate decisions in real time.

## Repository Layout

- `app/` – Qt widgets, actions, and application wiring.
- `common/`, `io/`, `render/`, `src/` – shared C++ components used by the desktop client.
- `train/` – synthetic dataset tooling and machine-learning pipelines.
- `docs/` – developer documentation, including AI model interface notes.
- `models/` – pre-trained TorchScript/ONNX artefacts consumed by the app.

## Quick Start

1. **Configure dependencies**
   - Install system prerequisites (CMake, a modern C++ toolchain, Qt 6, and optionally LibTorch if you plan to run the AI path inside the GUI).
   - Create a Python environment for the training utilities:
     ```bash
     python -m venv .venv
     .\.venv\Scripts\activate  # PowerShell on Windows; source .venv/bin/activate on POSIX
     pip install -r train/requirements.txt
     pip install torch onnx onnxscript numpy
     ```

2. **Generate test models**
   - Produce a deterministic TorchScript stub for pipeline smoke tests:
     ```bash
     python -m train.make_fixed_model
     ```
   - Train the synthetic strategy predictor and export TorchScript + ONNX artefacts:
     ```bash
     python -m train.train_strategy --output-dir models
     ```
     The command writes `strategy_v0.pt`, `strategy_v0.onnx`, a schema JSON description, and a model card into `models/`.

3. **Build the desktop app**
   - Point CMake at Qt and (optionally) LibTorch, then configure with `-DWITH_TORCH=ON` to enable AI-assisted planning. See `BUILD.md` for generator-specific tips.

## TorchAI Overview

TorchAI expects a `1 x 6` tensor per inference, ordered as:
1. Bounding box X (mm)
2. Bounding box Y (mm)
3. Bounding box Z (mm)
4. Surface area estimate (mm²)
5. User-selected step-over (mm)
6. Active tool diameter (mm)

The production model (`strategy_v0`) uses these features to emit logits for Raster vs Waterline, a raster hatch angle in degrees, and a step-over limited to 90 % of the tool diameter. See `docs/MODEL_CARD.md` for validation metrics, failure handling, and deployment notes.

## Training Notes

- Synthetic samples mirror the conventions used by `train/generate_synthetic.py` so offline experiments resemble the CAD stock used in demos.
- `train/train_strategy.py` fixes RNG seeds for reproducibility, applies early stopping, and exports both TorchScript (via tracing) and ONNX (opset 17). Feature scaling is embedded in the exported module, allowing the C++ integration to supply raw millimetre values.
- Adjust hyperparameters via CLI flags (e.g., `--samples`, `--epochs`, `--angle-weight`, `--step-weight`). Re-running the script overwrites the artefacts in `models/`.

## Next Steps

- Integrate on-device evaluation with recorded machining sessions to validate predictions against real geometry.
- Expand the feature vector (e.g., curvature summaries, material codes, or tool family identifiers) once the end-to-end pipeline can ingest them.
- Add automated checks that load the TorchScript and ONNX assets during CI to ensure schema drift is caught early.

## Security & Maintenance

- Treat `models/` contents as trusted code. TorchScript and ONNX graphs execute arbitrary compute; only distribute artefacts that come from a controlled build pipeline and verify their hashes before shipping.
- Keep third-party libraries (Qt, Assimp, LibTorch, ONNX Runtime, CadQuery, trimesh, etc.) on supported versions. Monitor upstream advisories and refresh your toolchain before publishing binaries or datasets.
- Avoid committing generated artefacts (build outputs, virtual environments, bytecode) to keep the repository free of host-specific paths or sensitive metadata; the supplied `.gitignore` covers the common cases.

For more detail on geometry generation, refer to `train/README.md`. For deployment guidance and fallback behaviour, see `docs/MODEL_CARD.md`.
