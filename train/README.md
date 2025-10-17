# Synthetic Dataset Tooling

The `train/` package generates small machining datasets composed of CAD solids,
their corresponding triangulated meshes, and strategy labels useful for
bootstrapping AI-driven CAM research.

## Requirements

Create a clean Python environment (3.10 or newer is recommended) and install the
dependencies pinned for repeatability:

```bash
python -m venv .venv
source .venv/bin/activate               # or .venv\Scripts\activate on Windows
pip install -r train/requirements.txt
```

Key packages:

* **CadQuery** `2.3.1` — parametric solid modelling used to author randomized stock.
* **NumPy** `1.26.4` — random sampling, numeric helpers.
* **trimesh** `4.2.3` — STL I/O and bounding-box calculations.

## Generator usage

```bash
python -m train.generate_synthetic --out data/synth --n 300 --seed 1337 --force
```

Arguments:

* `--out`: target directory that will contain `sample_XXXX` subfolders.
* `--n`: number of synthetic samples to emit (defaults to `300`).
* `--seed`: base RNG seed for reproducibility (`2025` if omitted).
* `--force`: overwrite individual sample folders when re-running.

Each sample directory contains:

```
sample_0001/
  ├── mesh.stl          # triangulated CadQuery export in millimetres
  └── meta.json         # metadata contract described below
```

The metadata schema follows the project contract:

```json
{
  "bbox": [x_mm, y_mm, z_mm],
  "material": "MDF",
  "tool_diameter_mm": 6.0,
  "label": {
    "strategy": "raster" | "waterline",
    "angle_deg": 0-180,
    "step_over_mm": float
  }
}
```

## Geometry recipe

For every sample the generator:

1. Builds a rectangular stock (60–140 mm × 40–110 mm × 15–60 mm).
2. Applies a random sequence of features (rectangular and circular pockets,
   raised bosses, vertical fillets, and chamfers) using CadQuery.
3. Exports the solid as `mesh.stl` with moderate tessellation tolerances.
4. Reloads the mesh via `trimesh` to compute the axis-aligned bounding box.
5. Chooses a toolpath label heuristic:
   * **Raster** with a random hatch angle when the part is relatively flat
     (z-span < 28% of the largest planar span).
   * **Waterline** when the part demonstrates tall / steep features, using a
     smaller step-over and neutral angle.
6. Writes `meta.json` with the final label and supporting metadata.

The heuristics are intentionally simple to keep the dataset reproducible without
requiring external CAM software. They can be extended to consult OpenCAMLib or
FreeCAD Path for verification in future iterations.

## Licensing

All generated assets and the accompanying scripts are released under the same
license as the main repository (see the repository root for details). The
synthesised geometry is purely algorithmic and contains no third-party models.

## Strategy predictor training

Install the additional dependencies required for the learning pipeline:

```bash
pip install torch onnx onnxscript numpy
```

Train the machining strategy model and emit the artefacts described in the project brief:

```bash
python -m train.train_strategy --output-dir models
```

The command writes `strategy_v0.pt`, `strategy_v0.onnx`, `strategy_v0.onnx.json`, and `strategy_v0.card.json` into the target directory. Re-run the script with different flags (see `--help`) to customise epochs, dataset size, or destinations.
