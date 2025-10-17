# Training Environment

The Training Environment manager prepares a self-contained Python stack that mirrors the runtime requirements for model training. Everything is stored in `runtime/` next to the application and can be regenerated at any time.

## What Gets Installed
- **Python 3.11** (embedded Windows build, downloaded automatically if a suitable system Python is unavailable)
- **Virtual environment** at `runtime/venv`
- Python packages:
  - `torch`, `torchvision`, `torchaudio` (CUDA wheels when an NVIDIA GPU is detected, otherwise CPU wheels)
  - `onnxruntime-gpu` (or `onnxruntime` for CPU-only setups)
  - `cadquery`, `build123d`
  - `trimesh`, `numpy`, `matplotlib`
- The environment footprint is approximately **3.5-6.0 GB** depending on whether CUDA toolkits are downloaded.

## CPU-Only Mode
The Training Environment dock exposes a CPU-only toggle (persisted in QSettings) that forces the manager to install CPU wheels even when a CUDA-capable GPU is discovered. This is useful for laptops with unsupported driver stacks or for minimising download size (~1.5 GB).

## Storage Layout
```
runtime/
+-- python/          # Embedded interpreter if downloaded
+-- venv/            # Virtual environment with pip-installed packages
+-- temp/            # Temporary download/extraction staging
```

## Re-running the Setup
Use the **Prepare/Repair Environment** button inside the "Training Environment" dock. The manager will:
1. Locate or download Python.
2. Recreate the virtual environment.
3. Upgrade `pip` and reinstall the packages listed above.
4. Persist the status in `QSettings` so the UI can show whether the environment is ready.

Cancelling the process politely terminates any active `pip` command; rerunning will pick up from scratch.

## GPU Detection
The dock reports the first CUDA device using `nvidia-smi`. If no compatible GPU is found, the manager defaults to CPU packages.

## Troubleshooting
- Ensure the application can reach `python.org` for the embedded download. If you must work offline, place the embeddable `.zip` in `runtime/python/` ahead of time.
- Running the manager from an elevated PowerShell prompt may be required if corporate policies block `Expand-Archive` from unsigned folders.
- The full log is visible in the dock and persisted for the current session.
