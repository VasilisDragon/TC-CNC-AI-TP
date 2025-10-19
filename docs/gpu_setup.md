# GPU Inference Setup

To use CUDA acceleration for either TorchScript or ONNX Runtime inference you
need the GPU-enabled binaries at build time and a driver/runtime visible at
run time. The steps below assume Windows + CMake presets, but the same layout
works on Linux with minor path adjustments.

## 1. Install NVIDIA Drivers

Make sure the host exposes a CUDA-capable GPU to the environment that runs the
app or the evaluation scripts:

1. Install the latest NVIDIA Game Ready / Studio driver (Windows) or CUDA
   driver package (Linux).
2. When working inside WSL, install the Windows driver **and** the `nvidia-smi`
   packages inside WSL so `/dev/dxg` and `/usr/lib/wsl/lib` are available.

You can validate visibility with `nvidia-smi` (system-wide) and
`torch.cuda.is_available()` / `onnxruntime.get_available_providers()` (see the
diagnostics section below).

## 2. TorchScript (LibTorch)

1. Download the matching LibTorch CUDA archive from
   <https://pytorch.org/get-started/locally/> (select the C++ / LibTorch option
   and CUDA toolkit version that matches your driver).
2. Extract the archive somewhere stable (e.g. `D:\deps\libtorch-cuda`).
3. Point CMake to the unpacked directory:
   - set `TORCH_DIR` in the environment before running CMake, or
   - pass `-DTORCH_DIR=D:/deps/libtorch-cuda` when configuring.
4. Configure with `-DWITH_TORCH=ON`. The top-level CMakeLists caches this in
   `AI_TORCH_ENABLED`; the CUDA-enabled build of LibTorch shares the same
   CMake package name as the CPU build.
5. Ensure the LibTorch `bin/` folder is added to `PATH` (Windows) or
   `LD_LIBRARY_PATH` (Linux) when launching the app so the CUDA runtime DLLs /
   SOs can be resolved at run time.

## 3. ONNX Runtime

1. Install the CUDA-flavoured ONNX Runtime package:
   - **vcpkg**: `vcpkg install onnxruntime[cuda]` (requires installing the CUDA
     toolkit first).
   - **Manual**: Download the `onnxruntime-win-x64-gpu` or
     `onnxruntime-linux-x64-gpu` archive from
     <https://github.com/microsoft/onnxruntime/releases> and extract it (e.g.
     to `D:\deps\onnxruntime-gpu`).
2. Append the root to `CMAKE_PREFIX_PATH` (or specify
   `-DONNXRUNTIME_ROOTDIR=...`) before running CMake.
3. Configure with `-DWITH_ONNXRUNTIME=ON`. When the CUDA provider is present, the
   build links `onnxruntime::onnxruntime_providers_cuda` automatically and
   `OnnxAI` can request `CUDAExecutionProvider` at runtime.
4. Add the ONNX Runtime `lib/` directory to the dynamic library search path
   (Windows: `PATH`, Linux: `LD_LIBRARY_PATH`).
5. Re-run CMake and check the configure log for
   `OnnxAI: available providers - CUDAExecutionProvider, CPUExecutionProvider`
   to confirm the GPU-enabled package was detected. If you only see
   `CPUExecutionProvider` (or `AzureExecutionProvider` on Windows), you are still
   linking against the CPU-only build.
6. The PyPI wheel bundles most CUDA libraries but still expects vendor DLLs /
   SOs such as `libcublasLt.so` to be visible at run time. Install the matching
   CUDA Toolkit (or copy the redistributable DLLs) and make sure their `lib/`
   directory is on the library search path, otherwise ONNX Runtime will emit
   errors like `Failed to load library libonnxruntime_providers_cuda.so:
   libcublasLt.so.11 not found` before falling back to the CPU provider.

## 4. Diagnostics and Verification

After building a CUDA-enabled configuration:

- Launch the application with `--loglevel=info` (or watch standard output) to
  see the new device diagnostics. `TorchAI` logs whether CUDA is available and
  which device has been selected; `OnnxAI` reports the ONNX Runtime providers
  detected.
- In a terminal, run:

  ```bash
  python tools/eval/run_eval.py \
      --dataset testdata/eval/smoke \
      --model models/test_run/strategy_v0.onnx \
      --output build/eval/smoke_gpu \
      --device cuda
  ```

  The script now lists the available providers and switches to the GPU if the
  provider is functional.

If CUDA cannot be initialised (driver mismatch, missing DLLs, etc.) both Torch
and ONNX paths fall back to CPU automatically. Check the log output for the
exact failure reason.
