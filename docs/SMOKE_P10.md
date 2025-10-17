# Smoke Test P10

1. Download a LibTorch build (CPU or CUDA). Point `TORCH_DIR` at the extracted directory and configure the project with `cmake -S . -B build -DWITH_TORCH=ON`. Confirm CMake reports that LibTorch was found.
2. Build and start the application. Import any mesh so a `render::Model` is active.
3. Choose **AI > Preferences...**. Verify the dialog shows the selected model name, absolute path, last modified timestamp, and the active device string (CPU or CUDA).
4. Toggle **Force CPU inference**. The device label should switch to `CPU (forced)` when checked. Click **Test Inference**; the status line should read “Inference succeeded.” and the console should log latency, strategy, angle, step-over, and device (for example `AI test inference: 2.84 ms via CUDA, Raster @ 45.0 deg, step 3.000 mm`).
5. If the machine has a CUDA-capable GPU, uncheck **Force CPU inference** and run **Test Inference** again. Confirm the device column and log entry now reference CUDA and that latency changes accordingly.
6. Accept the dialog. Reopen **AI > Preferences...** to confirm the force-CPU setting persisted. Finally, generate a toolpath and verify no warnings appear and the console still reports the AI-assisted decision.
