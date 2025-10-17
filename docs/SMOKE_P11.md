# Smoke Test P11

1. Install ONNX Runtime via vcpkg (`vcpkg install onnxruntime:x64-windows`). Optionally enable the manifest feature with `vcpkg install --feature-flags=manifests .:onnxruntime` from the repository root.
2. Place a Torch-like model pair in the app's `models` directory: copy `example.onnx` and create `example.onnx.json` alongside it. The JSON sidecar should specify the tensor names, e.g.
   ```json
   {
     "input": "features",
     "outputs": {
       "logits": "logits",
       "angle": "angle",
       "stepover": "step"
     }
   }
   ```
3. Configure the project with `-DWITH_ONNXRUNTIME=ON` (and provide `-DCMAKE_TOOLCHAIN_FILE`/`VCPKG_TARGET_TRIPLET` as needed). Confirm CMake reports that ONNX Runtime was found.
4. Build and launch the application. In the **AI Model** combo the ONNX asset should appear with a `[ONNX]` badge. Selecting it updates the device label (CPU or CUDA if the provider is available) without errors.
5. Open **AI > Preferences...**, toggle **Force CPU inference**, and verify the device label and Test button update accordingly. Press **Test Inference** to record latency; the log should emit a line similar to `[ONNX] 1.92 ms via CUDA, Raster @ 45.0 deg, step 3.000 mm`.
6. Return to the main window, select the ONNX model, and generate a toolpath. Confirm the console reports the strategy using the `[ONNX]` badge and the active device.
