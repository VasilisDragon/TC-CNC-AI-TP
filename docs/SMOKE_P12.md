# Smoke Test P12

**Objective:** confirm OpenCAMLib toolpaths are invoked and logged.

1. Build OpenCAMLib and configure the project with `-DWITH_OCL=ON` (see `docs/OCL_WINDOWS.md`). Run `cmake --build` to ensure linking succeeds.
2. Launch the application and load a moderately detailed mesh. Ensure the tool diameter and max depth per pass are reasonable (for example, a 6 mm tool with 1 mm max depth per pass).
3. Open **AI > Preferences...**, disable **Force CPU inference** if you plan to test CUDA, then close the dialog with **OK**.
4. Trigger toolpath generation. Watch the Console for a line similar to `OCL waterline path generated in 120.45 ms`. If the console prints an *OCL error* banner and falls back to the raster placeholder, revisit the OCL installation.
5. Switch the AI strategy to Raster (either via the model or Preferences) and generate again. Expect a banner such as `OCL drop-cutter path generated in 95.32 ms`.
6. Toggle **Force CPU inference** in Preferences, regenerate, and verify that latency updates and the banner still appears. This confirms the OCL adapter works regardless of the inference device.
