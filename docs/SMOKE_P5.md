# Smoke Test P5

1. Launch the app after building (`cmake --build build` then run `build/app/AIToolpathGenerator.exe`).
2. Open a sample OBJ/STL via File -> Open Model... and confirm the Model Browser updates with triangle count and "Toolpath: none".
3. In Toolpath Settings, keep defaults (all fields remain gray) and click Generate Toolpath.
4. Observe in the viewer: green solid lines track cutting moves, yellow dashed lines show rapids above the mesh; the Model Browser now lists segment count plus feed/spindle.
5. Use View -> Reset Camera to verify the toolpath stays overlaid while the camera refits the model.
