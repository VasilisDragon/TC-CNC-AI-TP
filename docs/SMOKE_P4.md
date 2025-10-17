# Smoke Test P4

1. **Configure & build**
   - `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release`
   - `cmake --build build`
2. **Run the app**
   - Launch `build/app/AIToolpathGenerator.exe`.
3. **Open a mesh**
   - File -> Open Model..., pick a small OBJ or STL (e.g. a cube).
   - Confirm the Model Browser shows the file name and triangle count (should match the mesh; a cube with triangulated faces reports 12 triangles).
4. **Verify camera controls**
   - Use View -> Reset Camera to refocus on the loaded model and ensure the viewer frames the mesh correctly.
