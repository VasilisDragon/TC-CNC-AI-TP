# Windows Packaging

This project ships a CPack/NSIS recipe that produces a standalone installer
containing the Qt runtime, Assimp, optional ONNX Runtime / LibTorch DLLs, and
bundled application assets (`tools.json`, `models/`, and `samples/`).

## Prerequisites
- Visual Studio 2022 with the Desktop development with C++ workload
- Qt 6.5 MSVC x64 build (make sure `Qt6_DIR` or `CMAKE_PREFIX_PATH` points to it)
- NSIS 3.08 or newer available on `PATH`
- (Optional) ONNX Runtime / LibTorch binaries if you plan to enable those features

## Configure the build
```powershell
cmake -S . -B build\windows-release `
      -G "Visual Studio 17 2022" `
      -A x64 `
      -DWITH_ONNXRUNTIME=ON `
      -DWITH_TORCH=OFF
```
Adjust the `WITH_*` options to match the features you want to ship. Leave them
`OFF` if the corresponding SDKs are not available.

## Build and package
```powershell
cmake --build build\windows-release --config Release --target package
```
The `package` target triggers `windeployqt`, copies dependent DLLs (Qt, Assimp,
ONNX Runtime/LibTorch when enabled), and assembles the NSIS installer. The final
installer is written next to the build tree, e.g.:
```
build\windows-release\AIToolpathGenerator-0.1.0-win64.exe
```

To inspect the staged install tree before packaging:
```powershell
cmake --install build\windows-release --config Release --prefix out\windows
```
The install tree in `out\windows` should contain `AIToolpathGenerator.exe`, the
Qt runtime, `tools.json`, `models`, and `samples/sample_part.stl`.

## Running the installer locally
Double-click the generated `AIToolpathGenerator-*.exe`. The first launch of the
installed application automatically opens the bundled sample part and displays a
welcome message guiding the user through the toolpath controls.