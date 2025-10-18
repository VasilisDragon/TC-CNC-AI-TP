# Build Doctor

This page collects the most common build-time issues we have seen on a fresh Windows 11 workstation. Work through these in order before filing a bug.

## Toolchain Setup
- **Visual Studio 2022**: install the "Desktop development with C++" workload, including the latest MSVC toolset and Windows 11 SDK. Always open the *x64 Native Tools Command Prompt* before running CMake/Ninja.
- **CMake + MSVC toolchain**: install the latest CMake from [cmake.org](https://cmake.org) and the Visual Studio 2022 Build Tools (C++ workload). The default presets target `vs2022-x64`, but you can add a Ninja preset if you keep that workflow.
- **Qt 6.5+**: install the MSVC 64-bit binary build. Set `Qt6_DIR` (or `Qt6Core_DIR`) to the `lib/cmake/Qt6` directory so that `find_package(Qt6 ...)` succeeds.

## Optional Backends
- **LibTorch (`-DWITH_TORCH=ON`)**: download the prebuilt Release+Debug zip that matches your MSVC version. Set `TORCH_DIR` to the root folder (the one containing `share/cmake/Torch`). The generator enables the backend only when `find_package(Torch)` succeeds.
- **ONNX Runtime (`-DWITH_ONNXRUNTIME=ON`)**: install the CPU package and point `onnxruntime_DIR` at its CMake directory (usually `<install>/lib/cmake/onnxruntime`).
- **OpenCAMLib (`-DWITH_OCL=ON`)**: provide `OCL_DIR` so the build can locate both the headers and the import library. Without it the target compiles with CPU stubs.

## Frequent Errors
- **`fatal error C1083: Cannot open include file: 'QtCore/qobject.h'`**  
  Qt was not located. Verify `Qt6_DIR` (see above) and that you are using the x64 Native Tools prompt.
- **`LINK : fatal error LNK1104: cannot open file 'libtorch.lib'`**  
  Torch backend requested but libraries missing. Either set `TORCH_DIR` correctly or disable the backend (`-DWITH_TORCH=OFF`).
- **`The C compiler identification is unknown`**  
  MSVC environment not initialised. Launch the VS "Developer Command Prompt" or run `vcvars64.bat` before invoking CMake.
- **`No known features for CXX compiler` while using Ninja**  
  Update CMake to 3.21+ and verify Ninja is the matching architecture (use the prebuilt release zip).
- **`error MSB8020` in Visual Studio**  
  The project is opened without the required MSVC toolset or Windows SDK. Install the latest toolset and retarget the solution.

## Quick Checklist
1. Run `scripts/verify_env.ps1` - it checks CMake, Ninja, `cl.exe`, Qt and optional SDK hints.
2. Use `cmake --preset vs2022-x64` (or your custom Ninja preset) to configure.
3. Build with `cmake --build --preset build-vs-debug` (or the release variant you need).
4. Launch the desktop app from `build/vs2022/app/Release/AIToolpathGenerator.exe`.
