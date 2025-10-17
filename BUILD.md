# Build Instructions

The project targets Windows 11 with MSVC, Qt 6, and vcpkg. The instructions below assume:

- Visual Studio 2022 with the *Desktop development with C++* workload
- vcpkg cloned locally (e.g. `C:\dev\vcpkg`)
- Ninja installed (ships with recent CMake or install separately)

## 1. Bootstrap vcpkg (first time only)

```powershell
cd C:\dev\vcpkg
.\bootstrap-vcpkg.bat
```

Integrate (optional but convenient):

```powershell
.\vcpkg integrate install
```

## 2. Install dependencies

```powershell
cd C:\dev\vcpkg
.\vcpkg install qtbase[opengl]:x64-windows qttools:x64-windows assimp:x64-windows
```

If STEP import support is desired, add the optional feature:

```powershell
.\vcpkg install assimp[step]:x64-windows
```

## 3. Configure the project

```powershell
$env:VCPKG_ROOT = "C:\dev\vcpkg"
cmake -G Ninja -S . -B build `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_BUILD_TYPE=Release
```

To configure a Debug build, change `-DCMAKE_BUILD_TYPE=Release` to `Debug`.

## 4. Build

```powershell
cmake --build build
```

The main executable (`AIToolpathGenerator.exe`) will be placed in `build\app`.

## 5. Run

```powershell
.\build\app\AIToolpathGenerator.exe
```

High-DPI scaling and the Fusion dark palette are enabled by default.

