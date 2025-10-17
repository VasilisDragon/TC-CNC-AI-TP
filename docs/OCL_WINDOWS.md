# OpenCAMLib on Windows

The OpenCAMLib (OCL) project does not currently ship binaries for Windows, but it can be compiled from source with Visual Studio. The steps below outline one supported workflow that works well with this repository.

## 1. Prerequisites

- Visual Studio 2022 with the *Desktop development with C++* workload
- Git and CMake (3.21 or newer)
- Python 3.11 (for generating SWIG interfaces, optional if you only need the static library)

## 2. Fetch the sources

```powershell
cd C:\dev
git clone https://github.com/aewallin/opencamlib.git
cd opencamlib
```

## 3. Configure the build

```powershell
mkdir build && cd build
cmake -G "Visual Studio 17 2022" `
      -DCMAKE_BUILD_TYPE=Release `
      -DBUILD_PYTHON_MODULE=OFF `
      ..
```

- Set `BUILD_PYTHON_MODULE=ON` only if you also need the Python bindings. The CNC toolpath adapter only requires the C++ library.
- The generated solution will place headers under `opencamlib/include` and the library under `build/lib/Release` after compilation.

## 4. Build

```powershell
cmake --build . --config Release --target OpenCAMLib
```

The resulting library will be located at:

- `C:\dev\opencamlib\build\lib\Release\OpenCAMLib.lib` (static)

## 5. Point CMake at OCL

Set the following environment variable before configuring this project:

```powershell
$env:OCL_DIR = "C:\dev\opencamlib"
```

Alternatively, pass the location explicitly when configuring:

```powershell
cmake -S . -B build `
  -DWITH_OCL=ON `
  -DOCL_DIR=C:\dev\opencamlib `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" `
  -DVCPKG_TARGET_TRIPLET=x64-windows
```

The top-level `CMakeLists.txt` looks for headers under `${OCL_DIR}/include` and the library in `${OCL_DIR}/lib` (Release builds place libraries in `lib/Release`; add that directory to `PATH` during development if you use DLLs).

## 6. Verifying integration

1. Reconfigure the project with `-DWITH_OCL=ON`.
2. Build `tp`; the linker should pick up `OpenCAMLib.lib` automatically.
3. At runtime the console will print an `OCL waterline path generated in ... ms` banner whenever the adapter runs successfully. If you only see the raster fallback messages, double-check your include/lib paths.
