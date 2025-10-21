set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_PREFIX_PATH "D:/Codex/CNCTC-main/build/vs2022/vcpkg_installed/x64-windows;D:/Codex/CNCTC-main/build/vs2022/vcpkg_installed/x64-windows/share")
find_package(OpenCASCADE REQUIRED COMPONENTS FoundationClasses ModelingData ModelingAlgorithms DataExchange)
message(STATUS "OpenCASCADE_FOUND=")
