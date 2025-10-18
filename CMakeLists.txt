cmake_minimum_required(VERSION 3.21)

project(AIToolpathGenerator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(GNUInstallDirs)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(Qt6 6.5 COMPONENTS Core Gui Widgets Network OpenGL OpenGLWidgets REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

option(WITH_TORCH "Enable TorchScript inference via LibTorch" OFF)
option(WITH_ONNXRUNTIME "Enable ONNX Runtime inference" OFF)
option(WITH_OCL "Enable OpenCAMLib toolpath generation" OFF)
option(WITH_EMBEDDED_TESTS "Embed diagnostics tests in the desktop application" OFF)

set(AI_TORCH_ENABLED OFF CACHE INTERNAL "Enable Torch integration" FORCE)
if (WITH_TORCH)
    if (NOT TORCH_DIR AND DEFINED ENV{TORCH_DIR})
        set(TORCH_DIR "$ENV{TORCH_DIR}")
    endif()
    if (TORCH_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${TORCH_DIR}")
        if (NOT Torch_DIR AND EXISTS "${TORCH_DIR}/share/cmake/Torch")
            set(Torch_DIR "${TORCH_DIR}/share/cmake/Torch")
        endif()
    endif()
    find_package(Torch QUIET)
    if (Torch_FOUND)
        message(STATUS "Found LibTorch at ${TORCH_INSTALL_PREFIX}")
        set(AI_TORCH_ENABLED ON CACHE INTERNAL "Enable Torch integration" FORCE)
    else()
        message(WARNING "WITH_TORCH enabled but LibTorch not found. Falling back to stub TorchAI.")
    endif()
endif()

set(AI_ONNX_ENABLED OFF CACHE INTERNAL "Enable ONNX integration" FORCE)
if (WITH_ONNXRUNTIME)
    find_package(onnxruntime CONFIG QUIET)
    if (onnxruntime_FOUND)
        message(STATUS "Found ONNX Runtime")
        set(AI_ONNX_ENABLED ON CACHE INTERNAL "Enable ONNX integration" FORCE)
    else()
        message(WARNING "WITH_ONNXRUNTIME enabled but ONNX Runtime not found. Skipping OnnxAI support.")
    endif()
endif()

set(TP_OCL_ENABLED OFF CACHE INTERNAL "Enable OCL integration" FORCE)
if (WITH_OCL)
    if (NOT OCL_DIR AND DEFINED ENV{OCL_DIR})
        set(OCL_DIR "$ENV{OCL_DIR}")
    endif()
    if (OCL_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${OCL_DIR}")
    endif()
    find_path(OCL_INCLUDE_DIR
              NAMES ocl/ocl.h OpenCamLib/ocl.h
              HINTS
                  ${OCL_DIR}
                  ${OCL_DIR}/include
              PATH_SUFFIXES include)
    find_library(OCL_LIBRARY
                 NAMES OpenCAMLib ocl
                 HINTS
                     ${OCL_DIR}
                     ${OCL_DIR}/lib
                     ${OCL_DIR}/lib64
                 PATH_SUFFIXES lib lib64)
    if (OCL_INCLUDE_DIR AND OCL_LIBRARY)
        message(STATUS "Found OpenCAMLib: ${OCL_LIBRARY}")
        set(TP_OCL_ENABLED ON CACHE INTERNAL "Enable OCL integration" FORCE)
    else()
        message(WARNING "WITH_OCL enabled but OpenCAMLib not found. OCL features disabled.")
    endif()
endif()

set(APP_ENABLED_BACKENDS_LIST "")
if (AI_TORCH_ENABLED)
    list(APPEND APP_ENABLED_BACKENDS_LIST "Torch")
endif()
if (AI_ONNX_ENABLED)
    list(APPEND APP_ENABLED_BACKENDS_LIST "ONNX")
endif()
if (TP_OCL_ENABLED)
    list(APPEND APP_ENABLED_BACKENDS_LIST "OCL")
endif()
if (APP_ENABLED_BACKENDS_LIST)
    string(JOIN ", " ENABLED_BACKENDS "${APP_ENABLED_BACKENDS_LIST}")
else()
    set(ENABLED_BACKENDS "None")
endif()

set(GIT_COMMIT_HASH "unknown")
find_package(Git QUIET)
if (Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH_OUTPUT
        ERROR_QUIET
        RESULT_VARIABLE GIT_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (GIT_RESULT EQUAL 0 AND GIT_COMMIT_HASH_OUTPUT)
        set(GIT_COMMIT_HASH "${GIT_COMMIT_HASH_OUTPUT}")
    endif()
endif()

string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%d %H:%M" UTC)
if (CMAKE_CONFIGURATION_TYPES)
    set(BUILD_CONFIG_FALLBACK "Multi-config")
else()
    set(BUILD_CONFIG_FALLBACK "${CMAKE_BUILD_TYPE}")
endif()

add_subdirectory(common)
add_subdirectory(render)
add_subdirectory(src/io)
add_subdirectory(src/ai)
add_subdirectory(src/tp)
if (WITH_EMBEDDED_TESTS)
    add_subdirectory(src/tests_core)
endif()
add_subdirectory(app)

if (AI_ONNX_ENABLED)
    find_package(Threads REQUIRED)
    add_executable(onnx_ai_smoke
        tests/onnx_ai_smoke.cpp
    )
    target_link_libraries(onnx_ai_smoke
        PRIVATE
            ai
            render
            Qt6::Core
            Threads::Threads
    )
    target_compile_definitions(onnx_ai_smoke PRIVATE AI_SMOKE_TEST=1)
endif()

include(CTest)

if (BUILD_TESTING)
    add_executable(basic_sanity_tests
        tests/basic_sanity.cpp
        src/tests_core/basic_toolpath_tests.cpp
    )
    target_include_directories(basic_sanity_tests
        PRIVATE
            ${CMAKE_SOURCE_DIR}/tests
            ${CMAKE_SOURCE_DIR}/src/tests_core
    )
    target_link_libraries(basic_sanity_tests
        PRIVATE
            common
    )

    add_executable(path_safety_tests
        tests/path_safety_tests.cpp
    )
    target_link_libraries(path_safety_tests
        PRIVATE
            tp
    )

    add_executable(headless_pipeline_tests
        tests/headless_pipeline.cpp
    )
    target_link_libraries(headless_pipeline_tests
        PRIVATE
            tp
            io
    )
    string(REPLACE "\\" "\\\\" CNCTC_SOURCE_DIR_ESCAPED "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(headless_pipeline_tests
        PRIVATE
            CNCTC_SOURCE_DIR="${CNCTC_SOURCE_DIR_ESCAPED}"
    )

    add_executable(feature_ai_tests
        tests/feature_ai_doctest.cpp
    )
    target_include_directories(feature_ai_tests
        PRIVATE
            ${CMAKE_SOURCE_DIR}/tests
    )
    target_link_libraries(feature_ai_tests
        PRIVATE
            ai
            tp
            render
            common
    )

    add_test(NAME basic_sanity COMMAND basic_sanity_tests)
    add_test(NAME path_safety COMMAND path_safety_tests)
    add_test(NAME headless_pipeline COMMAND headless_pipeline_tests)
    add_test(NAME feature_ai COMMAND feature_ai_tests)

    set_tests_properties(path_safety PROPERTIES LABELS fast)
    set_tests_properties(headless_pipeline PROPERTIES LABELS fast)
    set_tests_properties(basic_sanity PROPERTIES LABELS fast)
    set_tests_properties(feature_ai PROPERTIES LABELS fast)

    if (TARGET onnx_ai_smoke)
        add_test(NAME onnx_ai_smoke_test COMMAND onnx_ai_smoke)
        set_tests_properties(onnx_ai_smoke_test PROPERTIES LABELS fast)
    endif()
endif()

if (MSVC)
    add_compile_options(/W4 /permissive- /Zc:preprocessor /EHsc /bigobj)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

add_compile_definitions(
    $<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN;NOMINMAX;_USE_MATH_DEFINES;UNICODE;_UNICODE>
)

set(EXTRA_RUNTIME_DLLS "")
if (TARGET assimp::assimp)
    get_target_property(ASSIMP_SHARED_LOCATION assimp::assimp IMPORTED_LOCATION_RELEASE)
    if (NOT ASSIMP_SHARED_LOCATION)
        get_target_property(ASSIMP_SHARED_LOCATION assimp::assimp IMPORTED_LOCATION)
    endif()
    if (ASSIMP_SHARED_LOCATION)
        list(APPEND EXTRA_RUNTIME_DLLS "${ASSIMP_SHARED_LOCATION}")
    endif()
endif()

set(_vcpkg_runtime_search_paths "")
if (CMAKE_BINARY_DIR)
    list(APPEND _vcpkg_runtime_search_paths
        "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/bin"
        "${CMAKE_BINARY_DIR}/vcpkg_installed/x64-windows/debug/bin")
endif()
if (DEFINED VCPKG_INSTALLED_DIR)
    list(APPEND _vcpkg_runtime_search_paths
        "${VCPKG_INSTALLED_DIR}/x64-windows/bin"
        "${VCPKG_INSTALLED_DIR}/x64-windows/debug/bin")
endif()

set(_third_party_runtime_dlls
    "brotlicommon.dll"
    "brotlidec.dll"
    "brotlienc.dll"
    "bz2.dll"
    "double-conversion.dll"
    "freetype.dll"
    "harfbuzz.dll"
    "kubazip.dll"
    "libpng16.dll"
    "minizip.dll"
    "pcre2-16.dll"
    "poly2tri.dll"
    "pugixml.dll"
    "zlib1.dll"
    "zstd.dll"
)

foreach(_dll_name IN LISTS _third_party_runtime_dlls)
    string(REPLACE "." "_" _dll_cache_key "${_dll_name}")
    string(REPLACE "-" "_" _dll_cache_key "${_dll_cache_key}")
    string(TOUPPER "${_dll_cache_key}" _dll_cache_key)
    set(_dll_cache_var "CNCTC_RUNTIME_${_dll_cache_key}")
    unset(${_dll_cache_var} CACHE)
    if (_vcpkg_runtime_search_paths)
        find_file(${_dll_cache_var}
            NAMES "${_dll_name}"
            PATHS ${_vcpkg_runtime_search_paths}
            NO_DEFAULT_PATH)
    endif()
    if (NOT ${_dll_cache_var})
        find_file(${_dll_cache_var} NAMES "${_dll_name}")
    endif()
    if (${_dll_cache_var})
        get_filename_component(_dll_found_dir "${${_dll_cache_var}}" DIRECTORY)
        if (_dll_found_dir MATCHES "/debug/bin$" OR _dll_found_dir MATCHES "\\\\debug\\\\bin$")
            get_filename_component(_dll_found_parent "${_dll_found_dir}" DIRECTORY)
            get_filename_component(_dll_found_grand "${_dll_found_parent}" DIRECTORY)
            if (EXISTS "${_dll_found_grand}/bin/${_dll_name}")
                list(APPEND EXTRA_RUNTIME_DLLS "${_dll_found_grand}/bin/${_dll_name}")
            else()
                list(APPEND EXTRA_RUNTIME_DLLS "${${_dll_cache_var}}")
            endif()
        else()
            list(APPEND EXTRA_RUNTIME_DLLS "${${_dll_cache_var}}")
        endif()
    endif()
endforeach()

if (AI_ONNX_ENABLED AND TARGET onnxruntime::onnxruntime)
    get_target_property(ONNX_SHARED_LOCATION onnxruntime::onnxruntime IMPORTED_LOCATION_RELEASE)
    if (NOT ONNX_SHARED_LOCATION)
        get_target_property(ONNX_SHARED_LOCATION onnxruntime::onnxruntime IMPORTED_LOCATION)
    endif()
    if (ONNX_SHARED_LOCATION)
        list(APPEND EXTRA_RUNTIME_DLLS "${ONNX_SHARED_LOCATION}")
        get_filename_component(ONNX_DLL_DIR "${ONNX_SHARED_LOCATION}" DIRECTORY)
        if (EXISTS "${ONNX_DLL_DIR}")
            file(GLOB ONNX_PROVIDER_DLLS "${ONNX_DLL_DIR}/onnxruntime_providers_*.dll")
            list(APPEND EXTRA_RUNTIME_DLLS ${ONNX_PROVIDER_DLLS})
        endif()
    endif()
endif()

if (AI_TORCH_ENABLED AND TORCH_DLLS)
    foreach(TORCH_DLL IN LISTS TORCH_DLLS)
        if (TORCH_DLL MATCHES "\\.(dll|DLL)$" AND EXISTS "${TORCH_DLL}")
            list(APPEND EXTRA_RUNTIME_DLLS "${TORCH_DLL}")
        endif()
    endforeach()
endif()

if (EXTRA_RUNTIME_DLLS)
    list(REMOVE_DUPLICATES EXTRA_RUNTIME_DLLS)
    install(FILES ${EXTRA_RUNTIME_DLLS} DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

set(CPACK_GENERATOR "NSIS")
set(CPACK_PACKAGE_NAME "AIToolpathGenerator")
set(CPACK_PACKAGE_VENDOR "CNCTC")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AI-assisted toolpath generator demo application.")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
set(CPACK_NSIS_PACKAGE_NAME "${CPACK_PACKAGE_NAME}")
set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_NAME}")
set(CPACK_NSIS_URL_INFO_ABOUT "https://cnctc.dev/")
set(CPACK_NSIS_CONTACT "support@cnctc.dev")
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/resources/app.ico")
set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/resources/app.ico")
set(CPACK_NSIS_INSTALLED_ICON_NAME "${CMAKE_INSTALL_BINDIR}/AIToolpathGenerator.exe")
set(CPACK_NSIS_EXECUTABLES_DIRECTORY "${CMAKE_INSTALL_BINDIR}")
set(CPACK_PACKAGE_EXECUTABLES "AIToolpathGenerator" "AIToolpathGenerator")
set(CPACK_CREATE_DESKTOP_LINKS "AIToolpathGenerator")
set(CPACK_NSIS_MUI_FINISHPAGE_RUN "${CMAKE_INSTALL_BINDIR}/AIToolpathGenerator.exe")

# Try to locate the NSIS compiler automatically so the `package` target can run unattended.
find_program(CPACK_NSIS_EXECUTABLE
    NAMES makensis makensis.exe
    HINTS
        ENV NSIS_ROOT
        "C:/Program Files/NSIS"
        "C:/Program Files (x86)/NSIS"
    PATH_SUFFIXES bin
)
if (CPACK_NSIS_EXECUTABLE)
    message(STATUS "NSIS (makensis) detected at: ${CPACK_NSIS_EXECUTABLE}")
else()
    message(WARNING "NSIS (makensis) not found. Install NSIS or set NSIS_ROOT to enable the installer packaging target.")
endif()

include(CPack)
