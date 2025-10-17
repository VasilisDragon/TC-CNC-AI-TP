cmake_minimum_required(VERSION 3.21)

project(AIToolpathGenerator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

include(GNUInstallDirs)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

find_package(Qt6 6.5 COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

option(WITH_TORCH "Enable TorchScript inference via LibTorch" OFF)
option(WITH_ONNXRUNTIME "Enable ONNX Runtime inference" OFF)
option(WITH_OCL "Enable OpenCAMLib toolpath generation" OFF)

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
add_subdirectory(app)

if (AI_ONNX_ENABLED)
    add_executable(onnx_ai_smoke
        tests/onnx_ai_smoke.cpp
    )
    target_link_libraries(onnx_ai_smoke
        PRIVATE
            ai
            render
            Qt6::Core
    )
    target_compile_definitions(onnx_ai_smoke PRIVATE AI_SMOKE_TEST=1)
endif()

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

include(CPack)
