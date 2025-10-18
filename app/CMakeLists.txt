qt_add_executable(app
    ${CMAKE_SOURCE_DIR}/src/app/MainWindow.h
    ${CMAKE_SOURCE_DIR}/src/app/AiPreferencesDialog.h
    ${CMAKE_SOURCE_DIR}/src/app/TrainingNewModelDialog.h
    ${CMAKE_SOURCE_DIR}/src/app/TrainingSyntheticDataDialog.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/app/ToolpathSettingsWidget.h
    ${CMAKE_SOURCE_DIR}/src/app/MainWindow.cpp
    ${CMAKE_SOURCE_DIR}/src/app/AiPreferencesDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/app/TrainingNewModelDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/app/TrainingSyntheticDataDialog.cpp
    ${CMAKE_SOURCE_DIR}/src/app/AiPreferencesDialog.ui
    ${CMAKE_SOURCE_DIR}/src/app/TrainingNewModelDialog.ui
    ${CMAKE_SOURCE_DIR}/src/app/TrainingSyntheticDataDialog.ui
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ToolpathSettingsWidget.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

qt_add_resources(app app_resources
    ${CMAKE_SOURCE_DIR}/resources/app.qrc
)

set(app_deploy_script "")

# Locate Qt's deploy support helpers if they are available with the current Qt installation
set(_qt6_deploy_hints "")
if (DEFINED Qt6_DIR)
    list(APPEND _qt6_deploy_hints "${Qt6_DIR}" "${Qt6_DIR}/.." "${Qt6_DIR}/../..")
endif()
if (DEFINED QT6_INSTALL_PREFIX)
    list(APPEND _qt6_deploy_hints "${QT6_INSTALL_PREFIX}")
endif()
if (DEFINED CMAKE_PREFIX_PATH)
    list(APPEND _qt6_deploy_hints ${CMAKE_PREFIX_PATH})
endif()
if (DEFINED VCPKG_INSTALLED_DIR)
    list(APPEND _qt6_deploy_hints
        "${VCPKG_INSTALLED_DIR}"
        "${VCPKG_INSTALLED_DIR}/x64-windows"
        "${VCPKG_INSTALLED_DIR}/../vcpkg_installed"
    )
endif()
list(REMOVE_DUPLICATES _qt6_deploy_hints)

find_file(QT6_DEPLOY_SUPPORT_MODULE
    NAMES Qt6DeploySupport.cmake
    HINTS ${_qt6_deploy_hints}
    PATH_SUFFIXES lib/cmake/Qt6 Qt6
)

if (QT6_DEPLOY_SUPPORT_MODULE)
    include("${QT6_DEPLOY_SUPPORT_MODULE}")
else()
    find_program(_qt_windeployqt_executable
        NAMES windeployqt6 windeployqt
        HINTS ${_qt6_deploy_hints}
        PATH_SUFFIXES bin libexec tools/Qt6/bin tools/Qt/bin
    )
    if (_qt_windeployqt_executable)
        message(STATUS "Using windeployqt helper: ${_qt_windeployqt_executable}")
    else()
        message(WARNING "Qt6DeploySupport.cmake and windeployqt helper not found; Qt runtime deployment will be skipped.")
    endif()
endif()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/app)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/app/BuildInfo.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/app/BuildInfo.h
    @ONLY
)

set(QT_DEPLOY_BIN_DIR "${CMAKE_INSTALL_BINDIR}")

target_include_directories(app
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_sources(app
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/train/EnvManager.h
        ${CMAKE_SOURCE_DIR}/src/train/EnvManager.cpp
        ${CMAKE_SOURCE_DIR}/src/train/TrainingManager.h
        ${CMAKE_SOURCE_DIR}/src/train/TrainingManager.cpp
)

if (WITH_EMBEDDED_TESTS)
    target_sources(app
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src/app/DiagnosticsDialog.h
            ${CMAKE_SOURCE_DIR}/src/app/DiagnosticsDialog.cpp
    )
    target_link_libraries(app
        PRIVATE
            tests_core
    )
endif()

target_link_libraries(app
    PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Network
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        render
        io
        tp
        common
        ai
)

target_compile_definitions(app
    PRIVATE
        APP_BUILD_CONFIG="$<CONFIG>"
        WITH_EMBEDDED_TESTS=$<BOOL:${WITH_EMBEDDED_TESTS}>
)

set_target_properties(app PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE FALSE
    OUTPUT_NAME "AIToolpathGenerator"
)

if (QT6_DEPLOY_SUPPORT_MODULE)
    qt_generate_deploy_app_script(
        TARGET app
        OUTPUT_SCRIPT app_deploy_script
        NO_UNSUPPORTED_PLATFORM_ERROR
    )
elseif(_qt_windeployqt_executable)
    set(_qt_windeployqt_wrapper "${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_windeployqt.cmake")

    add_custom_command(TARGET app POST_BUILD
        COMMAND ${CMAKE_COMMAND}
            -DQT_WDEPLOYQT_EXECUTABLE="${_qt_windeployqt_executable}"
            -DQT_DEPLOY_TARGET="$<TARGET_FILE:app>"
            -DQT_DEPLOY_OUTPUT_DIR="$<TARGET_FILE_DIR:app>"
            -DQT_DEPLOY_CONFIG="$<CONFIG>"
            -P "${_qt_windeployqt_wrapper}"
        COMMENT "Deploying Qt runtime alongside app using windeployqt"
        COMMAND_EXPAND_LISTS
        VERBATIM
    )

    set(_qt_windeployqt_install_script "${CMAKE_CURRENT_BINARY_DIR}/windeployqt_install.cmake")
    set(QT_WDEPLOYQT_EXECUTABLE "${_qt_windeployqt_executable}")
    set(QT_DEPLOY_APP_FILENAME "AIToolpathGenerator${CMAKE_EXECUTABLE_SUFFIX}")
    set(QT_DEPLOY_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}")
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/windeployqt_install.cmake.in
        ${_qt_windeployqt_install_script}
        @ONLY
    )
    unset(QT_WDEPLOYQT_EXECUTABLE)
    unset(QT_DEPLOY_APP_FILENAME)
    unset(QT_DEPLOY_INSTALL_BINDIR)
    unset(QT_DEPLOY_INSTALL_BINDIR)
    set(app_deploy_script "${_qt_windeployqt_install_script}")
endif()

install(TARGETS app
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Ensure Qt runtime and plugins are copied next to the executable
if (app_deploy_script)
    install(SCRIPT ${app_deploy_script})
endif()

install(FILES
    ${CMAKE_SOURCE_DIR}/resources/tools.json
    DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/models/
    DESTINATION ${CMAKE_INSTALL_BINDIR}/models
    PATTERN ".gitkeep" EXCLUDE
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/samples/
    DESTINATION ${CMAKE_INSTALL_BINDIR}/samples
    FILES_MATCHING PATTERN "*.stl"
)

