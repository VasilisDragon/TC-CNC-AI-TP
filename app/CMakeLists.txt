qt_add_executable(app
    ../src/app/MainWindow.h
    ../src/app/AiPreferencesDialog.h
    ../src/app/TrainingNewModelDialog.h
    ../src/app/TrainingSyntheticDataDialog.h
    include/app/ToolpathSettingsWidget.h
    ../src/app/MainWindow.cpp
    ../src/app/AiPreferencesDialog.cpp
    ../src/app/TrainingNewModelDialog.cpp
    ../src/app/TrainingSyntheticDataDialog.cpp
    ../src/app/AiPreferencesDialog.ui
    ../src/app/TrainingNewModelDialog.ui
    ../src/app/TrainingSyntheticDataDialog.ui
    src/ToolpathSettingsWidget.cpp
    src/main.cpp
)

qt_add_resources(app app_resources
    ../resources/app.qrc
)

include(Qt6DeploySupport)

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
        ../src/train/EnvManager.h
        ../src/train/EnvManager.cpp
        ../src/train/TrainingManager.h
        ../src/train/TrainingManager.cpp
)

target_link_libraries(app
    PRIVATE
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
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
)

set_target_properties(app PROPERTIES
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE FALSE
    OUTPUT_NAME "AIToolpathGenerator"
)

qt_generate_deploy_app_script(
    TARGET app
    OUTPUT_SCRIPT app_deploy_script
    NO_UNSUPPORTED_PLATFORM_ERROR
)

install(TARGETS app
    BUNDLE DESTINATION .
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Ensure Qt runtime and plugins are copied next to the executable
install(SCRIPT ${app_deploy_script})

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

