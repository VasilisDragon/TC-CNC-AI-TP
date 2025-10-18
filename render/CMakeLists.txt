qt_add_library(render STATIC
    include/render/CameraController.h
    include/render/Model.h
    ../src/render/ModelViewerWidget.h
    include/render/Polyline.h
    ../src/render/ToolpathOverlay.h
    ../src/render/SimulationController.h
    src/CameraController.cpp
    src/Model.cpp
    ../src/render/ModelViewerWidget.cpp
    ../src/render/ToolpathOverlay.cpp
    ../src/render/SimulationController.cpp
)

target_include_directories(render
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/render
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/tp
)

target_link_libraries(render
    PUBLIC
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        glm::glm
        common
)

qt_add_resources(render "render_resources"
    PREFIX "/render"
    FILES
        shaders/flat.frag
        shaders/flat.vert
        shaders/polyline.frag
        shaders/polyline.vert
)


