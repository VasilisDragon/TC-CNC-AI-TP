add_library(common STATIC
    include/common/logging.h
    include/common/math.h
    include/common/units.h
    src/math.cpp
    src/logging.cpp
    ../src/common/Units.h
    ../src/common/Units.cpp
    ../src/common/Tool.h
    ../src/common/Tool.cpp
    ../src/common/ToolLibrary.h
    ../src/common/ToolLibrary.cpp
)

target_include_directories(common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(common
    PUBLIC
        Qt6::Core
        Qt6::Gui
)

target_compile_definitions(common
    PUBLIC
        QT_NO_KEYWORDS
)
