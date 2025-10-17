add_library(io STATIC
    ModelImporter.cpp
    ModelImporter.h
    ImportWorker.cpp
    ImportWorker.h
)

target_include_directories(io
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(io
    PUBLIC
        Qt6::Core
        assimp::assimp
        render
        common
)

target_compile_definitions(io
    PUBLIC
        QT_NO_KEYWORDS
)
