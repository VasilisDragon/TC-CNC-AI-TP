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

if (IO_OCCT_ENABLED)
    if (OCCT_INCLUDE_DIRS)
        target_include_directories(io
            PUBLIC
                ${OCCT_INCLUDE_DIRS}
        )
    endif()
    target_compile_definitions(io
        PUBLIC
            WITH_OCCT
    )
    if (OCCT_LIBRARIES)
        target_link_libraries(io
            PUBLIC
                ${OCCT_LIBRARIES}
        )
    endif()
endif()
