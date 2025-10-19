add_library(tp STATIC
    Toolpath.h
    ToolpathGenerator.h
    ToolpathGenerator.cpp
    GougeChecker.h
    GougeChecker.cpp
    Machine.h
    Machine.cpp
    GenerateWorker.h
    GenerateWorker.cpp
    IPost.h
    GRBLPost.h
    GRBLPost.cpp
    GCodeExporter.h
    GCodeExporter.cpp
    Stock.h
    Stock.cpp
    heightfield/UniformGrid.h
    heightfield/UniformGrid.cpp
    heightfield/HeightField.h
    heightfield/HeightField.cpp
    waterline/ZSlicer.h
    waterline/ZSlicer.cpp
    ocl/OclAdapter.h
    ocl/OclAdapter.cpp
)

target_include_directories(tp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tp
    PUBLIC
        glm::glm
        render
        ai
)

if (TP_OCL_ENABLED)
    target_compile_definitions(tp PUBLIC TP_WITH_OCL=1)
    if (OCL_INCLUDE_DIR)
        target_include_directories(tp
            PUBLIC
                ${OCL_INCLUDE_DIR}
        )
    endif()
    if (OCL_LIBRARY)
        target_link_libraries(tp
            PUBLIC
                ${OCL_LIBRARY}
        )
    endif()
else()
    target_compile_definitions(tp PUBLIC TP_WITH_OCL=0)
endif()
