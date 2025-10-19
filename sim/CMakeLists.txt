add_library(sim STATIC
    include/sim/StockGrid.h
    src/StockGrid.cpp
)

target_include_directories(sim
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/render/include
        ${CMAKE_SOURCE_DIR}/src/tp
)

target_link_libraries(sim
    PUBLIC
        Qt6::Gui
        glm::glm
        render
        tp
        common
)

target_compile_features(sim PUBLIC cxx_std_20)
