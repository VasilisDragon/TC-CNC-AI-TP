if (NOT WITH_EMBEDDED_TESTS)
    return()
endif()

add_library(tests_core STATIC
    TestsCore.cpp
    basic_toolpath_tests.cpp
)

target_include_directories(tests_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src/tests_core
        ${CMAKE_SOURCE_DIR}/tests/doctest
)

target_link_libraries(tests_core
    PUBLIC
        Qt6::Core
        tp
)

target_compile_definitions(tests_core
    PUBLIC
        WITH_EMBEDDED_TESTS=1
)
