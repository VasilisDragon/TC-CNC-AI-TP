add_library(ai STATIC
    IPathAI.h
    FeatureExtractor.h
    FeatureExtractor.cpp
    ModelCard.h
    ModelCard.cpp
    TorchAI.h
    TorchAI.cpp
    OnnxAI.h
    OnnxAI.cpp
    ModelManager.h
    ModelManager.cpp
)

target_include_directories(ai
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(ai
    PUBLIC
        Qt6::Core
        render
)

if (AI_TORCH_ENABLED)
    find_package(Torch REQUIRED)
    target_compile_definitions(ai PUBLIC AI_WITH_TORCH=1)
    target_include_directories(ai
        PUBLIC
            ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(ai
        PUBLIC
            ${TORCH_LIBRARIES}
    )
    set_property(TARGET ai PROPERTY CXX_STANDARD 17)
    set_property(TARGET ai PROPERTY CXX_STANDARD_REQUIRED ON)
    if (DEFINED TORCH_CXX_FLAGS)
        target_compile_options(ai PRIVATE ${TORCH_CXX_FLAGS})
    endif()
endif()

if (AI_ONNX_ENABLED)
    find_package(onnxruntime CONFIG REQUIRED)
    target_compile_definitions(ai PUBLIC AI_WITH_ONNXRUNTIME=1)
    target_link_libraries(ai
        PUBLIC
            onnxruntime::onnxruntime
    )
    if (TARGET onnxruntime::onnxruntime_providers_cuda)
        target_link_libraries(ai
            PUBLIC
                onnxruntime::onnxruntime_providers_cuda
        )
        target_compile_definitions(ai PUBLIC AI_ONNX_WITH_CUDA=1)
    endif()
endif()
