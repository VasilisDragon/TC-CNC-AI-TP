# Testing

## Configure
- Configure the project with CMake and enable testing: `cmake -S . -B build -DBUILD_TESTING=ON`.
- Build the tests from the build tree: `cmake --build build --target path_safety_tests headless_pipeline_tests`.

## Run
- From the build directory run the fast regression suite: `ctest -L fast`.
- To rebuild and re-run in one step: `cmake --build build --target test`.

## Fast Suite Contents
- `basic_sanity`: header-only doctest exercises core data structures without optional backends.
- `path_safety`: verifies generated cut motion stays above the sampled surface and rejects self-intersecting segments.
- `headless_pipeline`: exercises the importer -> AI decision -> toolpath -> G-code path using `samples/sample_part.stl`.
- `onnx_ai_smoke_test`: included automatically when ONNX Runtime support is enabled.
