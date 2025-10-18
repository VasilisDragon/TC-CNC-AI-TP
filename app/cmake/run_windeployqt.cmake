if(NOT DEFINED QT_DEPLOY_CONFIG)
    message(FATAL_ERROR "QT_DEPLOY_CONFIG must be provided.")
endif()

if(NOT DEFINED QT_WDEPLOYQT_EXECUTABLE OR QT_WDEPLOYQT_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "QT_WDEPLOYQT_EXECUTABLE must be provided.")
endif()

if(NOT DEFINED QT_DEPLOY_TARGET OR QT_DEPLOY_TARGET STREQUAL "")
    message(FATAL_ERROR "QT_DEPLOY_TARGET must be provided.")
endif()

if(NOT DEFINED QT_DEPLOY_OUTPUT_DIR OR QT_DEPLOY_OUTPUT_DIR STREQUAL "")
    message(FATAL_ERROR "QT_DEPLOY_OUTPUT_DIR must be provided.")
endif()

set(_qt_config_raw "${QT_DEPLOY_CONFIG}")
string(REPLACE "\"" "" _qt_config_stripped "${_qt_config_raw}")
string(TOLOWER "${_qt_config_stripped}" _qt_config_lower)

set(_windeploy_path "${QT_WDEPLOYQT_EXECUTABLE}")
string(REPLACE "\"" "" _windeploy_path "${_windeploy_path}")

set(_target_path "${QT_DEPLOY_TARGET}")
string(REPLACE "\"" "" _target_path "${_target_path}")

set(_output_dir "${QT_DEPLOY_OUTPUT_DIR}")
string(REPLACE "\"" "" _output_dir "${_output_dir}")

message(STATUS "run_windeployqt config='${_qt_config_raw}' stripped='${_qt_config_stripped}'")
message(STATUS "run_windeployqt exe='${_windeploy_path}' target='${_target_path}' dir='${_output_dir}'")

if(_qt_config_lower STREQUAL "debug")
    message(STATUS "Skipping windeployqt for Debug configuration.")
    return()
endif()

set(_qt_args
    --no-compiler-runtime
    --dir "${_output_dir}"
    "${_target_path}"
)

if(_qt_config_lower STREQUAL "release" OR _qt_config_lower STREQUAL "relwithdebinfo" OR _qt_config_lower STREQUAL "minsizerel")
    list(INSERT _qt_args 0 "--release")
endif()

execute_process(
    COMMAND "${_windeploy_path}" ${_qt_args}
    RESULT_VARIABLE _qt_result
    OUTPUT_VARIABLE _qt_stdout
    ERROR_VARIABLE _qt_stderr
)

if(NOT _qt_result EQUAL 0)
    message(FATAL_ERROR "windeployqt failed with exit code ${_qt_result}\nStdout:\n${_qt_stdout}\nStderr:\n${_qt_stderr}")
endif()
