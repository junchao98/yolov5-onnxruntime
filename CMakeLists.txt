cmake_minimum_required(VERSION 3.20.0)
project(yolo_ort)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
option(ENABLE_PERF  "inference perf" OFF)
option(ENABLE_OCL  "enable opencl ep" OFF)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)
#find_package(Qt5 COMPONENTS Widgets REQUIRED)

include_directories("include/")

add_executable(yolo_ort
               src/main.cpp
               src/detector.cpp
               src/utils.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(yolo_ort PRIVATE "${ONNXRUNTIME_DIR}/include/onnxruntime/core/session")
target_compile_features(yolo_ort PRIVATE cxx_std_14)

if (ENABLE_PERF)
    target_compile_definitions(yolo_ort PUBLIC ENABLE_PERF)
endif(ENABLE_PERF)

if (ENABLE_OCL)
    target_compile_definitions(yolo_ort PUBLIC ENABLE_OCL)
endif(ENABLE_OCL)

if (WIN32)
    target_link_libraries(yolo_ort "${ONNXRUNTIME_DIR}/build/Linux/${CMAKE_BUILD_TYPE}/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    target_link_libraries(yolo_ort "${ONNXRUNTIME_DIR}/build/Linux/${CMAKE_BUILD_TYPE}/libonnxruntime.so"
        ${OpenCV_LIBS}
        # yolo_ort Qt5::Core
        # yolo_ort Qt5::Widgets
    )
endif(UNIX)

