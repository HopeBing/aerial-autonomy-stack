cmake_minimum_required(VERSION 3.8)
project(yolo)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)

# ==============================================================================
# ONNX RUNTIME
# ==============================================================================
# Define the ABSOLUTE PATH to the ONNX Runtime library
set(ONNXRUNTIME_FULL_PATH "/usr/local/lib/python3.10/dist-packages/onnxruntime/capi/libonnxruntime.so.1.23.1")

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# A. Find Python Installation Path (used for finding headers and CUDA provider)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import os, onnxruntime; print(os.path.dirname(onnxruntime.__file__))"
  OUTPUT_VARIABLE ORT_PYTHON_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# B. Define Search Paths for Headers
set(ORT_SEARCH_INCLUDE_PATHS 
    "${ORT_PYTHON_DIR}/capi/include" 
    "${ORT_PYTHON_DIR}/include"
    "/usr/local/include/onnxruntime" 
)

# C. Find Headers (download fallback)
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h HINTS ${ORT_SEARCH_INCLUDE_PATHS})
if(NOT ONNXRUNTIME_INCLUDE_DIR)
    message(WARNING "Downloading missing ONNX headers...")
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import onnxruntime; print(onnxruntime.__version__)" OUTPUT_VARIABLE ORT_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(ORT_HEADER_URL_BASE "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ORT_VERSION}/include/onnxruntime/core/session")
    set(DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/onnxruntime_headers")
    file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")
    foreach(HEADER_FILE 
        "onnxruntime_cxx_api.h" "onnxruntime_c_api.h" "onnxruntime_cxx_inline.h" "onnxruntime_ep_c_api.h" "onnxruntime_float16.h"
    )
        file(DOWNLOAD "${ORT_HEADER_URL_BASE}/${HEADER_FILE}" "${DOWNLOAD_DIR}/${HEADER_FILE}")
    endforeach()
    set(ONNXRUNTIME_INCLUDE_DIR "${DOWNLOAD_DIR}")
endif()

# D. Define Search Paths for Optional Libraries (CUDA)
set(ORT_SEARCH_LIB_PATHS 
    "${ORT_PYTHON_DIR}/capi" 
    "/usr/local/lib" 
    "/usr/lib"
)
find_library(ONNXRUNTIME_CUDA_LIB NAMES onnxruntime_providers_cuda HINTS ${ORT_SEARCH_LIB_PATHS})
# ==============================================================================

add_executable(yolo_node src/yolo.cpp)

target_include_directories(yolo_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  ${OpenCV_INCLUDE_DIRS}
  ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link ROS Dependencies
ament_target_dependencies(yolo_node
  rclcpp
  sensor_msgs
  vision_msgs
  cv_bridge
)

# ==============================================================================

# Link System Dependencies
target_link_libraries(yolo_node
  ${OpenCV_LIBRARIES}
  ${ONNXRUNTIME_FULL_PATH} 
  dl # Required for dynamic linking functions, which ONNX Runtime uses
)

# Link CUDA provider if found
if(ONNXRUNTIME_CUDA_LIB)
    target_link_libraries(yolo_node ${ONNXRUNTIME_CUDA_LIB})
endif()

# ==============================================================================

install(TARGETS yolo_node DESTINATION lib/${PROJECT_NAME})

ament_package()