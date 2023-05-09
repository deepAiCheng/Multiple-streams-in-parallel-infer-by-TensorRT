cmake_minimum_required(VERSION 3.10)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(TensorRT REQUIRED)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)


include_directories(/usr/src/jetson_multimedia_api/include)

# tensorrt
include_directories(${TensorRT_INCLUDE_DIRS})
link_directories(${TensorRT_LIBRARIES})

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)



add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer cudart)


find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5_det yolov5_det.cpp ${SRCS} )
target_link_libraries(yolov5_det nvinfer)
target_link_libraries(yolov5_det cudart)
target_link_libraries(yolov5_det myplugins)
target_link_libraries(yolov5_det ${OpenCV_LIBS})


