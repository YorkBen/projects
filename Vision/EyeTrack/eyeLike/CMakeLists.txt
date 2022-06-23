PROJECT( eyelike_project )
set(OpenCV_DIR D:\Program Files (x86)\opencv-4.5.5\build)
FIND_PACKAGE( OpenCV REQUIRED )
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(src)
