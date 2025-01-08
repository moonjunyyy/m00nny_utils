cmake_minimum_required(VERSION 3.15)

# Set the project name
project(M00NNY_Utils VERSION 1.0.0 DESCRIPTION "A collection of utility functions" LANGUAGES CXX CUDA)

set(PREFIX_PATH           "/usr/local"      CACHE PATH   "Installation prefix")
set(LIBTORCH_PATH         "/usr/local"      CACHE PATH   "Path to LibTorch directory")
set(PYTHON_DEV_PATH       "/usr/local"      CACHE PATH   "Path to Python development files")
set(CUDA_TOOLKIT_PATH     "/usr/local/cuda" CACHE PATH   "Path to CUDA directory")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (CUDA_TOOLKIT_PATH)
    set(CMAKE_PREFIX_PATH "${CUDA_TOOLKIT_PATH};${CMAKE_PREFIX_PATH}")
    find_package(CUDA REQUIRED)
    include_directories("${CUDA_INCLUDE_DIRS}")
    message(STATUS "${CUDA_INCLUDE_DIRS}")
endif()
message(STATUS "")
message(STATUS "LibTorch path: ${LIBTORCH_PATH}")
if (LIBTORCH_PATH)
    set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")
    set(CMAKE_PREFIX_PATH "$${LIBTORCH_PATH};${Torch_DIR};{CMAKE_PREFIX_PATH}")
    find_package(Torch REQUIRED)
    message(STATUS "Torch include path: ${TORCH_INCLUDE_DIRS}")
endif()

message(STATUS "")
# Get the python include path
execute_process(
    COMMAND python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python include path: ${PYTHON_INCLUDE_PATH}")
# Get the python library path
execute_process(
    COMMAND python -c "from distutils.sysconfig import get_config_var; print(get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE PYTHON_LIB_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python library path: ${PYTHON_LIB_PATH}")
message(STATUS "")

# Set the build type
set(CMAKE_BUILD_TYPE "Release")
# Add the source files into a list
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# Loop through each source file
foreach(SOURCE_FILE IN LISTS SOURCES)
    message(STATUS "Source file: ${SOURCE_FILE}")
    # Get the name of the file without the extension (e.g. 'a' from src/a.cpp')
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
    # Create an shared library for each source file
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILE})
    # Set the output directory
    set_target_properties(${TARGET_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PREFIX_PATH}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${PREFIX_PATH}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${PREFIX_PATH}/bin"
    )
endforeach()