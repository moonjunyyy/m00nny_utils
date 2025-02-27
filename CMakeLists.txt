cmake_minimum_required(VERSION 3.18)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Set the project name
project(M00NNY_Utils VERSION 1.0.0 DESCRIPTION "A collection of utility functions" LANGUAGES CXX CUDA)
set(TARGET_NAME "m00nny_utils")

# Set the installation prefix
set(PREFIX_PATH "/usr/local" CACHE PATH "Installation prefix")

message("\n\n")
message("----------------------PYTHON----------------------")
find_package(Python REQUIRED COMPONENTS Development)
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python include directory:")
foreach(INCLUDE IN LISTS Python_INCLUDE_DIRS)
    message("\t${INCLUDE}")
endforeach()
message(STATUS "Python library directory:")
foreach(LIBRARY IN LISTS Python_LIBRARIES)
    message("\t${LIBRARY}")
endforeach()
message("\n\n")
include_directories("${Python_INCLUDE_DIRS}")

# Find LibTorch
message("---------------------LIBTORCH---------------------")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "Torch version:           ${TORCH_VERSION}")
message(STATUS "Torch include directory:")
foreach(INCLUDE IN LISTS TORCH_INCLUDE_DIRS)
    message("\t${INCLUDE}")
endforeach()
message(STATUS "Torch library directory:")
foreach(LIBRARY IN LISTS TORCH_LIBRARIES)
    message("\t${LIBRARY}")
endforeach()
message(STATUS "Torch Python library directory: \n\t${TORCH_PYTHON_LIBRARY}")
include_directories("${TORCH_INCLUDE_DIRS}")
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} ${TORCH_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${TORCH_CUDA_FLAGS}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("\n")
    message(STATUS "Build Type: Debug")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g")
    add_compile_definitions(LIBM00NNY_DEBUG)
    set(PREFIX_PATH "${PREFIX_PATH}/debug")
else()
    message("\n")
    message(STATUS "Build Type: Release")
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    set(PREFIX_PATH "${PREFIX_PATH}/release")
endif()

# Add the source files into a list
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
# Loop through each source file

message(STATUS "Source file: ${SOURCES}")
# Get the name of the file without the extension (e.g. 'a' from src/a.cpp')

# Create an shared library for each source file
add_library(${TARGET_NAME} SHARED ${SOURCES})
# Set the output directory
set_target_properties(${TARGET_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PREFIX_PATH}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${PREFIX_PATH}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${PREFIX_PATH}/bin"
)
# Link the library with the required libraries
target_link_libraries(${TARGET_NAME} PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(${TARGET_NAME} PUBLIC ${Python_LIBRARIES})