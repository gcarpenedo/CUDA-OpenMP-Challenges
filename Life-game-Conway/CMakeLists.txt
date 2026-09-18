cmake_minimum_required(VERSION 3.10)

project(LifeGameConway LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Build type option - default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Release, Debug)" FORCE)
endif()

# Set optimization flags based on build type
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall -Wextra")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -flto -Wall -Wextra")
endif()

# Source files
set(COMMON_SOURCES
    src/utils.cpp
    src/sequential_simulation.cpp
    src/parallel_simulation.cpp
    src/parallel_simulation2.cpp
    src/parallel_simulation3.cpp
    src/parallel_simulation4.cpp
)

# Headers
set(HEADERS
    src/grid.h
    src/rules.h
    src/utils.h
    src/simulation.h
)

# Create executable
add_executable(chal2 src/chal2.cpp ${COMMON_SOURCES} ${HEADERS})
add_executable(benchmark src/benchmark.cpp ${COMMON_SOURCES} ${HEADERS})

# Link OpenMP
target_link_libraries(chal2 PRIVATE OpenMP::OpenMP_CXX )
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)

# Include directories
target_include_directories(chal2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenMP_CXX_INCLUDE_DIRS})
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenMP_CXX_INCLUDE_DIRS})
