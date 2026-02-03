cmake_minimum_required(VERSION 3.0)
project(r2lsh)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Boost REQUIRED)
if (Boost_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Boost_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Boost_CXX_FLAGS}")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Boost dynamic-bitset is required")
endif()

# Add all your source files
set(SRCS
    src/gendef.cpp
    src/blk_file.cpp
    src/util.cpp
    src/random.cpp
    src/block_file.cpp
    src/b_node.cpp
    src/b_tree.cpp
    src/qalsh.cpp
    src/ann.cpp
    src/main.cpp
)



# Include Boost library

# Create the executable
add_executable(r2lsh ${SRCS})
add_definitions(-std=c++11 -O3 -lboost -Wall)

