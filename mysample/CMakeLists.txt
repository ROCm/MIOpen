cmake_minimum_required(VERSION 3.10)
project(winograd_conv_test LANGUAGES CXX)

# Set HIP as the compiler
find_package(HIP REQUIRED)

# Find MIOpen library
find_package(miopen REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define source file
set(SOURCE_FILE mi.cpp)

# Create executable
add_executable(winograd_conv ${SOURCE_FILE})

# Link HIP and MIOpen
target_link_libraries(winograd_conv PRIVATE hip::host MIOpen)
