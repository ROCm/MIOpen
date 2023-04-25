################################################################################
#
# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

include(FetchContent)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
        HALF
        URL https://downloads.sourceforge.net/project/half/half/2.2.0/half-2.2.0.zip?ts=gAAAAABjwYR1XMSTpEV_SFhJCBhkdlggkBl9O2GQawgvJmGwqYTAsgEacimyfdBymL6etwelv_kYY7BdlCpy4mgcqceBbymbeQ%3D%3D&r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fhalf%2Ffiles%2Flatest%2Fdownload
        URL_MD5 b169bcfda2c65e90f73167e06368a27a)


if (BUILD_TESTING)
    set(BUILD_GMOCK OFF CACHE INTERNAL "")
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(GoogleTest googletest)
endif()

set(BOOST_INCLUDE_LIBRARIES any container date_time filesystem hof interprocess range thread variant)
set(BOOST_ENABLE_CMAKE ON)

FetchContent_Declare(
        Boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.81.0)

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2)

FetchContent_MakeAvailable(HALF ${GoogleTest} Boost nlohmann_json)

if(NOT NMAKE_DIR)
    set(NMAKE_DIR "$ENV{NMAKE_DIR}")
endif()

find_program(NMAKE_EXECUTABLE NAMES nmake.exe REQUIRED HINTS "${NMAKE_DIR}")

include(ExternalProject)

ExternalProject_Add(
        bzip2
        GIT_REPOSITORY https://sourceware.org/git/bzip2.git
        GIT_TAG bzip2-1.0.8
        GIT_SHALLOW true
        UPDATE_DISCONNECTED true
        BUILD_IN_SOURCE true
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${NMAKE_EXECUTABLE} /f .\\makefile.msc bzip2
        INSTALL_COMMAND "")

ExternalProject_Get_Property(bzip2 install_dir)

add_library(bzip2::bzip2 INTERFACE IMPORTED GLOBAL)
add_dependencies(bzip2::bzip2 bzip2)
target_link_directories(bzip2::bzip2 INTERFACE ${install_dir}/src/bzip2)
target_link_libraries(bzip2::bzip2 INTERFACE libbz2.lib)
target_include_directories(bzip2::bzip2 INTERFACE ${install_dir}/src/bzip2)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} CMAKE_BINARY_DIR)

if(MIOPEN_ENABLE_SQLITE)
    ExternalProject_Add(
            sqlite3
            GIT_REPOSITORY https://github.com/sqlite/sqlite.git
            GIT_TAG version-3.40.0
            GIT_SHALLOW true
            UPDATE_DISCONNECTED true
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ${NMAKE_EXECUTABLE} /f ..\\sqlite3\\Makefile.msc TOP=..\\sqlite3
            INSTALL_COMMAND cmd.exe /c copy /y sqlite3.dll "${CMAKE_BINARY_DIR}\\bin" && copy /y sqlite3.lib "${CMAKE_BINARY_DIR}\\lib" && for %I in (sqlite3.h sqlite3ext.h) do copy /y %I "${CMAKE_BINARY_DIR}\\include"
    )

    ExternalProject_Get_Property(sqlite3 install_dir)

    add_library(sqlite3::sqlite3 INTERFACE IMPORTED GLOBAL)
    add_dependencies(sqlite3::sqlite3 sqlite3)
    target_link_directories(sqlite3::sqlite3 INTERFACE ${CMAKE_BINARY_DIR}\\lib)
    target_link_libraries(sqlite3::sqlite3 INTERFACE sqlite3.lib)
    target_include_directories(sqlite3::sqlite3 INTERFACE ${CMAKE_BINARY_DIR}\\include)
endif()

set(_CXX_FLAGS
        -Wno-undef
        -Wno-reserved-identifier
        -Wno-global-constructors
        -Wno-missing-noreturn
        -Wno-disabled-macro-expansion
        -Wno-used-but-marked-unused
        -Wno-switch-enum
        -Wno-zero-as-null-pointer-constant
        -Wno-unused-member-function
        -Wno-comma
        -Wno-old-style-cast
        -Wno-deprecated
        -Wno-ignored-attributes
        -Wno-incompatible-pointer-types
        -Wno-enum-constexpr-conversion
        -Wno-deprecated-builtins
        -Wno-enum-constexpr-conversion)

    target_compile_options(boost_atomic PRIVATE ${_CXX_FLAGS})
    target_compile_options(boost_chrono PRIVATE ${_CXX_FLAGS})
    target_compile_options(boost_container PRIVATE ${_CXX_FLAGS})
    target_compile_options(boost_date_time PRIVATE ${_CXX_FLAGS})
    target_compile_options(boost_exception PRIVATE ${_CXX_FLAGS})
    target_compile_options(boost_filesystem PRIVATE ${_CXX_FLAGS})
    target_compile_options(boost_thread PRIVATE ${_CXX_FLAGS})

if(BUILD_TESTING)
    target_compile_options(gtest PRIVATE ${_CXX_FLAGS})
    target_compile_options(gtest_main PRIVATE ${_CXX_FLAGS})
endif()

set(HALF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/_deps/half-src/include PARENT_SCOPE)
