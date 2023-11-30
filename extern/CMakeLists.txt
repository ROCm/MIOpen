################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
cmake_minimum_required(VERSION 3.25)

option(MIOPEN_USE_SYSTEM "Cumulative option to use dependencies pre-installed in the system" OFF)

option(MIOPEN_HALF_USE_SYSTEM "Use Half library installed in the system" ${MIOPEN_USE_SYSTEM})
option(MIOPEN_BOOST_USE_SYSTEM "Use Boost libraries installed in the system" ${MIOPEN_USE_SYSTEM})
option(MIOPEN_JSON_USE_SYSTEM "Use Nlohmann JSON installed in the system" ${MIOPEN_USE_SYSTEM})
option(MIOPEN_BZIP2_USE_SYSTEM "Use BZip2 library installed in the system" ${MIOPEN_USE_SYSTEM})
option(MIOPEN_SQLITE3_USE_SYSTEM "Use SQLite3 library installed in the system" ${MIOPEN_USE_SYSTEM})

include(FetchContent)
cmake_policy(SET CMP0135 NEW)

set(_DISABLE_INSTALL FALSE)
macro(install)
    if(NOT _DISABLE_INSTALL)
        _install(${ARGV})
    endif()
endmacro()

if(MIOPEN_HALF_USE_SYSTEM)
    set(__half_find_package_args FIND_PACKAGE_ARGS REQUIRED)
endif()

FetchContent_Declare(
        HALF
        URL https://downloads.sourceforge.net/project/half/half/2.2.0/half-2.2.0.zip?ts=gAAAAABjwYR1XMSTpEV_SFhJCBhkdlggkBl9O2GQawgvJmGwqYTAsgEacimyfdBymL6etwelv_kYY7BdlCpy4mgcqceBbymbeQ%3D%3D&r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fhalf%2Ffiles%2Flatest%2Fdownload
        URL_MD5 b169bcfda2c65e90f73167e06368a27a
        SYSTEM
        ${__half_find_package_args})

FetchContent_MakeAvailable(HALF)

set(BOOST_INCLUDE_LIBRARIES any container date_time filesystem hof interprocess range thread variant CACHE INTERNAL "")

if(MIOPEN_BOOST_USE_SYSTEM)
    set(__boost_find_package_args FIND_PACKAGE REQUIRED COMPONENTS ${BOOST_INCLUDE_LIBRARIES})
endif()

FetchContent_Declare(
        Boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.83.0
        SYSTEM
        ${__boost_find_package_args})

if(MIOPEN_JSON_USE_SYSTEM)
    set(__nlohmann_json_find_package_args FIND_PACKAGE_ARGS REQUIRED)
endif()

# Store the current value of BUILD_SHARED_LIBS
set(__build_shared_libs ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

FetchContent_MakeAvailable(Boost)

# Restore the old value of BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "Type of libraries to build" FORCE)

set(JSON_MultipleHeaders ON CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_SystemInclude ON CACHE INTERNAL "")

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
        SYSTEM
        ${__nlohmann_json_find_package_args})

FetchContent_Declare(
        ZStd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG v1.5.5
        SYSTEM
        ${__zstd_find_package_args}
)

if(MIOPEN_ENABLE_AI_KERNEL_TUNING OR MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK)
    FetchContent_Declare(
            FunctionalPlus
            GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/FunctionalPlus.git
            GIT_TAG v0.2.18-p0
            SYSTEM
            ${__functional_plus_find_package_args}
    )
    set(FDEEP_BUILD_UNITTEST OFF CACHE INTERNAL "")
    FetchContent_Declare(
            frugally_deep
            GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/frugally-deep.git
            GIT_TAG 9683d557eb672ee2304f80f6682c51242d748a50
            SYSTEM
            ${__frugakky_deep_find_package_args}
    )
    FetchContent_Declare(
            eigen
            GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/eigen.git
            GIT_TAG 3.4.0
            SYSTEM
            ${__eigen_find_package_args}
    )
    # Store the current value of BUILD_SHARED_LIBS
    set(__build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF CACHE INTERNAL "")

    set(_DISABLE_INSTALL TRUE)
    FetchContent_MakeAvailable(ZStd nlohmann_json FunctionalPlus frugally_deep eigen)
    set(_DISABLE_INSTALL FALSE)

    # Restore the old value of BUILD_SHARED_LIBS
    set(BUILD_TESTING ${__build_testing} CACHE BOOL "" FORCE)
else()
    FetchContent_MakeAvailable(ZStd hlohmann_json)
endif()

include(ExternalProject)

if(MIOPEN_BZIP2_USE_SYSTEM)
    find_package(BZip2 1.0.8 REQUIRED)
else()
    if(NOT NMAKE_DIR)
        set(NMAKE_DIR "$ENV{NMAKE_DIR}")
    endif()

    find_program(NMAKE_EXECUTABLE NAMES nmake.exe REQUIRED HINTS "${NMAKE_DIR}")

    ExternalProject_Add(
            bzip2
            GIT_REPOSITORY https://sourceware.org/git/bzip2.git
            GIT_TAG bzip2-1.0.8
            GIT_SHALLOW true
            UPDATE_DISCONNECTED true
            BUILD_IN_SOURCE true
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ${NMAKE_EXECUTABLE} /f .\\makefile.msc lib
            INSTALL_COMMAND "")

    ExternalProject_Get_Property(bzip2 SOURCE_DIR)

    add_library(BZip2::BZip2 INTERFACE IMPORTED GLOBAL)
    add_dependencies(BZip2::BZip2 bzip2)
    target_link_directories(BZip2::BZip2 INTERFACE ${SOURCE_DIR})
    target_link_libraries(BZip2::BZip2 INTERFACE libbz2.lib)
    target_include_directories(BZip2::BZip2 INTERFACE ${SOURCE_DIR})
endif()

if(MIOPEN_ENABLE_SQLITE)
    if(MIOPEN_SQLITE3_USE_SYSTEM)
        find_package(PkgConfig)
        pkg_check_modules(SQLITE3 REQUIRED sqlite3)
    else()
        ExternalProject_Add(
                sqlite3
                GIT_REPOSITORY https://github.com/sqlite/sqlite.git
                GIT_TAG version-3.40.0
                GIT_SHALLOW true
                UPDATE_DISCONNECTED true
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ${NMAKE_EXECUTABLE} /f ..\\sqlite3\\Makefile.msc USE_AMALGAMATION=1 NO_TCL=1 TOP=..\\sqlite3 libsqlite3.lib
                INSTALL_COMMAND ""
        )

        ExternalProject_Get_Property(sqlite3 BINARY_DIR)

        # For compatibility with PkgConfig on Linux
        add_library(SQLite::SQLite3 INTERFACE IMPORTED GLOBAL)
        add_dependencies(SQLite::SQLite3 sqlite3)
        target_link_directories(SQLite::SQLite3 INTERFACE ${BINARY_DIR})
        target_link_libraries(SQLite::SQLite3 INTERFACE libsqlite3.lib)
        target_include_directories(SQLite::SQLite3 INTERFACE ${BINARY_DIR})
    endif()
endif()

target_compile_options(boost_container PRIVATE -Wno-incompatible-pointer-types)

set(HALF_INCLUDE_DIR ${CMAKE_BINARY_DIR}/_deps/half-src/include PARENT_SCOPE)

if(BUILD_TESTING)
    set(gtest_force_shared_crt ON CACHE INTERNAL "")
    set(BUILD_GMOCK OFF CACHE INTERNAL "")
    set(INSTALL_GTEST OFF CACHE INTERNAL "")

    FetchContent_Declare(
        GTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571
        SYSTEM
    )

    set(__build_shared_libs ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

    FetchContent_MakeAvailable(GTest)
    set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "Type of libraries to build" FORCE)

    target_compile_options(gtest PRIVATE -Wno-undef)
    target_compile_options(gtest_main PRIVATE -Wno-undef)
endif()
