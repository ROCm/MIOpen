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
option(MIOPEN_ZSTD_USE_SYSTEM "Use ZStd library installed in the system" ${MIOPEN_USE_SYSTEM})
if(MIOPEN_ENABLE_AI_KERNEL_TUNING OR MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK)
    option(MIOPEN_FUNCTIONAL_PLUS_USE_SYSTEM "Use Functional Plus installed in the system" ${MIOPEN_USE_SYSTEM})
    option(MIOPEN_FRUGALLY_DEEP_USE_SYSTEM "Use Frugally-deep installed in the system" ${MIOPEN_USE_SYSTEM})
    option(MIOPEN_EIGEN_USE_SYSTEM "Use Eigen installed in the system" ${MIOPEN_USE_SYSTEM})
endif()
if(BUILD_TESTING)
    option(MIOPEN_GOOGLETEST_USE_SYSTEM "Use GoogleTest installed in the system" ${MIOPEN_USE_SYSTEM})
endif()

include(FetchContent)
cmake_policy(SET CMP0135 NEW)

include(Tools.cmake)

if(MIOPEN_HALF_USE_SYSTEM)
    find_path(__half_include_dir half.hpp PATH_SUFFIXES half)
endif()

if(NOT __half_include_dir)
    FetchContent_Declare(
            half
            URL https://downloads.sourceforge.net/project/half/half/2.2.0/half-2.2.0.zip?ts=gAAAAABjwYR1XMSTpEV_SFhJCBhkdlggkBl9O2GQawgvJmGwqYTAsgEacimyfdBymL6etwelv_kYY7BdlCpy4mgcqceBbymbeQ%3D%3D&r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Fhalf%2Ffiles%2Flatest%2Fdownload
            URL_MD5 b169bcfda2c65e90f73167e06368a27a
            SYSTEM)

    FetchContent_MakeAvailable(half)

    FetchContent_GetProperties(half)
    set(__half_include_dir ${half_SOURCE_DIR}/include)
endif()

set(HALF_INCLUDE_DIR ${__half_include_dir} PARENT_SCOPE)

if(MIOPEN_BOOST_USE_SYSTEM)
    find_package(Boost 1.74 COMPONENTS filesystem system GLOBAL)
endif()

if(NOT Boost_FOUND)
    set(BOOST_INCLUDE_LIBRARIES any container date_time filesystem hof interprocess range thread variant CACHE INTERNAL "")
    FetchContent_Declare(
            Boost
            GIT_REPOSITORY https://github.com/boostorg/boost.git
            GIT_TAG boost-1.83.0
            SYSTEM)

    # Store the current value of BUILD_SHARED_LIBS
    set(__build_shared_libs ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

    FetchContent_MakeAvailable(Boost)

    # Restore the old value of BUILD_SHARED_LIBS
    set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "Type of libraries to build" FORCE)

    target_compile_options(boost_container PRIVATE -Wno-incompatible-pointer-types)
endif()

if(MIOPEN_JSON_USE_SYSTEM)
    set(__nlohmann_json_find_package_args FIND_PACKAGE_ARGS REQUIRED GLOBAL)
endif()

set(JSON_MultipleHeaders ON CACHE INTERNAL "")
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_SystemInclude ON CACHE INTERNAL "")

FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
        SYSTEM
        ${__nlohmann_json_find_package_args})

set(_DISABLE_CHECK_TOOLCHAIN_VAR TRUE)

if(MIOPEN_ENABLE_AI_KERNEL_TUNING OR MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK)

    if(MIOPEN_FUNCTIONAL_PLUS_USE_SYSTEM)
        set(__functional_plus_find_package_args FIND_PACKAGE_ARGS REQUIRED GLOBAL)
    endif()

    FetchContent_Declare(
            FunctionalPlus
            GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/FunctionalPlus.git
            GIT_TAG v0.2.18-p0
            SYSTEM
            ${__functional_plus_find_package_args}
    )

    if(MIOPEN_FRUGALLY_DEEP_USE_SYSTEM)
        set(__frugally_deep_find_package_args FIND_PACKAGE_ARGS REQUIRED GLOBAL)
    endif()

    set(FDEEP_BUILD_UNITTEST OFF CACHE INTERNAL "")
    FetchContent_Declare(
            frugally-deep
            GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/frugally-deep.git
            GIT_TAG 9683d557eb672ee2304f80f6682c51242d748a50
            SYSTEM
            ${__frugally_deep_find_package_args}
    )

    if(MIOPEN_EIGEN_USE_SYSTEM)
        set(__eigen_find_package_args FIND_PACKAGE_ARGS REQUIRED GLOBAL)
    endif()

    FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/eigen.git
            GIT_TAG 3.4.0
            SYSTEM
            ${__eigen_find_package_args}
    )
    # Store the current value of BUILD_SHARED_LIBS
    set(__build_testing ${BUILD_TESTING})
    set(BUILD_TESTING OFF CACHE INTERNAL "")

    set(_DISABLE_INSTALL TRUE)
    FetchContent_MakeAvailable(nlohmann_json Eigen3 FunctionalPlus frugally-deep)
    set(_DISABLE_INSTALL FALSE)

    # Restore the old value of BUILD_SHARED_LIBS
    set(BUILD_TESTING ${__build_testing} CACHE BOOL "" FORCE)
else()
    FetchContent_MakeAvailable(hlohmann_json)
endif()

if(MIOPEN_ZSTD_USE_SYSTEM)
    find_package(zstd GLOBAL)
endif()

if(NOT zstd_FOUND)
    FetchContent_Declare(
        zstd
        GIT_REPOSITORY https://github.com/facebook/zstd.git
        GIT_TAG v1.5.5
        SYSTEM
    )

    FetchContent_GetProperties(zstd)
    if(NOT zstd_POPULATED)
        FetchContent_Populate(zstd)
        add_subdirectory(${zstd_SOURCE_DIR}/build/cmake ${zstd_BINARY_DIR})
    endif()

    target_compile_options(libzstd_shared PRIVATE
        -Wno-strict-prototypes -Wno-unused-function)
    target_compile_options(libzstd_static PRIVATE
        -Wno-strict-prototypes -Wno-unused-function)
endif()

include(ExternalProject)

if(WIN32)
    find_program(NMAKE_EXECUTABLE NAMES nmake.exe REQUIRED HINTS "${NMAKE_DIR}" ENV NMAKE_DIR)
else()
    find_program(MAKE_EXECUTABLE NAMES make gmake REQUIRED)
endif()

if(MIOPEN_BZIP2_USE_SYSTEM)
    find_package(BZip2 1.0.8 GLOBAL)
endif()

if(NOT BZip2_FOUND)
    if(WIN32)
        set(__bzip2_build_command ${NMAKE_EXECUTABLE} /f .\\makefile.msc lib)
    else()
        set(__bzip2_build_command pwd && ${MAKE_EXECUTABLE} libbz2.a)
    endif()

    ExternalProject_Add(
            bzip2
            GIT_REPOSITORY https://sourceware.org/git/bzip2.git
            GIT_TAG bzip2-1.0.8
            GIT_SHALLOW true
            UPDATE_DISCONNECTED true
            BUILD_IN_SOURCE true
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ${__bzip2_build_command}
            INSTALL_COMMAND "")

    ExternalProject_Get_Property(bzip2 SOURCE_DIR)

    add_library(BZip2::BZip2 INTERFACE IMPORTED GLOBAL)
    add_dependencies(BZip2::BZip2 bzip2)
    target_link_directories(BZip2::BZip2 INTERFACE ${SOURCE_DIR})
    target_link_libraries(BZip2::BZip2 INTERFACE libbz2${CMAKE_STATIC_LIBRARY_SUFFIX})
    target_include_directories(BZip2::BZip2 INTERFACE ${SOURCE_DIR})
endif()

if(MIOPEN_ENABLE_SQLITE)
    if(MIOPEN_SQLITE3_USE_SYSTEM)
        find_package(SQLite3 GLOBAL)
    endif()

    if(NOT SQLite3_FOUND)
        set(__sqlite_repository https://github.com/sqlite/sqlite.git)
        set(__sqlite_tag version-3.40.0)
        if(WIN32)
            ExternalProject_Add(
                sqlite3
                GIT_REPOSITORY ${__sqlite_repository}
                GIT_TAG ${__sqlite_tag}
                GIT_SHALLOW true
                UPDATE_DISCONNECTED true
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ${NMAKE_EXECUTABLE} /f ..\\sqlite3\\Makefile.msc USE_AMALGAMATION=1 NO_TCL=1 TOP=..\\sqlite3 libsqlite3.lib
                INSTALL_COMMAND ""
            )
        else()
            ExternalProject_Add(
                sqlite3
                GIT_REPOSITORY ${__sqlite_repository}
                GIT_TAG ${__sqlite_tag}
                GIT_SHALLOW true
                CONFIGURE_HANDLED_BY_BUILD true
                UPDATE_DISCONNECTED true
                CONFIGURE_COMMAND ../sqlite3/configure --disable-tcl $<$<CONFIG:Debug>:--enable-debug>
                BUILD_COMMAND ${MAKE_EXECUTABLE} libsqlite3.la
                INSTALL_COMMAND ""
            )
        endif()

        ExternalProject_Get_Property(sqlite3 BINARY_DIR)

        # For compatibility with PkgConfig on Linux
        add_library(SQLite::SQLite3 INTERFACE IMPORTED GLOBAL)
        add_dependencies(SQLite::SQLite3 sqlite3)
        target_link_directories(SQLite::SQLite3 INTERFACE ${BINARY_DIR})
        target_link_libraries(SQLite::SQLite3 INTERFACE libsqlite3${CMAKE_STATIC_LIBRARY_SUFFIX})
        target_include_directories(SQLite::SQLite3 INTERFACE ${BINARY_DIR})
    endif()
endif()

if(BUILD_TESTING)
    if(MIOPEN_GOOGLETEST_USE_SYSTEM)
        set(__gtest_use_system FIND_PACKAGE_ARGS REQUIRED GLOBAL)
    endif()

    FetchContent_Declare(
        GTest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG f8d7d77c06936315286eb55f8de22cd23c188571
        SYSTEM
        ${__gtest_use_system}
    )

    set(INSTALL_GTEST OFF CACHE INTERNAL "")
    set(BUILD_GMOCK OFF CACHE INTERNAL "")

    if(WIN32)
        set(gtest_force_shared_crt ON CACHE INTERNAL "")
    endif()

    set(__build_shared_libs ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")

    FetchContent_MakeAvailable(GTest)

    set(BUILD_SHARED_LIBS ${__build_shared_libs} CACHE BOOL "Type of libraries to build" FORCE)

    if(NOT MIOPEN_GOOGLETEST_USE_SYSTEM)
        target_compile_options(gtest PRIVATE -Wno-undef)
        target_compile_options(gtest_main PRIVATE -Wno-undef)
    endif()

    include(GoogleTest)
endif()

set(_DISABLE_CHECK_TOOLCHAIN_VAR FALSE)