# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.2)

include(FetchContent)

option(MIOPEN_NO_DOWNLOAD
       "Disables downloading of any external dependencies" OFF
)

if(MIOPEN_NO_DOWNLOAD)
    set(FETCHCONTENT_FULLY_DISCONNECTED
        OFF
        CACHE BOOL "Don't attempt to download or update anything" FORCE
    )
endif()

# Dependencies where the local version should be used, if available
set(_miopen_all_local_deps
    composable_kernel
)
# Dependencies where we never look for a local version
set(_miopen_all_remote_deps
)

# miopen_add_dependency(
#   dep_name
#   [NO_LOCAL]
#   [VERSION version]
#   [FIND_PACKAGE_ARGS args...]
#   [COMPONENT component]
#   [PACKAGE_NAME
#      [package_name]
#      [DEB deb_package_name]
#      [RPM rpm_package_name]]
# )
function(miopen_add_dependency dep_name)
    set(options NO_LOCAL)
    set(oneValueArgs VERSION HASH)
    set(multiValueArgs FIND_PACKAGE_ARGS PACKAGE_NAME COMPONENTS)
    cmake_parse_arguments(
        PARSE
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    if(dep_name IN_LIST _miopen_all_local_deps)
        if(NOT PARSE_NO_LOCAL)
            find_package(
                ${dep_name} ${PARSE_VERSION} QUIET ${PARSE_FIND_PACKAGE_ARGS}
            )
        endif()
        if(NOT ${dep_name}_FOUND)
            message(STATUS "Did not find ${dep_name}, it will be built locally")
            _build_local()
        else()
            message(
                STATUS
                    "Found ${dep_name}: ${${dep_name}_DIR} (found version \"${${dependency_name}_VERSION}\")"
            )
            foreach(VAR IN LISTS ${dep_name}_EXPORT_VARS)
                set(${VAR}
                    ${${VAR}}
                    PARENT_SCOPE
                )
            endforeach()
        endif()
    elseif(dep_name IN_LIST _miopen_all_remote_deps)
        message(TRACE "Will build ${dep_name} locally")
        _build_local()
    else()
        message(WARNING "Unknown dependency: ${dep_name}")
        return()
    endif()
endfunction()

macro(_build_local)
    cmake_policy(PUSH)
    if(BUILD_VERBOSE)
        message(STATUS "=========== Adding ${dep_name} ===========")
    endif()
    _pushstate()
    set(CMAKE_MESSAGE_INDENT "[${dep_name}] ")
    cmake_language(
        CALL _fetch_${dep_name} "${PARSE_VERSION}" "${PARSE_HASH}"
    )
    _popstate()
    if(BUILD_VERBOSE)
        message(STATUS "=========== Added ${dep_name} ===========")
    endif()
    cmake_policy(POP)
    foreach(VAR IN LISTS ${dep_name}_EXPORT_VARS)
        set(${VAR}
            ${${VAR}}
            PARENT_SCOPE
        )
    endforeach()
endmacro()

function(_fetch_composable_kernel VERSION HASH)
    if(HASH)
        set(HASH_ARG HASH ${HASH})
    endif()

    message(STATUS "Fetching composable_kernel hash: ${HASH_ARG}")
    FetchContent_Declare(
        composable_kernel
        URL https://github.com/ROCm/composable_kernel/archive/${HASH}.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )

    _save_var(GPU_ARCHS)
    _save_var(CMAKE_BUILD_TYPE)
    _save_var(MIOPEN_REQ_LIBS_ONLY)
    _save_var(DISABLE_OFFLOAD_COMPRESS)
    _save_var(ENABLE_CLANG_CPP_CHECKS)
    _save_var(BUILD_SHARED_LIBS)


    # Use GPU_ARCHs for CK as it stops CK's cmake from including the tests.
    set(GPU_ARCHS "${GPU_TARGETS}" CACHE INTERNAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE INTERNAL "")
    set(MIOPEN_REQ_LIBS_ONLY ON CACHE INTERNAL "")
    set(DISABLE_OFFLOAD_COMPRESS ON CACHE INTERNAL "")
    set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
    set(ENABLE_CLANG_CPP_CHECKS OFF CACHE INTERNAL "")

    FetchContent_MakeAvailable(composable_kernel)

    _restore_var(CMAKE_BUILD_TYPE)
    _restore_var(GPU_ARCHS)
    _restore_var(MIOPEN_REQ_LIBS_ONLY)
    _restore_var(DISABLE_OFFLOAD_COMPRESS)
    _restore_var(ENABLE_CLANG_CPP_CHECKS)
    _restore_var(BUILD_SHARED_LIBS)


    set(MIOPEN_CK_INCLUDE_DIR ${composable_kernel_SOURCE_DIR}/include CACHE PATH "Path to ck includes")
    set(MIOPEN_CK_LIBRARY_INCLUDE_DIR ${composable_kernel_SOURCE_DIR}/library/include CACHE PATH "Path to ck library includes")
    set(MIOPEN_CK_BUILD_INCLUDE_DIR ${composable_kernel_BINARY_DIR}/include CACHE PATH "Path to ck build includes")


    _exclude_from_all(${composable_kernel_SOURCE_DIR})
    _mark_targets_as_system(${composable_kernel_SOURCE_DIR})
endfunction()

# Utility functions, pulled from rocroller repo
macro(_determine_git_tag PREFIX DEFAULT)
    if(HASH)
        set(GIT_TAG ${HASH})
    elseif(VERSION AND NOT "${PREFIX}" STREQUAL "FALSE")
        set(GIT_TAG ${PREFIX}${VERSION})
    else()
        set(GIT_TAG ${DEFAULT})
    endif()
endmacro()

macro(_save_var _name)
    if(DEFINED CACHE{${_name}})
        set(_old_cache_${_name} $CACHE{${_name}})
        unset(${_name} CACHE)
    endif()
    # We can't tell if a variable is referring to a cache or a regular variable.
    # To ensure this gets the value of the regular, variable, temporarily unset
    # the cache variable if it was set before checking for the regular variable.
    if(DEFINED ${_name})
        set(_old_${_name} ${${_name}})
    endif()
    if(DEFINED _old_cache_${_name})
        set(${_name}
            ${_old_cache_${_name}}
            CACHE INTERNAL ""
        )
    endif()
    if(DEFINED ENV{${_name}})
        set(_old_env_${_name} $ENV{${_name}})
    endif()
endmacro()

macro(_restore_var _name)
    if(DEFINED _old_${_name})
        set(${_name} ${_old_${_name}})
        unset(_old_${_name})
    else()
        unset(${_name})
    endif()
    if(DEFINED _old_cache_${_name})
        set(${_name}
            ${_old_cache_${_name}}
            CACHE INTERNAL ""
        )
        unset(_old_cache_${_name})
    else()
        unset(${_name} CACHE)
    endif()
    if(DEFINED _old_env_${_name})
        set(ENV{${_name}} ${_old_env_${_name}})
        unset(_old_env_${_name})
    else()
        unset(ENV{${_name}})
    endif()
endmacro()

## not actually a stack, but that shouldn't be relevant
macro(_pushstate)
    _save_var(CMAKE_CXX_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK)
    unset(CMAKE_CXX_CPPCHECK CACHE)
    _save_var(CMAKE_MESSAGE_INDENT)
    _save_var(CPACK_GENERATOR)
endmacro()

macro(_popstate)
    _restore_var(CPACK_GENERATOR)
    _restore_var(CMAKE_MESSAGE_INDENT)
    _restore_var(CMAKE_CXX_CPPCHECK)
endmacro()

macro(_exclude_from_all _dir)
    set_property(DIRECTORY ${_dir} PROPERTY EXCLUDE_FROM_ALL ON)
endmacro()

macro(_mark_targets_as_system _dirs)
    foreach(_dir ${_dirs})
        get_directory_property(_targets DIRECTORY ${_dir} BUILDSYSTEM_TARGETS)
        foreach(_target IN LISTS _targets)
            get_target_property(
                _includes ${_target} INTERFACE_INCLUDE_DIRECTORIES
            )
            if(_includes)
                target_include_directories(
                    ${_target} SYSTEM INTERFACE ${_includes}
                )
            endif()
        endforeach()
    endforeach()
endmacro()
