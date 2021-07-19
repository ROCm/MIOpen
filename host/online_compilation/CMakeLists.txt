set(CMAKE_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

## for online-compiling of HIP kernels
set(OLC_HIP_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")

## reset to avoid the C++ options from the parent project
set(CMAKE_CXX_FLAGS "")
message("Compiling options for library and kernels: ${CMAKE_CXX_FLAGS}")

# look for and register clang-offload-bundler
if(OLC_HIP_COMPILER MATCHES ".*clang\\+\\+$")
    find_program(OLC_OFFLOADBUNDLER_BIN clang-offload-bundler
        PATH_SUFFIXES bin
        PATHS
	    /opt/rocm/llvm
	    ${CMAKE_INSTALL_PREFIX}/llvm
    )
endif()

if(OLC_OFFLOADBUNDLER_BIN)
    message(STATUS "clang-offload-bundler found: ${OLC_OFFLOADBUNDLER_BIN}")
    set(OLC_OFFLOADBUNDLER_BIN "${OLC_OFFLOADBUNDLER_BIN}")
else()
    # look for and register extractkernel
    message(STATUS "clang-offload-bundler not found")

    find_program(EXTRACTKERNEL_BIN extractkernel
        PATH_SUFFIXES bin
        PATHS
            /opt/rocm/hip
            /opt/rocm/hcc
            /opt/rocm
	    ${CMAKE_INSTALL_PREFIX}/hip
            ${CMAKE_INSTALL_PREFIX}/hcc
            ${CMAKE_INSTALL_PREFIX}

    )
    if(EXTRACTKERNEL_BIN)
        message(STATUS "extractkernel found: ${EXTRACTKERNEL_BIN}")
        set(EXTRACTKERNEL_BIN "${EXTRACTKERNEL_BIN}")
    else()
        message(FATAL_ERROR "extractkernel not found")
    endif()
endif()

option(Boost_USE_STATIC_LIBS "Use boost static libraries" OFF)
set(BOOST_COMPONENTS filesystem)
add_definitions(-DBOOST_ALL_NO_LIB=1)
find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})

# HIP is always required
find_package(hip REQUIRED PATHS /opt/rocm)
message(STATUS "Build with HIP ${hip_VERSION}")
target_flags(HIP_COMPILER_FLAGS hip::device)
# Remove cuda arch flags
string(REGEX REPLACE --cuda-gpu-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")
string(REGEX REPLACE --offload-arch=[a-z0-9]+ "" HIP_COMPILER_FLAGS "${HIP_COMPILER_FLAGS}")

set(OLC_hip_VERSION_MAJOR "${hip_VERSION_MAJOR}")
set(OLC_hip_VERSION_MINOR "${hip_VERSION_MINOR}")
set(OLC_hip_VERSION_PATCH "${hip_VERSION_PATCH}")

option(ENABLE_DEBUG "Build to enable debugging" ON)
if(ENABLE_DEBUG)
    set(OLC_DEBUG 1)
else()
    set(OLC_DEBUG 0)
endif()

configure_file("${PROJECT_SOURCE_DIR}/host/online_compilation/include/config.h.in" "${PROJECT_BINARY_DIR}/host/online_compilation/include/config.h")

include_directories(BEFORE
    ${PROJECT_BINARY_DIR}/host/online_compilation/include
)

message(STATUS "Hip compiler flags: ${HIP_COMPILER_FLAGS}")

## HIP_COMPILER_FLAGS will be used for on-line compiling of the HIP kernels
add_definitions("-DHIP_COMPILER_FLAGS=${HIP_COMPILER_FLAGS}")

file(GLOB_RECURSE COMPOSABLE_KERNEL_INCLUDE_1 "${PROJECT_SOURCE_DIR}/composable_kernel/include/*/*.hpp")
file(GLOB COMPOSABLE_KERNEL_INCLUDE_2 "${PROJECT_SOURCE_DIR}/external/rocm/include/bfloat16_dev.hpp")
set(MCONV_KERNEL_INCLUDES
    ${COMPOSABLE_KERNEL_INCLUDE_1}
    ${COMPOSABLE_KERNEL_INCLUDE_2}
   )

file(GLOB_RECURSE MCONV_KERNELS "${PROJECT_SOURCE_DIR}/composable_kernel/src/kernel_wrapper/*.cpp")

add_kernels(${CMAKE_CURRENT_SOURCE_DIR} "${MCONV_KERNELS}")
add_kernel_includes(${CMAKE_CURRENT_SOURCE_DIR} "${MCONV_KERNEL_INCLUDES}")

set(ONLINE_COMPILATION_SOURCE
     ${PROJECT_BINARY_DIR}/kernel.cpp
     ${PROJECT_BINARY_DIR}/kernel_includes.cpp
)

include_directories(BEFORE
    ${PROJECT_BINARY_DIR}/host/online_compilation/include
    include
)

set(OLC_HIP_UTILITY_CPPS
    hip_utility/logger.cpp
    hip_utility/tmp_dir.cpp
    hip_utility/md5.cpp  
    hip_utility/exec_utils.cpp
    hip_utility/target_properties.cpp  
    hip_utility/handlehip.cpp
    hip_utility/kernel_build_params.cpp  
    hip_utility/hip_build_utils.cpp  
    hip_utility/hipoc_program.cpp  
    hip_utility/hipoc_kernel.cpp  
    hip_utility/kernel_cache.cpp  
    hip_utility/binary_cache.cpp
   )

list(APPEND OLC_SOURCES ${OLC_HIP_UTILITY_CPPS} ${OLC_HIP_UTILITY_HEADERS})

## addkernels provide the tool to create inlined kernels in one header
add_subdirectory(addkernels)

function(inline_kernels_src KERNELS KERNEL_INCLUDES)
    set(KERNEL_SRC_HPP_FILENAME batch_all.cpp.hpp)
    set(KERNEL_SRC_HPP_PATH ${PROJECT_BINARY_DIR}/inlined_kernels/${KERNEL_SRC_HPP_FILENAME})
    set(KERNEL_SRC_CPP_PATH ${PROJECT_BINARY_DIR}/inlined_kernels/batch_all.cpp)

    add_custom_command(
        OUTPUT ${KERNEL_SRC_HPP_PATH}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS addkernels ${KERNELS} ${KERNEL_INCLUDES}
        COMMAND $<TARGET_FILE:addkernels> -target ${KERNEL_SRC_HPP_PATH} -extern -source ${KERNELS}
	COMMENT "Inlining All kernels"
    )
    configure_file(kernels_batch.cpp.in ${KERNEL_SRC_CPP_PATH})
    list(APPEND OLC_SOURCES ${KERNEL_SRC_CPP_PATH} ${KERNEL_SRC_HPP_PATH})

    set(OLC_SOURCES ${OLC_SOURCES} PARENT_SCOPE)
endfunction()

inline_kernels_src("${MCONV_KERNELS}" "${MCONV_KERNEL_INCLUDES}")

list(APPEND ONLINE_COMPILATION_SOURCE ${OLC_SOURCES} ${PROJECT_BINARY_DIR}/olc_kernel_includes.h)

add_custom_command(
    OUTPUT ${PROJECT_BINARY_DIR}/olc_kernel_includes.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS addkernels ${MCONV_KERNEL_INCLUDES}
    COMMAND $<TARGET_FILE:addkernels> -no-recurse -guard GUARD_OLC_KERNEL_INCLUDES_HPP_ -target ${PROJECT_BINARY_DIR}/olc_kernel_includes.h -source ${MCONV_KERNEL_INCLUDES}
    COMMENT "Inlining HIP kernel includes"
  )

## the library target
add_library(online_compilation SHARED ${ONLINE_COMPILATION_SOURCE}) 

target_include_directories(online_compilation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/online_compilation/include/)
target_include_directories(online_compilation PRIVATE ${PROJECT_BINARY_DIR})
target_include_directories(online_compilation PRIVATE ${PROJECT_SOURCE_DIR}/external/half/include/)

target_link_libraries(online_compilation PRIVATE hip::device)
target_link_libraries(online_compilation INTERFACE hip::host)
target_link_libraries(online_compilation PRIVATE Boost::filesystem)

target_compile_features(online_compilation PUBLIC)
set_target_properties(online_compilation PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS online_compilation LIBRARY DESTINATION lib) 
