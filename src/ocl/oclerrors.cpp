#include <mlopen/errors.hpp>
#include <unordered_map>

namespace mlopen {

std::string OpenCLErrorMessage(int error, const std::string& msg)
{
    static std::unordered_map<int, std::string> error_map = 
    {
        { CL_SUCCESS, "Success" },
        { CL_DEVICE_NOT_FOUND, "Device Not Found" },
        { CL_DEVICE_NOT_AVAILABLE, "Device Not Available" },
        { CL_COMPILER_NOT_AVAILABLE, "Compiler Not Available" },
        { CL_MEM_OBJECT_ALLOCATION_FAILURE, "Mem Object Allocation Failure" },
        { CL_OUT_OF_RESOURCES, "Out Of Resources" },
        { CL_OUT_OF_HOST_MEMORY, "Out Of Host Memory" },
        { CL_PROFILING_INFO_NOT_AVAILABLE, "Profiling Info Not Available" },
        { CL_MEM_COPY_OVERLAP, "Mem Copy Overlap" },
        { CL_IMAGE_FORMAT_MISMATCH, "Image Format Mismatch" },
        { CL_IMAGE_FORMAT_NOT_SUPPORTED, "Image Format Not Supported" },
        { CL_BUILD_PROGRAM_FAILURE, "Build Program Failure" },
        { CL_MAP_FAILURE, "Map Failure" },
        { CL_MISALIGNED_SUB_BUFFER_OFFSET, "Misaligned Sub Buffer Offset" },
        { CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "Exec Status Error For Events In Wait List" },
        { CL_COMPILE_PROGRAM_FAILURE, "Compile Program Failure" },
        { CL_LINKER_NOT_AVAILABLE, "Linker Not Available" },
        { CL_LINK_PROGRAM_FAILURE, "Link Program Failure" },
        { CL_DEVICE_PARTITION_FAILED, "Device Partition Failed" },
        { CL_KERNEL_ARG_INFO_NOT_AVAILABLE, "Kernel Arg Info Not Available" }
    };
    return msg + error_map[error];
}
}
