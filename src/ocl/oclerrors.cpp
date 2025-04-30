/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/errors.hpp>
#include <unordered_map>

namespace miopen {

std::string OpenCLErrorMessage(int error, const std::string& msg)
{
    static const std::unordered_map<int, std::string> error_map = {
        {CL_SUCCESS, "Success"},
        {CL_DEVICE_NOT_FOUND, "Device Not Found"},
        {CL_DEVICE_NOT_AVAILABLE, "Device Not Available"},
        {CL_COMPILER_NOT_AVAILABLE, "Compiler Not Available"},
        {CL_MEM_OBJECT_ALLOCATION_FAILURE, "Mem Object Allocation Failure"},
        {CL_OUT_OF_RESOURCES, "Out Of Resources"},
        {CL_OUT_OF_HOST_MEMORY, "Out Of Host Memory"},
        {CL_PROFILING_INFO_NOT_AVAILABLE, "Profiling Info Not Available"},
        {CL_MEM_COPY_OVERLAP, "Mem Copy Overlap"},
        {CL_IMAGE_FORMAT_MISMATCH, "Image Format Mismatch"},
        {CL_IMAGE_FORMAT_NOT_SUPPORTED, "Image Format Not Supported"},
        {CL_BUILD_PROGRAM_FAILURE, "Build Program Failure"},
        {CL_MAP_FAILURE, "Map Failure"},
        {CL_MISALIGNED_SUB_BUFFER_OFFSET, "Misaligned Sub Buffer Offset"},
        {CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST, "Exec Status Error For Events In Wait List"},
        {CL_COMPILE_PROGRAM_FAILURE, "Compile Program Failure"},
        {CL_LINKER_NOT_AVAILABLE, "Linker Not Available"},
        {CL_LINK_PROGRAM_FAILURE, "Link Program Failure"},
        {CL_DEVICE_PARTITION_FAILED, "Device Partition Failed"},
        {CL_KERNEL_ARG_INFO_NOT_AVAILABLE, "Kernel Arg Info Not Available"},
        {CL_INVALID_VALUE, "Invalid Value"},
        {CL_INVALID_DEVICE_TYPE, "Invalid Device Type"},
        {CL_INVALID_PLATFORM, "Invalid Platform"},
        {CL_INVALID_DEVICE, "Invalid Device"},
        {CL_INVALID_CONTEXT, "Invalid Context"},
        {CL_INVALID_QUEUE_PROPERTIES, "Invalid Queue Properties"},
        {CL_INVALID_COMMAND_QUEUE, "Invalid Command Queue"},
        {CL_INVALID_HOST_PTR, "Invalid Host Ptr"},
        {CL_INVALID_MEM_OBJECT, "Invalid Mem Object"},
        {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "Invalid Image Format Descriptor"},
        {CL_INVALID_IMAGE_SIZE, "Invalid Image Size"},
        {CL_INVALID_SAMPLER, "Invalid Sampler"},
        {CL_INVALID_BINARY, "Invalid Binary"},
        {CL_INVALID_BUILD_OPTIONS, "Invalid Build Options"},
        {CL_INVALID_PROGRAM, "Invalid Program"},
        {CL_INVALID_PROGRAM_EXECUTABLE, "Invalid Program Executable"},
        {CL_INVALID_KERNEL_NAME, "Invalid Kernel Name"},
        {CL_INVALID_KERNEL_DEFINITION, "Invalid Kernel Definition"},
        {CL_INVALID_KERNEL, "Invalid Kernel"},
        {CL_INVALID_ARG_INDEX, "Invalid Arg Index"},
        {CL_INVALID_ARG_VALUE, "Invalid Arg Value"},
        {CL_INVALID_ARG_SIZE, "Invalid Arg Size"},
        {CL_INVALID_KERNEL_ARGS, "Invalid Kernel Args"},
        {CL_INVALID_WORK_DIMENSION, "Invalid Work Dimension"},
        {CL_INVALID_WORK_GROUP_SIZE, "Invalid Work Group Size"},
        {CL_INVALID_WORK_ITEM_SIZE, "Invalid Work Item Size"},
        {CL_INVALID_GLOBAL_OFFSET, "Invalid Global Offset"},
        {CL_INVALID_EVENT_WAIT_LIST, "Invalid Event Wait List"},
        {CL_INVALID_EVENT, "Invalid Event"},
        {CL_INVALID_OPERATION, "Invalid Operation"},
        {CL_INVALID_GL_OBJECT, "Invalid Gl Object"},
        {CL_INVALID_BUFFER_SIZE, "Invalid Buffer Size"},
        {CL_INVALID_MIP_LEVEL, "Invalid Mip Level"},
        {CL_INVALID_GLOBAL_WORK_SIZE, "Invalid Global Work Size"},
        {CL_INVALID_PROPERTY, "Invalid Property"},
        {CL_INVALID_IMAGE_DESCRIPTOR, "Invalid Image Descriptor"},
        {CL_INVALID_COMPILER_OPTIONS, "Invalid Compiler Options"},
        {CL_INVALID_LINKER_OPTIONS, "Invalid Linker Options"},
        {CL_INVALID_DEVICE_PARTITION_COUNT, "Invalid Device Partition Count"}};
    if(error_map.count(error) > 0)
    {
        return msg + " " + error_map.at(error);
    }
    else
    {
        return msg + "Unknown OpenCL error " + std::to_string(error);
    }
}
} // namespace miopen
