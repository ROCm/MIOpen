#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include "float_types.h"
#include "MIOpenCol2Im3d.cl"  // The path to your OpenCL kernel file

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// Utility function to read kernel file
std::string readKernelFile(const std::string& filename) {
    std::ifstream file(filename);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int main() {
    cl_int err;

    // Initialize OpenCL context, device, and command queue
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    // Get platform and device
    err = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Kernel source
    std::string kernel_source = readKernelFile("MIOpenCol2Im3d.cl");
    const char* source_str = kernel_source.c_str();

    // Create program
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, nullptr, &err);

    // Build the program
    err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "Col2Im3d", &err);

    // Allocate and initialize memory buffers
    const int col_size = 1024; // Example size
    const int im_size = 1024;  // Example size
    cl_mem col_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, col_size * sizeof(float), nullptr, &err);
    cl_mem im_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, im_size * sizeof(float), nullptr, &err);

    // Dummy data for example - normally you would initialize with your actual data
    std::vector<float> col(col_size, 1.0f); // Initialize 'col' with 1.0
    std::vector<float> im(im_size, 0.0f);   // Initialize 'im' with 0.0

    // Write data to buffer
    err = clEnqueueWriteBuffer(queue, col_buffer, CL_TRUE, 0, col_size * sizeof(float), col.data(), 0, nullptr, nullptr);

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &col_buffer);
    // ... Set other kernel arguments here ...
    err = clSetKernelArg(kernel, 20, sizeof(cl_mem), &im_buffer);

    // Define global work size
    size_t global_work_size[1] = { static_cast<size_t>(col_size) };

    // Launch the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);

    // Read the result back to host memory
    err = clEnqueueReadBuffer(queue, im_buffer, CL_TRUE, 0, im_size * sizeof(float), im.data(), 0, nullptr, nullptr);

    // Release resources
    clReleaseMemObject(col_buffer);
    clReleaseMemObject(im_buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}

