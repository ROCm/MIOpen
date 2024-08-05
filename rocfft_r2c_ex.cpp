// Copyright (C) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <complex>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>
#include <rocfft/rocfft.h>

#include "../../../shared/CLI11.hpp"
#include "examplekernels.h"
#include "exampleutils.h"
#include <stdexcept>

int main(int argc, char* argv[])
{
    std::cout << "rocfft double-precision real/complex transform\n" << std::endl;

    // Length of transform:
    std::vector<size_t> length = {8};

    // Gpu device id:
    size_t deviceId = 0;

    // Command-line options:
    CLI::App app{"rocfft sample command line options"};
    app.add_option("--device", deviceId, "Select a specific device id")->default_val(0);
    CLI::Option* opt_outofplace =
        app.add_flag("-o, --outofplace", "Perform an out-of-place transform");
    CLI::Option* opt_inverse = app.add_flag("-i, --inverse", "Perform an inverse transform");
    app.add_option(
        "--length", length, "Lengths of the transform separated by spaces (eg: --length 4 4)");

    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    // Placeness for the transform
    if(rocfft_setup() != rocfft_status_success)
        throw std::runtime_error("rocfft_setup failed.");
    const rocfft_result_placement place =
        *opt_outofplace ? rocfft_placement_notinplace : rocfft_placement_inplace;
    const bool inplace = place == rocfft_placement_inplace;

    // Direction of transform
    const rocfft_transform_type direction =
        *opt_inverse ? rocfft_transform_type_real_inverse : rocfft_transform_type_real_forward;
    const bool forward = direction == rocfft_transform_type_real_forward;

    // Set up the strides and buffer size for the real values:
    std::vector<size_t> rstride = {1};
    for(unsigned int i = 1; i < length.size(); ++i)
    {
        // In-place transforms need space for two extra real values in the contiguous
        // direction.
        auto val = (length[i - 1] + ((inplace && i == 1) ? 2 : 0)) * rstride[i - 1];
        rstride.push_back(val);
    }
    // NB: not tight, but hey
    const size_t real_size = length[length.size() - 1] * rstride[rstride.size() - 1];
    std::vector<double> rdata(real_size); // host storage

    // The complex data length is half + 1 of the real data length in the contiguous
    // dimensions.  Since rocFFT is column-major, this is the first index.
    std::vector<size_t> clength = length;
    clength[0]                  = clength[0] / 2 + 1;
    std::vector<size_t> cstride = {1};
    for(unsigned int i = 1; i < clength.size(); ++i)
    {
        cstride.push_back(clength[i - 1] * cstride[i - 1]);
    }
    const size_t complex_size = clength[clength.size() - 1] * cstride[cstride.size() - 1];
    std::vector<hipDoubleComplex> cdata(complex_size); // host storage

    // Based on the direction, we set the input and output parameters appropriately.
    const size_t isize  = forward ? real_size : complex_size;
    const size_t ibytes = isize * (forward ? sizeof(double) : sizeof(hipDoubleComplex));
    const std::vector<size_t> ilength = forward ? length : clength;
    const std::vector<size_t> istride = forward ? rstride : cstride;

    const size_t osize  = forward ? complex_size : real_size;
    const size_t obytes = osize * (forward ? sizeof(hipDoubleComplex) : sizeof(double));
    const std::vector<size_t> olength = forward ? clength : length;
    const std::vector<size_t> ostride = forward ? cstride : rstride;

    // Print information about the transform:
    std::cout << "direction: ";
    if(forward)
        std::cout << "forward\n";
    else
        std::cout << "inverse\n";
    std::cout << "length:";
    for(const auto i : length)
        std::cout << " " << i;
    std::cout << "\n";
    if(inplace)
        std::cout << "in-place transform\n";
    else
        std::cout << "out-of-place transform\n";
    std::cout << "deviceID: " << deviceId << "\n";
    std::cout << "input length:";
    for(auto i : ilength)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "input buffer stride:";
    for(auto i : istride)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "input buffer size: " << ibytes << "\n";

    std::cout << "output length:";
    for(auto i : olength)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "output buffer stride:";
    for(auto i : ostride)
        std::cout << " " << i;
    std::cout << "\n";
    std::cout << "output buffer size: " << obytes << "\n";
    std::cout << std::endl;

    // Set the device:
    if(hipSetDevice(deviceId) != hipSuccess)
        throw std::runtime_error("hipSetDevice failed.");

    // Create HIP device object and initialize data
    // Kernels are provided in examplekernels.h
    void* gpu_in          = nullptr;
    hipError_t hip_status = hipMalloc(&gpu_in, inplace ? std::max(ibytes, obytes) : ibytes);
    if(hip_status != hipSuccess)
        throw std::runtime_error("device error");

    if(forward)
    {
        initreal_cm(length, istride, gpu_in);
    }
    else
    {
        init_hermitiancomplex_cm(length, ilength, istride, gpu_in);
    }

    // Print the input:
    std::cout << "input:\n";
    if(forward)
    {
        hip_status = hipMemcpy(rdata.data(), gpu_in, ibytes, hipMemcpyDeviceToHost);
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMemcpy failed.");
        printbuffer_cm(rdata, ilength, istride, 1, isize);
    }
    else
    {
        hip_status = hipMemcpy(cdata.data(), gpu_in, ibytes, hipMemcpyDeviceToHost);
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMemcpy failed.");
        printbuffer_cm(cdata, ilength, istride, 1, isize);

        // Check that the buffer is Hermitian symmetric:
        check_symmetry_cm(cdata, length, istride, 1, isize);
    }

    // rocfft_status can be used to capture API status info
    rocfft_status rc = rocfft_status_success;

    // Create the a descrition struct to set data layout:
    rocfft_plan_description gpu_description = nullptr;
    rc                                      = rocfft_plan_description_create(&gpu_description);
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to create plan description");

    rc = rocfft_plan_description_set_data_layout(
        gpu_description,
        // input data format:
        forward ? rocfft_array_type_real : rocfft_array_type_hermitian_interleaved,
        // output data format:
        forward ? rocfft_array_type_hermitian_interleaved : rocfft_array_type_real,
        nullptr,
        nullptr,
        istride.size(), // input stride length
        istride.data(), // input stride data
        0,              // input batch distance
        ostride.size(), // output stride length
        ostride.data(), // output stride data
        0);             // ouptut batch distance
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to set data layout");

    // We can also pass "nullptr" instead of a description; rocFFT will use reasonable
    // default parameters.  If the data isn't contiguous, we need to set strides, etc,
    // using the description.

    // Create the FFT plan:
    rocfft_plan gpu_plan = nullptr;
    rc                   = rocfft_plan_create(&gpu_plan,
                            place,
                            direction,
                            rocfft_precision_double,
                            length.size(),    // Dimension
                            length.data(),    // lengths
                            1,                // Number of transforms
                            gpu_description); // Description
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to create plan");

    // Get the execution info for the fft plan (in particular, work memory requirements):
    rocfft_execution_info planinfo = nullptr;
    rc                             = rocfft_execution_info_create(&planinfo);
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to create execution info");

    size_t workbuffersize = 0;
    rc                    = rocfft_plan_get_work_buffer_size(gpu_plan, &workbuffersize);
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to get work buffer size");

    // If the transform requires work memory, allocate a work buffer:
    void* wbuffer = nullptr;
    if(workbuffersize > 0)
    {
        hip_status = hipMalloc(&wbuffer, workbuffersize);
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMalloc failed");

        rc = rocfft_execution_info_set_work_buffer(planinfo, wbuffer, workbuffersize);
        if(rc != rocfft_status_success)
            throw std::runtime_error("failed to set work buffer");
    }

    // If the transform is out-of-place, allocate the output buffer as well:
    void* gpu_out = inplace ? gpu_in : nullptr;
    if(!inplace)
    {
        hip_status = hipMalloc(&gpu_out, obytes);
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMalloc failed");
    }

    // Execute the GPU transform:
    rc = rocfft_execute(gpu_plan,         // plan
                        (void**)&gpu_in,  // in_buffer
                        (void**)&gpu_out, // out_buffer
                        planinfo);        // execution info
    if(rc != rocfft_status_success)
        throw std::runtime_error("failed to execute");

    // Get the output from the device and print to cout:
    std::cout << "output:\n";
    if(forward)
    {
        hip_status = hipMemcpy(cdata.data(), gpu_out, obytes, hipMemcpyDeviceToHost);
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMemcpy failed.");
        printbuffer_cm(cdata, olength, ostride, 1, osize);
    }
    else
    {
        hip_status = hipMemcpy(rdata.data(), gpu_out, obytes, hipMemcpyDeviceToHost);
        if(hip_status != hipSuccess)
            throw std::runtime_error("hipMemcpy failed.");
        printbuffer_cm(rdata, olength, ostride, 1, osize);
    }

    // Clean up: free GPU memory:
    if(hipFree(gpu_in) != hipSuccess)
        throw std::runtime_error("hipFree failed.");

    if(!inplace)
    {
        if(hipFree(gpu_out) != hipSuccess)
            throw std::runtime_error("hipFree failed.");
    }
    if(wbuffer != nullptr)
    {
        if(hipFree(wbuffer) != hipSuccess)
            throw std::runtime_error("hipFree failed.");
    }

    // Clean up: destroy plans:
    if(rocfft_execution_info_destroy(planinfo) != rocfft_status_success)
        throw std::runtime_error("rocfft_execution_info_destroy failed.");
    planinfo = nullptr;
    if(rocfft_plan_description_destroy(gpu_description) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_description_destroy failed.");
    gpu_description = nullptr;
    if(rocfft_plan_destroy(gpu_plan) != rocfft_status_success)
        throw std::runtime_error("rocfft_plan_destroy failed.");
    gpu_plan = nullptr;

    rocfft_cleanup();
    return 0;
}
