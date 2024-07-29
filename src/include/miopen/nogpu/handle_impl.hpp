/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_NOGPU_HANDLE_IMPL_HPP_
#define GUARD_MIOPEN_NOGPU_HANDLE_IMPL_HPP_
namespace miopen {

struct HandleImpl
{
    using StreamPtr = std::shared_ptr<typename std::remove_pointer<hipStream_t>::type>;

    HandleImpl() : ctx() {}

    void elapsed_time(hipEvent_t start, hipEvent_t stop)
    {
        if(enable_profiling)
            hipEventElapsedTime(&this->profiling_result, start, stop);
    }

    std::function<void(hipEvent_t, hipEvent_t)> elapsed_time_handler()
    {
        return std::bind(
            &HandleImpl::elapsed_time, this, std::placeholders::_1, std::placeholders::_2);
    }

    bool enable_profiling = false;
    StreamPtr stream      = nullptr;
    rocblas_handle_ptr rhandle_;
    hipblasLt_handle_ptr hip_blasLt_handle;
    float profiling_result = 0.0;
    int device             = -1;
    std::string device_name;
    std::size_t num_cu             = 0;
    std::size_t local_mem_size     = 0;
    std::size_t global_mem_size    = 0;
    std::size_t img3d_max_width    = 0;
    std::size_t warp_size          = 64;
    std::size_t max_mem_alloc_size = 0;
    Allocator allocator{};
    KernelCache cache;
    std::int64_t ctx;
    TargetProperties target_properties;
};
} // namespace miopen
#endif // GUARD_MIOPEN_NOGPU_HANDLE_IMPL_HPP_
