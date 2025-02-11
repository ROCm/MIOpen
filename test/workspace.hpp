/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#pragma once

#include <hip/hip_runtime.h>

#include "get_handle.hpp"

#define HIP_CHECK(exp)                                                                 \
    if((exp) != hipSuccess)                                                            \
    {                                                                                  \
        MIOPEN_LOG_E(#exp "failed at line: " << __LINE__ << " in file: " << __FILE__); \
    }

class Workspace
{

    // RAII class for hip allocations
    class GPUBuffer
    {
    public:
        GPUBuffer() = default;

        explicit GPUBuffer(size_t num_bytes) : sz_(num_bytes)
        {
            if(num_bytes > 0)
            {
                HIP_CHECK(hipMalloc(&buf_, num_bytes));
                assert(buf_ != nullptr);
            }
            else
            {
                buf_ = nullptr;
            }
        }

        ~GPUBuffer() { FreeBuf(); }

        void* ptr() { return buf_; }
        void* ptr() const { return buf_; }

        auto size() const { return sz_; }

        GPUBuffer(const GPUBuffer&) = delete;
        GPUBuffer& operator=(const GPUBuffer&) = delete;

        GPUBuffer(GPUBuffer&& that) noexcept : buf_(that.buf_), sz_(that.sz_)
        {
            that.buf_ = nullptr; // take over ownership
            that.sz_  = 0;
        }

        GPUBuffer& operator=(GPUBuffer&& that) noexcept
        {
            FreeBuf();
            std::swap(this->buf_, that.buf_);
            std::swap(this->sz_, that.sz_);
            return *this;
        }

    private:
        void FreeBuf()
        {
            HIP_CHECK(hipFree(buf_));
            buf_ = nullptr;
            sz_  = 0;
        }

        void* buf_ = nullptr;
        size_t sz_ = 0;
    };

    // for use in miopen .*GetWorkSpaceSize() methods where a pointer to size_t is
    // passed to capture the size. Must call AdjustToSize() after calling such a method
    size_t* SizePtr() { return &sz_; }

    void AdjustToSize()
    {
        if(sz_ != gpu_buf_.size())
        {
            gpu_buf_ = GPUBuffer(sz_);
        }
    }

public:
    explicit Workspace(size_t sz = 0) : sz_(sz) { AdjustToSize(); }

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&&)                 = default;
    Workspace& operator=(Workspace&&) = default;

    size_t size() const { return sz_; }

    void resize(size_t sz_in_bytes)
    {
        sz_ = sz_in_bytes;
        AdjustToSize();
    }

    auto ptr() const { return gpu_buf_.ptr(); }

    auto ptr() { return gpu_buf_.ptr(); }

    template <typename V>
    void Write(const V& vec)
    {
        using T = typename V::value_type;
        resize(vec.size() * sizeof(T));
        HIP_CHECK(hipMemcpy(this->ptr(), &vec[0], size(), hipMemcpyHostToDevice));
    }

    template <typename V>
    void ReadTo(V& vec) const
    {
        using T = typename V::value_type;
        if(vec.size() * sizeof(T) != size())
        {
            MIOPEN_LOG_E("vector of wrong size passed");
            std::abort();
        }
        HIP_CHECK(hipMemcpy(&vec[0], ptr(), size(), hipMemcpyDeviceToHost));
    }

    template <typename V>
    V Read() const
    {
        using T         = typename V::value_type;
        size_t num_elem = size() / sizeof(T);
        V ret(num_elem);
        ReadTo(ret);
        return ret;
    }

private:
    // const miopen::Handle& handle_;
    // miopen::Allocator::ManageDataPtr data_{};
    GPUBuffer gpu_buf_{};
    size_t sz_{};
};
