/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
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

#include <miopen/db_path.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/problem_description.hpp>
#include <miopen/miopen.h>

#include <string>

namespace miopen {
struct ConvolutionDescriptor;
struct Handle;
struct TensorDescriptor;

struct ConvolutionUserBuffers
{
    union
    {
        struct Fwd
        {
            ConstData_t x;
            ConstData_t w;
            Data_t y;
        } fwd;
        struct Bwd
        {
            Data_t dx;
            ConstData_t w;
            ConstData_t dy;
        } bwd;
        struct WrW
        {
            ConstData_t dx;
            Data_t dw;
            ConstData_t dy;
        } wrw;
    } io;
    Data_t workSpace;
    size_t workSpaceSize;
    ConstData_t bias;
    ConvolutionUserBuffers(Data_t w, size_t s, ConstData_t b = nullptr)
        : io({{nullptr, nullptr, nullptr}}), workSpace(w), workSpaceSize(s), bias(b)
    {
    }
    ConvolutionUserBuffers() : ConvolutionUserBuffers(nullptr, 0, nullptr) {}
    void SetFwd(ConstData_t x, ConstData_t w, Data_t y)
    {
        io.fwd.x = x;
        io.fwd.y = y;
        io.fwd.w = w;
    }
    void SetBwd(Data_t dx, ConstData_t w, ConstData_t dy)
    {
        io.bwd.dx = dx;
        io.bwd.dy = dy;
        io.bwd.w  = w;
    }
    void SetWrW(ConstData_t dx, Data_t dw, ConstData_t dy)
    {
        io.wrw.dx = dx;
        io.wrw.dy = dy;
        io.wrw.dw = dw;
    }
};

/// A leftover of the legacy design, houses problem config,
/// environmental context (e.g. HW/SW platform) and solver-specific state.
///
/// TODO: These three entities should be made separate.
struct ConvolutionContext : ProblemDescription, ExecutionContext
{
    // Solution-specific
    std::string general_compile_options;

    ConvolutionContext() = default;
    ConvolutionContext(conv::Direction dir) : ProblemDescription(dir) {}
    ConvolutionContext(const TensorDescriptor& in,
                       const TensorDescriptor& weights,
                       const TensorDescriptor& out,
                       const ConvolutionDescriptor& conv,
                       conv::Direction dir,
                       int bias_ = 0)
        : ProblemDescription(in, weights, out, conv, dir, bias_)
    {
    }
    ConvolutionContext(const ProblemDescription& problem) : ProblemDescription(problem) {}

    void SetupFloats();

public:
    bool is_for_generic_search = false;

    inline void SetBufs(const ConvolutionUserBuffers& bufs) { _bufs = bufs; }
    inline const ConvolutionUserBuffers& GetBufs() const { return _bufs; }

private:
    ConvolutionUserBuffers _bufs;
};

} // namespace miopen
