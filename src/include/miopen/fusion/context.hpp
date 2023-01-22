/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

namespace miopen {

struct FusionContext : miopen::ExecutionContext
{
    FusionContext(Handle& handle) : ExecutionContext(&handle) {}

    // ConvolutionContext GetConvContext(size_t idx, conv::Direction dir, const FusionDescription&
    // problem) const
    ConvolutionContext GetConvContext(size_t idx, conv::Direction dir) const
    {
        MIOPEN_THROW("Do Something");
        return {};
#if 0
        const auto conv_prob = problem.GetConvProblem(idx, dir);
        if(dir == conv::Direction::Forward)
        {
            auto ctx = ConvolutionContext{conv_prob.conv_problem, *this};
            ctx.SetStream(&this->GetStream());
            ctx.DetectRocm();
            ctx.SetupFloats();
            return ctx;
        }
        else
        {
            MIOPEN_THROW(miopenStatusNotImplemented);
        }
#endif
    }
    bool is_for_generic_search = false;
};

} // namespace miopen
