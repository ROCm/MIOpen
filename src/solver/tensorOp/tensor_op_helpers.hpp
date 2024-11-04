/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <miopen/tensorOp/problem_description.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/datatype.hpp>

namespace miopen {

namespace solver {

namespace tensorOp {

inline void GetCommonParams(KernelBuildParameters& build_params,
                            miopen::tensorOp::ProblemDescription problem,
                            bool is64bSupported)
{
    build_params.Define("MIOPEN_TYPE", miopen::GetDataType(problem.GetBTensorDesc().GetType()));

    switch(problem.GetTensorOp())
    {
    case 0: build_params.Define("MIOPEN_TENSOR_OP", "miopenAdd"); break;
    case 1: build_params.Define("MIOPEN_TENSOR_OP", "miopenMul"); break;
    case 2: build_params.Define("MIOPEN_TENSOR_OP", "miopenMin"); break;
    case 3: build_params.Define("MIOPEN_TENSOR_OP", "miopenMax"); break;
    }

    if(is64bSupported && problem.GetATensorDesc().AllDimsFitIntoInt())
    {
        build_params.Define("DIM_TYPE", "uint32_t");
    }
    else
    {
        build_params.Define("DIM_TYPE", "uint64_t");
    }
    // current workaround
    build_params.Define("MIOPEN_USE_FP16", std::to_string(0));
    build_params.Define("MIOPEN_USE_FP32", std::to_string(1));
}

inline void
GetRDBLCKandREADTYPE(size_t len, miopenDataType_t type, size_t& RD_BLCK, std::string& READ_TYPE)
{
    RD_BLCK                     = (len % 4 == 0) ? 4 : (len % 2 == 0) ? 2 : 1;
    const std::string data_type = GetDataType(type);
    READ_TYPE                   = (RD_BLCK == 1) ? data_type : data_type + std::to_string(RD_BLCK);
}

} // namespace tensorOp

} // namespace solver

} // namespace miopen
