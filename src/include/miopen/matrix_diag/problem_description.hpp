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

#include <miopen/names.hpp>
#include <miopen/miopen.h>
#include <miopen/activ.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

struct NetworkConfig;

namespace matrix_diag {

bool IsValidMatrixDiag(const TensorDescriptor& padDesc,
                       const TensorDescriptor& diagDesc,
                       const TensorDescriptor& outDesc,
                       int64_t diagOffset0,
                       int64_t diagOffset1,
                       const std::string& refer_name,
                       const std::string& pad_alias_name,
                       const std::string& diag_alias_name,
                       const std::string& out_alias_name);

struct MatrixSetDiagForwardProblemDescription : ProblemDescriptionBase
{
    MatrixSetDiagForwardProblemDescription(const TensorDescriptor& inputDesc_,
                                           const TensorDescriptor& diagDesc_,
                                           const TensorDescriptor& outputDesc_,
                                           const int64_t diagOffset0_,
                                           const int64_t diagOffset1_,
                                           const miopenMatrixDiagAlignMode_t align_,
                                           const std::string refer_name_ = "MatrixSetDiagForward",
                                           const std::string input_alias_name_  = "Input",
                                           const std::string diag_alias_name_   = "Diagonal",
                                           const std::string output_alias_name_ = "Output")
        : inputDesc(inputDesc_),
          diagDesc(diagDesc_),
          outputDesc(outputDesc_),
          diagOffset0(diagOffset0_),
          diagOffset1(diagOffset1_),
          align(align_),
          refer_name(refer_name_),
          input_alias_name(input_alias_name_),
          diag_alias_name(diag_alias_name_),
          output_alias_name(output_alias_name_)
    {
        IsValidMatrixDiag(inputDesc,
                          diagDesc,
                          outputDesc,
                          diagOffset0,
                          diagOffset1,
                          refer_name,
                          input_alias_name,
                          diag_alias_name,
                          output_alias_name);
    }

    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    miopenMatrixDiagAlignMode_t GetAlign() const { return align; }

    bool IsAllContiguous() const
    {
        if(!diagDesc.IsContiguous())
            return false;
        if(!inputDesc.IsContiguous())
            return false;
        if(!outputDesc.IsContiguous())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    const TensorDescriptor inputDesc;
    const TensorDescriptor diagDesc;
    const TensorDescriptor outputDesc;
    const int64_t diagOffset0;
    const int64_t diagOffset1;
    const miopenMatrixDiagAlignMode_t align;

    const std::string refer_name;
    const std::string input_alias_name;
    const std::string diag_alias_name;
    const std::string output_alias_name;

    NetworkConfig MakeForwardNetworkConfig() const;
};

struct MatrixDiagPartForwardProblemDescription : ProblemDescriptionBase
{
    MatrixDiagPartForwardProblemDescription(const TensorDescriptor& inputDesc_,
                                            const TensorDescriptor& padDesc_,
                                            const TensorDescriptor& outputDesc_,
                                            const int64_t diagOffset0_,
                                            const int64_t diagOffset1_,
                                            const miopenMatrixDiagAlignMode_t align_,
                                            const std::string refer_name_ = "MatrixDiagPartForward",
                                            const std::string input_alias_name_  = "Input",
                                            const std::string pad_alias_name_    = "Padding",
                                            const std::string output_alias_name_ = "Output")
        : inputDesc(inputDesc_),
          padDesc(padDesc_),
          outputDesc(outputDesc_),
          diagOffset0(diagOffset0_),
          diagOffset1(diagOffset1_),
          align(align_),
          refer_name(refer_name_),
          input_alias_name(input_alias_name_),
          pad_alias_name(pad_alias_name_),
          output_alias_name(output_alias_name_)
    {
        IsValidMatrixDiag(padDesc,
                          outputDesc,
                          inputDesc,
                          diagOffset0,
                          diagOffset1,
                          refer_name,
                          pad_alias_name,
                          output_alias_name,
                          input_alias_name);
    }

    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }
    miopenMatrixDiagAlignMode_t GetAlign() const { return align; }

    bool IsAllContiguous() const
    {
        if(!padDesc.IsContiguous())
            return false;
        if(!inputDesc.IsContiguous())
            return false;
        if(!outputDesc.IsContiguous())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

protected:
    const TensorDescriptor inputDesc;
    const TensorDescriptor padDesc;
    const TensorDescriptor outputDesc;
    const int64_t diagOffset0;
    const int64_t diagOffset1;
    const miopenMatrixDiagAlignMode_t align;

    const std::string refer_name;
    const std::string input_alias_name;
    const std::string pad_alias_name;
    const std::string output_alias_name;

    NetworkConfig MakeForwardNetworkConfig() const;
};

} // namespace matrix_diag

} // namespace miopen
