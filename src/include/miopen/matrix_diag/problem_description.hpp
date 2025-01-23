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
#include <miopen/kernel_info.hpp>
#include <miopen/kernel_build_params.hpp>

namespace miopen {

struct NetworkConfig;

namespace solver::matrix_diag {
KernelInfo make_hip_kernel(std::vector<size_t> localsize,
                           std::vector<size_t> gridsize,
                           std::string kernel_file,
                           std::string kernel_name,
                           KernelBuildParameters build_params);
} // namespace solver::matrix_diag

namespace matrix_diag {

bool IsValidMatrixDiag(const TensorDescriptor& padDesc,
                       const TensorDescriptor& diagDesc,
                       const TensorDescriptor& outDesc,
                       int64_t diagOffset0,
                       int64_t diagOffset1,
                       const std::string& refer_name,
                       const std::string& pad_alias_name,
                       const std::string& diag_alias_name,
                       const std::string& out_alias_name,
                       bool hasPad,
                       bool hasDiag);

struct MatrixDiagProblemDescriptionBase : ProblemDescriptionBase
{
    MatrixDiagProblemDescriptionBase(const int64_t diagOffset0_,
                                     const int64_t diagOffset1_,
                                     const miopenMatrixDiagAlignMode_t align_)
        : diagOffset0(diagOffset0_), diagOffset1(diagOffset1_), align(align_)
    {
    }

    miopenMatrixDiagAlignMode_t GetAlign() const { return align; }

    bool IsOneDiagonal() const
    {
        if(diagOffset0 != diagOffset1)
            return false;
        return true;
    }

protected:
    int64_t diagOffset0;
    int64_t diagOffset1;
    const miopenMatrixDiagAlignMode_t align;
};

struct MatrixSetDiagForwardProblemDescription : MatrixDiagProblemDescriptionBase
{
    MatrixSetDiagForwardProblemDescription(const TensorDescriptor& inputDesc_,
                                           const TensorDescriptor& diagDesc_,
                                           const TensorDescriptor& outputDesc_,
                                           const int64_t diagOffset0_,
                                           const int64_t diagOffset1_,
                                           const miopenMatrixDiagAlignMode_t align_,
                                           const std::string refer_name_,
                                           const std::string input_alias_name_,
                                           const std::string diag_alias_name_,
                                           const std::string output_alias_name_,
                                           const bool hasInput)
        : MatrixDiagProblemDescriptionBase(diagOffset0_, diagOffset1_, align_),
          inputDesc(inputDesc_),
          diagDesc(diagDesc_),
          outputDesc(outputDesc_),
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
                          output_alias_name,
                          hasInput,
                          true);
    }

    const TensorDescriptor& GetInputDesc() const { return inputDesc; }
    const TensorDescriptor& GetDiagDesc() const { return diagDesc; }
    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsAllContiguous() const
    {
        if(!inputDesc.IsContiguous())
            return false;
        if(!diagDesc.IsContiguous())
            return false;
        if(!outputDesc.IsContiguous())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor inputDesc;
    const TensorDescriptor diagDesc;
    const TensorDescriptor outputDesc;

    const std::string refer_name;
    const std::string input_alias_name;
    const std::string diag_alias_name;
    const std::string output_alias_name;
};

struct MatrixSetDiagBackwardProblemDescription : MatrixDiagProblemDescriptionBase
{
    MatrixSetDiagBackwardProblemDescription(const TensorDescriptor& outputGradDesc_,
                                            const TensorDescriptor& inputGradDesc_,
                                            const TensorDescriptor& diagGradDesc_,
                                            const int64_t diagOffset0_,
                                            const int64_t diagOffset1_,
                                            const miopenMatrixDiagAlignMode_t align_)
        : MatrixDiagProblemDescriptionBase(diagOffset0_, diagOffset1_, align_),
          outputGradDesc(outputGradDesc_),
          inputGradDesc(inputGradDesc_),
          diagGradDesc(diagGradDesc_)
    {
        IsValidMatrixDiag(outputGradDesc,
                          outputGradDesc,
                          inputGradDesc,
                          diagOffset0,
                          diagOffset1,
                          "MatrixSetDiagBackward",
                          "Output gradient",
                          "",
                          "Input gradient",
                          true,
                          false);
        IsValidMatrixDiag(outputGradDesc,
                          diagGradDesc,
                          outputGradDesc,
                          diagOffset0,
                          diagOffset1,
                          "MatrixSetDiagBackward",
                          "",
                          "Diagonal gradient",
                          "Output gradient",
                          false,
                          true);
    }

    const TensorDescriptor& GetOutputGradDesc() const { return outputGradDesc; }
    const TensorDescriptor& GetInputGradDesc() const { return inputGradDesc; }
    const TensorDescriptor& GetDiagGradDesc() const { return diagGradDesc; }

    bool IsAllContiguous() const
    {
        if(!outputGradDesc.IsContiguous())
            return false;
        if(!inputGradDesc.IsContiguous())
            return false;
        if(!diagGradDesc.IsContiguous())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor outputGradDesc;
    const TensorDescriptor inputGradDesc;
    const TensorDescriptor diagGradDesc;
};

struct MatrixDiagPartForwardProblemDescription : MatrixDiagProblemDescriptionBase
{
    MatrixDiagPartForwardProblemDescription(const TensorDescriptor& inputDesc_,
                                            const TensorDescriptor& padDesc_,
                                            const TensorDescriptor& outputDesc_,
                                            const int64_t diagOffset0_,
                                            const int64_t diagOffset1_,
                                            const miopenMatrixDiagAlignMode_t align_,
                                            const std::string refer_name_,
                                            const std::string input_alias_name_,
                                            const std::string pad_alias_name_,
                                            const std::string output_alias_name_,
                                            const bool hasPad)
        : MatrixDiagProblemDescriptionBase(diagOffset0_, diagOffset1_, align_),
          inputDesc(inputDesc_),
          padDesc(padDesc_),
          outputDesc(outputDesc_),
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
                          input_alias_name,
                          hasPad,
                          true);
    }

    const TensorDescriptor& GetOutputDesc() const { return outputDesc; }

    bool IsAllContiguous() const
    {
        if(!inputDesc.IsContiguous())
            return false;
        if(!padDesc.IsContiguous())
            return false;
        if(!outputDesc.IsContiguous())
            return false;
        return true;
    }

    NetworkConfig MakeNetworkConfig() const override;

private:
    const TensorDescriptor inputDesc;
    const TensorDescriptor padDesc;
    const TensorDescriptor outputDesc;

    const std::string refer_name;
    const std::string input_alias_name;
    const std::string pad_alias_name;
    const std::string output_alias_name;
};

} // namespace matrix_diag

} // namespace miopen
