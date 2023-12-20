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
#include <miopen/matrixOps.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor.hpp>

#include <array>
#include <initializer_list>

extern "C" miopenStatus_t miopenInitGemmDescriptor(miopenGemmDescriptor_t* gemmDesc,
                                                   long long int m_,
                                                   long long int n_,
                                                   long long int k_,
                                                   long long int strideA_,
                                                   long long int strideB_,
                                                   long long int strideC_,
                                                   bool isColMajor_,
                                                   bool transA_,
                                                   bool transB_)
{
    MIOPEN_LOG_FUNCTION(m_, n_, k_, strideA_, strideB_, strideC_, isColMajor_, transA_, transB_);
    return miopen::try_([&] {
        miopen::deref(gemmDesc) = new miopen::matrixOps::GemmDescriptor{
            m_, n_, k_, strideA_, strideB_, strideC_, isColMajor_, transA_, transB_};
    });
}

extern "C" miopenStatus_t miopenDestroyGemmDescriptor(miopenGemmDescriptor_t gemmDesc)
{

    MIOPEN_LOG_FUNCTION(gemmDesc);
    return miopen::try_([&] { miopen_destroy_object(gemmDesc); });
}

extern "C" miopenStatus_t
miopenInitMatrixAdditionDescriptor(miopenMatrixAdditionDescriptor_t* MatrixAdditionDesc,
                                   long long int m_,
                                   long long int n_,
                                   long long int k_,
                                   long long int strideC_,
                                   long long int strideD_,
                                   long long int strideE_,
                                   bool isColMajor_,
                                   bool transA_,
                                   bool transB_)
{
    MIOPEN_LOG_FUNCTION(m_, n_, k_, strideC_, strideD_, strideE_, isColMajor_, transA_, transB_);
    return miopen::try_([&] {
        miopen::deref(MatrixAdditionDesc) = new miopen::matrixOps::MatrixAdditionDescriptor{
            m_, n_, k_, strideC_, strideD_, strideE_, isColMajor_, transA_, transB_};
    });
}

extern "C" miopenStatus_t
miopenDestroyMatrixAdditionDescriptor(miopenMatrixAdditionDescriptor_t MatrixAdditionDesc)
{

    MIOPEN_LOG_FUNCTION(MatrixAdditionDesc);
    return miopen::try_([&] { miopen_destroy_object(MatrixAdditionDesc); });
}
