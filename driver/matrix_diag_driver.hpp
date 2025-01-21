/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACTORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "driver.hpp"
#include "mloMatrixDiagHost.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"

#include <../test/ford.hpp>
#include <../test/verify.hpp>

#include <cstdint>
#include <miopen/miopen.h>

template <typename T>
class MatrixDiagDriver : public Driver
{
public:
    MatrixDiagDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&diagDesc);
        miopenCreateTensorDescriptor(&padDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&diagGradDesc);

        data_type = miopen_type<T>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<int> GetTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    T GetTolerance();
    int VerifyBackward() override;
    int VerifyForward() override;
    ~MatrixDiagDriver() override
    {
        miopenDestroyTensorDescriptor(diagDesc);
        miopenDestroyTensorDescriptor(padDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(diagGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    int64_t k0, k1;
    miopenMatrixDiagAlignMode_t align;

    miopenTensorDescriptor_t diagDesc;
    miopenTensorDescriptor_t padDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t diagGradDesc;

    std::unique_ptr<GPUMem> diag_dev;
    std::unique_ptr<GPUMem> pad_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> diag_grad_dev;

    std::vector<T> diag;
    std::vector<T> pad;
    std::vector<T> output;
    std::vector<T> output_grad;
    std::vector<T> diag_grad;

    std::vector<T> output_host;
    std::vector<T> diag_grad_host;
};

template <typename T>
int MatrixDiagDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    auto diagTensorParam = inflags.GetValueTensor("diagonal-shape");
    auto diag_length     = diagTensorParam.lengths;
    if(diag_length.empty())
    {
        std::cout << "Diagonal tensor must not be empty";
        return miopenStatusBadParm;
    }

    auto padTensorParam = inflags.GetValueTensor("padding-shape");
    auto pad_length     = padTensorParam.lengths;
    if(pad_length.empty())
    {
        std::cout << "Padding tensor must not be empty";
        return miopenStatusBadParm;
    }

    auto align_str = inflags.GetValueStr("align");
    if(align_str != "LEFT_RIGHT" && align_str != "RIGHT_LEFT" && align_str != "LEFT_LEFT" &&
       align_str != "RIGHT_RIGHT")
    {
        std::cout << "Invalid align value";
        return miopenStatusBadParm;
    }

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagDriver<T>::GetandSetData()
{
    auto diag_length = inflags.GetValueVectorUint64("diagonal-shape");
    auto pad_length  = inflags.GetValueVectorUint64("padding-shape");
    k0               = inflags.GetValueInt("k0");
    k1               = inflags.GetValueInt("k1");
    auto num_rows    = inflags.GetValueUint64("num-rows");
    auto num_cols    = inflags.GetValueUint64("num-cols");
    auto out_length  = diag_length;
    if(k0 != k1)
        out_length.push_back(0);
    out_length[out_length.size() - 2] = num_rows;
    out_length[out_length.size() - 1] = num_cols;

    if(SetTensorNd(diagDesc, diag_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing diagonal tensor: " + inflags.GetValueStr("diagonal-shape") +
                     ".");
    if(SetTensorNd(padDesc, pad_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing padding tensor: " + inflags.GetValueStr("padding-shape") + ".");
    if(SetTensorNd(outputDesc, out_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor");

    if(SetTensorNd(outputGradDesc, out_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output gradient tensor");
    if(SetTensorNd(diagGradDesc, diag_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing diag gradient tensor");

    auto align_str = inflags.GetValueStr("align");
    if(align_str == "LEFT_RIGHT")
        align = MIOPEN_MATRIX_ALIGN_LEFT_RIGHT;
    else if(align_str == "RIGHT_LEFT")
        align = MIOPEN_MATRIX_ALIGN_RIGHT_LEFT;
    else if(align_str == "LEFT_LEFT")
        align = MIOPEN_MATRIX_ALIGN_LEFT_LEFT;
    else if(align_str == "RIGHT_RIGHT")
        align = MIOPEN_MATRIX_ALIGN_RIGHT_RIGHT;
    else
        MIOPEN_THROW("Error parsing align mode");

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward MatrixDiag (Default=1)", "int");
    inflags.AddInputFlag(
        "diagonal-shape", 'D', "2,4", "Shape of diagonal tensor (Default=2,4)", "vector<uint>");
    inflags.AddInputFlag(
        "padding-shape", 'P', "1", "Shape of padding tensor (Default=1)", "vector<uint>");
    inflags.AddInputFlag(
        "k0",
        'k',
        "0",
        "Diagonal first offset. Positive value means superdiagonal, 0 refers to the main "
        "diagonal, and negative value means subdiagonals. k can be a single integer (for a single "
        "diagonal) or a pair of integers specifying the low and high ends of a matrix band. The "
        "first diagonal offset must not be larger than the second. (Default=0)",
        "int");
    inflags.AddInputFlag(
        "k1",
        'K',
        "0",
        "Diagonal second offset. Positive value means superdiagonal, 0 refers to the main "
        "diagonal, and negative value means subdiagonals. k can be a single integer (for a single "
        "diagonal) or a pair of integers specifying the low and high ends of a matrix band. The "
        "second diagonal offset must not be smaller than the first. (Default=0)",
        "int");
    inflags.AddInputFlag("num-rows",
                         'R',
                         "0",
                         "The number of rows of the output matrix. If it is not provided, the op "
                         "assumes the output matrix is a square matrix and infers the matrix size "
                         "from k and the innermost dimension of diagonal. (Default=0)",
                         "uint");
    inflags.AddInputFlag("num-cols",
                         'C',
                         "0",
                         "The number of columns of the output matrix. If it is not provided, the "
                         "op assumes the output matrix is a square matrix and infers the matrix "
                         "size from k and the innermost dimension of diagonal. (Default=0)",
                         "uint");
    inflags.AddInputFlag(
        "align",
        'a',
        "LEFT_RIGHT",
        "An optional string from: \"LEFT_RIGHT\", \"RIGHT_LEFT\", \"LEFT_LEFT\", \"RIGHT_RIGHT\". "
        "Defaults to \"RIGHT_LEFT\". Some diagonals are shorter than max_diag_len and need to be "
        "padded. align is a string specifying how superdiagonals and subdiagonals should be "
        "aligned, respectively. There are four possible alignments: \"RIGHT_LEFT\" (default), "
        "\"LEFT_RIGHT\", \"LEFT_LEFT\", and \"RIGHT_RIGHT\". \"RIGHT_LEFT\" aligns superdiagonals "
        "to the right (left-pads the row) and subdiagonals to the left (right-pads the row). It is "
        "the packing format LAPACK uses. cuSPARSE uses \"LEFT_RIGHT\", which is the opposite "
        "alignment.",
        "str");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "0", "Verify Each Layer (Default=0)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagDriver<T>::AllocateBuffersAndCopy()
{
    size_t diag_sz = GetTensorSize(diagDesc);
    size_t pad_sz  = GetTensorSize(padDesc);
    size_t out_sz  = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    diag_dev        = std::unique_ptr<GPUMem>(new GPUMem(ctx, diag_sz, sizeof(T)));
    pad_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, pad_sz, sizeof(T)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(T)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(T)));
    diag_grad_dev   = std::unique_ptr<GPUMem>(new GPUMem(ctx, diag_sz, sizeof(T)));

    diag        = std::vector<T>(diag_sz);
    pad         = std::vector<T>(pad_sz);
    output      = std::vector<T>(out_sz, std::numeric_limits<T>::quiet_NaN());
    output_grad = std::vector<T>(out_sz);
    diag_grad   = std::vector<T>(diag_sz, std::numeric_limits<T>::quiet_NaN());

    output_host    = std::vector<T>(out_sz, std::numeric_limits<T>::quiet_NaN());
    diag_grad_host = std::vector<T>(diag_sz, std::numeric_limits<T>::quiet_NaN());

    for(int i = 0; i < diag_sz; i++)
        diag[i] = prng::gen_A_to_B<T>(static_cast<T>(-1e-5), static_cast<T>(1e-6));

    for(int i = 0; i < pad_sz; i++)
        pad[i] = prng::gen_A_to_B<T>(static_cast<T>(-1e-5), static_cast<T>(1e-6));

    for(int i = 0; i < out_sz; i++)
        output_grad[i] = prng::gen_A_to_B<T>(static_cast<T>(-1e-5), static_cast<T>(1e-6));

    if(diag_dev->ToGPU(GetStream(), diag.data()) != 0)
    {
        std::cerr << "Error copying (diag) to GPU, size: " << diag_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(pad_dev->ToGPU(GetStream(), pad.data()) != 0)
    {
        std::cerr << "Error copying (pad) to GPU, size: " << pad_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(output_dev->ToGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output) to GPU, size: " << output_dev->GetSize() << std::endl;
        return miopenStatusAllocFailed;
    }

    if(output_grad_dev->ToGPU(GetStream(), output_grad.data()) != 0)
    {
        std::cerr << "Error copying (output_grad) to GPU, size: " << output_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusAllocFailed;
    }

    if(diag_grad_dev->ToGPU(GetStream(), diag_grad.data()) != 0)
    {
        std::cerr << "Error copying (diag_grad) to GPU, size: " << diag_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusAllocFailed;
    }

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagDriver<T>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenMatrixDiagForward(GetHandle(),
                                diagDesc,
                                diag_dev.get(),
                                padDesc,
                                pad_dev.get(),
                                outputDesc,
                                output_dev.get(),
                                k0,
                                k1,
                                align);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Forward MatrixDiag Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MatrixDiag Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(output_dev->FromGPU(GetStream(), output.data()) != 0)
    {
        std::cerr << "Error copying (output_dev) from GPU, size: " << output_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagDriver<T>::RunForwardCPU()
{
    return mloMatrixSetDiagForwardRunHost(padDesc,
                                          diagDesc,
                                          outputDesc,
                                          pad.data(),
                                          diag.data(),
                                          output_host.data(),
                                          k0,
                                          k1,
                                          true,
                                          align);
}

template <typename T>
int MatrixDiagDriver<T>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenMatrixDiagBackward(GetHandle(),
                                 outputGradDesc,
                                 output_grad_dev.get(),
                                 diagGradDesc,
                                 diag_grad_dev.get(),
                                 k0,
                                 k1,
                                 align);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            std::cout << "Wall-clock Time Backward MatrixDiag Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward MatrixDiag Elapsed: " << kernel_average_time << " ms"
                  << std::endl;
    }

    if(diag_grad_dev->FromGPU(GetStream(), diag_grad.data()) != 0)
    {
        std::cerr << "Error copying (diag_grad_dev) from GPU, size: " << diag_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagDriver<T>::RunBackwardCPU()
{
    return mloMatrixDiagPartForwardRunHost<T>(outputGradDesc,
                                              nullptr,
                                              diagGradDesc,
                                              output_grad.data(),
                                              nullptr,
                                              diag_grad_host.data(),
                                              k0,
                                              k1,
                                              align);
}

template <typename T>
int MatrixDiagDriver<T>::VerifyForward()
{
    RunForwardCPU();

    auto error_output = miopen::rms_range(output_host, output);

    if(error_output != 0)
    {
        std::cout << "Backward MatrixDiag Output FAILED: " << error_output << " != " << 0
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Backward MatrixDiag Output Verifies OK on CPU reference (" << error_output
                  << " = " << 0 << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagDriver<T>::VerifyBackward()
{
    RunBackwardCPU();

    auto error_diag_grad = miopen::rms_range(diag_grad_host, diag_grad);

    if(error_diag_grad != 0)
    {
        std::cout << "Backward MatrixDiag Diagonal Gradient FAILED: " << error_diag_grad
                  << " != " << 0 << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward MatrixDiag Diagonal Gradient Verifies OK on CPU reference ("
                  << error_diag_grad << " = " << 0 << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
