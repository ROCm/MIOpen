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
class MatrixDiagPartDriver : public Driver
{
public:
    MatrixDiagPartDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputDesc);
        miopenCreateTensorDescriptor(&padDesc);
        miopenCreateTensorDescriptor(&outputDesc);
        miopenCreateTensorDescriptor(&outputGradDesc);
        miopenCreateTensorDescriptor(&inputGradDesc);

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
    ~MatrixDiagPartDriver() override
    {
        miopenDestroyTensorDescriptor(inputDesc);
        miopenDestroyTensorDescriptor(padDesc);
        miopenDestroyTensorDescriptor(outputDesc);
        miopenDestroyTensorDescriptor(outputGradDesc);
        miopenDestroyTensorDescriptor(inputGradDesc);
    }

private:
    InputFlags inflags;

    int forw;

    int64_t k0, k1;
    miopenMatrixDiagAlignMode_t align;

    miopenTensorDescriptor_t inputDesc;
    miopenTensorDescriptor_t padDesc;
    miopenTensorDescriptor_t outputDesc;
    miopenTensorDescriptor_t outputGradDesc;
    miopenTensorDescriptor_t inputGradDesc;

    std::unique_ptr<GPUMem> input_dev;
    std::unique_ptr<GPUMem> pad_dev;
    std::unique_ptr<GPUMem> output_dev;
    std::unique_ptr<GPUMem> output_grad_dev;
    std::unique_ptr<GPUMem> input_grad_dev;

    std::vector<T> input;
    std::vector<T> pad;
    std::vector<T> output;
    std::vector<T> output_grad;
    std::vector<T> input_grad;

    std::vector<T> output_host;
    std::vector<T> input_grad_host;
};

template <typename T>
int MatrixDiagPartDriver<T>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    auto input_length = inflags.GetValueVectorUint64("input-shape");
    if(input_length.size() < 2)
    {
        std::cout << "Input tensor must have at least 2 dimensions";
        return miopenStatusBadParm;
    }

    auto pad_length = inflags.GetValueVectorUint64("padding-shape");
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
int MatrixDiagPartDriver<T>::GetandSetData()
{
    auto input_length = inflags.GetValueVectorUint64("input-shape");
    auto pad_length   = inflags.GetValueVectorUint64("padding-shape");
    k0                = inflags.GetValueInt("k0");
    k1                = inflags.GetValueInt("k1");
    auto M            = input_length[input_length.size() - 2];
    auto N            = input_length[input_length.size() - 1];
    auto max_diag_len = std::min(M + std::min(k1, 0L), N + std::min(-k0, 0L));
    auto num_diags    = k1 - k0 + 1;
    auto out_length   = input_length;
    if(k0 == k1)
    {
        out_length.pop_back();
        out_length.back() = max_diag_len;
    }
    else
    {
        out_length[out_length.size() - 2] = num_diags;
        out_length[out_length.size() - 1] = max_diag_len;
    }

    if(SetTensorNd(inputDesc, input_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input tensor: " + inflags.GetValueStr("input-shape") + ".");
    if(SetTensorNd(padDesc, pad_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing padding tensor: " + inflags.GetValueStr("padding-shape") + ".");
    if(SetTensorNd(outputDesc, out_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output tensor");

    if(SetTensorNd(outputGradDesc, out_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing output gradient tensor");
    if(SetTensorNd(inputGradDesc, input_length, data_type) != miopenStatusSuccess)
        MIOPEN_THROW("Error parsing input gradient tensor");

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
int MatrixDiagPartDriver<T>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward MatrixDiagPart (Default=1)", "int");
    inflags.AddInputFlag(
        "input-shape", 'I', "2,3,4", "Shape of input tensor (Default=2,3,4)", "vector<uint>");
    inflags.AddInputFlag(
        "padding-shape", 'P', "1", "Shape of padding tensor (Default=1)", "vector<uint>");
    inflags.AddInputFlag(
        "k0",
        'k',
        "0",
        "Diagonal first offset. Positive value means superinput, 0 refers to the main "
        "input, and negative value means subinputs. k can be a single integer (for a "
        "single "
        "input) or a pair of integers specifying the low and high ends of a matrix band. The "
        "first input offset must not be larger than the second. (Default=0)",
        "int");
    inflags.AddInputFlag(
        "k1",
        'K',
        "0",
        "Diagonal second offset. Positive value means superinput, 0 refers to the main "
        "input, and negative value means subinputs. k can be a single integer (for a "
        "single "
        "input) or a pair of integers specifying the low and high ends of a matrix band. The "
        "second input offset must not be smaller than the first. (Default=0)",
        "int");
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
int MatrixDiagPartDriver<T>::AllocateBuffersAndCopy()
{
    size_t input_sz = GetTensorSize(inputDesc);
    size_t pad_sz   = GetTensorSize(padDesc);
    size_t out_sz   = GetTensorSize(outputDesc);

    uint32_t ctx = 0;

    input_dev       = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(T)));
    pad_dev         = std::unique_ptr<GPUMem>(new GPUMem(ctx, pad_sz, sizeof(T)));
    output_dev      = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(T)));
    output_grad_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(T)));
    input_grad_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, input_sz, sizeof(T)));

    input       = std::vector<T>(input_sz);
    pad         = std::vector<T>(pad_sz, static_cast<T>(0));
    output      = std::vector<T>(out_sz, std::numeric_limits<T>::quiet_NaN());
    output_grad = std::vector<T>(out_sz, static_cast<T>(1));
    input_grad  = std::vector<T>(input_sz, std::numeric_limits<T>::quiet_NaN());

    output_host     = std::vector<T>(out_sz, std::numeric_limits<T>::quiet_NaN());
    input_grad_host = std::vector<T>(input_sz, std::numeric_limits<T>::quiet_NaN());

    for(int i = 0; i < input_sz; i++)
        input[i] = prng::gen_A_to_B<T>(static_cast<T>(-1e-5), static_cast<T>(1e-6));

    if(input_dev->ToGPU(GetStream(), input.data()) != 0)
    {
        std::cerr << "Error copying (input) to GPU, size: " << input_dev->GetSize() << std::endl;
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

    if(input_grad_dev->ToGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying (input_grad) to GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusAllocFailed;
    }

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagPartDriver<T>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenMatrixDiagPartForward(GetHandle(),
                                    inputDesc,
                                    input_dev->GetMem(),
                                    padDesc,
                                    pad_dev->GetMem(),
                                    outputDesc,
                                    output_dev->GetMem(),
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
            std::cout << "Wall-clock Time Forward MatrixDiagPart Elapsed: " << t.gettime_ms() / iter
                      << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Forward MatrixDiagPart Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
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
int MatrixDiagPartDriver<T>::RunForwardCPU()
{
    return mloMatrixDiagPartForwardRunHost(inputDesc,
                                           padDesc,
                                           outputDesc,
                                           input.data(),
                                           pad.data(),
                                           output_host.data(),
                                           k0,
                                           k1,
                                           align);
}

template <typename T>
int MatrixDiagPartDriver<T>::RunBackwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenMatrixDiagPartBackward(GetHandle(),
                                     outputGradDesc,
                                     output_grad_dev->GetMem(),
                                     inputGradDesc,
                                     input_grad_dev->GetMem(),
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
            std::cout << "Wall-clock Time Backward MatrixDiagPart Elapsed: "
                      << t.gettime_ms() / iter << " ms" << std::endl;

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        std::cout << "GPU Kernel Time Backward MatrixDiagPart Elapsed: " << kernel_average_time
                  << " ms" << std::endl;
    }

    if(input_grad_dev->FromGPU(GetStream(), input_grad.data()) != 0)
    {
        std::cerr << "Error copying (input_grad_dev) from GPU, size: " << input_grad_dev->GetSize()
                  << std::endl;
        return miopenStatusInternalError;
    }

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagPartDriver<T>::RunBackwardCPU()
{
    return mloMatrixSetDiagForwardRunHost<T>(nullptr,
                                             outputGradDesc,
                                             inputGradDesc,
                                             nullptr,
                                             output_grad.data(),
                                             input_grad_host.data(),
                                             k0,
                                             k1,
                                             true,
                                             align);
}

template <typename T>
int MatrixDiagPartDriver<T>::VerifyForward()
{
    RunForwardCPU();

    auto error_output = miopen::rms_range(output_host, output);

    if(error_output != 0)
    {
        std::cout << "Forward MatrixDiagPart Output FAILED: " << error_output << " != " << 0
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward MatrixDiagPart Output Verifies OK on CPU reference (" << error_output
                  << " = " << 0 << ')' << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename T>
int MatrixDiagPartDriver<T>::VerifyBackward()
{
    RunBackwardCPU();

    auto error_input_grad = miopen::rms_range(input_grad_host, input_grad);

    if(error_input_grad != 0)
    {
        std::cout << "Backward MatrixDiagPart Diagonal Gradient FAILED: " << error_input_grad
                  << " != " << 0 << std::endl;
        return EC_VerifyBwd;
    }
    else
    {
        std::cout << "Backward MatrixDiagPart Diagonal Gradient Verifies OK on CPU reference ("
                  << error_input_grad << " = " << 0 << ')' << std::endl;
    }

    return miopenStatusSuccess;
}
