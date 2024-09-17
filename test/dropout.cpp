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

#include "driver.hpp"
#include "dropout_util.hpp"
#include "get_handle.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"
#include "random.hpp"

#define DROPOUT_DEBUG_CTEST 0
// Workaround for issue #1128
#define DROPOUT_SINGLE_CTEST 1

template <class T>
struct verify_forward_dropout
{
    tensor<T> input;
    tensor<T> output;
    std::vector<unsigned char> rsvsp;
    miopen::DropoutDescriptor DropoutDesc;
    miopen::TensorDescriptor noise_shape;
    size_t in_offset;
    size_t out_offset;
    size_t rsvsp_offset;
    bool use_rsvsp;
    typename std::vector<unsigned char>::iterator rsvsp_ptr;

    verify_forward_dropout(const miopen::DropoutDescriptor& pDropoutDesc,
                           const miopen::TensorDescriptor& pNoiseShape,
                           const tensor<T>& pinput,
                           const tensor<T>& poutput,
                           std::vector<unsigned char>& prsvsp,
                           size_t pin_offset,
                           size_t pout_offset,
                           size_t prsvsp_offset,
                           bool puse_rsvsp = true)
    {
        DropoutDesc  = pDropoutDesc;
        noise_shape  = pNoiseShape;
        input        = pinput;
        output       = poutput;
        rsvsp        = prsvsp;
        in_offset    = pin_offset;
        out_offset   = pout_offset;
        rsvsp_offset = prsvsp_offset;
        use_rsvsp    = puse_rsvsp;
        rsvsp_ptr    = prsvsp.begin();
    }

    tensor<T> cpu() const
    {
        size_t states_size = DropoutDesc.stateSizeInBytes / sizeof(rocrand_state_xorwow);
        auto states_cpu    = std::vector<rocrand_state_xorwow>(states_size);
        InitKernelStateEmulator(states_cpu, DropoutDesc);

        auto out_cpu   = output;
        auto rsvsp_cpu = rsvsp;

        DropoutForwardVerify<T>(get_handle(),
                                DropoutDesc,
                                input.desc,
                                input.data,
                                out_cpu.desc,
                                out_cpu.data,
                                rsvsp_cpu,
                                states_cpu,
                                in_offset,
                                out_offset,
                                rsvsp_offset);

        return out_cpu;
    }

    tensor<T> gpu() const
    {
        auto&& handle  = get_handle();
        auto out_gpu   = output;
        auto rsvsp_dev = handle.Write(rsvsp);
        auto in_dev    = handle.Write(input.data);
        auto out_dev   = handle.Write(output.data);

        DropoutDesc.DropoutForward(handle,
                                   input.desc,
                                   input.desc,
                                   in_dev.get(),
                                   output.desc,
                                   out_dev.get(),
                                   use_rsvsp ? rsvsp_dev.get() : nullptr,
                                   rsvsp.size(),
                                   in_offset,
                                   out_offset,
                                   rsvsp_offset);

        out_gpu.data   = handle.Read<T>(out_dev, output.data.size());
        auto rsvsp_gpu = handle.Read<unsigned char>(rsvsp_dev, rsvsp.size());

        std::copy(rsvsp_gpu.begin(), rsvsp_gpu.end(), rsvsp_ptr);
        return out_gpu;
    }

    void fail(int badtensor) const
    {
        std::cout << "Forward Dropout: " << std::endl;
        std::cout << "Input tensor: " << input.desc.ToString() << std::endl;
        switch(badtensor)
        {
        case(0): std::cout << "Output tensor failed verification." << std::endl; break;
        case(1): std::cout << "Reservespace failed verification." << std::endl; break;
        default: break;
        }
    }
};

template <class T>
struct verify_backward_dropout
{
    tensor<T> din;
    tensor<T> dout;
    std::vector<unsigned char> rsvsp;
    miopen::DropoutDescriptor DropoutDesc;

    size_t in_offset;
    size_t out_offset;
    size_t rsvsp_offset;
    bool use_rsvsp;

    verify_backward_dropout(const miopen::DropoutDescriptor& pDropoutDesc,
                            const tensor<T>& pdin,
                            const tensor<T>& pdout,
                            const std::vector<unsigned char>& prsvsp,
                            size_t pin_offset,
                            size_t pout_offset,
                            size_t prsvsp_offset,
                            bool puse_rsvsp = true)
    {
        DropoutDesc  = pDropoutDesc;
        din          = pdin;
        dout         = pdout;
        rsvsp        = prsvsp;
        in_offset    = pin_offset;
        out_offset   = pout_offset;
        rsvsp_offset = prsvsp_offset;
        use_rsvsp    = puse_rsvsp;
    }

    tensor<T> cpu() const
    {
        auto din_cpu   = din;
        auto rsvsp_cpu = rsvsp;

        DropoutBackwardVerify<T>(DropoutDesc,
                                 dout.desc,
                                 dout.data,
                                 din_cpu.desc,
                                 din_cpu.data,
                                 rsvsp_cpu,
                                 in_offset,
                                 out_offset,
                                 rsvsp_offset);

        return din_cpu;
    }

    tensor<T> gpu() const
    {
        auto&& handle = get_handle();
        auto din_gpu  = din;

        auto din_dev   = handle.Write(din.data);
        auto dout_dev  = handle.Write(dout.data);
        auto rsvsp_dev = handle.Write(rsvsp);

        DropoutDesc.DropoutBackward(handle,
                                    din.desc,
                                    dout.desc,
                                    dout_dev.get(),
                                    din.desc,
                                    din_dev.get(),
                                    use_rsvsp ? rsvsp_dev.get() : nullptr,
                                    rsvsp.size(),
                                    in_offset,
                                    out_offset,
                                    rsvsp_offset);

        din_gpu.data = handle.Read<T>(din_dev, din.data.size());
        return din_gpu;
    }

    void fail(int = 0) const
    {
        std::cout << "Backward Dropout: " << std::endl;
        std::cout << "Doutput tensor: " << dout.desc.ToString() << std::endl;
    }
};

template <class T>
struct dropout_driver : test_driver
{
    std::vector<std::vector<int>> input_dims;
    float dropout_rate{};
    unsigned long long seed{};
    bool mask{};
    std::vector<int> in_dim{};
    int rng_mode_cmd = 0;

    dropout_driver()
    {
        input_dims                                              = get_sub_tensor();
        std::set<std::vector<int>> get_inputs_set               = get_inputs(1);
        std::set<std::vector<int>> get_3d_conv_input_shapes_set = get_3d_conv_input_shapes(1);

// Workaround for issue #1128
#if DROPOUT_SINGLE_CTEST
        input_dims.resize(1);
        add(in_dim, "input-dim", generate_data(input_dims));
        add(dropout_rate, "dropout", generate_data({float(0.5)}));
        add(seed, "seed", generate_data({0x0ULL}));
        add(mask, "use-mask", generate_data({false}));
        add(rng_mode_cmd, "rng-mode", generate_data({0}));
#else
#define DROPOUT_LARGE_CTEST 0
#if DROPOUT_LARGE_CTEST
        input_dims.insert(input_dims.end(), get_inputs_set.begin(), get_inputs_set.end());
        input_dims.insert(input_dims.end(),
                          get_3d_conv_input_shapes_set.begin(),
                          get_3d_conv_input_shapes_set.end());
#else
        auto itr = get_inputs_set.begin();
        for(std::size_t i = 0; i < get_inputs_set.size(); itr++, i++)
            if(i % 6 == 0)
                input_dims.push_back(*itr);

        itr = get_3d_conv_input_shapes_set.begin();
        for(std::size_t i = 0; i < get_3d_conv_input_shapes_set.size(); itr++, i++)
            if(i % 3 == 0)
                input_dims.push_back(*itr);
#endif

        add(in_dim, "input-dim", generate_data(input_dims));
        add(dropout_rate, "dropout", generate_data({float(0.0), float(0.5), float(1.0)}));
        add(seed, "seed", generate_data({0x0ULL, 0xFFFFFFFFFFFFFFFFULL}));
        add(mask, "use-mask", generate_data({false, true}));
        add(rng_mode_cmd, "rng-mode", generate_data({0}));
#endif
    }

    void run()
    {
        miopen::DropoutDescriptor DropoutDesc;
        uint64_t max_value       = miopen_type<T>{} == miopenHalf ? 5 : 17;
        auto&& handle            = get_handle();
        auto in                  = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        miopenRNGType_t rng_mode = miopenRNGType_t(rng_mode_cmd);

        size_t stateSizeInBytes = std::min(size_t(MAX_PRNG_STATE), handle.GetImage3dMaxWidth()) *
                                  sizeof(rocrand_state_xorwow);
        size_t reserveSpaceSizeInBytes = in.desc.GetElementSize() * sizeof(bool);
        size_t total_mem =
            2 * (2 * in.desc.GetNumBytes() + reserveSpaceSizeInBytes) + stateSizeInBytes;
        size_t device_mem = handle.GetGlobalMemorySize();
#if !DROPOUT_DEBUG_CTEST
        if(total_mem >= device_mem)
        {
#endif
            show_command();
            std::cout << "Config requires " << total_mem
                      << " Bytes to write all necessary tensors to GPU. GPU has " << device_mem
                      << " Bytes of memory." << std::endl;
#if !DROPOUT_DEBUG_CTEST
        }
#else
        std::cout << "Input tensor requires " << in.desc.GetElementSize() << " Bytes of memory."
                  << std::endl;
        std::cout << "Output tensor requires " << in.desc.GetElementSize() << " Bytes of memory."
                  << std::endl;
        std::cout << "reserveSpace requires " << reserveSpaceSizeInBytes << " Bytes of memory."
                  << std::endl;
        std::cout << "PRNG state space requires " << stateSizeInBytes << " Bytes of memory."
                  << std::endl;
#endif
        if(total_mem >= device_mem)
        {
            return;
        }

        auto reserveSpace = std::vector<unsigned char>(in.desc.GetElementSize());
        if(mask)
        {
            for(size_t i = 0; i < in.desc.GetElementSize(); i++)
            {
                reserveSpace[i] =
                    static_cast<unsigned char>(prng::gen_canonical<float>() > dropout_rate);
            }
        }

        DropoutDesc.dropout          = dropout_rate;
        DropoutDesc.stateSizeInBytes = stateSizeInBytes;
        DropoutDesc.seed             = seed;
        DropoutDesc.use_mask         = mask;
        DropoutDesc.rng_mode         = rng_mode;

        auto state_buf      = handle.Create<unsigned char>(stateSizeInBytes);
        DropoutDesc.pstates = state_buf.get();
        DropoutDesc.InitPRNGState(
            handle, DropoutDesc.pstates, DropoutDesc.stateSizeInBytes, DropoutDesc.seed);
#if DROPOUT_DEBUG_CTEST
        std::cout <<
#if MIOPEN_BACKEND_OPENCL
            "Use OpenCL backend."
#elif MIOPEN_BACKEND_HIP
            "Use HIP backend."
#endif
                  << std::endl;
#endif

        auto out = tensor<T>{in_dim};
        verify(verify_forward_dropout<T>{DropoutDesc, in.desc, in, out, reserveSpace, 0, 0, 0});

        auto dout = tensor<T>{in_dim}.generate(tensor_elem_gen_integer{max_value});
        auto din  = tensor<T>{in_dim};
        verify(verify_backward_dropout<T>{DropoutDesc, din, dout, reserveSpace, 0, 0, 0});
        if(!mask)
        {
            verify(verify_forward_dropout<T>{
                DropoutDesc, in.desc, in, out, reserveSpace, 0, 0, 0, false});
            verify(
                verify_backward_dropout<T>{DropoutDesc, din, dout, reserveSpace, 0, 0, 0, false});
        }
    }
};

int main(int argc, const char* argv[]) { test_drive<dropout_driver>(argc, argv); }
