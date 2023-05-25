/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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

#include <miopen/handle.hpp>
#include <miopen/miopen.h>
#include <miopen/util_sol.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/batched_transpose_sol.hpp>
#include <miopen/invoker.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/env.hpp>
#include <boost/optional.hpp>
#include <random>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <tuple> // std::ignore
#include "test.hpp"

#include "half.hpp"
using half_float::half;
#include <miopen/hip_float8.h>
using float8  = miopen_f8::hip_f8<miopen_f8::hip_f8_type::fp8>;
using bfloat8 = miopen_f8::hip_f8<miopen_f8::hip_f8_type::bf8>;

#include "driver.hpp"
#include "random.hpp"

template <typename IntType>
uint32_t psudo_rng(const IntType& x, const uint32_t& idx, const uint32_t& seed)
{
    // https://github.com/ROCmSoftwarePlatform/tensorflow-upstream/blob/fp8_quant/tensorflow/stream_executor/rocm/rocm_helpers.cu.cc#L39
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    if(sizeof(IntType) == 4)
        drop_bits ^= x >> 16;
    drop_bits = ((drop_bits & 31) << 11) | (drop_bits >> 5);
    drop_bits *= 0x7000149;
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (idx * 229791) ^ seed);
    return rng;
}

template <typename T>
void cvt_pixel_with_flag(T& /*src*/,
                         const uint32_t& /*flag*/,
                         const uint32_t& /*stoch*/,
                         const uint32_t& /*idx*/,
                         const uint32_t& /*seed*/)
{
}

template <>
void cvt_pixel_with_flag<uint16_t>(ushort& src,
                                   const uint32_t& flag,
                                   const uint32_t& stoch,
                                   const uint32_t& idx,
                                   const uint32_t& seed)
{
    if(flag == BT_FLAG_CVT_FP16_FP8_FP16)
    {
        half src_fp = *reinterpret_cast<half*>(&src);
        float8 cvt_fp;
        if(stoch)
        {
            cvt_fp = float8(
                src_fp, miopen_f8::hip_f8_rounding_mode::stochastic, psudo_rng(src, idx, seed));
        }
        else
        {
            cvt_fp = float8(src_fp);
        }
        half dst_fp = static_cast<half>(cvt_fp);
        src         = *reinterpret_cast<uint16_t*>(&dst_fp);
    }
    else if(flag == BT_FLAG_CVT_FP16_BF8_FP16)
    {
        half src_fp = *reinterpret_cast<half*>(&src);
        bfloat8 cvt_fp;
        if(stoch)
        {
            cvt_fp = bfloat8(
                src_fp, miopen_f8::hip_f8_rounding_mode::stochastic, psudo_rng(src, idx, seed));
        }
        else
        {
            cvt_fp = bfloat8(src_fp);
        }
        half dst_fp = static_cast<half>(cvt_fp);
        src         = *reinterpret_cast<uint16_t*>(&dst_fp);
    }
    /*
        else if(flag == BT_FLAG_CVT_BF16_FP8_BF16)
        {
            bfloat16 src_fp = *reinterpret_cast<bfloat16*>(&src);
            float8 cvt_fp   = static_cast<float8>(src_fp);
            bfloat16 dst_fp = static_cast<bfloat16>(cvt_fp);
            src             = *reinterpret_cast<uint16_t*>(&dst_fp);
        }
        else if(flag == BT_FLAG_CVT_BF16_BF8_BF16)
        {
            bfloat16 src_fp = *reinterpret_cast<bfloat16*>(&src);
            bfloat8 cvt_fp  = static_cast<bfloat8>(src_fp);
            bfloat16 dst_fp = static_cast<bfloat16>(cvt_fp);
            src             = *reinterpret_cast<uint16_t*>(&dst_fp);
        }
    */
}

template <>
struct miopen_type<uint8_t> : std::integral_constant<miopenDataType_t, miopenInt8>
{
};

template <>
struct miopen_type<uint16_t> : std::integral_constant<miopenDataType_t, miopenHalf>
{
};

template <typename T>
void cpu_nchw2nhwc(T* dst,
                   T* src,
                   uint64_t N,
                   uint64_t C,
                   uint64_t H,
                   uint64_t W,
                   uint32_t flag,
                   uint32_t stoch,
                   uint32_t seed)
{
    for(uint64_t i_n = 0; i_n < N; i_n++)
    {
        for(uint64_t i_h = 0; i_h < H; i_h++)
        {
            for(uint64_t i_w = 0; i_w < W; i_w++)
            {
                for(uint64_t i_c = 0; i_c < C; i_c++)
                {
                    uint64_t idx_nhwc = i_n * H * W * C + i_h * W * C + i_w * C + i_c;
                    uint64_t idx_nchw = i_n * C * H * W + i_c * H * W + i_h * W + i_w;
                    T v               = src[idx_nchw];
                    cvt_pixel_with_flag(v, flag, stoch, static_cast<uint32_t>(idx_nhwc), seed);
                    dst[idx_nhwc] = v;
                }
            }
        }
    }
}

template <typename T>
void cpu_nhwc2nchw(T* dst,
                   T* src,
                   uint64_t N,
                   uint64_t C,
                   uint64_t H,
                   uint64_t W,
                   uint32_t flag,
                   uint32_t stoch,
                   uint32_t seed)
{
    for(uint64_t i_n = 0; i_n < N; i_n++)
    {
        for(uint64_t i_c = 0; i_c < C; i_c++)
        {
            for(uint64_t i_h = 0; i_h < H; i_h++)
            {
                for(uint64_t i_w = 0; i_w < W; i_w++)
                {
                    uint64_t idx_nhwc = i_n * H * W * C + i_h * W * C + i_w * C + i_c;
                    uint64_t idx_nchw = i_n * C * H * W + i_c * H * W + i_h * W + i_w;
                    T v               = src[idx_nhwc];
                    cvt_pixel_with_flag(v, flag, stoch, static_cast<uint32_t>(idx_nchw), seed);
                    dst[idx_nchw] = v;
                }
            }
        }
    }
}

template <typename T, typename TRANSPOSE_SOL>
struct cpu_transpose
{
};

template <typename T>
struct cpu_transpose<T, miopen::TransposeSolutionDefault2Nhwc>
{
    static void run(T* dst,
                    T* src,
                    uint64_t N,
                    uint64_t C,
                    uint64_t H,
                    uint64_t W,
                    uint32_t flag,
                    uint32_t stoch,
                    uint32_t seed)
    {
        cpu_nchw2nhwc<T>(dst, src, N, C, H, W, flag, stoch, seed);
    }
};

template <typename T>
struct cpu_transpose<T, miopen::TransposeSolutionNhwc2Default>
{
    static void run(T* dst,
                    T* src,
                    uint64_t N,
                    uint64_t C,
                    uint64_t H,
                    uint64_t W,
                    uint32_t flag,
                    uint32_t stoch,
                    uint32_t seed)
    {
        cpu_nhwc2nchw<T>(dst, src, N, C, H, W, flag, stoch, seed);
    }
};

template <typename TRANSPOSE_SOL>
struct transpose_str
{
};

template <>
struct transpose_str<miopen::TransposeSolutionDefault2Nhwc>
{
    static std::string get() { return "nchw2nhwc"; }
};

template <>
struct transpose_str<miopen::TransposeSolutionNhwc2Default>
{
    static std::string get() { return "nhwc2nchw"; }
};

enum tensor_layout_t
{
    miopen_tensor_layout_nchw,
    miopen_tensor_layout_ncdhw,
    miopen_tensor_layout_nhwc,
    miopen_tensor_layout_ndhwc,
};

std::string tensor_layout_to_string(tensor_layout_t layout)
{
    std::string layout_string("N/A");
    if(layout == miopen_tensor_layout_nchw)
        layout_string = "NCHW";
    else if(layout == miopen_tensor_layout_ncdhw)
        layout_string = "NCDHW";
    else if(layout == miopen_tensor_layout_nhwc)
        layout_string = "NHWC";
    else if(layout == miopen_tensor_layout_ndhwc)
        layout_string = "NDHWC";
    else
        MIOPEN_THROW("Unsupported tensor layout");
    return layout_string;
}

template <typename T>
struct to_miopen_data_type
{
};

template <>
struct to_miopen_data_type<float>
{
    static miopenDataType_t get() { return miopenFloat; }
};

template <>
struct to_miopen_data_type<uint16_t>
{
    static miopenDataType_t get() { return miopenHalf; } // we actually didn't calculate 16bit float
};

template <>
struct to_miopen_data_type<uint8_t>
{
    static miopenDataType_t get() { return miopenInt8; }
};

#define RAND_INTEGER_MAX 120
#define RAND_INTEGER_MIN -88

static int gen_rand_integer()
{
    static const int inited = []() -> int {
        std::srand(std::time(nullptr));
        return 1;
    }();
    std::ignore = inited;
    return GET_RAND();
}

template <typename T>
void rand_tensor(tensor<T>& t, int max, int min)
{
    // use integer to random.
    for(int i = 0; i < t.data.size(); i++)
        t[i] = static_cast<T>(gen_rand_integer() % (max - min) + min);
}

template <>
void rand_tensor<uint16_t>(tensor<uint16_t>& t, int max, int min)
{
    // use half to random.
    for(int i = 0; i < t.data.size(); i++)
    {
        half v = static_cast<half>(gen_rand_integer() % (max - min) + min);
        t[i]   = *reinterpret_cast<uint16_t*>(&v);
    }
}

template <typename T>
bool compare_equal(T r1, T r2)
{
    return r1 == r2;
}

template <>
bool compare_equal<float>(float r1, float r2)
{
    return miopen::float_equal(r1, r2);
}

template <typename T>
bool verify_tensor(tensor<T>& t_gpu, tensor<T>& t_cpu)
{
    if(t_gpu.data.size() != t_cpu.data.size())
    {
        MIOPEN_LOG_E("size not equal, should not happen");
        return false;
    }
    auto idx          = miopen::mismatch_idx(t_gpu.data, t_cpu.data, compare_equal<T>);
    bool valid_result = idx >= miopen::range_distance(t_cpu);

    if(!valid_result)
    {
        std::cout << "diff at:" << idx << ", gpu:" << t_gpu[idx] << ", cpu:" << t_cpu[idx]
                  << std::endl;
    }
    return valid_result;
}

struct transpose_base
{
    miopenHandle_t handle{};
#if MIOPEN_BACKEND_OPENCL
    cl_command_queue q{};
#endif

    transpose_base()
    {
        miopenCreate(&handle);
#if MIOPEN_BACKEND_OPENCL
        miopenGetStream(handle, &q);
#endif
    }
    ~transpose_base() { miopenDestroy(handle); }

    static std::vector<uint32_t> get_image_size() { return {1, 9, 14}; }

    static std::vector<uint32_t> get_channel_size() { return {3, 8, 14}; }

    static std::vector<uint32_t> get_batch_size() { return {1, 2}; }

    template <typename F>
    void iterate_transpose(F f)
    {
        std::vector<uint32_t> channel_list = get_channel_size();
        std::vector<uint32_t> image_list   = get_image_size();
        std::vector<uint32_t> batch_list   = get_batch_size();
        channel_list.push_back(gen_rand_integer() % 13 + 29);
        image_list.push_back(gen_rand_integer() % 13 + 15);
        batch_list.push_back(gen_rand_integer() % 4 + 3);

        for(uint32_t c : channel_list)
        {
            for(uint32_t h : image_list)
            {
                for(uint32_t w : image_list)
                {
                    for(uint32_t n : batch_list)
                    {
                        f(n, c, h, w);
                    }
                }
            }
        }
    }
};

struct transpose_invoke_param : public miopen::InvokeParams
{
    ConstData_t src = nullptr;
    Data_t dst      = nullptr;
    uint32_t stoch  = 0;
    uint32_t seed   = 0;

    transpose_invoke_param(ConstData_t src_, Data_t dst_, uint32_t stoch_, uint32_t seed_)
        : src(src_), dst(dst_), stoch(stoch_), seed(seed_)
    {
    }
    transpose_invoke_param(
        miopen::InvokeType type_, ConstData_t src_, Data_t dst_, uint32_t stoch_, uint32_t seed_)
        : InvokeParams{type_}, src(src_), dst(dst_), stoch(stoch_), seed(seed_)
    {
    }
};

template <typename T, typename TRANSPOSE_SOL, uint32_t FLAG = 0, uint32_t STOCH = 0>
struct transpose_test : transpose_base
{
    void run()
    {
        auto run_transpose = [this](uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
            int tensor_sz = n * c * h * w;
            std::vector<int> tensor_len({static_cast<int>(n),
                                         static_cast<int>(c),
                                         static_cast<int>(h),
                                         static_cast<int>(w)});

            std::vector<int> tensor_strides;

            std::string layout_default = miopen::tensor_layout_get_default(4);
            std::string layout_string  = tensor_layout_to_string(miopen_tensor_layout_nchw);

            miopen::tensor_layout_to_strides(
                tensor_len, layout_default, layout_string, tensor_strides);

            tensor<T> t_src(tensor_len, tensor_strides);
            tensor<T> t_dst(tensor_len, tensor_strides);
            tensor<T> t_dst_gpu(tensor_len, tensor_strides);
            unsigned long int rand_int_max = miopen::EnvvarValue("RAND_MAX", RAND_INTEGER_MAX);
            unsigned long int rand_int_min = miopen::EnvvarValue("RAND_MIN", RAND_INTEGER_MIN);
            rand_tensor(t_src, static_cast<int>(rand_int_max), static_cast<int>(rand_int_min));
#if MIOPEN_BACKEND_OPENCL
            cl_context cl_ctx;
            clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &cl_ctx, nullptr);
            cl_int status = CL_SUCCESS;
            cl_mem src_dev =
                clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(T) * tensor_sz, nullptr, &status);
            cl_mem dst_dev =
                clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, sizeof(T) * tensor_sz, nullptr, nullptr);
            status |= clEnqueueWriteBuffer(q,
                                           src_dev,
                                           CL_TRUE,
                                           0,
                                           sizeof(T) * tensor_sz,
                                           t_src.data.data(),
                                           0,
                                           nullptr,
                                           nullptr);
            EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
            void* src_dev;
            void* dst_dev;
            EXPECT(hipMalloc(&src_dev, sizeof(T) * tensor_sz) == hipSuccess);
            EXPECT(hipMalloc(&dst_dev, sizeof(T) * tensor_sz) == hipSuccess);
            EXPECT(hipMemcpy(
                       src_dev, t_src.data.data(), sizeof(T) * tensor_sz, hipMemcpyHostToDevice) ==
                   hipSuccess);
#endif
            uint32_t seed = 0;
            if(STOCH)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
                seed = distribution(gen);
            }
            // run cpu
            cpu_transpose<T, TRANSPOSE_SOL>::run(
                t_dst.data.data(), t_src.data.data(), n, c, h, w, FLAG, STOCH, seed);

            const auto invoke_param = transpose_invoke_param{
                DataCast(static_cast<const void*>(src_dev)), DataCast(dst_dev), STOCH, seed};

            miopen::ExecutionContext ctx;
            ctx.SetStream(&miopen::deref(this->handle));
            ctx.DetectRocm();
            // ctx.SetupFloats();

            TRANSPOSE_SOL transpose_sol(
                ctx, to_miopen_data_type<T>::get(), n, c, h, w, FLAG, STOCH, seed);

            std::vector<miopen::BatchedTransposeParam> all_params =
                transpose_sol.GetApplicableParams();
            int cnt_param   = 0;
            int total_param = all_params.size();
            for(auto& param : all_params)
            {
                transpose_sol.kernel_param_heuristic = param;
                std::vector<OpKernelArg> opArgs      = transpose_sol.GetKernelArg();

                boost::optional<miopen::InvokerFactory> invoker_factory(
                    [=](const std::vector<miopen::Kernel>& kernels) mutable {
                        return [=](const miopen::Handle& handle,
                                   const miopen::AnyInvokeParams& primitive_param) mutable {
                            decltype(auto) invoke_params =
                                primitive_param.CastTo<transpose_invoke_param>();

                            const auto k = handle.Run(kernels[0]);

                            opArgs[0]  = OpKernelArg(invoke_params.dst);
                            opArgs[1]  = OpKernelArg(invoke_params.src);
                            opArgs[11] = OpKernelArg(invoke_params.stoch);
                            opArgs[12] = OpKernelArg(invoke_params.seed);

                            k(opArgs);
                        };
                    });

                std::vector<miopen::solver::KernelInfo> construction_params{
                    transpose_sol.GetKernelInfo()};

                const auto invoker = miopen::deref(this->handle)
                                         .PrepareInvoker(*invoker_factory, construction_params);

                // run gpu
                invoker(miopen::deref(this->handle), invoke_param);

#if MIOPEN_BACKEND_OPENCL
                status = clEnqueueReadBuffer(q,
                                             dst_dev,
                                             CL_TRUE,
                                             0,
                                             sizeof(T) * tensor_sz,
                                             t_dst_gpu.data.data(),
                                             0,
                                             nullptr,
                                             nullptr);
                EXPECT(status == CL_SUCCESS);
#elif MIOPEN_BACKEND_HIP
                EXPECT(hipMemcpy(t_dst_gpu.data.data(),
                                 dst_dev,
                                 sizeof(T) * tensor_sz,
                                 hipMemcpyDeviceToHost) == hipSuccess);
#endif

                // we expect excact match, since use integer
                bool valid_result = verify_tensor(t_dst_gpu, t_dst);

                std::cout << "[" << transpose_str<TRANSPOSE_SOL>::get() << ", b" << (sizeof(T) * 8)
                          << " ] "
                          << "n:" << n << ", c:" << c << ", h:" << h << ", w:" << w
                          << ", flag:" << FLAG << ", stoch:" << STOCH << ", valid:" << valid_result
                          << ", kernel:" << transpose_sol.GetKernelName() << " [" << cnt_param
                          << "/" << total_param << "]" << std::endl;

                EXPECT(valid_result == true);
                cnt_param++;
            }
#if MIOPEN_BACKEND_OPENCL
            clReleaseMemObject(src_dev);
            clReleaseMemObject(dst_dev);
#elif MIOPEN_BACKEND_HIP
            hipFree(src_dev);
            hipFree(dst_dev);
#endif
        };

        iterate_transpose(run_transpose);
    }
};

int main()
{
    run_test<transpose_test<float, miopen::TransposeSolutionDefault2Nhwc>>();
    run_test<transpose_test<uint16_t, miopen::TransposeSolutionDefault2Nhwc>>();
    run_test<transpose_test<uint16_t,
                            miopen::TransposeSolutionDefault2Nhwc,
                            BT_FLAG_CVT_FP16_FP8_FP16,
                            0>>();
    run_test<transpose_test<uint16_t,
                            miopen::TransposeSolutionDefault2Nhwc,
                            BT_FLAG_CVT_FP16_FP8_FP16,
                            1>>();
    run_test<transpose_test<uint16_t,
                            miopen::TransposeSolutionDefault2Nhwc,
                            BT_FLAG_CVT_FP16_BF8_FP16,
                            0>>();
    run_test<transpose_test<uint16_t,
                            miopen::TransposeSolutionDefault2Nhwc,
                            BT_FLAG_CVT_FP16_BF8_FP16,
                            1>>();
    run_test<transpose_test<uint8_t, miopen::TransposeSolutionDefault2Nhwc>>();

    run_test<transpose_test<float, miopen::TransposeSolutionNhwc2Default>>();
    run_test<transpose_test<uint16_t, miopen::TransposeSolutionNhwc2Default>>();
    run_test<transpose_test<uint16_t,
                            miopen::TransposeSolutionNhwc2Default,
                            BT_FLAG_CVT_FP16_FP8_FP16,
                            0>>();
    run_test<transpose_test<uint16_t,
                            miopen::TransposeSolutionNhwc2Default,
                            BT_FLAG_CVT_FP16_FP8_FP16,
                            1>>();
    run_test<transpose_test<uint16_t,
                            miopen::TransposeSolutionNhwc2Default,
                            BT_FLAG_CVT_FP16_BF8_FP16,
                            0>>();
    run_test<transpose_test<uint16_t,
                            miopen::TransposeSolutionNhwc2Default,
                            BT_FLAG_CVT_FP16_BF8_FP16,
                            1>>();
    run_test<transpose_test<uint8_t, miopen::TransposeSolutionNhwc2Default>>();
}
