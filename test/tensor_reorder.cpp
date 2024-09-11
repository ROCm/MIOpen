/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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
#include <miopen/tensor_reorder_util.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>
#include <miopen/general_tensor_reorder_sol.hpp>
#include <miopen/invoker.hpp>
#include <miopen/invoke_params.hpp>
#include <boost/optional.hpp>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "test.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "get_handle.hpp"
#include "workspace.hpp"

template <typename T>
void cpu_tensor_reorder(T* dst,
                        T* src,
                        uint64_t dim_0,
                        uint64_t dim_1,
                        uint64_t dim_2,
                        uint64_t dim_3,
                        uint64_t order_0,
                        uint64_t order_1,
                        uint64_t order_2,
                        uint64_t order_3)
{
    const uint64_t src_dim[4] = {dim_0, dim_1, dim_2, dim_3};
    const uint64_t dst_dim[4] = {
        src_dim[order_0], src_dim[order_1], src_dim[order_2], src_dim[order_3]};

    const uint64_t src_stride[4] = {
        src_dim[1] * src_dim[2] * src_dim[3], src_dim[2] * src_dim[3], src_dim[3], 1};
    const uint64_t dst_stride[4] = {
        dst_dim[1] * dst_dim[2] * dst_dim[3], dst_dim[2] * dst_dim[3], dst_dim[3], 1};

    uint64_t itr_src_dim[4] = {0, 0, 0, 0};
    uint64_t itr_dst_dim[4] = {0, 0, 0, 0};

    for(itr_src_dim[0] = 0; itr_src_dim[0] < src_dim[0]; itr_src_dim[0]++)
    {
        for(itr_src_dim[1] = 0; itr_src_dim[1] < src_dim[1]; itr_src_dim[1]++)
        {
            for(itr_src_dim[2] = 0; itr_src_dim[2] < src_dim[2]; itr_src_dim[2]++)
            {
                for(itr_src_dim[3] = 0; itr_src_dim[3] < src_dim[3]; itr_src_dim[3]++)
                {
                    itr_dst_dim[0] = itr_src_dim[order_0];
                    itr_dst_dim[1] = itr_src_dim[order_1];
                    itr_dst_dim[2] = itr_src_dim[order_2];
                    itr_dst_dim[3] = itr_src_dim[order_3];

                    uint64_t idx_src =
                        itr_src_dim[0] * src_stride[0] + itr_src_dim[1] * src_stride[1] +
                        itr_src_dim[2] * src_stride[2] + itr_src_dim[3] * src_stride[3];
                    uint64_t idx_dst =
                        itr_dst_dim[0] * dst_stride[0] + itr_dst_dim[1] * dst_stride[1] +
                        itr_dst_dim[2] * dst_stride[2] + itr_dst_dim[3] * dst_stride[3];

                    dst[idx_dst] = src[idx_src];
                }
            }
        }
    }
}

template <typename T>
struct cpu_reorder
{
    static void run(T* dst,
                    T* src,
                    uint64_t dim_0,
                    uint64_t dim_1,
                    uint64_t dim_2,
                    uint64_t dim_3,
                    uint64_t order_0,
                    uint64_t order_1,
                    uint64_t order_2,
                    uint64_t order_3)
    {
        cpu_tensor_reorder<T>(
            dst, src, dim_0, dim_1, dim_2, dim_3, order_0, order_1, order_2, order_3);
    }
};

struct reorder_str
{
    static std::string get(uint32_t order_0, uint32_t order_1, uint32_t order_2, uint32_t order_3)
    {
        return ("r" + std::to_string(order_0) + std::to_string(order_1) + std::to_string(order_2) +
                std::to_string(order_3));
    }
};

std::string
supported_reorder_to_string(uint32_t order_0, uint32_t order_1, uint32_t order_2, uint32_t order_3)
{
    std::string layout_string("N/A");
    // NOLINTBEGIN(*-braces-around-statements)
    if((order_0 == 0) && (order_1 == 1) && (order_2 == 3) && (order_3 == 2))
        layout_string = "r0132";
    else if((order_0 == 0) && (order_1 == 2) && (order_2 == 1) && (order_3 == 3))
        layout_string = "r0213";
    else if((order_0 == 0) && (order_1 == 2) && (order_2 == 3) && (order_3 == 1))
        layout_string = "r0231";
    else if((order_0 == 0) && (order_1 == 3) && (order_2 == 1) && (order_3 == 2))
        layout_string = "r0312";
    else if((order_0 == 0) && (order_1 == 3) && (order_2 == 2) && (order_3 == 1))
        layout_string = "r0321";
    else if((order_0 == 1) && (order_1 == 0) && (order_2 == 2) && (order_3 == 3))
        layout_string = "r1023";
    else if((order_0 == 1) && (order_1 == 0) && (order_2 == 3) && (order_3 == 2))
        layout_string = "r1032";
    else if((order_0 == 1) && (order_1 == 2) && (order_2 == 0) && (order_3 == 3))
        layout_string = "r1203";
    else if((order_0 == 1) && (order_1 == 2) && (order_2 == 3) && (order_3 == 0))
        layout_string = "r1230";
    else if((order_0 == 1) && (order_1 == 3) && (order_2 == 0) && (order_3 == 2))
        layout_string = "r1302";
    else if((order_0 == 1) && (order_1 == 3) && (order_2 == 2) && (order_3 == 0))
        layout_string = "r1320";
    else if((order_0 == 2) && (order_1 == 0) && (order_2 == 1) && (order_3 == 3))
        layout_string = "r2013";
    else if((order_0 == 2) && (order_1 == 0) && (order_2 == 3) && (order_3 == 1))
        layout_string = "r2031";
    else if((order_0 == 2) && (order_1 == 1) && (order_2 == 0) && (order_3 == 3))
        layout_string = "r2103";
    else if((order_0 == 2) && (order_1 == 1) && (order_2 == 3) && (order_3 == 0))
        layout_string = "r2130";
    else if((order_0 == 2) && (order_1 == 3) && (order_2 == 0) && (order_3 == 1))
        layout_string = "r2301";
    else if((order_0 == 2) && (order_1 == 3) && (order_2 == 1) && (order_3 == 0))
        layout_string = "r2310";
    else if((order_0 == 3) && (order_1 == 0) && (order_2 == 1) && (order_3 == 2))
        layout_string = "r3012";
    else if((order_0 == 3) && (order_1 == 0) && (order_2 == 2) && (order_3 == 1))
        layout_string = "r3021";
    else if((order_0 == 3) && (order_1 == 1) && (order_2 == 0) && (order_3 == 2))
        layout_string = "r3102";
    else if((order_0 == 3) && (order_1 == 1) && (order_2 == 2) && (order_3 == 0))
        layout_string = "r3120";
    else if((order_0 == 3) && (order_1 == 2) && (order_2 == 0) && (order_3 == 1))
        layout_string = "r3201";
    else if((order_0 == 3) && (order_1 == 2) && (order_2 == 1) && (order_3 == 0))
        layout_string = "r3210";
    else
        MIOPEN_THROW("Unsupported reorder layout");
    // NOLINTEND(*-braces-around-statements)
    return layout_string;
}

template <typename T>
struct to_miopen_data_type
{
};

template <>
struct to_miopen_data_type<double>
{
    static miopenDataType_t get() { return miopenDouble; }
};

template <>
struct to_miopen_data_type<float>
{
    static miopenDataType_t get() { return miopenFloat; }
};

template <>
struct to_miopen_data_type<half_float::half>
{
    static miopenDataType_t get() { return miopenHalf; } // we actually didn't calculate 16bit float
};

template <>
struct to_miopen_data_type<int8_t>
{
    static miopenDataType_t get() { return miopenInt8; }
};

template <>
struct to_miopen_data_type<bfloat16>
{
    static miopenDataType_t get() { return miopenBFloat16; }
};

static constexpr int RAND_INTEGER_MAX = 120;
static constexpr int RAND_INTEGER_MIN = -88;

template <typename T>
void rand_tensor_integer(tensor<T>& t, int max = RAND_INTEGER_MAX, int min = RAND_INTEGER_MIN)
{
    // use integer to random.
    for(size_t i = 0; i < t.data.size(); i++)
        t[i] = static_cast<T>(prng::gen_A_to_B(min, max));
}

template <typename T>
bool compare_equal(T r1, T r2)
{
    return r1 == r2;
}

template <>
bool compare_equal<double>(double r1, double r2)
{
    return miopen::float_equal(r1, r2);
}

template <>
bool compare_equal<float>(float r1, float r2)
{
    return miopen::float_equal(r1, r2);
}

template <typename T>
bool verify_tensor(tensor<T>& t_gpu, tensor<T>& t_cpu)
{
    EXPECT(t_gpu.data.size() == t_cpu.data.size());
    auto idx          = miopen::mismatch_idx(t_gpu.data, t_cpu.data, compare_equal<T>);
    bool valid_result = idx >= miopen::range_distance(t_cpu);

    if(!valid_result)
    {
        std::cout << "diff at:" << idx << ", gpu:" << t_gpu[idx] << ", cpu:" << t_cpu[idx]
                  << std::endl;
    }
    return valid_result;
}

struct tensor_reorder_base_driver : test_driver
{

    static std::vector<uint32_t> get_dim_3_size() { return {1, 9}; }
    static std::vector<uint32_t> get_dim_2_size() { return {1, 9}; }
    static std::vector<uint32_t> get_dim_1_size() { return {3, 8}; }
    static std::vector<uint32_t> get_dim_0_size() { return {1, 2}; }

    template <typename F>
    void iterate_reorder(F f)
    {
        std::vector<uint32_t> dim_3_list = get_dim_3_size();
        std::vector<uint32_t> dim_2_list = get_dim_2_size();
        std::vector<uint32_t> dim_1_list = get_dim_1_size();
        std::vector<uint32_t> dim_0_list = get_dim_0_size();

        dim_3_list.push_back(prng::gen_off_range(29, 13));
        dim_2_list.push_back(prng::gen_off_range(29, 13));
        dim_1_list.push_back(prng::gen_off_range(15, 13));
        dim_0_list.push_back(prng::gen_off_range(3, 4));

        constexpr int all_possible_order[23][4] = {
            {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3},
            {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3},
            {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2},
            {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};

        for(auto order : all_possible_order)
        {
            for(uint32_t dim_3 : dim_3_list)
            {
                for(uint32_t dim_2 : dim_2_list)
                {
                    for(uint32_t dim_1 : dim_1_list)
                    {
                        for(uint32_t dim_0 : dim_0_list)
                        {
                            f(dim_0, dim_1, dim_2, dim_3, order[0], order[1], order[2], order[3]);
                        }
                    }
                }
            }
        }
    }
};

struct reorder_invoke_param : public miopen::InvokeParams
{
    ConstData_t src = nullptr;
    Data_t dst      = nullptr;

    reorder_invoke_param(ConstData_t src_, Data_t dst_) : src(src_), dst(dst_) {}
    reorder_invoke_param(miopen::InvokeType type_, ConstData_t src_, Data_t dst_)
        : InvokeParams{type_}, src(src_), dst(dst_)
    {
    }

    Data_t GetWorkspace() const { return nullptr; }
    std::size_t GetWorkspaceSize() const { return 0; }
};
template <typename T>
struct tensor_reorder_driver : tensor_reorder_base_driver
{
    // NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
    void run()
    {
        auto run_reorder = [](uint32_t dim_0,
                              uint32_t dim_1,
                              uint32_t dim_2,
                              uint32_t dim_3,
                              uint32_t order_0,
                              uint32_t order_1,
                              uint32_t order_2,
                              uint32_t order_3) {
            int tensor_sz = dim_0 * dim_1 * dim_2 * dim_3;
            std::vector<int> tensor_len({static_cast<int>(dim_0),
                                         static_cast<int>(dim_1),
                                         static_cast<int>(dim_2),
                                         static_cast<int>(dim_3)});

            std::vector<int> tensor_strides;

            std::string layout_default = miopen::tensor_layout_get_default(4);
            std::string layout_string = miopen::TensorDescriptor::LayoutEnumToStr(miopenTensorNCHW);
            std::string reorder_string =
                supported_reorder_to_string(order_0, order_1, order_2, order_3);

            miopen::tensor_layout_to_strides(
                tensor_len, layout_default, layout_string, tensor_strides);

            tensor<T> t_src(tensor_len, tensor_strides);
            tensor<T> t_dst(tensor_len, tensor_strides);
            tensor<T> t_dst_gpu(tensor_len, tensor_strides);
            rand_tensor_integer(t_src);

            auto& handle = get_handle();
            miopen::ExecutionContext ctx;
            ctx.SetStream(&handle);
            // ctx.SetupFloats();
            auto reorder_sol = MakeTensorReorderAttributes(ctx,
                                                           to_miopen_data_type<T>::get(),
                                                           dim_0,
                                                           dim_1,
                                                           dim_2,
                                                           dim_3,
                                                           order_0,
                                                           order_1,
                                                           order_2,
                                                           order_3);
            EXPECT(reorder_sol != nullptr);
            size_t workspace_size = reorder_sol->IsSkippable() ? sizeof(T) * tensor_sz
                                                               : reorder_sol->GetOutputTensorSize();
            Workspace wspace{workspace_size};

            auto src_dev = handle.Write(t_src.data);

            const auto invoke_param         = reorder_invoke_param{src_dev.get(), wspace.ptr()};
            std::vector<OpKernelArg> opArgs = reorder_sol->GetKernelArg();
            boost::optional<miopen::InvokerFactory> invoker_factory(
                [=](const std::vector<miopen::Kernel>& kernels) mutable {
                    return [=](const miopen::Handle& handle,
                               const miopen::AnyInvokeParams& primitive_param) mutable {
                        decltype(auto) invoke_params =
                            primitive_param.CastTo<reorder_invoke_param>();
                        const auto k = handle.Run(kernels[0]);
                        opArgs[0]    = OpKernelArg(invoke_params.dst);
                        opArgs[1]    = OpKernelArg(invoke_params.src);
                        k(opArgs);
                    };
                });
            std::vector<miopen::solver::KernelInfo> construction_params{
                reorder_sol->GetKernelInfo()};
            const auto invoker = handle.PrepareInvoker(*invoker_factory, construction_params);
            // run gpu
            invoker(handle, invoke_param);
            // run cpu
            cpu_reorder<T>::run(t_dst.data.data(),
                                t_src.data.data(),
                                dim_0,
                                dim_1,
                                dim_2,
                                dim_3,
                                order_0,
                                order_1,
                                order_2,
                                order_3);
            invoker_factory = boost::none;

            t_dst_gpu.data = wspace.Read<decltype(t_dst_gpu.data)>();

            // we expect excact match, since use integer
            bool valid_result = verify_tensor(t_dst_gpu, t_dst);
            std::cout << "[" << reorder_str::get(order_0, order_1, order_2, order_3) << ", b"
                      << (sizeof(T) * 8) << " ] "
                      << "dim_0:" << dim_0 << ", dim_1:" << dim_1 << ", dim_2:" << dim_2
                      << ", dim_3:" << dim_3 << ", valid:" << valid_result << std::endl;
            EXPECT(valid_result == true);
        };

        iterate_reorder(run_reorder);
    }
    // NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)
};

template <template <class...> class Driver>
void test_tensor_reorder(int argc, const char* argv[])
{
    std::vector<std::string> as(argv + 1, argv + argc);
    as.emplace_back("--float");
    for(auto&& arg : as)
    {
        if(arg == "--all")
        {
            test_drive_impl<Driver<double>>(argv[0], as);
            test_drive_impl<Driver<float>>(argv[0], as);
            test_drive_impl<Driver<half_float::half>>(argv[0], as);
            test_drive_impl<Driver<int8_t>>(argv[0], std::move(as));
            break;
        }
        if(arg == "--double")
        {
            test_drive_impl<Driver<double>>(argv[0], std::move(as));
            break;
        }
        if(arg == "--float")
        {
            test_drive_impl<Driver<float>>(argv[0], std::move(as));
            break;
        }
        if(arg == "--half")
        {
            test_drive_impl<Driver<half_float::half>>(argv[0], std::move(as));
            break;
        }
        if(arg == "--int8")
        {
            test_drive_impl<Driver<int8_t>>(argv[0], std::move(as));
            break;
        }
    }
}

int main(int argc, const char* argv[]) { test_tensor_reorder<tensor_reorder_driver>(argc, argv); }
