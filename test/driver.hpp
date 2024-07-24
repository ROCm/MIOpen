/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#ifndef GUARD_MIOPEN_TEST_DRIVER_HPP
#define GUARD_MIOPEN_TEST_DRIVER_HPP

#include "args.hpp"
#include "get_handle.hpp"
#include "network_data.hpp"
#include "serialize.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"

#include <functional>
#include <deque>
#include <half/half.hpp>
#include <type_traits>
#include <miopen/filesystem.hpp>
#include <miopen/functional.hpp>
#include <miopen/expanduser.hpp>
#include <miopen/md5.hpp>
#include <miopen/type_name.hpp>
#include <miopen/env.hpp>
#include <miopen/rank.hpp>
#include <miopen/bfloat16.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

namespace env = miopen::env;

template <class U, class T>
constexpr std::is_same<T, U> is_same(const T&)
{
    return {};
}

struct tensor_elem_gen_checkboard_sign
{
    template <class... Ts>
    double operator()(Ts... Xs) const
    {
        std::array<uint64_t, sizeof...(Ts)> dims = {{Xs...}};
        return std::accumulate(dims.begin(),
                               dims.end(),
                               true,
                               [](int init, uint64_t x) -> int { return init != (x % 2); })
                   ? 1
                   : -1;
    }
};

template <class V, class... Ts>
auto is_const_cpu(const V& v, Ts&&... xs) -> decltype(v.cpu(xs...), std::true_type{})
{
    return {};
}

template <class V, class... Ts>
auto is_const_cpu(V& v, Ts&&... xs) -> decltype(v.cpu(xs...), std::false_type{})
{
    return {};
}

// Run cpu in parallel if it can be ran as const
template <class V, class... Ts>
auto cpu_async(const V& v, Ts&&... xs) -> std::future<decltype(v.cpu(xs...))>
{
    return detach_async([&] { return v.cpu(xs...); });
}

template <class V, class... Ts>
auto cpu_async(V& v, Ts&&... xs) -> std::future<decltype(v.cpu(xs...))>
{
    return std::async(std::launch::deferred, [&] { return v.cpu(xs...); });
}

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_VERIFY_CACHE_PATH)

struct test_driver
{
    test_driver()                   = default;
    test_driver(const test_driver&) = delete;
    test_driver& operator=(const test_driver&) = delete;

    struct argument
    {
        std::function<void(std::vector<std::string>)> write_value;
        std::function<std::string()> read_value;
        std::vector<std::function<void()>> post_write_actions;
        std::vector<std::function<void(std::function<void()>)>> data_sources;
        std::string type;
        std::string name;

        // Function may refer to the argument by reference so this needs to be noncopyable
        argument()                = default;
        argument(const argument&) = delete;
        argument& operator=(const argument&) = delete;

        void post_write()
        {
            for(const auto& pw : post_write_actions)
            {
                pw();
            }
        }
        void write(std::vector<std::string> c)
        {
            write_value(c);
            post_write();
        }

        template <class Source, class T>
        void add_source(Source src, T& x)
        {
            data_sources.push_back([=, &x](std::function<void()> callback) {
                for(auto y : src()) // NOLINT
                {
                    x = T(y);
                    post_write();
                    callback();
                }
            });
        }
    };

    static std::string compute_cache_path()
    {
        auto s = env::value(MIOPEN_VERIFY_CACHE_PATH);
        if(s.empty())
            return "~/.cache/miopen/tests";
        else
            return s;
    }

    std::string program_name;
    std::deque<argument> arguments;
    std::unordered_map<std::string, std::size_t> argument_index;
    int cache_version      = 1;
    std::string cache_path = compute_cache_path();
    miopenDataType_t type  = miopenFloat;
    bool full_set          = false;
    int limit_set          = 2;
    int dataset_id         = 0;
    bool verbose           = false;
    double tolerance       = 80;
    bool time              = false;
    int batch_factor       = 0;
    bool no_validate       = false;
    int repeat             = 1;
    bool rethrow           = false;
    bool disabled_cache    = false;
    bool dry_run           = false;
    int config_iter_start  = 0;
    int iteration          = 0;

    argument& get_argument(const std::string& s)
    {
        assert(arguments.at(argument_index.at(s)).name == s);
        return arguments.at(argument_index.at(s));
    }

    bool has_argument(const std::string& arg) const { return argument_index.count(arg) > 0; }

    template <class Visitor>
    void parse(Visitor v)
    {
        v(full_set, {"--all"}, "Run all tests");
        v(limit_set, {"--limit"}, "Limits the number of generated test elements");
        v(dataset_id,
          {"--dataset"},
          "Identifies the data set used for generation of tests (default=0)");
        v(verbose, {"--verbose", "-v"}, "Run verbose mode");
        v(tolerance, {"--tolerance", "-t"}, "Set test tolerance");
        v(time, {"--time"}, "Time the kernel on GPU");
        v(batch_factor, {"--batch-factor", "-n"}, "Set batch factor");
        v(no_validate,
          {"--disable-validation"},
          "Disable cpu validation, so only gpu version is ran");
        v(repeat, {"--repeat"}, "Repeat the tests");
        v(rethrow, {"--rethrow"}, "Rethrow any exceptions found during verify");
        v(cache_path, {"--verification-cache", "-C"}, "Path to verification cache");
        v(disabled_cache, {"--disable-verification-cache"}, "Disable verification cache");
        v(dry_run, {"--dry-run"}, "Dry run. Does not run the test, just prints the command.");
        v(config_iter_start,
          {"--config-iter-start", "-i"},
          "index of config at which to start a test."
          "Can be used to restart a test after a failing config.");
    }

    struct per_arg
    {
        template <class T, class Action>
        void operator()(T& x, argument& a, Action action) const
        {
            action(x, a);
        }
    };

    template <class T, class... Fs>
    void add(T& x, std::string name, Fs... fs)
    {
        argument_index.insert(std::make_pair(name, arguments.size()));
        arguments.emplace_back();

        argument& arg   = arguments.back();
        arg.name        = name;
        arg.type        = miopen::get_type_name<T>();
        arg.write_value = [&](std::vector<std::string> params) { args::write_value{}(x, params); };
        arg.read_value  = [&] { return args::read_value{}(x); };
        miopen::each_args(std::bind(per_arg{}, std::ref(x), std::ref(arg), std::placeholders::_1),
                          fs...);
        // assert(get_argument(name).name == name);
    }

    void show_help()
    {
        std::cout << "Driver arguments: " << std::endl;
        this->parse([&](const auto& var, std::initializer_list<std::string> x, std::string help) {
            std::cout << std::endl;
            std::string prefix = "    ";
            for(const std::string& a : x)
            {
                std::cout << prefix;
                std::cout << a;
                prefix = ", ";
            }
            if(not is_same<bool>(var))
                std::cout << " [" << miopen::get_type_name(var) << "]";
            std::cout << std::endl;
            std::cout << "        " << help << std::endl;
        });
        std::cout << std::endl;
        std::cout << "Test inputs: " << std::endl;
        for(auto&& arg : this->arguments)
        {
            std::cout << "    --" << arg.name;
            if(not arg.type.empty())
                std::cout << " [" << arg.type << "]";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::string get_command_args()
    {
        std::stringstream ss;
        switch(this->type)
        {
        case miopenHalf: ss << "--half "; break;
        case miopenBFloat16: ss << "--bfloat16 "; break;
        case miopenInt8: ss << "--int8 "; break;
        case miopenInt32: ss << "--int32 "; break;
        case miopenInt64: ss << "--int64 "; break;
        case miopenFloat: ss << "--float "; break;
        case miopenDouble: ss << "--double "; break;
        case miopenFloat8: ss << "--float8"; break;
        case miopenBFloat8: ss << "--bfloat8"; break;
        }
        for(auto&& arg : this->arguments)
        {
            std::string value = arg.read_value();
            if(not value.empty())
            {
                ss << "--" << arg.name << " ";
                if(value != arg.name)
                    ss << value << " ";
            }
        }
        return ss.str();
    }

    std::vector<std::string> get_config()
    {
        std::vector<std::string> ret;

        switch(this->type)
        {
        case miopenHalf: ret.emplace_back("--half"); break;
        case miopenBFloat16: ret.emplace_back("--bf16"); break;
        case miopenInt8: ret.emplace_back("--int8"); break;
        case miopenInt32: ret.emplace_back("--int32"); break;
        case miopenInt64: ret.emplace_back("--int64"); break;
        case miopenFloat: ret.emplace_back("--float"); break;
        case miopenDouble: ret.emplace_back("--double"); break;
        case miopenFloat8: ret.emplace_back("--float8"); break;
        case miopenBFloat8: ret.emplace_back("--bfloat8"); break;
        }

        for(auto&& arg : this->arguments)
        {
            std::string value = arg.read_value();
            std::vector<std::string> value_vector;
            boost::split(value_vector, value, boost::is_any_of(" "), boost::token_compress_on);
            if(not value.empty())
            {
                ret.emplace_back("--" + arg.name);
                if(value != arg.name)
                    ret.insert(ret.end(), value_vector.begin(), value_vector.end());
            }
        }

        return ret;
    }

    void show_command()
    {
        std::cout << this->program_name << " ";
        std::cout << get_command_args() << std::endl;
    }

    template <class X, class G>
    struct generate_tensor_t
    {
        std::function<std::set<X>()> get_data;
        G tensor_elem_gen;

        template <class T>
        void operator()(T& x, argument& arg) const
        {
            arg.add_source(get_data, x);
            G g = tensor_elem_gen;
            arg.post_write_actions.push_back([&x, g] { tensor_generate{}(x, g); });
        }
    };

    template <class X, class G>
    generate_tensor_t<X, G> generate_tensor(std::set<X> dims, X single, G g)
    {
        return {[=]() -> std::set<X> {
                    if(full_set)
                        return dims;
                    else
                        return {single};
                },
                g};
    }

    template <class X, class G>
    generate_tensor_t<std::vector<X>, G>
    generate_tensor(std::set<std::vector<X>> dims, std::initializer_list<X> single, G g)
    {
        return generate_tensor<std::vector<X>, G>(dims, single, g);
    }

    template <class F, class G>
    auto lazy_generate_tensor(F f, G g) -> generate_tensor_t<miopen::range_value<decltype(f())>, G>
    {
        return {[=]() -> decltype(f()) {
                    if(full_set)
                        return f();
                    else
                        return {*f().begin()};
                },
                g};
    }

    template <class F, class X, class G>
    generate_tensor_t<X, G> lazy_generate_tensor(F f, X single, G g)
    {
        return {[=]() -> std::set<X> {
                    if(full_set)
                        return f();
                    else
                        return {single};
                },
                g};
    }

    template <class F, class X, class G>
    generate_tensor_t<std::vector<X>, G>
    lazy_generate_tensor(F f, std::initializer_list<X> single, G g)
    {
        return lazy_generate_tensor<F, std::vector<X>, G>(f, single, g);
    }

    template <class F, class G>
    generate_tensor_t<std::vector<int>, G> get_tensor(F gen_shapes, G gen_value)
    {
        return lazy_generate_tensor([=] { return gen_shapes(batch_factor); }, gen_value);
    }

    template <class G = tensor_elem_gen_integer>
    generate_tensor_t<std::vector<int>, G>
    get_bn_spatial_input_tensor(G tensor_elem_gen = tensor_elem_gen_integer{})
    {
        return lazy_generate_tensor(
            [=] { return get_bn_spatial_inputs(batch_factor); }, {4, 64, 28, 28}, tensor_elem_gen);
    }

    template <class G = tensor_elem_gen_integer>
    generate_tensor_t<std::vector<int>, G>
    get_bn_peract_input_tensor(G tensor_elem_gen = tensor_elem_gen_integer{})
    {
        return lazy_generate_tensor(
            [=] { return get_bn_peract_inputs(batch_factor); }, {16, 32, 8, 8}, tensor_elem_gen);
    }

    template <class G = tensor_elem_gen_integer>
    generate_tensor_t<std::vector<int>, G>
    get_input_tensor(G tensor_elem_gen = tensor_elem_gen_integer{})
    {
        return lazy_generate_tensor(
            [=] { return get_inputs(batch_factor); }, {16, 32, 8, 8}, tensor_elem_gen);
    }

    template <class G = tensor_elem_gen_integer>
    generate_tensor_t<std::vector<int>, G>
    get_3d_bn_spatial_input_tensor(G tensor_elem_gen = tensor_elem_gen_integer{})
    {
        return lazy_generate_tensor([=] { return get_3d_bn_spatial_inputs(batch_factor); },
                                    {16, 32, 8, 8, 8},
                                    tensor_elem_gen);
    }

    template <class G = tensor_elem_gen_integer>
    generate_tensor_t<std::vector<int>, G>
    get_3d_bn_peract_input_tensor(G tensor_elem_gen = tensor_elem_gen_integer{})
    {
        return lazy_generate_tensor([=] { return get_3d_bn_peract_inputs(batch_factor); },
                                    {16, 32, 8, 8, 8},
                                    tensor_elem_gen);
    }

    template <class G = tensor_elem_gen_integer>
    generate_tensor_t<std::vector<int>, G>
    get_weights_tensor(G tensor_elem_gen = tensor_elem_gen_integer{})
    {
        return lazy_generate_tensor(
            [=] { return get_weights(batch_factor); }, {64, 32, 5, 5}, tensor_elem_gen);
    }

    template <class X>
    struct generate_data_t
    {
        std::function<X()> get_data;
        template <class T>
        void operator()(T& x, argument& arg) const
        {
            arg.add_source(get_data, x);
        }
        // This is necessary to reuse lambdas provided by generate_data*()
        // in the lambdas generated by the generate_multi_data*().
        template <class T>
        std::vector<T> operator()(const T&) const
        {
            return get_data();
        }
    };

    template <class T>
    generate_data_t<std::vector<T>> generate_data(std::vector<T> dims, T single)
    {
        return {[=]() -> std::vector<T> {
            if(full_set)
                return dims;
            else
                return {single};
        }};
    }

    template <class T>
    generate_data_t<std::vector<T>>
    generate_data_limited(std::vector<T> dims, int limit_multiplier, T single)
    {
        return {[=]() -> std::vector<T> {
            if(full_set)
            {
                if(limit_set > 0)
                {
                    auto endpoint =
                        std::min(static_cast<int>(dims.size()), limit_set * limit_multiplier);
                    std::vector<T> subvec(dims.cbegin(), dims.cbegin() + endpoint);
                    return subvec;
                }
                else
                    return dims;
            }
            else
            {
                return {single};
            }
        }};
    }

    template <class T>
    generate_data_t<std::vector<T>> generate_data(std::initializer_list<T> dims)
    {
        return generate_data(std::vector<T>(dims));
    }

    template <class T>
    generate_data_t<std::vector<std::vector<T>>>
    generate_data(std::initializer_list<std::initializer_list<T>> dims)
    {
        return generate_data(std::vector<std::vector<T>>(dims.begin(), dims.end()));
    }

    template <class T>
    generate_data_t<std::vector<T>> generate_data(std::vector<T> dims)
    {
        return {[=]() -> std::vector<T> {
            if(full_set)
                return dims;
            else
                return {dims.front()};
        }};
    }

    template <class T>
    generate_data_t<std::vector<T>> generate_multi_data(std::vector<std::vector<T>> multi_dims)
    {
        return {[=]() -> std::vector<T> { return generate_data(multi_dims.at(dataset_id))(T{}); }};
    }

    template <class T>
    generate_data_t<std::vector<T>> generate_data_limited(std::vector<T> dims, int limit_multiplier)
    {
        return {[=]() -> std::vector<T> {
            if(full_set)
            {
                if(limit_set > 0)
                {
                    auto endpoint =
                        std::min(static_cast<int>(dims.size()), limit_set * limit_multiplier);
                    std::vector<T> subvec(dims.cbegin(), dims.cbegin() + endpoint);
                    return subvec;
                }
                else
                {
                    return dims;
                }
            }
            else
            {
                return {dims.front()};
            }
        }};
    }

    template <class T>
    generate_data_t<std::vector<T>>
    generate_multi_data_limited(std::vector<std::vector<T>> multi_dims, int limit_multiplier)
    {
        return {[=]() -> std::vector<T> {
            return generate_data_limited(multi_dims.at(dataset_id), limit_multiplier)(T{});
        }};
    }

    template <class F, class T>
    auto lazy_generate_data(F f, T single) -> generate_data_t<decltype(f())>
    {
        return {[=]() -> decltype(f()) {
            if(full_set)
                return f();
            else
                return {single};
        }};
    }

    template <class F>
    auto lazy_generate_data(F f) -> generate_data_t<decltype(f())>
    {
        return {[=]() -> decltype(f()) {
            if(full_set)
                return f();
            else
                return {f().front()};
        }};
    }

    template <class T>
    generate_data_t<std::vector<T>> generate_single(T single)
    {
        return {[=]() -> std::vector<T> { return {single}; }};
    }

    template <class X>
    struct set_value_t
    {
        X value;
        template <class T>
        void operator()(T& x, argument& arg) const
        {
            auto y          = value;
            arg.type        = "";
            arg.write_value = [&x, y](std::vector<std::string> as) {
                if(not as.empty())
                    throw std::runtime_error("Argument should not have any additional parameters");
                x = y;
            };
            arg.read_value = [&x, &arg, y]() -> std::string {
                if(x == y)
                    return arg.name;
                else
                    return "";
            };
        }
    };

    template <class T>
    set_value_t<T> set_value(T x)
    {
        return {x};
    }

    set_value_t<bool> flag() { return set_value(true); }

    auto verify_reporter()
    {
        return [=](bool pass,
                   std::vector<double> error,
                   const auto& out_cpu,
                   const auto& out_gpu,
                   auto fail) {
            if(not pass or verbose)
            {
                if(not error.empty() or not pass)
                {
                    if(not verbose)
                        show_command();

                    if(not error.empty())
                        std::cout << (pass ? "error: " : "FAILED: ") << error.front() << std::endl;
                    else
                        std::cout << "FAILED: " << std::endl;

                    if(not verbose)
                    {
                        std::cout << "Iteration: " << this->iteration << std::endl;
                        fail(-1);
                    }
                }

                auto mxdiff = miopen::max_diff(out_cpu, out_gpu);
                std::cout << "Max diff: " << mxdiff << std::endl;
                //            auto max_idx = miopen::mismatch_diff(out_cpu, out_gpu, mxdiff);
                //            std::cout << "Max diff at " << max_idx << ": " << out_cpu[max_idx] <<
                //            " !=
                //            " << out_gpu[max_idx] << std::endl;

                if(miopen::range_zero(out_cpu))
                    std::cout << "Cpu data is all zeros" << std::endl;
                if(miopen::range_zero(out_gpu))
                    std::cout << "Gpu data is all zeros" << std::endl;

                auto idx = miopen::mismatch_idx(out_cpu, out_gpu, miopen::float_equal);
                if(idx < miopen::range_distance(out_cpu))
                {
                    std::cout << "Mismatch at " << idx << ": " << out_cpu[idx]
                              << " != " << out_gpu[idx] << std::endl;
                }

                auto cpu_nan_idx = find_idx(out_cpu, miopen::not_finite);
                if(cpu_nan_idx >= 0)
                {
                    std::cout << "Non finite number found in cpu at " << cpu_nan_idx << ": "
                              << out_cpu[cpu_nan_idx] << std::endl;
                }

                auto gpu_nan_idx = find_idx(out_gpu, miopen::not_finite);
                if(gpu_nan_idx >= 0)
                {
                    std::cout << "Non finite number found in gpu at " << gpu_nan_idx << ": "
                              << out_gpu[gpu_nan_idx] << std::endl;
                }
            }
            else if(miopen::range_zero(out_cpu) and miopen::range_zero(out_gpu) and
                    (miopen::range_distance(out_cpu) != 0))
            {
                show_command();
                std::cout << "Warning: Both CPU and GPU data is all zero" << std::endl;
                fail(-1);
            }
            return true;
        };
    }

    template <class CpuRange, class GpuRange, class Compare, class Report, class Fail>
    bool compare_and_report(
        const CpuRange& out_cpu, const GpuRange& out_gpu, Compare compare, Report report, Fail fail)
    {
        std::vector<double> error;
        bool pass = compare(error, out_cpu, out_gpu);
        return report(pass, error, out_cpu, out_gpu, fail);
    }

    template <class... CpuRanges, class... GpuRanges, class Compare, class Report, class Fail>
    bool compare_and_report(const std::tuple<CpuRanges...>& out_cpu,
                            const std::tuple<GpuRanges...>& out_gpu,
                            Compare compare,
                            Report report,
                            Fail fail)
    {
        static_assert(sizeof...(CpuRanges) == sizeof...(GpuRanges), "Cpu and gpu mismatch");
        return miopen::sequence([&](auto... is) {
            bool continue_ = true;
            miopen::each_args(
                [&](auto i) {
                    // cppcheck-suppress knownConditionTrueFalse
                    if(continue_)
                    {
                        continue_ = this->compare_and_report(
                            std::get<i>(out_cpu), std::get<i>(out_gpu), compare, report, [&](int) {
                                return fail(i);
                            });
                    }
                },
                is...);
            return continue_;
        })(std::integral_constant<std::size_t, sizeof...(CpuRanges)>{});
    }

    bool is_cache_disabled() const
    {
        if(disabled_cache)
            return true;
        auto p = miopen::ExpandUser(cache_path) / ".disabled";
        return miopen::fs::exists(p);
    }

    template <class V, class... Ts>
    auto run_cpu(bool retry, bool& miss, V& v, Ts&&... xs) -> std::future<decltype(v.cpu(xs...))>
    {
        using result_type = decltype(v.cpu(xs...));
        if(is_cache_disabled() or not is_const_cpu(v, xs...))
            return cpu_async(v, xs...);
        auto key = miopen::get_type_name<V>() + "-" + miopen::md5(get_command_args());
        auto p   = miopen::ExpandUser(cache_path) / std::to_string(cache_version);
        if(!miopen::fs::exists(p))
            miopen::fs::create_directories(p);
        auto f = p / key;
        if(miopen::fs::exists(f) and not retry)
        {
            miss = false;
            return detach_async([=] {
                result_type result;
                load(f.string(), result);
                return result;
            });
        }
        else
        {
            miss = true;
            return then(cpu_async(v, xs...), [=](auto data) {
                save(f.string(), data);
                return data;
            });
        }
    }

    template <class V>
    void adjust_parameters_impl(miopen::rank<0>, V&&)
    {
    }

    /// Winograd algorithm has worse precision than Direct and Gemm.
    /// Winograd-specific precision loss is roughly 2+2 bits.
    /// Let's adjust tolerance (only for FP32 WrW for now).
    template <class V>
    auto adjust_parameters_impl(miopen::rank<1>, V&& v)
        -> decltype(v.stats, v.is_conv_wrw_f32, void())
    {
        if(v.is_conv_wrw_f32 && v.stats->algorithm == miopenConvolutionAlgoWinograd)
            tolerance *= 16.0;
    }

    template <class V>
    auto adjust_parameters(V&& v) -> decltype(adjust_parameters_impl(miopen::rank<1>{}, v))
    {
        return adjust_parameters_impl(miopen::rank<1>{}, v);
    }

    template <class F, class V, class... Ts>
    auto verify_impl(F&& f, V&& v, Ts&&... xs)
        -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
    {
        decltype(v.cpu(xs...)) cpu;
        decltype(v.gpu(xs...)) gpu;

        if(verbose or time)
            show_command();

        try
        {
            auto&& h = get_handle();
            // Compute cpu
            std::future<decltype(v.cpu(xs...))> cpuf;
            bool cache_miss = true;
            if(not no_validate)
            {
                cpuf = run_cpu(false, cache_miss, v, xs...);
            }
            // Compute gpu
            if(time)
            {
                h.EnableProfiling();
                h.ResetKernelTime();
            }
            gpu = v.gpu(xs...);
            adjust_parameters(v);

            if(time)
            {
                std::cout << "Kernel time: " << h.GetKernelTime() << " ms" << std::endl;
                h.EnableProfiling(false);
            }
            // Validate
            if(!no_validate)
            {
                cpu         = cpuf.get();
                auto report = this->verify_reporter();
                bool retry  = true;
                if(not cache_miss)
                {
                    retry             = false;
                    auto report_retry = [&](bool pass,
                                            std::vector<double> error,
                                            const auto& out_cpu,
                                            const auto& out_gpu,
                                            auto fail) {
                        if(not pass)
                        {
                            retry = true;
                            return false;
                        }
                        return report(pass, error, out_cpu, out_gpu, fail);
                    };
                    compare_and_report(
                        cpu, gpu, f, report_retry, [&](int mode) { v.fail(mode, xs...); });
                    // cppcheck-suppress knownConditionTrueFalse
                    if(retry)
                    {
                        std::cout << "Warning: verify cache failed, rerunning cpu." << std::endl;
                        cpu = run_cpu(retry, cache_miss, v, xs...).get();
                    }
                }
                if(retry)
                    compare_and_report(cpu, gpu, f, report, [&](int mode) { v.fail(mode, xs...); });
            }

            if(verbose or time)
                v.fail(std::integral_constant<int, -1>{}, xs...);
        }
        catch(const std::exception& ex)
        {
            show_command();
            std::cout << "FAILED: " << ex.what() << std::endl;
            v.fail(-1, xs...);
            if(rethrow)
                throw;
        }
        catch(...)
        {
            show_command();
            std::cout << "FAILED with unknown exception" << std::endl;
            v.fail(-1, xs...);
            if(rethrow)
                throw;
        }
        if(no_validate)
        {
            return std::make_pair(gpu, gpu);
        }
        else
        {
            return std::make_pair(cpu, gpu);
        }
    }

    template <class V, class... Ts>
    auto verify_eps(V&& v, Ts&&... xs) -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
    {
        return verify_impl(
            [&](std::vector<double>& error, auto&& cpu, auto&& gpu) {
                CHECK(miopen::range_distance(cpu) == miopen::range_distance(gpu));

                double threshold = v.epsilon() * tolerance;
                error            = {miopen::rms_range(cpu, gpu)};
                return error.front() <= threshold;
            },
            v,
            xs...);
    }

    template <class V, class... Ts>
    auto verify(V&& v, Ts&&... xs) -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
    {
        return verify_impl(
            [&](std::vector<double>& error, auto&& cpu, auto&& gpu) {
                CHECK(miopen::range_distance(cpu) == miopen::range_distance(gpu));

                using value_type = miopen::range_value<decltype(gpu)>;
                double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
                error            = {miopen::rms_range(cpu, gpu)};
                return error.front() <= threshold;
            },
            v,
            xs...);
    }

    template <class V, class... Ts>
    auto verify_equals(V&& v, Ts&&... xs) -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
    {
        return verify_impl(
            [&](auto&, auto&& cpu, auto&& gpu) {
                auto idx = miopen::mismatch_idx(cpu, gpu, miopen::float_equal);
                return idx >= miopen::range_distance(cpu);
            },
            v,
            xs...);
    }

    template <class Derived>
    void base_run()
    {
        if(this->iteration >= this->config_iter_start)
        {
            if(this->dry_run)
            {
                std::cout << "Iteration: " << this->iteration << std::endl;
                show_command();
            }
            else
            {
                prng::reset_seed();
                static_cast<Derived*>(this)->run();
                prng::reset_seed();
            }
        }
        this->iteration++;
    }
};

template <class Iterator, class Action>
void run_data(Iterator start, Iterator last, Action a)
{
    if(start == last)
    {
        a();
        return;
    }

    auto&& sources = (*start)->data_sources;
    if(sources.empty())
    {
        run_data(std::next(start), last, a);
    }
    else
    {
        for(auto&& src : sources)
        {
            src([=] { run_data(std::next(start), last, a); });
        }
    }
}

struct keyword_set
{
    std::set<std::string>* value;
    keyword_set(std::set<std::string>& x) : value(&x) {}
    template <class T>
    void operator()(T&&, std::initializer_list<std::string> x, std::string) const
    {
        value->insert(x);
    }
};

struct parser
{
    args::string_map* m;
    parser(args::string_map& x) : m(&x) {}
    template <class T>
    void operator()(T& x, std::initializer_list<std::string> keywords, std::string) const
    {
        for(auto&& keyword : keywords)
        {
            if(m->count(keyword) > 0)
            {
                try
                {
                    args::write_value{}(x, (*m)[keyword]);
                    return;
                }
                catch(...)
                {
                    std::cerr << "Invalid argument: " << keyword << std::endl;
                    throw;
                }
            }
        }
    }

    void operator()(bool& x, std::initializer_list<std::string> keywords, std::string) const
    {
        for(auto&& keyword : keywords)
        {
            if(m->count(keyword) > 0)
            {
                x = true;
                return;
            }
        }
    }
};

template <class Driver>
void check_unparsed_args(Driver& d,
                         std::unordered_map<std::string, std::vector<std::string>>& arg_map,
                         std::set<std::string>& keywords)
{
    for(auto&& p : arg_map)
    {
        if(p.first.empty())
        {
            std::cerr << "Unused arguments: " << std::endl;
            for(auto&& s : p.second)
                std::cerr << "    " << s << std::endl;
            std::abort();
        }
        else if(keywords.count(p.first) == 0)
        {
            assert(p.first.length() > 2);
            auto name = p.first.substr(2);
            try
            {
                auto&& arg = d.get_argument(name);
                arg.write(p.second);
            }
            catch(const std::exception& ex)
            {
                std::cerr << "Invalid argument: " << name << std::endl;
                std::cerr << "With parameters: " << std::endl;
                for(auto&& s : p.second)
                    std::cerr << "    " << s << std::endl;
                std::cerr << ex.what() << std::endl;
                std::abort();
            }
            catch(...)
            {
                std::cerr << "Invalid argument: " << name << std::endl;
                std::cerr << "With parameters: " << std::endl;
                for(auto&& s : p.second)
                    std::cerr << "    " << s << std::endl;
                throw;
            }
        }
    }
}
#if(MIOPEN_TEST_DRIVER_MODE == 2)
template <class Driver>
std::vector<typename Driver::argument*>
get_data_args(Driver& d, std::unordered_map<std::string, std::vector<std::string>>& arg_map)
{
    // Run data on arguments that are not passed in
    std::vector<typename Driver::argument*> data_args;
    for(auto&& arg : d.arguments)
    {
        if(arg_map.count("--" + arg.name) == 0)
        {
            data_args.push_back(&arg);
        }
    }

    return data_args;
}

template <class Driver>
void set_driver_datatype(Driver& d,
                         std::unordered_map<std::string, std::vector<std::string>>& arg_map)
{
    if(arg_map.count("--half") > 0)
    {
        d.type = miopenHalf;
    }
    else if(arg_map.count("--int8") > 0)
    {
        d.type = miopenInt8;
    }
    else if(arg_map.count("--double") > 0)
    {
        d.type = miopenDouble;
    }
    else
    {
        d.type = miopenFloat;
    }
}

template <class Driver>
std::vector<std::vector<std::string>>
build_configs(Driver& d,
              std::unordered_map<std::string, std::vector<std::string>>& arg_map,
              std::set<std::string>& keywords)
{
    std::cout << "Building configs...";
    std::vector<std::vector<std::string>> configs;

    d.parse(parser{arg_map});
    check_unparsed_args<Driver>(d, arg_map, keywords);
    std::vector<typename Driver::argument*> data_args = get_data_args<Driver>(d, arg_map);

    run_data(data_args.begin(), data_args.end(), [&] {
        prng::reset_seed();
        std::vector<std::string> config = d.get_config();
        configs.push_back(config);
        prng::reset_seed();
    });
    std::cout << " done." << std::endl;
    return configs;
}

template <class Driver>
std::unordered_map<std::string, std::vector<std::string>>
create_arg_map(Driver& d, std::set<std::string>& keywords, std::vector<std::string>& args)
{
    d.parse(keyword_set{keywords});
    return args::parse(args, [&](std::string x) {
        return (keywords.count(x) > 0) or
               ((x.compare(0, 2, "--") == 0) and d.has_argument(x.substr(2)));
    });
}

// simple rolling average equation taken from
// https://stackoverflow.com/questions/12636613/how-to-calculate-moving-average-without-keeping-the-count-and-data-total
inline double approxRollingAverage(double avg, double new_sample, int N)
{
    avg -= avg / N;
    avg += new_sample / N;
    return avg;
}

template <class Driver>
void run_config(std::vector<std::string>& config,
                std::unordered_map<std::string, std::vector<std::string>>& arg_map,
                std::string& program_name,
                std::set<std::string>& keywords,
                int& test_repeat_count)
{
    Driver config_driver{};
    config_driver.program_name = program_name;
    auto config_arg_map        = create_arg_map<Driver>(config_driver, keywords, config);
    set_driver_datatype<Driver>(config_driver, config_arg_map);
    config_driver.parse(parser{config_arg_map});
    check_unparsed_args<Driver>(config_driver, config_arg_map, keywords);

    std::vector<typename Driver::argument*> config_data_args =
        get_data_args<Driver>(config_driver, config_arg_map);

    if(arg_map.count("--verbose") > 0)
    {
        config_driver.verbose = true;
    }

    if(arg_map.count("--disable-verification-cache") > 0)
    {
        config_driver.disabled_cache = true;
    }

    for(int j = 0; j < test_repeat_count; j++)
    {
        run_data(config_data_args.begin(), config_data_args.end(), [&] {
            prng::reset_seed();
            config_driver.run();
            prng::reset_seed();
        });
    }
}

template <class Driver>
void test_drive_impl_2(std::string program_name, std::vector<std::string> as)
{
    Driver d{};
    d.program_name = program_name;

    std::set<std::string> keywords{"--help", "-h", "--half", "--float", "--double", "--int8"};
    auto arg_map          = create_arg_map<Driver>(d, keywords, as);
    int test_repeat_count = d.repeat;

    // Show help
    if((arg_map.count("-h") > 0) or (arg_map.count("--help") > 0))
    {
        d.show_help();
        return;
    }

    set_driver_datatype<Driver>(d, arg_map);

    std::vector<std::vector<std::string>> configs = build_configs<Driver>(d, arg_map, keywords);
    size_t config_count                           = configs.size();
    double running_average                        = 0;

    // iterate through and run configs
    for(size_t i = d.config_iter_start; i < config_count; ++i)
    {
        std::cout << "Config " << i + 1 << "/" << config_count << std::endl;
        auto start = std::chrono::high_resolution_clock::now(); // Record start time
        run_config<Driver>(configs[i], arg_map, program_name, keywords, test_repeat_count);
        auto finish = std::chrono::high_resolution_clock::now(); // Record end time
        std::chrono::duration<double> elapsed = finish - start;
        if(i == 0)
        {
            running_average = elapsed.count();
        }
        else
        {
            running_average = approxRollingAverage(running_average, elapsed.count(), config_count);
        }

        std::cout << "Elapsed time: " << elapsed.count() << " s"
                  << ", "
                  << "Running Average: " << running_average << " s" << std::endl;
    }
}
#endif
template <class Driver>
void test_drive_impl_1(std::string program_name, std::vector<std::string> as)
{
    Driver d{};
    d.program_name = program_name;

    std::cout << program_name << " ";
    for(const auto& str : as)
        std::cout << str << " ";
    std::cout << std::endl;

    std::set<std::string> keywords{
        "--help", "-h", "--half", "--float", "--double", "--int8", "--bfloat16"};
    d.parse(keyword_set{keywords});
    auto arg_map = args::parse(as, [&](std::string x) {
        return (keywords.count(x) > 0) or
               ((x.compare(0, 2, "--") == 0) and d.has_argument(x.substr(2)));
    });

    if(arg_map.count("--half") > 0)
    {
        d.type = miopenHalf;
    }
    else if(arg_map.count("--int8") > 0)
    {
        d.type = miopenInt8;
    }
    else if(arg_map.count("--bfloat16") > 0)
    {
        d.type = miopenBFloat16;
    }
    else if(arg_map.count("--double") > 0)
    {
        d.type = miopenDouble;
    }
    else
    {
        d.type = miopenFloat;
    }

    // Show help
    if((arg_map.count("-h") > 0) or (arg_map.count("--help") > 0))
    {
        d.show_help();
        return;
    }

    d.parse(parser{arg_map});

    for(auto&& p : arg_map)
    {
        if(p.first.empty())
        {
            std::cerr << "Unused arguments: " << std::endl;
            for(auto&& s : p.second)
                std::cerr << "    " << s << std::endl;
            std::abort();
        }
        else if(keywords.count(p.first) == 0)
        {
            assert(p.first.length() > 2);
            auto name = p.first.substr(2);
            try
            {
                auto&& arg = d.get_argument(name);
                arg.write(p.second);
            }
            catch(const std::exception& ex)
            {
                std::cerr << "Invalid argument: " << name << std::endl;
                std::cerr << "With parameters: " << std::endl;
                for(auto&& s : p.second)
                    std::cerr << "    " << s << std::endl;
                std::cerr << ex.what() << std::endl;
                std::abort();
            }
            catch(...)
            {
                std::cerr << "Invalid argument: " << name << std::endl;
                std::cerr << "With parameters: " << std::endl;
                for(auto&& s : p.second)
                    std::cerr << "    " << s << std::endl;
                throw;
            }
        }
    }

    // Run data on arguments that are not passed in
    std::vector<typename Driver::argument*> data_args;
    for(auto&& arg : d.arguments)
    {
        if(arg_map.count("--" + arg.name) == 0)
        {
            data_args.push_back(&arg);
        }
    }

    prng::reset_seed();
    for(int i = 0; i < d.repeat; i++)
    {
        d.iteration = 0;
        run_data(data_args.begin(), data_args.end(), [&] { d.template base_run<Driver>(); });
    }
}

template <class Driver>
void test_drive_impl(std::string program_name, std::vector<std::string> as)
{
#if(MIOPEN_TEST_DRIVER_MODE == 2)
    std::cout << "MIOPEN_TEST_DRIVER_MODE 2." << std::endl;
    test_drive_impl_2<Driver>(program_name, as);
#else
    test_drive_impl_1<Driver>(program_name, as);
#endif
}

template <class Driver>
void test_drive(int argc, const char* argv[], const char* program_name = nullptr)
{
    std::string name(program_name ? program_name : argv[0]);
    std::vector<std::string> as(argv + (program_name ? 0 : 1), argv + argc);
    test_drive_impl<Driver>(name, std::move(as));
}

template <template <class...> class Driver>
void test_drive(int argc, const char* argv[], const char* program_name = nullptr)
{
    std::string name(program_name ? program_name : argv[0]);
    std::vector<std::string> as(argv + (program_name ? 0 : 1), argv + argc);

    for(auto&& arg : as)
    {
        if(arg == "--half")
        {
            test_drive_impl<Driver<half_float::half>>(name, std::move(as));
            return;
        }
        if(arg == "--int8")
        {
            test_drive_impl<Driver<int8_t>>(name, std::move(as));
            return;
        }
        if(arg == "--float")
        {
            test_drive_impl<Driver<float>>(name, std::move(as));
            return;
        }
        if(arg == "--bfloat16")
        {
            test_drive_impl<Driver<bfloat16>>(name, std::move(as));
            return;
        }
        if(arg == "--double")
        {
            test_drive_impl<Driver<double>>(name, std::move(as));
            return;
        }
    }

    // default datatype
    test_drive_impl<Driver<float>>(name, std::move(as));
}

#endif // GUARD_MIOPEN_TEST_DRIVER_HPP
