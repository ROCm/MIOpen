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

#include "args.hpp"
#include "network_data.hpp"
#include "tensor_holder.hpp"
#include "test.hpp"
#include "verify.hpp"

#include <functional>
#include <miopen/functional.hpp>

template <class Test_Driver_Private_TypeName_>
const std::string& get_type_name()
{
    static std::string name;

    if(name.empty())
    {
#ifdef _MSC_VER
        name = typeid(Test_Driver_Private_TypeName_).name();
        name = name.substr(7);
#else
        const char parameter_name[] = "Test_Driver_Private_TypeName_ =";

        name = __PRETTY_FUNCTION__;

        auto begin  = name.find(parameter_name) + sizeof(parameter_name);
#if(defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7)
        auto length = name.find_last_of(",") - begin;
#else
        auto length = name.find_first_of("];", begin) - begin;
#endif
        name        = name.substr(begin, length);
#endif
    }

    return name;
}

struct rand_gen
{
    double operator()(int n, int c, int h, int w) const
    {
        return double((547 * n + 701 * c + 877 * h + 1049 * w + 173) % 17);
    };
};

struct test_driver
{
    test_driver()                   = default;
    test_driver(const test_driver&) = delete;
    test_driver& operator=(const test_driver&) = delete;

    struct argument
    {
        std::function<void(std::vector<std::string>)> write_value;
        std::vector<std::function<void()>> post_write_actions;
        std::vector<std::function<void(std::function<void()>)>> data_sources;
        std::string type;

        void post_write()
        {
            for(auto pw : post_write_actions)
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
                for(auto&& y : src())
                {
                    x = T(y);
                    post_write();
                    callback();
                }

            });
        }
    };

    std::unordered_map<std::string, argument> arguments;
    bool full_set    = false;
    bool verbose     = false;
    double tolerance = 80;
    int batch_factor = 0;
    bool no_validate = false;

    template <class Visitor>
    void parse(Visitor v)
    {
        v(full_set, {"--all"}, "Run all tests");
        v(verbose, {"--verbose", "-v"}, "Run verbose mode");
        v(tolerance, {"--tolerance", "-t"}, "Set test tolerance");
        v(batch_factor, {"--batch-factor", "-n"}, "Set batch factor");
        v(no_validate,
          {"--disable-validation"},
          "Disable cpu validation, so only gpu version is ran");
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
        arguments.insert(std::make_pair(name, argument{}));

        argument& arg   = arguments[name];
        arg.type        = get_type_name<T>();
        arg.write_value = [&](std::vector<std::string> params) { args::write_value{}(x, params); };
        miopen::each_args(std::bind(per_arg{}, std::ref(x), std::ref(arg), std::placeholders::_1),
                          fs...);
    }

    struct generate_tensor_t
    {
        std::function<std::set<std::vector<int>>()> get_data;
        template <class T>
        void operator()(T& x, argument& arg) const
        {
            arg.add_source(get_data, x);
            arg.post_write_actions.push_back([&x] { tensor_generate{}(x, rand_gen{}); });
        }
    };

    generate_tensor_t generate_tensor(std::set<std::vector<int>> dims, std::vector<int> single)
    {
        return {[=]() -> std::set<std::vector<int>> {
            if(full_set)
                return dims;
            else
                return {single};
        }};
    }

    template <class F>
    generate_tensor_t lazy_generate_tensor(F f, std::vector<int> single)
    {
        return {[=]() -> std::set<std::vector<int>> {
            if(full_set)
                return f();
            else
                return {single};
        }};
    }

    generate_tensor_t get_bn_spatial_input_tensor()
    {
        return lazy_generate_tensor([=] { return get_bn_spatial_inputs(batch_factor); },
                                    {4, 64, 28, 28});
    }

    generate_tensor_t get_bn_peract_input_tensor()
    {
        return lazy_generate_tensor([=] { return get_bn_peract_inputs(batch_factor); },
                                    {16, 32, 8, 8});
    }

    generate_tensor_t get_input_tensor()
    {
        return lazy_generate_tensor([=] { return get_inputs(batch_factor); }, {16, 32, 8, 8});
    }

    generate_tensor_t get_weights_tensor()
    {
        return lazy_generate_tensor([=] { return get_weights(batch_factor); }, {64, 32, 5, 5});
    }

    template <class X>
    struct generate_data_t
    {
        std::function<std::vector<X>()> get_data;
        template <class T>
        void operator()(T& x, argument& arg) const
        {
            arg.add_source(get_data, x);
        }
    };

    template <class T>
    generate_data_t<T> generate_data(std::vector<T> dims, T single)
    {
        return {[=]() -> std::vector<T> {
            if(full_set)
                return dims;
            else
                return {single};
        }};
    }

    template <class T>
    generate_data_t<T> generate_data(std::initializer_list<T> dims)
    {
        return generate_data(std::vector<T>(dims));
    }

    template <class T>
    generate_data_t<std::vector<T>>
    generate_data(std::initializer_list<std::initializer_list<T>> dims)
    {
        return generate_data(std::vector<std::vector<T>>(dims.begin(), dims.end()));
    }

    template <class T>
    generate_data_t<T> generate_data(std::vector<T> dims)
    {
        return {[=]() -> std::vector<T> {
            if(full_set)
                return dims;
            else
                return {dims.front()};
        }};
    }

    template <class T>
    generate_data_t<T> generate_single(T single)
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
            arg.write_value = [&x, y](std::vector<std::string>) { x = y; };
        }
    };

    template <class T>
    set_value_t<T> set_value(T x)
    {
        return {x};
    }

    set_value_t<bool> flag() { return set_value(true); }

    template <class CpuRange, class GpuRange, class Fail>
    std::pair<CpuRange, GpuRange> verify_check(CpuRange out_cpu, GpuRange out_gpu, Fail fail)
    {
        CHECK(miopen::range_distance(out_cpu) == miopen::range_distance(out_gpu));

        using value_type = miopen::range_value<decltype(out_gpu)>;
        double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
        auto error       = miopen::rms_range(out_cpu, out_gpu);
        if(not(error <= threshold) or verbose)
        {
            std::cout << (verbose ? "error: " : "FAILED: ") << error << std::endl;
            if(not verbose)
                fail(-1);

            auto mxdiff = miopen::max_diff(out_cpu, out_gpu);
            std::cout << "Max diff: " << mxdiff << std::endl;
            //            auto max_idx = miopen::mismatch_diff(out_cpu, out_gpu, mxdiff);
            //            std::cout << "Max diff at " << max_idx << ": " << out_cpu[max_idx] << " !=
            //            " << out_gpu[max_idx] << std::endl;

            if(miopen::range_zero(out_cpu))
                std::cout << "Cpu data is all zeros" << std::endl;
            if(miopen::range_zero(out_gpu))
                std::cout << "Gpu data is all zeros" << std::endl;

            auto idx = miopen::mismatch_idx(out_cpu, out_gpu, miopen::float_equal);
            if(idx < miopen::range_distance(out_cpu))
                std::cout << "Mismatch at " << idx << ": " << out_cpu[idx] << " != " << out_gpu[idx]
                          << std::endl;

            auto cpu_nan_idx = find_idx(out_cpu, miopen::not_finite);
            if(cpu_nan_idx >= 0)
                std::cout << "Non finite number found in cpu at " << cpu_nan_idx << ": "
                          << out_cpu[cpu_nan_idx] << std::endl;

            auto gpu_nan_idx = find_idx(out_gpu, miopen::not_finite);
            if(gpu_nan_idx >= 0)
                std::cout << "Non finite number found in gpu at " << gpu_nan_idx << ": "
                          << out_gpu[gpu_nan_idx] << std::endl;
        }
        else if(miopen::range_zero(out_cpu) and miopen::range_zero(out_gpu))
        {
            std::cout << "Warning: data is all zero" << std::endl;
            fail(-1);
        }
        return std::make_pair(std::move(out_cpu), std::move(out_gpu));
    }

    struct verify_check_t
    {
        template <class Self, class CpuRange, class GpuRange, class Fail, class I>
        auto operator()(Self self, CpuRange out_cpu, GpuRange out_gpu, Fail fail, I i) const
            MIOPEN_RETURNS(self->verify_check(std::get<I{}>(out_cpu),
                                              std::get<I{}>(out_gpu),
                                              std::bind(fail, i)))
    };

    struct verify_check_make_tuples
    {
        template <class... Ts>
        auto operator()(Ts... xs) const
            MIOPEN_RETURNS(std::make_pair(std::make_tuple(std::move(xs.first)...),
                                          std::make_tuple(std::move(xs.second)...)))
    };

    template <class... CpuRanges, class... GpuRanges, class Fail>
    std::pair<std::tuple<CpuRanges...>, std::tuple<GpuRanges...>>
    verify_check(std::tuple<CpuRanges...> out_cpu, std::tuple<GpuRanges...> out_gpu, Fail fail)
    {
        static_assert(sizeof...(CpuRanges) == sizeof...(GpuRanges), "Cpu and gpu mismatch");
        return miopen::sequence(miopen::by(verify_check_make_tuples{},
                                           std::bind(verify_check_t{},
                                                     this,
                                                     std::move(out_cpu),
                                                     std::move(out_gpu),
                                                     fail,
                                                     std::placeholders::_1)))(
            std::integral_constant<std::size_t, sizeof...(CpuRanges)>{});
    }

    template <class V, class... Ts>
    auto verify(V&& v, Ts&&... xs) -> decltype(std::make_pair(v.cpu(xs...), v.gpu(xs...)))
    {
        if(verbose)
            v.fail(std::integral_constant<int, -1>{}, xs...);
        try
        {
            if(no_validate)
            {
                auto gpu = v.gpu(xs...);
                return std::make_pair(gpu, gpu);
            }
            else
                return verify_check(
                    v.cpu(xs...), v.gpu(xs...), [&](int mode) { v.fail(mode, xs...); });
        }
        catch(const std::exception& ex)
        {
            std::cout << "FAILED: " << ex.what() << std::endl;
            v.fail(-1, xs...);
            throw;
        }
        catch(...)
        {
            std::cout << "FAILED with unknown exception" << std::endl;
            v.fail(-1, xs...);
            throw;
        }
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
        for(auto&& src : sources)
        {
            src([=] { run_data(std::next(start), last, a); });
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

struct show_help
{
    template <class T>
    void operator()(const T&, std::initializer_list<std::string> x, std::string help) const
    {
        std::cout << std::endl;
        std::string prefix = "    ";
        for(std::string a : x)
        {
            std::cout << prefix;
            std::cout << a;
            prefix = ", ";
        }
        if(not std::is_same<T, bool>{})
            std::cout << " [" << get_type_name<T>() << "]";
        std::cout << std::endl;
        std::cout << "        " << help << std::endl;
    }
};

template <class Driver>
void test_drive(int argc, const char* argv[])
{
    std::vector<std::string> as(argv + 1, argv + argc);
    Driver d{};

    std::set<std::string> keywords{"--help", "-h"};
    d.parse(keyword_set{keywords});
    auto arg_map = args::parse(as, [&](std::string x) {
        return (keywords.count(x) > 0) or
               ((x.compare(0, 2, "--") == 0) and d.arguments.count(x.substr(2)) > 0);
    });

    // Show help
    if(arg_map.count("-h") or arg_map.count("--help"))
    {
        std::cout << "Driver arguments: " << std::endl;
        d.parse(show_help{});
        std::cout << std::endl;
        std::cout << "Test inputs: " << std::endl;
        for(auto&& p : d.arguments)
        {
            std::cout << "    --" << p.first;
            if(not p.second.type.empty())
                std::cout << " [" << p.second.type << "]";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        return;
    }

    d.parse(parser{arg_map});

    for(auto&& p : arg_map)
    {
        if(keywords.count(p.first) == 0)
        {
            assert(p.first.length() > 2);
            auto name = p.first.substr(2);
            try
            {
                auto&& arg = d.arguments.at(name);
                arg.write(p.second);
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
    for(auto&& p : d.arguments)
    {
        if(arg_map.count("--" + p.first) == 0)
        {
            data_args.push_back(&p.second);
        }
    }

    run_data(data_args.begin(), data_args.end(), [&] { d.run(); });
}
