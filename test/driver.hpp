
#include "args.hpp"
#include "tensor_holder.hpp"
#include "network_data.hpp"

#include <functional>

struct rand_gen
{
    double operator()(int n, int c, int h, int w) const
    {
        return double((547*n+701*c+877*h+1049*w+173)%17);
    };
};

struct test_driver
{
    test_driver()=default;
    test_driver(const test_driver&)=delete;
    test_driver& operator=(const test_driver&)=delete;

    struct argument
    {
        std::function<void(std::vector<std::string>)> write_value;
        std::vector<std::function<void()>> post_write_actions;
        std::vector<std::function<void(std::function<void()>)>> data_sources;

        void post_write()
        {
            for(auto pw:post_write_actions)
            {
                pw();
            }
        }
        void write(std::vector<std::string> c)
        {
            write_value(c);
            post_write();
        }

        template<class Source, class T>
        void add_source(Source src, T& x)
        {
            data_sources.push_back([=, &x](std::function<void()> callback)
            {
                for(auto&& y:src())
                {
                    x = T(y);
                    post_write();
                    callback();
                }
                
            });
        }
    };

    std::unordered_map<std::string, argument> arguments;
    bool full_set = false;

    template<class ArgMap>
    void parse(const ArgMap& m)
    {
        if (m.count("--all") > 0) full_set = true;
    }


    struct per_arg
    {
        template<class T, class Action>
        void operator()(T& x, argument& a, Action action) const
        {
            action(x, a);
        }
    };

    template<class T, class... Fs>
    void add(T& x, std::string name, Fs... fs)
    {
        arguments.insert(std::make_pair(name, argument{}));

        argument& arg = arguments[name];
        arg.write_value = [&](std::vector<std::string> params)
        {
            args::write_value{}(x, params);
        };
        mlopen::each_args(std::bind(per_arg{}, std::ref(x), std::ref(arg), std::placeholders::_1), fs...);
    }

    struct generate_tensor_t
    {
        std::function<std::set<std::vector<int>>()> get_data;
        template<class T>
        void operator()(T& x, argument& arg) const
        {
            arg.add_source(get_data, x);
            arg.post_write_actions.push_back([&x]
            {
                tensor_generate{}(x, rand_gen{});
            });
        }
    };

    generate_tensor_t generate_tensor(std::set<std::vector<int>> dims, std::vector<int> single)
    {
        return {[=]() -> std::set<std::vector<int>> {
            if (full_set) return dims; 
            else return {single};
        }};
    }

    generate_tensor_t get_input_tensor()
    {
        return generate_tensor(get_inputs(), {16, 32, 8, 8});
    }

    generate_tensor_t get_weights_tensor()
    {
        return generate_tensor(get_weights(), {64, 32, 5, 5});
    }

    template<class X>
    struct generate_data_t
    {
        std::function<std::vector<X>()> get_data;
        template<class T>
        void operator()(T& x, argument& arg) const
        {
            arg.add_source(get_data, x);
        }
    };

    template<class T>
    generate_data_t<T> generate_data(std::vector<T> dims, T single)
    {
        return {[=]() -> std::vector<T> {
            if (full_set) return dims; 
            else return {single};
        }};
    }

    template<class T>
    generate_data_t<T> generate_single(T single)
    {
        return {[=]() -> std::vector<T> {
            return {single};
        }};
    }
};

template<class Iterator, class Action>
void run_data(Iterator start, Iterator last, Action a)
{
    if (start == last) 
    {
        a();
        return;
    }

    auto&& sources = (*start)->data_sources;
    if (sources.empty())
    {
        run_data(std::next(start), last, a);
    }
    else for(auto&& src:sources)
    {
        src([=]
        {
            run_data(std::next(start), last, a);
        });
    }

}

template<class Driver>
void test_drive(int argc, const char *argv[])
{
    std::vector<std::string> as(argv+1, argv+argc);
    Driver d{};

    const std::set<std::string> keywords{"--help", "-h", "--all"};
    auto arg_map = args::parse(as, [&](std::string x)
    {
        return 
            (keywords.count(x) > 0) or 
            ((x.compare(0, 2, "--") == 0) and d.arguments.count(x.substr(2)) > 0);
    });

    d.parse(arg_map);

    for(auto&& p:arg_map)
    {
        if (keywords.count(p.first) == 0)
        {
            auto name = p.first.substr(2);
            auto&& arg = d.arguments[name];
            arg.write(p.second);
        }
    }

    // Run data on arguments that are not passed in
    std::vector<typename Driver::argument*> data_args; 
    for(auto&& p:d.arguments)
    {
        if (arg_map.count("--" + p.first) == 0)
        {
            data_args.push_back(&p.second);
        }
    }

    run_data(data_args.begin(), data_args.end(), [&]
    {
        d.run();
    });
}

