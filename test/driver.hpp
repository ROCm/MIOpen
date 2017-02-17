
#include "args.hpp"
#include "tensor_holder.hpp"
#include "network_data.hpp"

struct rand_gen
{
    double operator()(int n, int c, int h, int w) const
    {
        return double((547*n+701*c+877*h+1049*w+173)%17);
    };
};

struct test_driver
{
    struct argument
    {
        std::function<void(std::vector<std::string>)> write_value;
        std::vector<std::function<void()>> post_write_actions;
        // std::string name;
    };

    std::unordered_map<std::string, argument> arguments;


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
        std::cout << "Add: " << name << std::endl;
        argument arg;
        arg.write_value = [&](std::vector<std::string> params)
        {
            args::write_value{}(x, params);
        };
        mlopen::each_args(std::bind(per_arg{}, std::ref(x), std::ref(arg), std::placeholders::_1), fs...);
        arguments.insert(std::make_pair(name, arg));
    }

    struct generate_tensor_t
    {
        template<class T>
        void operator()(T& x, argument& arg) const
        {
            arg.post_write_actions.push_back([&]
            {
                tensor_generate{}(x, rand_gen{});
            });
        }
    };

    // Tensors: {16, 32, 8, 8}, {64, 32, 5, 5}
    generate_tensor_t generate_tensor()
    {
        return {};
    }
};

template<class Driver>
void test_drive(int argc, const char *argv[])
{
    std::vector<std::string> as(argv+1, argv+argc);
    Driver d{};

    auto arg_map = args::parse(as, [&](std::string x)
    {
        return (x.compare(0, 2, "--") == 0) and d.arguments.count(x.substr(2)) > 0;
    });

    for(auto&& p:arg_map)
    {
        auto name = p.first.substr(2);
        auto&& arg = d.arguments[name];
        arg.write_value(p.second);
        for(auto post_write:arg.post_write_actions)
        {
            post_write();
        }

    }

    d.run();
}

