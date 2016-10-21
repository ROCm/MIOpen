
#include "args.hpp"
#include "tensor_holder.hpp"
#include "network_data.hpp"

template<class F>
struct test_driver
{
    std::vector<int> input_dims;
    std::vector<int> weights_dims;

    template<class V>
    void visit(V v)
    {
        v(input_dims, "--input");
        v(weights_dims, "--weights");
    }

    void run()
    {
        auto g0 = [](int, int, int, int) { return 0; };
        auto g1 = [](int, int, int, int) { return 1; };
        auto g_id = [](int, int, int h, int w) { return h == w ? 1 : 0; };
        auto g = [](int n, int c, int h, int w)
        {
            double x = (547*n+701*c+877*h+1049*w+173)%1223;
            return x/691.0;
        };
        (void)g0;
        (void)g1;
        (void)g_id;
        (void)g;

        if (input_dims.empty() && weights_dims.empty())
        {
        #if MLOPEN_TEST_ALL
            printf("verify_all\n");
            generate_all<float, network_visitor>(F{}, g0, g1, g_id, g);
        #else
            printf("verify_one\n");
            generate_one<float>(F{}, {16, 32, 8, 8}, {64, 32, 5, 5}, g);
        #endif
        }
        else
        {
            generate_one<float>(F{}, input_dims, weights_dims, g);
        }
    }
};

template<class F>
void test_drive(int argc, const char *argv[])
{
    args::parse<test_driver<F>>(argc, argv);
}


