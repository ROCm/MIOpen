
#include "args.hpp"
#include "tensor_holder.hpp"
#include "network_data.hpp"

template<class F, class Base>
struct test_driver : Base
{
    struct empty_args
    {
        template<class T>
        void operator()(bool& b, T&& x) const
        {
            b = b && x.empty();
        }
    };

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

        bool empty = true;
        this->visit(std::bind(empty_args{}, std::ref(empty), std::placeholders::_1));

        if (empty)
        {
        #if MLOPEN_TEST_ALL
            printf("verify_all\n");
        #ifdef NDEBUG
            this->template generate_all<float>(F{}, g0, g1, g_id, g);
        #else
            this->template generate_all<float>(F{}, g);
        #endif
        #else
            printf("verify_one\n");
            this->template generate_default<float>(F{}, g);
        #endif
        }
        else
        {
            this->template generate_one<float>(F{}, g);
        }
    }
};

struct unary_input
{
    std::vector<int> input_dims;

    template<class V>
    void visit(V v)
    {
        v(input_dims, "--input");
    }

    template<class T, class F, class... Gs>
    void generate_all(F f, Gs... gs)
    {
        generate_unary_all<T>(f, gs...);
    }

    template<class T, class F, class G>
    void generate_default(F f, G g)
    {
        generate_unary_one<T>(f, {16, 32, 8, 8}, g);
    }

    template<class T, class F, class G>
    void generate_one(F f, G g)
    {
        std::reverse(input_dims.begin(), input_dims.end());
        generate_unary_one<T>(f, input_dims, g);
    }
};

struct binary_input
{
    std::vector<int> input_dims;
    std::vector<int> weights_dims;

    template<class V>
    void visit(V v)
    {
        v(input_dims, "--input");
        v(weights_dims, "--weights");
    }

    template<class T, class F, class... Gs>
    void generate_all(F f, Gs... gs)
    {
        generate_binary_all<T>(f, gs...);
    }

    template<class T, class F, class G>
    void generate_default(F f, G g)
    {
        generate_binary_one<T>(f, {16, 32, 8, 8}, {64, 32, 5, 5}, g);
    }

    template<class T, class F, class G>
    void generate_one(F f, G g)
    {
        std::reverse(input_dims.begin(), input_dims.end());
        std::reverse(weights_dims.begin(), weights_dims.end());
        generate_binary_one<T>(f, input_dims, weights_dims, g);
    }
};


template<class F, class Base>
void test_drive(int argc, const char *argv[])
{
    args::parse<test_driver<F, Base>>(argc, argv);
}


