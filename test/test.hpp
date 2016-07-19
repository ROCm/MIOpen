#include <cassert>
#include <cstdio>
#include <cstdlib>

void failed(const char * msg, const char* file, int line)
{
    printf("FAILED: %s: %s:%i\n", msg, file, line);
    std::abort();
}

#define CHECK(...) if (!(__VA_ARGS__)) failed(#__VA_ARGS__, __FILE__, __LINE__)

template<class T>
void run_test()
{
    T t = {};
    t.run();
}
