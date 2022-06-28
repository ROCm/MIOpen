#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>

//Todo: Reduced duplicated code from cache.cpp
//E.g. declare extern check_cache() in cache. cpp and declare dependency on cache.cpp?

// Demonstrate some basic assertions.
//Todo: Include a simple task
//TEST_F()?: use the same data configuration for multiple tests

#ifdef __cplusplus
extern "C" {
#endif

extern int check_cache(void);

#ifdef __cplusplus
}
#endif

TEST(HelloTest, BasicAssertions)
{

    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);

    // Check if we can access MIOpen internals
    auto idx = 0;
    for(const auto& solver_id :
        miopen::solver::GetSolversByPrimitive(miopen::solver::Primitive::Convolution))
    {
        std::ignore = solver_id;
        ++idx;
    }
    EXPECT_GT(idx, 0);

    //Test Case for cache.cpp
    EXPECT_EQ(check_cache(), 0);
}

