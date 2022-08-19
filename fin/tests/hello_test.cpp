#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/solver_id.hpp>

// Demonstrate some basic assertions.
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
}
