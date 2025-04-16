#include <gtest/gtest.h>

#include <miopen/miopen.h>
#include <miopen/errors.hpp>

#include "gtest_common.hpp"
#include "unit_conv_solver.hpp"

using Tol = miopen::unit_tests::Tolerances;

namespace {
std::vector<Gpu> gpus{Gpu::gfx900,
                      Gpu::gfx906,
                      Gpu::gfx908,
                      Gpu::gfx90A,
                      Gpu::gfx94X,
                      Gpu::gfx950,
                      Gpu::gfx103X,
                      Gpu::gfx110X,
                      Gpu::gfx120X};

std::vector<miopenDataType_t> types{miopenHalf,
                                    miopenFloat,
                                    miopenInt32,
                                    miopenInt8,
                                    miopenBFloat16,
                                    miopenDouble,
                                    miopenFloat8_fnuz,
                                    miopenBFloat8_fnuz,
                                    miopenInt64};

float GetValue(Gpu gpu, miopenDataType_t type)
{
    Tol def;
    float val = 0.5 / static_cast<float>(gpu) + static_cast<float>(type);
    EXPECT_NE(val, def.Get(Gpu::gfx900, miopenFloat));
    return val;
}
} // namespace

TEST(CPU_UnitConvSolverToleranceTests_NONE, testThrows)
{
    Tol tol;

    EXPECT_NO_THROW(tol.Get(Gpu::gfx90A, miopenHalf));
    EXPECT_ANY_THROW(tol.Get(Gpu::gfx908 | Gpu::gfx90A, miopenHalf));
    EXPECT_ANY_THROW(tol.Get(static_cast<Gpu>(static_cast<int>(Gpu::gfxLast) << 1), miopenHalf));
}

TEST(CPU_UnitConvSolverToleranceTests_NONE, testSetAllUnique)
{
    Tol tol;

    for(auto g : gpus)
    {
        for(auto t : types)
        {
            auto val = GetValue(g, t);
            EXPECT_NE(val, tol.Get(g, t));
            tol.Set(g, t, val);
        }
    }

    for(auto g : gpus)
    {
        for(auto t : types)
        {
            std::vector<std::pair<Gpu, miopenDataType_t>> matches;
            auto val = GetValue(g, t);

            for(auto gg : gpus)
            {
                for(auto tt : types)
                {
                    if(val == tol.Get(gg, tt))
                    {
                        matches.push_back({gg, tt});
                    }
                }
            }

            EXPECT_EQ(1, matches.size());
        }
    }
}

TEST(CPU_UnitConvSolverToleranceTests_NONE, testSetMulti)
{
    Tol tol;

    float test_val = tol.Get(Gpu::gfx94X, miopenHalf);

    const float gfx103X_miopenFloat = (test_val += 1.0);
    tol.Set(Gpu::gfx103X, miopenFloat, gfx103X_miopenFloat);
    EXPECT_EQ(gfx103X_miopenFloat, tol.Get(Gpu::gfx103X, miopenFloat));

    const float gfx103X_gfx120X_miopenFloat = (test_val += 1.0);
    tol.Set(Gpu::gfx103X | Gpu::gfx120X, miopenFloat, gfx103X_gfx120X_miopenFloat);
    EXPECT_EQ(gfx103X_gfx120X_miopenFloat, tol.Get(Gpu::gfx103X, miopenFloat));
    EXPECT_EQ(gfx103X_gfx120X_miopenFloat, tol.Get(Gpu::gfx120X, miopenFloat));

    const float gfx103X_gfx120X_miopenHalf = (test_val += 1.0);
    tol.Set(Gpu::gfx103X | Gpu::gfx120X, miopenHalf, gfx103X_gfx120X_miopenHalf);
    EXPECT_EQ(gfx103X_gfx120X_miopenHalf, tol.Get(Gpu::gfx103X, miopenHalf));
    EXPECT_EQ(gfx103X_gfx120X_miopenHalf, tol.Get(Gpu::gfx120X, miopenHalf));
    EXPECT_EQ(gfx103X_gfx120X_miopenFloat, tol.Get(Gpu::gfx103X, miopenFloat));
    EXPECT_EQ(gfx103X_gfx120X_miopenFloat, tol.Get(Gpu::gfx120X, miopenFloat));

    const float gfxX4_miopenHalf = (test_val += 1.0);
    tol.Set(Gpu::gfx900 | Gpu::gfx908 | Gpu::gfx94X | Gpu::gfx120X, miopenHalf, gfxX4_miopenHalf);
    EXPECT_EQ(gfxX4_miopenHalf, tol.Get(Gpu::gfx900, miopenHalf));
    EXPECT_EQ(gfxX4_miopenHalf, tol.Get(Gpu::gfx908, miopenHalf));
    EXPECT_EQ(gfxX4_miopenHalf, tol.Get(Gpu::gfx94X, miopenHalf));
    EXPECT_EQ(gfx103X_gfx120X_miopenHalf, tol.Get(Gpu::gfx103X, miopenHalf));
    EXPECT_EQ(gfxX4_miopenHalf, tol.Get(Gpu::gfx120X, miopenHalf));
    EXPECT_EQ(gfx103X_gfx120X_miopenFloat, tol.Get(Gpu::gfx103X, miopenFloat));
    EXPECT_EQ(gfx103X_gfx120X_miopenFloat, tol.Get(Gpu::gfx120X, miopenFloat));

    const float all = (test_val += 1.0);
    tol.Set(Gpu::All, miopenFloat, all);
    EXPECT_EQ(gfxX4_miopenHalf, tol.Get(Gpu::gfx900, miopenHalf));
    EXPECT_EQ(gfxX4_miopenHalf, tol.Get(Gpu::gfx908, miopenHalf));
    EXPECT_EQ(gfxX4_miopenHalf, tol.Get(Gpu::gfx94X, miopenHalf));
    EXPECT_EQ(gfx103X_gfx120X_miopenHalf, tol.Get(Gpu::gfx103X, miopenHalf));
    EXPECT_EQ(gfxX4_miopenHalf, tol.Get(Gpu::gfx120X, miopenHalf));
    EXPECT_EQ(all, tol.Get(Gpu::gfx900, miopenFloat));
    EXPECT_EQ(all, tol.Get(Gpu::gfx906, miopenFloat));
    EXPECT_EQ(all, tol.Get(Gpu::gfx908, miopenFloat));
    EXPECT_EQ(all, tol.Get(Gpu::gfx90A, miopenFloat));
    EXPECT_EQ(all, tol.Get(Gpu::gfx94X, miopenFloat));
    EXPECT_EQ(all, tol.Get(Gpu::gfx950, miopenFloat));
    EXPECT_EQ(all, tol.Get(Gpu::gfx103X, miopenFloat));
    EXPECT_EQ(all, tol.Get(Gpu::gfx110X, miopenFloat));
    EXPECT_EQ(all, tol.Get(Gpu::gfx120X, miopenFloat));
}
