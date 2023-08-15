/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_TEST_RANDOM_HPP
#define GUARD_MIOPEN_TEST_RANDOM_HPP

#include <miopen/env.hpp>
#include <iostream>
#include <cstdlib>
#include <random>

std::minstd_rand& get_minstd_gen()
{
    static thread_local std::minstd_rand minstd_gen{[]() {
        auto external_seed = miopen::EnvvarValue("MIOPEN_DRIVER_PRNG_SEED", 100500);

        auto seed = external_seed == 0
                        ? std::random_device{}()
                        : static_cast<std::random_device::result_type>(external_seed);
        std::cout << "Random seed: " << seed << "\n";
        return seed;
    }()};

    return minstd_gen;
}

bool use_legacy_prng()
{
    static bool legacy_prng = miopen::IsEnvvarValueEnabled("MIOPEN_USE_LEGACY_PRNG");
    return legacy_prng;
}

// template <typename T>
// inline T FRAND()
// {
//
//     return use_legacy_prng() ? static_cast<T>(rand() / (static_cast<double>(RAND_MAX)))
//                              : std::generate_canonical<T, 1>(get_minstd_gen());
// }

/// Basically, this is a copy of driver/random.hpp. Why:
/// We want to have the same functionality as implemented in driver/random.hpp,
/// But we want this functionality to be independent, so changes in tests won't affect the driver
/// and vice versa. This independency could be important, because, for example, the driver
/// implements its own cache of verification data and change or GET_RAND() would break it.

inline int GET_RAND() { return use_legacy_prng() ? rand() : get_minstd_gen()(); }

// template <typename T>
// inline T RAN_GEN(T A, T B)
// {
//     return static_cast<T>(FRAND<double>() * (B - A)) + A;
// }

#endif
