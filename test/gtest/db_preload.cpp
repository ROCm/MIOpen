/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include <chrono>
#include <future>
#include <miopen/db_preload.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/handle.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/tmp_dir.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <mutex>
#include <ostream>
#include <shared_mutex>
#include <thread>

namespace {
template <class Waiter = void (*)()>
void Produce(
    const miopen::fs::path& filename, miopen::DbPreloadStates& states, Waiter&& waiter = []() {})
{
    miopen::TryStartPreloadingDbs(
        [&]() {
            miopen::StartPreloadingDb(
                filename,
                [&](const miopen::fs::path& path) -> miopen::PreloadedDb {
                    auto ret =
                        std::make_unique<miopen::ReadonlyRamDb>(miopen::DbKinds::PerfDb, path);
                    ret->Prefetch(false);
                    waiter();
                    return ret;
                },
                states);
        },
        states);
}

template <class PreWaiter = void (*)(), class PostWaiter = void (*)()>
void Consume(
    int consumers_count,
    const miopen::fs::path& filename,
    miopen::ReadonlyRamDb::Instances& dbs,
    PreWaiter&& pre_waiter   = []() {},
    PostWaiter&& post_waiter = []() {})
{
    std::vector<std::thread> consumers;
    consumers.reserve(consumers_count);

    std::shared_mutex preloaded;
    auto owner_lock = std::unique_lock<std::shared_mutex>(preloaded);

    for(auto i = 0; i < consumers_count; ++i)
    {
        consumers.emplace_back([&]() {
            auto consumer_lock = std::shared_lock<std::shared_mutex>(preloaded);
            std::ignore =
                miopen::ReadonlyRamDb::GetCached(miopen::DbKinds::KernelDb, filename, false, dbs);
        });
    }

    pre_waiter();
    owner_lock.unlock();
    post_waiter();

    for(auto&& consumer : consumers)
        consumer.join();
}
} // namespace

TEST(SmokeCPUDbPreloadTrivialNONE, Basic)
{
    miopen::DbPreloadStates states;
    miopen::TmpDir dir;
    auto filename = dir.path / "db.txt";

    Produce(filename, states);
    ASSERT_NE(nullptr, miopen::GetPreloadedReadonlyRamDb(filename, states));
}

TEST(SmokeCPUDbPreloadTrivialNONE, Fallback)
{
    miopen::DbPreloadStates states;
    miopen::TmpDir dir;
    auto filename = dir.path / "db.txt";

    ASSERT_EQ(nullptr, miopen::GetPreloadedReadonlyRamDb(filename, states));
}

struct TestCase
{
    int consumers;
};

std::ostream& operator<<(std::ostream& s, const TestCase& test_case)
{
    s << "consumers: " << test_case.consumers;
    return s;
}

class CPUDbPreloadNONE : public testing::TestWithParam<TestCase>
{
};

TEST_P(CPUDbPreloadNONE, Sequential)
{
    auto param = GetParam();

    miopen::DbPreloadStates states;
    miopen::ReadonlyRamDb::Instances dbs{{}, &states};
    miopen::TmpDir dir;
    auto filename = dir.path / "db.txt";

    Consume(param.consumers, filename, dbs);

    std::promise<void> promise;
    auto future = promise.get_future();

    Produce(filename, states, [&]() {
        promise.set_value(); // promise.set_value_at_thread_exit();
    });

    Consume(param.consumers, filename, dbs, [&]() { future.get(); });
}

TEST_P(CPUDbPreloadNONE, Parallel)
{
    auto param = GetParam();

    miopen::DbPreloadStates states;
    miopen::ReadonlyRamDb::Instances dbs{{}, &states};
    miopen::TmpDir dir;
    auto filename = dir.path / "db.txt";

    std::shared_mutex producer_mutex;
    auto producer_mutex_owner_lock = std::unique_lock<std::shared_mutex>(producer_mutex);

    Produce(filename, states, [&]() {
        // just synchronisation
        std::ignore = std::shared_lock<std::shared_mutex>(producer_mutex);
    });

    Consume(
        param.consumers,
        filename,
        dbs,
        []() {},
        [&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
            producer_mutex_owner_lock.unlock();
        });
}

INSTANTIATE_TEST_SUITE_P(Smoke, CPUDbPreloadNONE, testing::Values(TestCase{1}, TestCase{4}));
