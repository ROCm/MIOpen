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

#include <miopen/db_preload.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/handle.hpp>
#include <miopen/readonlyramdb.hpp>
#include <miopen/tmp_dir.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <ostream>
#include <shared_mutex>
#include <string>
#include <thread>

namespace {
template <class Filenames, class Waiter = void (*)(const miopen::fs::path&)>
void Produce(
    const Filenames& filenames, miopen::DbPreloadStates& states, Waiter&& waiter = [](auto) {})
{
    states.TryStartPreloadingDbs([&]() {
        for(const auto& filename : filenames)
        {
            states.StartPreloadingDb(
                filename,
                [waiter = std::move(waiter)](const miopen::stop_token& stop,
                                             const miopen::fs::path& path) -> miopen::PreloadedDb {
                    auto ret =
                        std::make_unique<miopen::ReadonlyRamDb>(miopen::DbKinds::PerfDb, path);
                    ret->Prefetch(false, stop);
                    waiter(path);
                    return ret;
                });
        }
    });
}

template <class Filenames, class PreWaiter = void (*)(), class PostWaiter = void (*)()>
void Consume(
    int consumer_count,
    const Filenames& filenames,
    miopen::ReadonlyRamDb::Instances& dbs,
    PreWaiter&& pre_waiter   = []() {},
    PostWaiter&& post_waiter = []() {})
{
    if(filenames.size() != 1)
        ASSERT_LE(consumer_count, filenames.size());

    auto filename = [&](auto i) -> const auto&
    {
        return filenames.size() == 1 ? filenames[0] : filenames[i];
    };

    std::vector<std::thread> consumers;
    consumers.reserve(consumer_count);

    std::shared_mutex preloaded;
    auto owner_lock = std::unique_lock<std::shared_mutex>(preloaded);

    for(auto i = 0; i < consumer_count; ++i)
    {
        consumers.emplace_back([&dbs, &preloaded, i, filename = std::move(filename)]() {
            // just synchronisation
            std::ignore = std::shared_lock<std::shared_mutex>(preloaded);
            std::ignore = miopen::ReadonlyRamDb::GetCached(
                miopen::DbKinds::KernelDb, filename(i), false, dbs);
        });
    }

    pre_waiter();
    owner_lock.unlock();
    post_waiter();

    for(auto&& consumer : consumers)
        consumer.join();
}

auto GenerateFilenames(int producer_count, const miopen::fs::path& directory)
{
    std::vector<miopen::fs::path> filenames;
    filenames.reserve(producer_count);
    for(auto i = 0; i < producer_count; ++i)
        filenames.emplace_back(directory / ("db" + std::to_string(i) + ".txt"));
    return filenames;
}
} // namespace

TEST(CPU_Smoke_DbPreloadTrivial_NONE, Cleanup)
{
    miopen::DbPreloadStates states;
    const miopen::TmpDir dir;
    const auto filenames = GenerateFilenames(1, dir.path);

    Produce(filenames, states);
}

TEST(CPU_Smoke_DbPreloadTrivial_NONE, Stop)
{
    bool stoped = false;

    {
        miopen::DbPreloadStates states;
        const miopen::TmpDir dir;
        const auto filenames = GenerateFilenames(1, dir.path);

        states.TryStartPreloadingDbs([&]() {
            auto preloader = [&stoped](const miopen::stop_token& stop,
                                       const miopen::fs::path& path) -> miopen::PreloadedDb {
                auto ret = std::make_unique<miopen::ReadonlyRamDb>(miopen::DbKinds::PerfDb, path);
                ret->Prefetch(false, stop);
                while(!stop.stop_requested())
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                stoped = true;
                return ret;
            };

            states.StartPreloadingDb(filenames[0], std::move(preloader));
        });
    }

    ASSERT_TRUE(stoped);
}

TEST(CPU_Smoke_DbPreloadTrivial_NONE, Basic)
{
    miopen::DbPreloadStates states;
    const miopen::TmpDir dir;
    const auto filenames = GenerateFilenames(1, dir.path);

    Produce(filenames, states);
    ASSERT_NE(nullptr, states.GetPreloadedReadonlyRamDb(filenames[0]));
}

TEST(CPU_Smoke_DbPreloadTrivial_NONE, Full)
{
    auto states = std::make_shared<miopen::DbPreloadStates>();
    miopen::ReadonlyRamDb::Instances dbs{{}, states};
    const miopen::TmpDir dir;
    const auto filenames = GenerateFilenames(1, dir.path);

    Produce(filenames, *states);
    std::ignore =
        miopen::ReadonlyRamDb::GetCached(miopen::DbKinds::KernelDb, filenames[0], false, dbs);
}

TEST(CPU_Smoke_DbPreloadTrivial_NONE, Fallback)
{
    miopen::DbPreloadStates states;
    const miopen::TmpDir dir;
    const auto filenames = GenerateFilenames(1, dir.path);

    ASSERT_EQ(nullptr, states.GetPreloadedReadonlyRamDb(filenames[0]));
}

struct TestCase
{
    int consumers;
    int producers;

    friend std::ostream& operator<<(std::ostream& s, const TestCase& test_case)
    {
        s << "consumers: " << test_case.consumers << ", ";
        s << "producers: " << test_case.producers;
        return s;
    }

    std::string ToTestName() const
    {
        return std::to_string(consumers) + "x" + std::to_string(producers);
    }
};

class CPU_DbPreload_NONE : public testing::TestWithParam<TestCase>
{
};

TEST_P(CPU_DbPreload_NONE, Sequential)
{
    auto param = GetParam();

    auto states = std::make_shared<miopen::DbPreloadStates>();
    miopen::ReadonlyRamDb::Instances dbs{{}, states};
    const miopen::TmpDir dir;
    const auto filenames = GenerateFilenames(param.producers, dir.path);

    Consume(param.consumers, filenames, dbs);

    std::map<miopen::fs::path, std::promise<void>> promises;
    for(auto&& filename : filenames)
        promises.emplace(filename, std::promise<void>());

    Produce(filenames, *states, [&](auto&& path) { promises.at(path).set_value(); });

    Consume(param.consumers, filenames, dbs, [&]() {
        for(auto&& pair : promises)
            pair.second.get_future().get();
    });
}

TEST_P(CPU_DbPreload_NONE, Parallel)
{
    auto param = GetParam();

    auto states = std::make_shared<miopen::DbPreloadStates>();
    miopen::ReadonlyRamDb::Instances dbs{{}, states};
    const miopen::TmpDir dir;
    const auto filenames = GenerateFilenames(param.consumers, dir.path);

    std::shared_mutex producer_mutex;
    auto producer_mutex_owner_lock = std::unique_lock<std::shared_mutex>(producer_mutex);

    Produce(filenames, *states, [&](auto) {
        // just synchronisation
        std::ignore = std::shared_lock<std::shared_mutex>(producer_mutex);
    });

    Consume(
        param.consumers,
        filenames,
        dbs,
        []() {},
        [&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds{10});
            producer_mutex_owner_lock.unlock();
        });
}

INSTANTIATE_TEST_SUITE_P(
    Smoke,
    CPU_DbPreload_NONE,
    testing::Values(TestCase{1, 1}, TestCase{4, 1}, TestCase{1, 4}, TestCase{4, 4}),
    [](const testing::TestParamInfo<TestCase>& info) { return info.param.ToTestName(); });
