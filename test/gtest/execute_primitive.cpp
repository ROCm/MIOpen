/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <miopen/filesystem.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/invoker.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/names.hpp>
#include <miopen/problem_description_base.hpp>
#include <miopen/solver.hpp>
#include <miopen/temp_file.hpp>

namespace fs = miopen::fs;

#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
#include <miopen/sqlite_db.hpp>
#endif

#include "get_handle.hpp"

#include <gtest/gtest_common.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <string>

struct TestResults
{
    bool regular_selected             = false;
    bool tunable_selected             = false;
    bool search_invoked               = false;
    bool get_default_perf_cfg_invoked = false;
    std::uint32_t total_get_solutions = 0;
    std::uint32_t perf_cfg_value      = 0;
};

struct TestProblemDescriptionTag
{
    // In real code this type should not have any data
    fs::path pdb_path;
    fs::path updb_path;
};

struct TestProblemDescription : miopen::ProblemDescriptionBase,
                                TestProblemDescriptionTag
#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
    ,
                                miopen::SQLiteSerializable<TestProblemDescription>
#endif
{
    std::uint32_t net_config;
    TestResults* test_results;

    auto MakeNetworkConfig() const -> miopen::NetworkConfig override
    {
        return miopen::NetworkConfig{std::to_string(net_config)};
    }

    void Serialize(std::ostream& stream) const { stream << net_config; }

#if MIOPEN_ENABLE_SQLITE && MIOPEN_USE_SQLITE_PERFDB
    static std::string table_name() { return "config"; }
#endif

    template <class TSelf>
    static void Visit(TSelf&& self, std::function<void(std::string, std::string)> visitor)
    {
    }

    template <class TSelf>
    static void Visit(TSelf&& self, std::function<void(std::int64_t, std::string)> visitor)
    {
        visitor(static_cast<std::int32_t>(self.net_config), "net_config");
    }

    template <class TSelf, class Visitor>
    static void VisitAll(TSelf&& self, Visitor visitor)
    {
        Visit(std::forward<TSelf>(self),
              [&](int64_t value, std::string name) { visitor(value, name); });
        Visit(std::forward<TSelf>(self),
              [&](std::string value, std::string name) { visitor(value, name); });
    }

    friend auto GetDb(const miopen::ExecutionContext&, const TestProblemDescriptionTag& problem)
        -> miopen::PerformanceDb
    {
        return {miopen::DbKinds::PerfDb, problem.pdb_path, problem.updb_path};
    }
};

struct TestPerfConfig : miopen::solver::PerfConfig
{
    static constexpr std::uint32_t default_value  = 1234;
    static constexpr std::uint32_t searched_value = 4321;

    std::uint32_t check_value;

    TestPerfConfig(std::uint32_t check_value_ = 0) : check_value(check_value_) {}

    void Serialize(std::ostream& stream) const override {}
    auto Deserialize(const std::string&) -> bool override { return true; }
};

static auto MakeNoopSolution() -> miopen::solver::ConvSolution
{
    miopen::solver::ConvSolution sol{};
    sol.invoker_factory = [](auto&&) { return [](auto&&, auto&&) {}; };
    return sol;
}

struct RegularTestSolver
    : miopen::solver::NonTunableSolverBase<miopen::ExecutionContext, TestProblemDescription>
{
    auto SolverDbId() const -> const std::string& override
    {
        return GetSolverDbId<RegularTestSolver>();
    }

    auto IsApplicable(const miopen::ExecutionContext&, const TestProblemDescription&) const
        -> bool override
    {
        return true;
    }

    auto GetSolution(const miopen::ExecutionContext&, const TestProblemDescription& problem) const
        -> miopen::solver::ConvSolution override
    {
        problem.test_results->regular_selected = true;
        problem.test_results->total_get_solutions++;
        return MakeNoopSolution();
    }
};

struct TunableTestSolver : miopen::solver::TunableSolverMixin<miopen::ExecutionContext,
                                                              TestProblemDescription,
                                                              TestPerfConfig>
{
    auto SolverDbId() const -> const std::string& override
    {
        return GetSolverDbId<TunableTestSolver>();
    }
    auto IsApplicable(const miopen::ExecutionContext&, const TestProblemDescription&) const
        -> bool override
    {
        return true;
    }

    auto GetDefaultPerformanceConfig(const miopen::ExecutionContext&,
                                     const TestProblemDescription& problem) const
        -> TestPerfConfig override
    {
        problem.test_results->get_default_perf_cfg_invoked = true;
        return {TestPerfConfig::default_value};
    }

    auto IsValidPerformanceConfig(const miopen::ExecutionContext&,
                                  const TestProblemDescription&,
                                  const TestPerfConfig&) const -> bool override
    {
        return true;
    }

    auto Search(const miopen::ExecutionContext&,
                const TestProblemDescription& problem,
                const miopen::AnyInvokeParams&) const -> TestPerfConfig override
    {
        problem.test_results->search_invoked = true;
        return {TestPerfConfig::searched_value};
    }

    auto GetSolution(const miopen::ExecutionContext&,
                     const TestProblemDescription& problem,
                     const TestPerfConfig& raw_perf_cfg) const
        -> miopen::solver::ConvSolution override
    {
        const auto& perf_cfg = dynamic_cast<const TestPerfConfig&>(raw_perf_cfg);

        problem.test_results->perf_cfg_value   = perf_cfg.check_value;
        problem.test_results->tunable_selected = true;
        problem.test_results->total_get_solutions++;
        return MakeNoopSolution();
    }
};

struct ExecutePrimitive : testing::Test
{
};

auto CallExecutePrimitive(const miopen::ExecutionContext& ctx) -> TestResults
{
    static std::uint32_t call_id = 0; // this is to distinguish between test cases

    miopen::TempFile pdb{"test_pdb"};
    miopen::TempFile updb{"test_fdb"};

    TestResults test_results{};
    TestProblemDescription problem{};
    problem.test_results = &test_results;
    problem.pdb_path     = pdb;
    problem.updb_path    = updb;
    problem.net_config   = call_id++;

    constexpr auto solvers =
        miopen::solver::SolverContainer<TunableTestSolver, RegularTestSolver>{};
    solvers.ExecutePrimitive(ctx, problem, miopen::AlgorithmName{"test::algo"}, {});

    return test_results;
}

TEST(CPU_ExecutePrimitive_NONE, NoParams)
{
    auto ctx = miopen::ExecutionContext{&get_handle()};

    auto results = CallExecutePrimitive(ctx);

    ASSERT_EQ(results.total_get_solutions, 1);
    ASSERT_FALSE(results.regular_selected);
    ASSERT_TRUE(results.tunable_selected);
    ASSERT_TRUE(results.get_default_perf_cfg_invoked);
    ASSERT_FALSE(results.search_invoked);
    ASSERT_EQ(results.perf_cfg_value, TestPerfConfig::default_value);
}

TEST(CPU_ExecutePrimitiveSearchUpdate_NONE, SearchUpdate)
{
    auto ctx      = miopen::ExecutionContext{&get_handle()};
    ctx.do_search = true;
    ctx.db_update = true;

    auto results = CallExecutePrimitive(ctx);

    ASSERT_EQ(results.total_get_solutions, 1);
    ASSERT_FALSE(results.regular_selected);
    ASSERT_TRUE(results.tunable_selected);
    ASSERT_FALSE(results.get_default_perf_cfg_invoked);
    ASSERT_TRUE(results.search_invoked);
    ASSERT_EQ(results.perf_cfg_value, TestPerfConfig::searched_value);
}
