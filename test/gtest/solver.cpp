// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <miopen/find_solution.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/temp_file.hpp>

#include "random.hpp"
#include "../get_handle.hpp"

namespace {
class TrivialTestSolver final : public miopen::solver::conv::ConvSolver
{
public:
    static const char* FileName() { return "TrivialTestSolver"; }

    const std::string& SolverDbId() const override { return GetSolverDbId<TrivialTestSolver>(); }

    bool IsApplicable(const miopen::ExecutionContext&,
                      const miopen::conv::ProblemDescription& problem) const override
    {
        return problem.GetInWidth() == 1;
    }

    miopen::solver::ConvSolution GetSolution(const miopen::ExecutionContext&,
                                             const miopen::conv::ProblemDescription&) const override
    {
        miopen::solver::ConvSolution ret;
        miopen::solver::KernelInfo kernel;

        kernel.kernel_file  = FileName();
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }
};

struct TestConfig : miopen::solver::PerfConfigBase<TestConfig>
{
    std::string str;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.str, "str");
    }
};

class SearchableTestSolver final : public miopen::solver::conv::ConvTunableSolver<TestConfig>
{
public:
    static int searches_done() { return _serches_done; }
    static const char* FileName() { return "SearchableTestSolver"; }
    static const char* NoSearchFileName() { return "SearchableTestSolver.NoSearch"; }

    const std::string& SolverDbId() const override { return GetSolverDbId<SearchableTestSolver>(); }

    bool IsApplicable(const miopen::ExecutionContext&,
                      const miopen::conv::ProblemDescription&) const override
    {
        return true;
    }

    TestConfig GetDefaultPerformanceConfig(const miopen::ExecutionContext&,
                                           const miopen::conv::ProblemDescription&) const override
    {
        TestConfig config{};
        config.str = NoSearchFileName();
        return config;
    }

    bool IsValidPerformanceConfig(const miopen::ExecutionContext&,
                                  const miopen::conv::ProblemDescription&,
                                  const TestConfig&) const override
    {
        return true;
    }

    TestConfig Search(const miopen::ExecutionContext&,
                      const miopen::conv::ProblemDescription&,
                      const miopen::AnyInvokeParams&) const override
    {
        TestConfig config;
        config.str = FileName();
        _serches_done++;
        return config;
    }

    miopen::solver::ConvSolution GetSolution(const miopen::ExecutionContext&,
                                             const miopen::conv::ProblemDescription&,
                                             const TestConfig& config) const override
    {

        miopen::solver::ConvSolution ret;
        miopen::solver::KernelInfo kernel;

        kernel.kernel_file  = config.str;
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }

private:
    static int _serches_done; // NOLINT (cppcoreguidelines-avoid-non-const-global-variables)
};

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
int SearchableTestSolver::_serches_done = 0;

static miopen::solver::ConvSolution FindSolution(const miopen::ExecutionContext& ctx,
                                                 const miopen::conv::ProblemDescription& problem,
                                                 const miopen::fs::path& db_path)
{
    miopen::PlainTextDb db(miopen::DbKinds::PerfDb, db_path);

    const auto solvers = miopen::solver::SolverContainer<TrivialTestSolver, SearchableTestSolver>{};

    return solvers.SearchForAllSolutions(ctx, problem, db, {}, 1).front();
}

template <class TInstance>
class StaticContainer
{
public:
    inline static TInstance& Instance()
    {
        // NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
        static TInstance data{};
        return data;
    }
};

struct TestParams
{
    std::string expected_kernel;
    std::vector<size_t> in;
    std::function<void(miopen::ExecutionContext&)> context_filter = [](miopen::ExecutionContext&) {
    };
};

using TestCase =
    std::tuple<std::vector<TestParams>,
               std::vector<TestParams>>; // pre-check parameters and post-check parameters

auto GenCases()
{
    return std::vector<TestCase>{std::make_tuple<std::vector<TestParams>, std::vector<TestParams>>(
        {TestParams{TrivialTestSolver::FileName(), {1, 1, 1, 1}, [](miopen::ExecutionContext&) {}},
         TestParams{TrivialTestSolver::FileName(),
                    {1, 1, 1, 1},
                    [](miopen::ExecutionContext& c) { c.do_search = true; }},
         TestParams{SearchableTestSolver::NoSearchFileName(),
                    {1, 1, 1, 2},
                    [](miopen::ExecutionContext& c) { c.do_search = false; }},
         TestParams{SearchableTestSolver::FileName(),
                    {1, 1, 1, 2},
                    [](miopen::ExecutionContext& c) { c.do_search = true; }}},
        {TestParams{
             SearchableTestSolver::FileName(), {1, 1, 1, 2}, [](miopen::ExecutionContext&) {}},
         TestParams{SearchableTestSolver::FileName(),
                    {1, 1, 1, 2},
                    [](miopen::ExecutionContext& c) { c.do_search = true; }}})};
}

auto GetCases()
{
    static auto cases = GenCases();
    return cases;
}

} // namespace

class CPU_SolverTest_NONE : public ::testing::TestWithParam<TestCase>
{
public:
    void SetUp() override { prng::reset_seed(); }

    void Run()
    {
        auto [pre_check_param, post_check_param] = GetParam();
        const miopen::TempFile db_path("miopen.tests.solver");

        for(auto const& param : pre_check_param)
        {
            ConstructTest(db_path, param.expected_kernel.c_str(), param.in, param.context_filter);
        }

        const auto& searchable_solver = StaticContainer<const SearchableTestSolver>::Instance();
        const auto searches           = SearchableTestSolver::searches_done();

        // Should read in both cases: result is already in DB, solver is searchable.

        for(auto const& param : post_check_param)
        {
            ConstructTest(db_path, param.expected_kernel.c_str(), param.in, param.context_filter);
        }

        // Checking no more searches were done.
        EXPECT_EQ(searches, searchable_solver.searches_done());
    }

protected:
    void ConstructTest(
        const miopen::fs::path& db_path,
        const char* expected_kernel,
        const std::vector<size_t>& in,
        const std::function<void(miopen::ExecutionContext&)>& context_filler =
            [](miopen::ExecutionContext&) {}) const
    {
        const auto problem =
            miopen::conv::ProblemDescription{miopen::TensorDescriptor{miopenFloat, in},
                                             miopen::TensorDescriptor{miopenFloat, in},
                                             miopen::TensorDescriptor{miopenFloat, in},
                                             miopen::ConvolutionDescriptor{},
                                             miopen::conv::Direction::Forward};
        auto ctx = miopen::ExecutionContext{};
        ctx.SetStream(&get_handle());
        context_filler(ctx);

        const auto sol = FindSolution(ctx, problem, db_path);

        EXPECT_TRUE(sol.construction_params.size() > 0);
        EXPECT_EQ(sol.construction_params[0].kernel_file, expected_kernel);
    }
};

TEST_P(CPU_SolverTest_NONE, TestSolver) { Run(); }

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_SolverTest_NONE,
                         ::testing::ValuesIn(GetCases()),
                         [](auto const&) { return "SearchesDoneTest"; });
