/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <miopen/convolution.hpp>
#include <miopen/db.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/invoke_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/solver.hpp>
#include <miopen/temp_file.hpp>

#include <cstdlib>
#include <functional>
#include <sstream>
#include <typeinfo>

#include "get_handle.hpp"
#include "test.hpp"

namespace miopen {
namespace tests {
class TrivialTestSolver final : public solver::ConvSolver
{
public:
    static const char* FileName() { return "TrivialTestSolver"; }

    const std::string& SolverDbId() const override { return GetSolverDbId<TrivialTestSolver>(); }

    bool IsApplicable(const ConvolutionContext& context) const override
    {
        return context.in_width == 1;
    }

    solver::ConvSolution GetSolution(const ConvolutionContext&) const
    {
        solver::ConvSolution ret;
        solver::KernelInfo kernel;

        kernel.kernel_file  = FileName();
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }
};

struct TestConfig : solver::Serializable<TestConfig>
{
    std::string str;

    template <class Self, class F>
    static void Visit(Self&& self, F f)
    {
        f(self.str, "str");
    }
};

class SearchableTestSolver final : public solver::ConvTunableSolver<TestConfig>
{
public:
    static int searches_done() { return _serches_done; }
    static const char* FileName() { return "SearchableTestSolver"; }
    static const char* NoSearchFileName() { return "SearchableTestSolver.NoSearch"; }

    const std::string& SolverDbId() const override { return GetSolverDbId<SearchableTestSolver>(); }

    bool IsApplicable(const ConvolutionContext& context) const override
    {
        std::ignore = context;
        return true;
    }

    TestConfig GetDefaultPerformanceConfig(const ConvolutionContext&) const override
    {
        TestConfig config{};
        config.str = NoSearchFileName();
        return config;
    }

    bool IsValidPerformanceConfig(const ConvolutionContext&, const TestConfig&) const override
    {
        return true;
    }

    TestConfig Search(const ConvolutionContext&, const AnyInvokeParams&) const override
    {
        TestConfig config;
        config.str = FileName();
        _serches_done++;
        return config;
    }

    solver::ConvSolution GetSolution(const ConvolutionContext&,
                                     const TestConfig& config) const override
    {

        solver::ConvSolution ret;
        solver::KernelInfo kernel;

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

static solver::ConvSolution FindSolution(const ConvolutionContext& ctx, const std::string& db_path)
{
    PlainTextDb db(db_path);

    const auto solvers = solver::SolverContainer<TrivialTestSolver, SearchableTestSolver>{};

    return solvers.SearchForAllSolutions(ctx, db, {}, 1).front();
}

class SolverTest
{
public:
    void Run() const
    {
        const TempFile db_path("miopen.tests.solver");

        ConstructTest(db_path, TrivialTestSolver::FileName(), {0, 0, 0, 1});

        ConstructTest(db_path,
                      TrivialTestSolver::FileName(),
                      {0, 0, 0, 1},
                      [](ConvolutionContext& c) { c.do_search = true; });

        ConstructTest(db_path,
                      SearchableTestSolver::NoSearchFileName(),
                      {0, 0, 0, 0},
                      [](ConvolutionContext& c) { c.do_search = false; });

        ConstructTest(db_path,
                      SearchableTestSolver::FileName(),
                      {0, 0, 0, 0},
                      [](ConvolutionContext& c) { c.do_search = true; });

        const auto& searchable_solver = StaticContainer<const SearchableTestSolver>::Instance();
        const auto searches           = SearchableTestSolver::searches_done();

        // Should read in both cases: result is already in DB, solver is searchable.
        ConstructTest(
            db_path, SearchableTestSolver::FileName(), {0, 0, 0, 0}, [](ConvolutionContext&) {});

        ConstructTest(db_path,
                      SearchableTestSolver::FileName(),
                      {0, 0, 0, 0},
                      [](ConvolutionContext& c) { c.do_search = true; });

        // Checking no more searches were done.
        EXPECT_EQUAL(searches, searchable_solver.searches_done());
    }

private:
    static void ConstructTest(
        const std::string& db_path,
        const char* expected_kernel,
        const std::initializer_list<size_t>& in,
        const std::function<void(ConvolutionContext&)>& context_filler = [](ConvolutionContext&) {})
    {
        auto ctx = ConvolutionContext{TensorDescriptor{miopenFloat, in},
                                      TensorDescriptor{miopenFloat, in},
                                      TensorDescriptor{miopenFloat, in},
                                      ConvolutionDescriptor{},
                                      conv::Direction::Forward};
        ctx.SetStream(&get_handle());
        context_filler(ctx);

        const auto sol = FindSolution(ctx, db_path);

        EXPECT_OP(sol.construction_params.size(), >, 0);
        EXPECT_EQUAL(sol.construction_params[0].kernel_file, expected_kernel);
    }
};
} // namespace tests
} // namespace miopen

int main() { miopen::tests::SolverTest().Run(); }
