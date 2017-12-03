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

#include <miopen/solver.hpp>

#include <cstdlib>
#include <functional>
#include <sstream>
#include <typeinfo>

#include "get_handle.hpp"
#include <miopen/mlo_internal.hpp>
#include "temp_file_path.hpp"
#include "test.hpp"

namespace miopen {
namespace tests {
class TrivialSlowTestSolver : public solver::SolverBase<ConvolutionContext>
{
    public:
    static const char* FileName() { return "TrivialSlowTestSolver"; }
    bool IsFast(const ConvolutionContext& context) const { return context.in_height == 1; }
    bool IsApplicable(const ConvolutionContext& context) const { return context.in_width == 1; }

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

class TrivialTestSolver : public solver::SolverBase<ConvolutionContext>
{
    public:
    static const char* FileName() { return "TrivialTestSolver"; }
    bool IsApplicable(const ConvolutionContext& context) const { return context.in_width == 1; }

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

class SearchableTestSolver : public solver::SolverBase<ConvolutionContext>
{
    public:
    static int searches_done() { return _serches_done; }
    static const char* FileName() { return "SearchableTestSolver"; }
    static const char* NoSearchFileName() { return "SearchableTestSolver.NoSearch"; }

    TestConfig GetPerformanceConfig(const ConvolutionContext&) const
    {
        TestConfig config{};
        config.str = NoSearchFileName();
        return config;
    }

    bool IsValidPerformanceConfig(const ConvolutionContext&, const TestConfig&) const
    {
        return true;
    }

    TestConfig Search(const ConvolutionContext&) const
    {
        TestConfig config;
        config.str = FileName();
        _serches_done++;
        return config;
    }

    solver::ConvSolution GetSolution(const ConvolutionContext&, const TestConfig& config) const
    {

        solver::ConvSolution ret;
        solver::KernelInfo kernel;

        kernel.kernel_file  = config.str;
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }

    private:
    static int _serches_done;
};

int SearchableTestSolver::_serches_done = 0;

class TrivialConstruct : public mlo_construct_direct2D
{
    public:
    TrivialConstruct(const char* db_path, int dir, bool do_bias = false)
        : mlo_construct_direct2D(dir, do_bias)
    {
        _db_path = db_path;
    }

    solver::ConvSolution FindSolution()
    {
        // clang-format off
        return miopen::solver::SearchForSolution<
            TrivialSlowTestSolver,
            TrivialTestSolver,
            SearchableTestSolver
        >(_search_params, this->GetDbRecord());
        // clang-format on
    }

    protected:
    std::string db_path() const { return _db_path; }
};

class SolverTest
{
    public:
    void Run() const
    {
        TempFilePath db_path("/tmp/miopen.tests.solver.XXXXXX");

        ConstructTest(db_path, TrivialSlowTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.setInputDescr("", "", 0, 0, 1, 1, 0, 0, 0, 0);
        });
        ConstructTest(db_path, TrivialTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.setInputDescr("", "", 0, 0, 0, 1, 0, 0, 0, 0);
        });
        ConstructTest(db_path, TrivialTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.setInputDescr("", "", 0, 0, 0, 1, 0, 0, 0, 0);
            c.doSearch(true);
        });
        ConstructTest(db_path,
                      SearchableTestSolver::NoSearchFileName(),
                      [](mlo_construct_direct2D& c) { c.doSearch(false); });
        ConstructTest(db_path, SearchableTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.doSearch(true);
        });

        const auto& searchable_solver = StaticContainer<const SearchableTestSolver>::Instance();
        const auto searches           = miopen::tests::SearchableTestSolver::searches_done();

        // Should read in both cases: result is already in DB, solver is searchable.
        ConstructTest(db_path, SearchableTestSolver::FileName(), [](mlo_construct_direct2D&) {});
        ConstructTest(db_path, SearchableTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.doSearch(true);
        });
        // Checking no more searches were done.
        EXPECT_EQUAL(searches, searchable_solver.searches_done());
    }

    private:
    void ConstructTest(const char* db_path,
                       const char* expected_kernel,
                       std::function<void(mlo_construct_direct2D&)> context_filler) const
    {
        TrivialConstruct construct(db_path, 1);
        construct.setStream(&get_handle());

        context_filler(construct);
        mloConstruct(construct);

        EXPECT_EQUAL(construct.getKernelFile(), expected_kernel);
    }
};
} // namespace tests
} // namespace miopen

int main() { miopen::tests::SolverTest().Run(); }
