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

#include <stdlib.h>
#include <functional>
#include <sstream>
#include <typeinfo>
#include <unistd.h>

#include "get_handle.hpp"
#include "miopen/mlo_internal.hpp"
#include "miopen/solver.hpp"
#include "test.hpp"

namespace miopen {
namespace tests {
class TrivialSlowTestSolver : public solver::Solver
{
    public:
    static const char* FileName() { return "TrivialSlowTestSolver"; }
    const char* SolverId() const override { return FileName(); }
    bool IsFast(const ConvolutionContext& context) const override { return context.in_height == 1; }
    bool IsApplicable(const ConvolutionContext& context) const override
    {
        return context.in_width == 1;
    }

    solver::ConvSolution GetSolution(const ConvolutionContext&,
                                     const solver::PerformanceConfig&) const override
    {
        solver::ConvSolution ret;
        solver::KernelInfo kernel;

        kernel.kernel_file  = SolverId();
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }
};

class TrivialTestSolver : public solver::Solver
{
    public:
    static const char* FileName() { return "TrivialTestSolver"; }
    const char* SolverId() const override { return FileName(); }
    bool IsApplicable(const ConvolutionContext& context) const override
    {
        return context.in_width == 1;
    }

    solver::ConvSolution GetSolution(const ConvolutionContext&,
                                     const solver::PerformanceConfig&) const override
    {
        solver::ConvSolution ret;
        solver::KernelInfo kernel;

        kernel.kernel_file  = SolverId();
        kernel.comp_options = " ";
        ret.construction_params.push_back(kernel);

        return ret;
    }
};

class TestConfig : public solver::PerformanceConfig
{
    public:
    std::string str;

    void Serialize(std::ostream& s) const override { s << str; }

    bool Deserialize(const std::string& s) override
    {
        std::istringstream ss(s);
        std::string temp;

        if(!std::getline(ss, temp))
            return false;

        str = temp;
        return true;
    }
};

class SearchableTestSolver : public solver::Solver
{
    public:
    static int searches_done() { return _serches_done; }
    static const char* FileName() { return "SearchableTestSolver"; }
    static const char* NoSearchFileName() { return "SearchableTestSolver.NoSearch"; }
    const char* SolverId() const override { return FileName(); }
    bool IsSearchable() const override { return true; }
    std::unique_ptr<solver::PerformanceConfig> PerformanceConfigImpl() const override
    {
        return miopen::make_unique<TestConfig>();
    }

    void InitPerformanceConfigImpl(const ConvolutionContext&,
                                   solver::PerformanceConfig& config_) const override
    {
        auto& config = dynamic_cast<TestConfig&>(config_);
        config.str   = NoSearchFileName();
    }

    void Search(const ConvolutionContext&, solver::PerformanceConfig& config_) const override
    {
        auto& config = dynamic_cast<TestConfig&>(config_);
        config.str   = SolverId();
        _serches_done++;
    }

    solver::ConvSolution GetSolution(const ConvolutionContext&,
                                     const solver::PerformanceConfig& config_) const override
    {
        const auto& config = dynamic_cast<const TestConfig&>(config_);

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
    TrivialConstruct(int dir, bool do_bias = false) : mlo_construct_direct2D(dir, do_bias)
    {
        auto temp_fd = mkstemp(_temp_file);
        EXPECT(temp_fd != -1);
        close(temp_fd);
    }

    ~TrivialConstruct() { std::remove(_temp_file); }

    const std::vector<std::reference_wrapper<const solver::Solver>>& SolverStore() const override
    {
        static const std::vector<std::reference_wrapper<const solver::Solver>> data{
            StaticContainer<const TrivialSlowTestSolver>::Instance(),
            StaticContainer<const TrivialTestSolver>::Instance(),
            StaticContainer<const SearchableTestSolver>::Instance(),
        };

        return data;
    }

    protected:
    std::string db_path() override { return _temp_file; }

    private:
    char _temp_file[32] = "/tmp/miopen.tests.solver.XXXXXX";
};

class SolverTest
{
    public:
    void Run() const
    {
        TrivialConstruct construct(1);

        ConstructTest(construct, TrivialSlowTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.setInputDescr("", "", 0, 0, 1, 1, 0, 0, 0, 0);
        });
        ConstructTest(construct, TrivialTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.setInputDescr("", "", 0, 0, 0, 1, 0, 0, 0, 0);
        });
        ConstructTest(construct,
                      SearchableTestSolver::NoSearchFileName(),
                      [](mlo_construct_direct2D& c) { c.doSearch(false); });
        ConstructTest(construct, SearchableTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.doSearch(true);
        });

        const auto& searchable_solver = StaticContainer<const SearchableTestSolver>::Instance();
        const auto searches           = searchable_solver.searches_done();

        // Should read in both cases: result is already in DB, solver is searchable.
        ConstructTest(construct, SearchableTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.doSearch(false);
        });
        ConstructTest(construct, SearchableTestSolver::FileName(), [](mlo_construct_direct2D& c) {
            c.doSearch(true);
        });
        // Checking no more searches were done.
        EXPECT_EQUAL(searches, searchable_solver.searches_done());
    }

    private:
    void ConstructTest(mlo_construct_direct2D& construct,
                       const char* expected_kernel,
                       std::function<void(mlo_construct_direct2D&)> context_filler) const
    {
        construct.setStream(&get_handle());
        construct.setInputDescr("", "", 0, 0, 0, 0, 0, 0, 0, 0);

        context_filler(construct);
        construct.mloConstruct();

        EXPECT_EQUAL(construct.getKernelFile(), expected_kernel);
    }
};
} // namespace tests
} // namespace miopen

int main() { miopen::tests::SolverTest().Run(); }
