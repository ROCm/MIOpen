/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include "test.hpp"
#include "driver.hpp"
#include "get_handle.hpp"
#include "lib_env_var.hpp"
#include "workspace.hpp"

#include <miopen/miopen.h>

#include <miopen/convolution.hpp>
#include <miopen/solution.hpp>

#include <miopen/solver_id.hpp>

#include <nlohmann/json.hpp>

#include <vector>

MIOPEN_LIB_ENV_VAR(MIOPEN_LOG_LEVEL)

namespace miopen {
struct Find2Test : test_driver
{
    tensor<float> x;
    tensor<float> w;
    tensor<float> y;
    Allocator::ManageDataPtr x_dev;
    Allocator::ManageDataPtr w_dev;
    Allocator::ManageDataPtr y_dev;

    miopenProblemDirection_t direction = miopenProblemDirectionForward;
    // --input 16,192,28,28 --weights 32,192,5,5 --filter 2,2,1,1,1,1,
    miopen::ConvolutionDescriptor filter = {
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};
    int tune;
    bool preallocate;
    std::size_t workspace_limit;
    bool attach_binaries;

    Find2Test()
    {
        add(attach_binaries, "attach_binaries", generate_data({0, 1}));

        add(direction,
            "direction",
            generate_data({
                miopenProblemDirectionForward,
                miopenProblemDirectionBackward,
                miopenProblemDirectionBackwardWeights,
            }));

        add(workspace_limit,
            "workspace_limit",
            generate_data({
                std::numeric_limits<std::size_t>::max(),
                static_cast<size_t>(0),
            }));

        add(tune, "tune", generate_data({0, 1}));
        add(preallocate, "preallocate", generate_data({0, 1}));
    }

    void run()
    {
        ReleaseMemory();
        GenerateTensors();
        TestConv();
    }

private:
    void ReleaseMemory()
    {
        x_dev = nullptr;
        w_dev = nullptr;
        y_dev = nullptr;

        x = {};
        w = {};
        y = {};
    }

    void GenerateTensors()
    {
        auto& handle_deref = get_handle();

        x = tensor<float>{16, 192, 28, 28}.generate(tensor_elem_gen_integer{17});
        w = tensor<float>{32, 192, 5, 5}.generate(tensor_elem_gen_integer{17});
        y = tensor<float>{filter.GetForwardOutputTensor(x.desc, w.desc)};

        x_dev = handle_deref.Write(x.data);
        w_dev = handle_deref.Write(w.data);
        y_dev = handle_deref.Write(y.data);
    }

    void TestConv()
    {
        miopenHandle_t handle = &get_handle();
        miopenProblem_t problem;

        EXPECT_EQUAL(miopenCreateConvProblem(&problem, &filter, direction), miopenStatusSuccess);

        AddConvTensorDescriptors(problem);

        std::ignore          = TestFindSolutions(handle, problem);
        const auto solutions = TestFindSolutionsWithOptions(handle, problem);

        TestSolutionAttributes(solutions);
        TestRunSolutions(handle, solutions);

        EXPECT_EQUAL(miopenDestroyProblem(problem), miopenStatusSuccess);
    }

    void AddConvTensorDescriptors(miopenProblem_t problem)
    {
        std::cerr << "Creating conv tensor descriptos..." << std::endl;

        auto test_set_tensor_descriptor = [problem](miopenTensorArgumentId_t name,
                                                    TensorDescriptor& desc) {
            EXPECT_EQUAL(miopenSetProblemTensorDescriptor(problem, name, &desc),
                         miopenStatusSuccess);
        };

        test_set_tensor_descriptor(miopenTensorConvolutionX, x.desc);
        test_set_tensor_descriptor(miopenTensorConvolutionW, w.desc);
        test_set_tensor_descriptor(miopenTensorConvolutionY, y.desc);

        // adding x descriptor again to validate that error is produced
        EXPECT_EQUAL(miopenSetProblemTensorDescriptor(problem, miopenTensorConvolutionX, &x.desc),
                     miopenStatusBadParm);

        std::cerr << "Created conv tensor descriptos." << std::endl;
    }

    std::vector<miopenSolution_t> TestFindSolutions(miopenHandle_t handle, miopenProblem_t problem)
    {
        std::cerr << "Testing miopenFindSolutions..." << std::endl;

        auto solutions = std::vector<miopenSolution_t>{};
        std::size_t found;

        solutions.resize(100);

        EXPECT_EQUAL(miopenFindSolutions(
                         handle, problem, nullptr, solutions.data(), &found, solutions.size()),
                     miopenStatusSuccess);
        EXPECT_OP(found, >=, 0);

        solutions.resize(found);

        std::cerr << "Finished testing miopenFindSolutions." << std::endl;
        return solutions;
    }

    std::vector<miopenSolution_t> TestFindSolutionsWithOptions(miopenHandle_t handle,
                                                               miopenProblem_t problem)
    {
        std::cerr << "Testing miopenFindSolutions with options..." << std::endl;

        auto solutions    = std::vector<miopenSolution_t>{};
        std::size_t found = 0;

        solutions.resize(100);

        {
            miopenFindOptions_t options;

            EXPECT_EQUAL(miopenCreateFindOptions(&options), miopenStatusSuccess);

            EXPECT_EQUAL(miopenSetFindOptionTuning(options, tune), miopenStatusSuccess);
            EXPECT_EQUAL(miopenSetFindOptionResultsOrder(options, miopenFindResultsOrderByTime),
                         miopenStatusSuccess);
            EXPECT_EQUAL(miopenSetFindOptionWorkspaceLimit(options, workspace_limit),
                         miopenStatusSuccess);
            EXPECT_EQUAL(miopenSetFindOptionAttachBinaries(options, attach_binaries),
                         miopenStatusSuccess);

            Allocator::ManageDataPtr workspace_dev;

            if(preallocate)
            {
                std::size_t workspace_max = 0;
                switch(direction)
                {
                case miopenProblemDirectionForward:
                    EXPECT_EQUAL(miopenConvolutionForwardGetWorkSpaceSize(
                                     handle, &x.desc, &w.desc, &filter, &y.desc, &workspace_max),
                                 miopenStatusSuccess);
                    break;
                case miopenProblemDirectionBackward:
                    EXPECT_EQUAL(miopenConvolutionBackwardDataGetWorkSpaceSize(
                                     handle, &y.desc, &w.desc, &filter, &x.desc, &workspace_max),
                                 miopenStatusSuccess);
                    break;
                case miopenProblemDirectionBackwardWeights:
                    EXPECT_EQUAL(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                                     handle, &y.desc, &x.desc, &filter, &w.desc, &workspace_max),
                                 miopenStatusSuccess);
                    break;
                default: MIOPEN_THROW(miopenStatusNotImplemented);
                }

                const auto workspace_size = std::min(workspace_limit, workspace_max);
                Workspace wspace{workspace_size};

                EXPECT_EQUAL(
                    miopenSetFindOptionPreallocatedWorkspace(options, wspace.ptr(), wspace.size()),
                    miopenStatusSuccess);

                EXPECT_EQUAL(miopenSetFindOptionPreallocatedTensor(
                                 options, miopenTensorConvolutionX, x_dev.get()),
                             miopenStatusSuccess);

                EXPECT_EQUAL(miopenSetFindOptionPreallocatedTensor(
                                 options, miopenTensorConvolutionW, w_dev.get()),
                             miopenStatusSuccess);

                EXPECT_EQUAL(miopenSetFindOptionPreallocatedTensor(
                                 options, miopenTensorConvolutionY, y_dev.get()),
                             miopenStatusSuccess);
            }

            std::cerr << "Testing with: ";
            std::cerr << (tune != 0 ? "tuning" : "no tuning") << ", ";
            std::cerr << (attach_binaries ? "attached binaries" : "no binaries") << ", ";
            std::cerr << workspace_limit << " ws limit";

            EXPECT_EQUAL(miopenFindSolutions(
                             handle, problem, options, solutions.data(), &found, solutions.size()),
                         miopenStatusSuccess);

            EXPECT_EQUAL(miopenDestroyFindOptions(options), miopenStatusSuccess);
        }

        EXPECT_OP(found, >=, 0);
        solutions.resize(found);

        std::cerr << "Finished testing miopenFindSolutions with options." << std::endl;
        return solutions;
    }

    void TestSolutionAttributes(const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing miopenGetSolution<Attribute>..." << std::endl;

        for(const auto& solution : solutions)
        {
            float time;
            std::size_t workspace_size;
            uint64_t solver_id;
            miopenConvAlgorithm_t algo;

            EXPECT_EQUAL(miopenGetSolutionTime(solution, &time), miopenStatusSuccess);
            EXPECT_EQUAL(miopenGetSolutionWorkspaceSize(solution, &workspace_size),
                         miopenStatusSuccess);
            EXPECT_EQUAL(miopenGetSolutionSolverId(solution, &solver_id), miopenStatusSuccess);
            EXPECT_EQUAL(miopenGetSolverIdConvAlgorithm(solver_id, &algo), miopenStatusSuccess);
        }

        std::cerr << "Finished testing miopenGetSolution<Attribute>." << std::endl;
    }

    void TestRunSolutions(miopenHandle_t handle, const std::vector<miopenSolution_t>& solutions)
    {
        std::cerr << "Testing solution functions..." << std::endl;

        miopenTensorDescriptor_t x_desc = &x.desc, w_desc = &w.desc, y_desc = &y.desc;

        for(const auto& solution : solutions)
        {
            uint64_t solver_id;
            EXPECT_EQUAL(miopenGetSolutionSolverId(solution, &solver_id), miopenStatusSuccess);
            std::cerr << "Testing solver " << solver_id << " (" << solver::Id(solver_id).ToString()
                      << ")" << std::endl;

            miopenTensorArgumentId_t names[3] = {
                miopenTensorConvolutionX, miopenTensorConvolutionW, miopenTensorConvolutionY};
            void* buffers[3]                        = {x_dev.get(), w_dev.get(), y_dev.get()};
            miopenTensorDescriptor_t descriptors[3] = {x_desc, w_desc, y_desc};

            TestRunSolution(handle, solution, 3, names, descriptors, buffers);

            // Save-load cycle
            std::cerr << "Testing miopenGetSolutionSize..." << std::endl;
            std::size_t solution_size;
            EXPECT_EQUAL(miopenGetSolutionSize(solution, &solution_size), miopenStatusSuccess);
            EXPECT_OP(solution_size, >, 0);

            auto solution_binary = std::vector<char>{};
            solution_binary.resize(solution_size);

            std::cerr << "Testing miopenSaveSolution..." << std::endl;
            EXPECT_EQUAL(miopenSaveSolution(solution, solution_binary.data()), miopenStatusSuccess);
            std::cerr << "Destroying original solution..." << std::endl;
            EXPECT_EQUAL(miopenDestroySolution(solution), miopenStatusSuccess);

            std::cerr << "Testing miopenLoadSolution..." << std::endl;
            miopenSolution_t read_solution;
            EXPECT_EQUAL(
                miopenLoadSolution(&read_solution, solution_binary.data(), solution_binary.size()),
                miopenStatusSuccess);

            TestRunSolution(handle, read_solution, 3, names, descriptors, buffers);
            EXPECT_EQUAL(miopenDestroySolution(read_solution), miopenStatusSuccess);
        }

        std::cerr << "Finished testing solution functions." << std::endl;
    }

    void TestRunSolution(miopenHandle_t handle,
                         miopenSolution_t solution,
                         std::size_t num_arguments,
                         const miopenTensorArgumentId_t* names,
                         miopenTensorDescriptor_t* descriptors,
                         void** buffers)
    {
        std::cerr << "Running a solution..." << std::endl;

        std::size_t workspace_size;
        EXPECT_EQUAL(miopenGetSolutionWorkspaceSize(solution, &workspace_size),
                     miopenStatusSuccess);

        Workspace wspace{workspace_size};

        const auto checked_run_solution = [&](miopenTensorDescriptor_t* descriptors_) {
            auto arguments = std::make_unique<miopenTensorArgument_t[]>(num_arguments);

            for(auto i = 0; i < num_arguments; ++i)
            {
                arguments[i].id         = names[i];
                arguments[i].descriptor = descriptors_ != nullptr ? &descriptors_[i] : nullptr;
                arguments[i].buffer     = buffers[i];
            }

            EXPECT_EQUAL(miopenRunSolution(
                             handle, solution, 3, arguments.get(), wspace.ptr(), wspace.size()),
                         miopenStatusSuccess);
        };

        // Without descriptors
        checked_run_solution(nullptr);
        // With descriptors
        checked_run_solution(descriptors);

        std::cerr << "Ran a solution." << std::endl;
    }
};
} // namespace miopen

int main(int argc, const char* argv[])
{
    lib_env::update(MIOPEN_LOG_LEVEL, 6);
    test_drive<miopen::Find2Test>(argc, argv);
}
