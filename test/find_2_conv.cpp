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

#include <miopen/miopen.h>

#include <miopen/convolution.hpp>

#include <vector>

namespace miopen {
struct Find2Test : test_driver
{
    tensor<float> x;
    tensor<float> w;
    tensor<float> y;
    Allocator::ManageDataPtr x_dev;
    Allocator::ManageDataPtr w_dev;
    Allocator::ManageDataPtr y_dev;
    // --input 16,192,28,28 --weights 32,192,5,5 --filter 2,2,1,1,1,1,
    miopen::ConvolutionDescriptor filter = {
        2, miopenConvolution, miopenPaddingDefault, {1, 1}, {1, 1}, {1, 1}};

    Find2Test()
    {
        x = {16, 192, 28, 28};
        w = {32, 192, 5, 5};
        y = tensor<float>{filter.GetForwardOutputTensor(x.desc, w.desc)};
    }

    void run()
    {
        auto& handle_deref = get_handle();

        x_dev = handle_deref.Write(x.data);
        w_dev = handle_deref.Write(w.data);
        y_dev = handle_deref.Write(y.data);

        miopenHandle_t handle;
        miopenProblem_t problem;
        miopenSearchOptions_t options;

        deref(&handle) = &handle_deref;
        EXPECT_EQUAL(miopenCreateProblem(&problem), miopenStatusSuccess);

        {
            miopenConvolutionDescriptor_t conv;
            miopen::deref(&conv) = new ConvolutionDescriptor{filter};
            EXPECT_EQUAL(
                miopenSetProblemOperatorDescriptor(problem, conv, miopenProblemDirectionForward),
                miopenStatusSuccess);
            EXPECT_EQUAL(miopenDestroyConvolutionDescriptor(conv), miopenStatusSuccess);
        }

        auto test_set_tensor_descriptor = [problem](miopenTensorName_t name,
                                                    const TensorDescriptor& desc) {
            miopenTensorDescriptor_t api_desc;
            miopen::deref(&api_desc) = new TensorDescriptor{desc};
            EXPECT_EQUAL(miopenSetProblemTensorDescriptor(problem, name, api_desc),
                         miopenStatusSuccess);
            EXPECT_EQUAL(miopenDestroyTensorDescriptor(api_desc), miopenStatusSuccess);
        };

        test_set_tensor_descriptor(miopenTensorConvolutionX, x.desc);
        test_set_tensor_descriptor(miopenTensorConvolutionW, w.desc);
        test_set_tensor_descriptor(miopenTensorConvolutionY, y.desc);

        auto solutions = std::vector<miopenSolution_t>{};
        std::size_t found;

        solutions.resize(100);

        // Without options
        EXPECT_EQUAL(miopenFindSolutions(
                         handle, problem, nullptr, solutions.data(), &found, solutions.size()),
                     miopenStatusSuccess);
        EXPECT_OP(found, >=, 0);

        // With options
        const auto checked_set_options = [&](auto option, auto value) {
            using Type = decltype(value);
            EXPECT_EQUAL(miopenSetSearchOption(options, option, sizeof(Type), &value),
                         miopenStatusSuccess);
        };

        EXPECT_EQUAL(miopenCreateSearchOptions(&options), miopenStatusSuccess);
        checked_set_options(miopenSearchOptionExhaustiveSearch, static_cast<int>(0));
        checked_set_options(miopenSearchOptionResultsOrder, miopenSearchResultsOrderByTime);
        checked_set_options(miopenSearchOptionWorkspaceLimit,
                            std::numeric_limits<std::size_t>::max());

        EXPECT_EQUAL(miopenFindSolutions(
                         handle, problem, options, solutions.data(), &found, solutions.size()),
                     miopenStatusSuccess);
        EXPECT_OP(found, >=, 0);

        EXPECT_EQUAL(miopenDestroySearchOptions(options), miopenStatusSuccess);

        solutions.resize(found);

        miopenTensorName_t names[3] = {
            miopenTensorConvolutionX, miopenTensorConvolutionW, miopenTensorConvolutionY};
        void* buffers[3] = {x_dev.get(), w_dev.get(), y_dev.get()};

        {
            miopenTensorDescriptor_t x_desc, w_desc, y_desc;
            miopen::deref(&x_desc)                  = new TensorDescriptor{x.desc};
            miopen::deref(&w_desc)                  = new TensorDescriptor{w.desc};
            miopen::deref(&y_desc)                  = new TensorDescriptor{y.desc};
            miopenTensorDescriptor_t descriptors[3] = {x_desc, w_desc, y_desc};

            const auto checked_run_solution =
                [&](auto&& solution, auto&& descriptors_, auto&& workspace, auto&& workspace_size) {
                    EXPECT_EQUAL(miopenRunSolution(handle,
                                                   solution,
                                                   3,
                                                   names,
                                                   descriptors_,
                                                   buffers,
                                                   workspace.get(),
                                                   workspace_size),
                                 miopenStatusSuccess);
                };

            for(const auto& solution : solutions)
            {
                const auto checked_get_attr = [&](auto name, auto& value) {
                    using Type = std::remove_reference_t<decltype(value)>;
                    EXPECT_EQUAL(
                        miopenGetSolutionAttribute(solution, name, sizeof(Type), &value, nullptr),
                        miopenStatusSuccess);
                };

                float time;
                std::size_t workspace_size;
                checked_get_attr(miopenSolutionAttributeTime, time);
                checked_get_attr(miopenSolutionAttributeWorkspaceSize, workspace_size);

                auto workspace     = std::vector<char>(workspace_size);
                auto workspace_dev = workspace_size != 0 ? handle_deref.Write(workspace) : nullptr;

                // Without descriptors
                checked_run_solution(solution, nullptr, workspace_dev, workspace_size);
                // With descriptors
                checked_run_solution(solution, descriptors, workspace_dev, workspace_size);

                // Save-load cycle
                std::size_t solution_size;
                EXPECT_EQUAL(miopenGetSolutionSize(solution, &solution_size), miopenStatusSuccess);

                auto solution_binary = std::vector<char>{};
                solution_binary.resize(solution_size);

                EXPECT_EQUAL(miopenSaveSolution(solution, solution_binary.data()),
                             miopenStatusSuccess);
                EXPECT_EQUAL(miopenDestroySolution(solution), miopenStatusSuccess);

                miopenSolution_t read_solution;
                EXPECT_EQUAL(miopenLoadSolution(
                                 &read_solution, solution_binary.data(), solution_binary.size()),
                             miopenStatusSuccess);

                // Without descriptors
                checked_run_solution(read_solution, nullptr, workspace_dev, workspace_size);
                // With descriptors
                checked_run_solution(read_solution, descriptors, workspace_dev, workspace_size);
                EXPECT_EQUAL(miopenDestroySolution(read_solution), miopenStatusSuccess);
            }

            EXPECT_EQUAL(miopenDestroyTensorDescriptor(x_desc), miopenStatusSuccess);
            EXPECT_EQUAL(miopenDestroyTensorDescriptor(w_desc), miopenStatusSuccess);
            EXPECT_EQUAL(miopenDestroyTensorDescriptor(y_desc), miopenStatusSuccess);
        }

        EXPECT_EQUAL(miopenDestroyProblem(problem), miopenStatusSuccess);
    }
};
} // namespace miopen

int main(int argc, const char* argv[]) { test_drive<miopen::Find2Test>(argc, argv); }
