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

#include <iostream>
#include <cassert>
#include <miopen/conv/heuristics/ai_heuristics.hpp>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_3d_kernel_tuning_utils.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/handle.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/implicitgemm_ck_util.hpp>

#include <miopen/conv/solvers.hpp>

using namespace miopen::solver::conv;

// Define a reusable ProblemDescription for all tests
// Example 3D convolution: NCDHW = 2x3x8x8x8, K=4, ZYX=3x3x3
miopen::conv::ProblemDescription GetReusableProblemDescription(
    miopenDataType_t dataType         = miopenFloat,
    miopen::conv::Direction direction = miopen::conv::Direction::BackwardWeights)
{
    // N, C, D, H, W
    std::vector<int> in_lengths = {2, 3, 8, 8, 8};
    // K, C, Z, Y, X
    std::vector<int> weights_lengths = {4, 3, 3, 3, 3};
    // N, K, Do, Ho, Wo
    std::vector<int> out_lengths = {2, 4, 6, 6, 6};

    miopen::TensorDescriptor in_desc(dataType, in_lengths);
    miopen::TensorDescriptor weights_desc(dataType, weights_lengths);
    miopen::TensorDescriptor out_desc(dataType, out_lengths);

    std::vector<int> pads              = {0, 0, 0};
    std::vector<int> strides           = {1, 1, 1};
    std::vector<int> dilations         = {1, 1, 1};
    std::vector<int> trans_output_pads = {0, 0, 0};

    miopen::ConvolutionDescriptor conv_desc(
        3, miopenConvolution, miopenPaddingDefault, pads, strides, dilations, trans_output_pads);

    return miopen::conv::ProblemDescription(in_desc, weights_desc, out_desc, conv_desc, direction);
}

void TestGetFeatures3D()
{

    auto problem     = GetReusableProblemDescription();
    int max_cu       = 304;
    std::string arch = "gfx942";

    auto features = miopen::solver::conv::GetFeatures3D(problem, max_cu, arch);

    // Check expected size
    if(features.size() != 22)
    {
        std::cerr << "GetFeatures3D: Unexpected feature vector size: " << features.size()
                  << std::endl;
        std::abort();
    }
}

void CheckGetFeatures3D_Values(const std::vector<float>& features,
                               miopen::conv::Direction direction)
{
    // For backward directions, input/output roles are swapped.
    // See ProblemInterpreter in problem_description_interpreter.hpp

    // Expected values for Forward
    int expected_in_c = 3, expected_in_d = 8, expected_in_h = 8, expected_in_w = 8;
    int expected_out_k = 4, expected_out_d = 6, expected_out_h = 6, expected_out_w = 6;
    int expected_batch_n       = 2;
    int expected_in_left_pad_d = 0, expected_in_left_pad_h = 0, expected_in_left_pad_w = 0;
    int expected_stride_d = 1, expected_stride_h = 1, expected_stride_w = 1;
    int expected_fil_d = 3, expected_fil_h = 3, expected_fil_w = 3;

    // For backward directions, swap input/output
    bool is_forward = (direction == miopen::conv::Direction::Forward);

    int in_c    = is_forward ? expected_in_c : expected_out_k;
    int in_d    = is_forward ? expected_in_d : expected_out_d;
    int in_h    = is_forward ? expected_in_h : expected_out_h;
    int in_w    = is_forward ? expected_in_w : expected_out_w;
    int out_k   = is_forward ? expected_out_k : expected_in_c;
    int out_d   = is_forward ? expected_out_d : expected_in_d;
    int out_h   = is_forward ? expected_out_h : expected_in_h;
    int out_w   = is_forward ? expected_out_w : expected_in_w;
    int batch_n = expected_batch_n; // batch size is not swapped

    // Now check features
    if(features[0] != in_c)
    {
        std::cerr << "InputChannelC mismatch: got " << features[0] << ", expected " << in_c << "\n";
        std::abort();
    }
    if(features[1] != in_d)
    {
        std::cerr << "InputDepthDi mismatch: got " << features[1] << ", expected " << in_d << "\n";
        std::abort();
    }
    if(features[2] != in_h)
    {
        std::cerr << "InputHeightHi mismatch: got " << features[2] << ", expected " << in_h << "\n";
        std::abort();
    }
    if(features[3] != in_w)
    {
        std::cerr << "InputWidthWi mismatch: got " << features[3] << ", expected " << in_w << "\n";
        std::abort();
    }
    if(features[4] != out_k)
    {
        std::cerr << "OutputChannelK mismatch: got " << features[4] << ", expected " << out_k
                  << "\n";
        std::abort();
    }
    if(features[5] != out_d)
    {
        std::cerr << "OutputDepthDo mismatch: got " << features[5] << ", expected " << out_d
                  << "\n";
        std::abort();
    }
    if(features[6] != out_h)
    {
        std::cerr << "OutputHeightHo mismatch: got " << features[6] << ", expected " << out_h
                  << "\n";
        std::abort();
    }
    if(features[7] != out_w)
    {
        std::cerr << "OutputWidthWo mismatch: got " << features[7] << ", expected " << out_w
                  << "\n";
        std::abort();
    }
    if(features[8] != expected_fil_d)
    {
        std::cerr << "FilterDepthZ mismatch: got " << features[8] << ", expected " << expected_fil_d
                  << "\n";
        std::abort();
    }
    if(features[9] != expected_fil_h)
    {
        std::cerr << "FilterHeightY mismatch: got " << features[9] << ", expected "
                  << expected_fil_h << "\n";
        std::abort();
    }
    if(features[10] != expected_fil_w)
    {
        std::cerr << "FilterWidthX mismatch: got " << features[10] << ", expected "
                  << expected_fil_w << "\n";
        std::abort();
    }
    if(features[11] != expected_in_left_pad_d)
    {
        std::cerr << "InputLeftPadD mismatch: got " << features[11] << ", expected "
                  << expected_in_left_pad_d << "\n";
        std::abort();
    }
    if(features[12] != expected_in_left_pad_h)
    {
        std::cerr << "InputLeftPadH mismatch: got " << features[12] << ", expected "
                  << expected_in_left_pad_h << "\n";
        std::abort();
    }
    if(features[13] != expected_in_left_pad_w)
    {
        std::cerr << "InputLeftPadW mismatch: got " << features[13] << ", expected "
                  << expected_in_left_pad_w << "\n";
        std::abort();
    }
    if(features[14] != expected_stride_d)
    {
        std::cerr << "StrideD mismatch: got " << features[14] << ", expected " << expected_stride_d
                  << "\n";
        std::abort();
    }
    if(features[15] != expected_stride_h)
    {
        std::cerr << "StrideH mismatch: got " << features[15] << ", expected " << expected_stride_h
                  << "\n";
        std::abort();
    }
    if(features[16] != expected_stride_w)
    {
        std::cerr << "StrideW mismatch: got " << features[16] << ", expected " << expected_stride_w
                  << "\n";
        std::abort();
    }
    if(features[17] != batch_n)
    {
        std::cerr << "BatchN mismatch: got " << features[17] << ", expected " << batch_n << "\n";
        std::abort();
    }
    if(features[18] != 0.0f)
    {
        std::cerr << "InputLayout (should be NCDHW) mismatch: got " << features[18]
                  << ", expected 0.0\n";
        std::abort();
    }
    if(features[19] != 0.0f)
    {
        std::cerr << "FilterLayout (should be NCDHW) mismatch: got " << features[19]
                  << ", expected 0.0\n";
        std::abort();
    }
    if(features[20] != 0.0f)
    {
        std::cerr << "OutputLayout (should be NCDHW) mismatch: got " << features[20]
                  << ", expected 0.0\n";
        std::abort();
    }
    if(features[21] != static_cast<float>(miopenFloat))
    {
        std::cerr << "DataType mismatch: got " << features[21] << ", expected "
                  << static_cast<float>(miopenFloat) << "\n";
        std::abort();
    }
}

void TestGetFeatures3D_ValueChecks()
{

    int max_cu       = 304;
    std::string arch = "gfx942";
    // Test value checks for all convolution directions
    {
        const std::vector<miopen::conv::Direction> directions = {
            miopen::conv::Direction::Forward,
            miopen::conv::Direction::BackwardData,
            miopen::conv::Direction::BackwardWeights};
        for(const auto direction : directions)
        {
            auto problem  = GetReusableProblemDescription(miopenFloat, direction);
            auto features = miopen::solver::conv::GetFeatures3D(problem, max_cu, arch);

            // Check expected size
            if(features.size() != 22)
            {
                std::cerr << "GetFeatures3D: Unexpected feature vector size for direction "
                          << static_cast<int>(direction) << ": " << features.size() << std::endl;
                std::abort();
            }

            CheckGetFeatures3D_Values(features, direction);
        }
    }
}

void TestGetFeatures3D_Directions()
{
    int max_cu       = 304;
    std::string arch = "gfx942";

    // Forward
    auto problem_fwd = GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::Forward);
    auto features_fwd = miopen::solver::conv::GetFeatures3D(problem_fwd, max_cu, arch);

    // BackwardData
    auto problem_bwd =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardData);
    auto features_bwd = miopen::solver::conv::GetFeatures3D(problem_bwd, max_cu, arch);

    // BackwardWeights
    auto problem_wrw =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);
    auto features_wrw = miopen::solver::conv::GetFeatures3D(problem_wrw, max_cu, arch);
    if(features_fwd.size() != features_bwd.size() || features_fwd.size() != features_wrw.size())
    {
        std::cerr << "GetFeatures3D: Feature vector sizes do not match for different directions."
                  << std::endl;
        std::abort();
    }
    // TODO: since we use separate heuristics per direction, the direction feature always gets
    // dropped. if this changes, we should add checks for the direction feature.
}

void TestGetFeatures3D_DataTypes()
{
    // get base problem description
    auto problem     = GetReusableProblemDescription();
    int max_cu       = 304;
    std::string arch = "gfx942";

    // 3. Different Data Types
    auto problem_f  = GetReusableProblemDescription(miopenFloat);
    auto features_f = miopen::solver::conv::GetFeatures3D(problem_f, max_cu, arch);
    if(features_f[21] != static_cast<float>(miopenFloat))
    {
        std::cerr << "GetFeatures3D: DataType mismatch for miopenFloat." << std::endl;
        std::abort();
    }

    // miopenHalf
    auto problem_h  = GetReusableProblemDescription(miopenHalf);
    auto features_h = miopen::solver::conv::GetFeatures3D(problem_h, max_cu, arch);
    if(features_h[21] != static_cast<float>(miopenHalf))
    {
        std::cerr << "GetFeatures3D: DataType mismatch for miopenHalf." << std::endl;
        std::abort();
    }

    // miopenBFloat16
    auto problem_b  = GetReusableProblemDescription(miopenBFloat16);
    auto features_b = miopen::solver::conv::GetFeatures3D(problem_b, max_cu, arch);
    if(features_b[21] != static_cast<float>(miopenBFloat16))
    {
        std::cerr << "GetFeatures3D: DataType mismatch for miopenBFloat16." << std::endl;
        std::abort();
    }
}

void TestTokenizeKernel()
{
    auto tokens = miopen::solver::conv::TokenizeKernel("type_param1_param2");
    if(tokens.size() != 3 || tokens[0] != "type" || tokens[1] != "param1" || tokens[2] != "param2")
    {
        std::cerr << "TokenizeKernel failed on normal input." << std::endl;
        std::abort();
    }
    auto empty = miopen::solver::conv::TokenizeKernel("");
    if(!empty.empty())
    {
        std::cerr << "TokenizeKernel failed on empty input." << std::endl;
        std::abort();
    }
}

void TestFilterHeuristicKernels()
{
    std::vector<std::string> kernels = {"typeA_param1", "typeB_param2", "typeA_param3"};
    std::vector<int> indexes;
    std::vector<std::vector<std::string>> tokens;
    miopen::solver::conv::FilterHeuristicKernels("typeA", kernels, indexes, tokens);

    if(indexes.size() != 2 || tokens.size() != 2)
    {
        // Check if the filtering worked correctly,
        // i.e., we got 2 kernels of type "typeA" and 2 corresponding indexes.
        std::cerr << "FilterHeuristicKernels: Incorrect filtering." << std::endl;
        std::abort();
    }
    if(indexes[0] != 0 || indexes[1] != 2)
    {
        std::cerr << "FilterHeuristicKernels: Incorrect indexes." << std::endl;
        std::abort();
    }
}

void TestGenerateSplitK()
{
    auto split_ks             = miopen::solver::conv::GenerateSplitK(8);
    std::vector<int> expected = {1, 2, 4, 8};
    if(split_ks != expected)
    {
        std::cerr << "GenerateSplitK: Incorrect output." << std::endl;
        std::abort();
    }
}

void TestExpandKernelParamsWithSplitK()
{
    std::vector<std::vector<std::string>> kernels = {{"typeA", "p1"}, {"typeB", "p2"}};
    std::vector<int> indexes                      = {0, 1};
    std::vector<int> split_ks                     = miopen::solver::conv::GenerateSplitK(8);
    auto [expanded, mapping] =
        miopen::solver::conv::ExpandKernelParamsWithSplitK(kernels, indexes, split_ks);

    if(expanded.size() != 8 || mapping.size() != 8)
    {
        std::cerr << "ExpandKernelParamsWithSplitK: Incorrect expansion." << std::endl;
        std::abort();
    }

    // Build expected expanded and mapping
    std::vector<std::vector<std::string>> expected_expanded = {
        {"typeA", "p1", "1"},
        {"typeA", "p1", "2"},
        {"typeA", "p1", "4"},
        {"typeA", "p1", "8"},
        {"typeB", "p2", "1"},
        {"typeB", "p2", "2"},
        {"typeB", "p2", "4"},
        {"typeB", "p2", "8"},
    };
    std::vector<std::pair<int, int>> expected_mapping = {
        {0, 1}, {0, 2}, {0, 4}, {0, 8}, {1, 1}, {1, 2}, {1, 4}, {1, 8}};

    // Check contents
    for(size_t i = 0; i < expanded.size(); ++i)
    {
        if(expanded[i] != expected_expanded[i])
        {
            std::cerr << "ExpandKernelParamsWithSplitK: Expanded content mismatch at " << i
                      << std::endl;
            std::abort();
        }
        if(mapping[i] != expected_mapping[i])
        {
            std::cerr << "ExpandKernelParamsWithSplitK: Mapping content mismatch at " << i
                      << std::endl;
            std::abort();
        }
    }
}

void TestRunParameterPredictionModel()
{
    // 1. Real context
    miopen::Handle handle;
    miopen::ExecutionContext ctx(&handle);

    // Print device info
    std::string device_name = handle.GetDeviceName();
    int max_cu              = handle.GetMaxComputeUnits();
    std::cout << "Device name: " << device_name << std::endl;
    std::cout << "Max compute units: " << max_cu << std::endl;

    // 2. ProblemDescription (reuse from other tests)
    auto problem =
        GetReusableProblemDescription(miopenFloat, miopen::conv::Direction::BackwardWeights);

    // 3. fill_valid_kernels function
    std::function<std::vector<std::string>(const miopen::conv::ProblemDescription&)>
        fill_valid_kernels = [&ctx](const miopen::conv::ProblemDescription& problem) {
            miopen::solver::conv::ConvHipImplicitGemm3DGroupWrwXdlops solver;
            if(!solver.IsApplicable(ctx, problem))
                return std::vector<std::string>{};
            auto perf_cfg = solver.GetDefaultPerformanceConfig(ctx, problem);
            auto solution = solver.GetSolution(ctx, problem, perf_cfg);
            std::vector<std::string> kernel_names;
            for(const auto& cp : solution.construction_params)
                kernel_names.push_back(cp.kernel_name);
            return kernel_names;
        };

    // 4. Prepare outputs
    std::vector<std::string> valid_kernels;
    int index = 0, split_k = 1;
    std::string kernel_id;
    std::string solver_name = "ConvHipImplicitGemm3DGroupWrwXdlops";

    // 5. Call the model
    bool result = miopen::solver::conv::RunParameterPredictionModel<float>(
        ctx, problem, valid_kernels, index, split_k, kernel_id, fill_valid_kernels, solver_name);

    // 6. Check outputs
    if(!result)
    {
        std::cerr << "RunParameterPredictionModel: Model did not return a valid result."
                  << std::endl;
        std::abort();
    }
    if(index < 0 || split_k < 1 || kernel_id.empty())
    {
        std::cerr << "RunParameterPredictionModel: Output values not set as expected." << std::endl;
        std::abort();
    }
    std::cout << "RunParameterPredictionModel: index=" << index << ", split_k=" << split_k
              << ", kernel_id=" << kernel_id << std::endl;
}

int main()
{
    // test related to GetFeatures3D
    TestGetFeatures3D();
    TestGetFeatures3D_ValueChecks();
    TestGetFeatures3D_Directions();
    TestGetFeatures3D_DataTypes();

    // test related to kernel tuning utils
    TestTokenizeKernel();
    TestFilterHeuristicKernels();
    TestGenerateSplitK();
    TestExpandKernelParamsWithSplitK();

    // test of main prediction model
    TestRunParameterPredictionModel();

    std::cout << "All tests passed." << std::endl;
    return 0;
}
