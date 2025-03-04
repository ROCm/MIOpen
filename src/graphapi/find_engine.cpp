
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

#include <miopen/errors.hpp>
#include <miopen/miopen.h>
#include <miopen/graphapi/engine.hpp>
#include <miopen/graphapi/matmul.hpp>
#include <miopen/graphapi/opgraph.hpp>
#include <miopen/graphapi/pointwise.hpp>
#include <miopen/graphapi/reduction.hpp>
#include <miopen/graphapi/reshape.hpp>
#include <miopen/graphapi/rng.hpp>
#include <miopen/graphapi/util.hpp>
#include <miopen/graphapi/variant_pack.hpp>
#include <miopen/graphapi/convolution.hpp>
#include <miopen/graphapi/conv_bias_res_add_activ_forward_executor.hpp>
#include <miopen/utility/scope.hpp>

namespace miopen {
namespace graphapi {

GraphPatternMatcher::~GraphPatternMatcher() = default;

class ConvBiasResAddActive_Fwd_Pattern : public GraphPatternMatcher
{
    struct OperationPointwiseWithOneVirtualInput
    {
        Tensor* concreteTensor;
        Tensor* virtualTensor;
        float concreteAlpha;
        float virtualAlpha;

        OperationPointwiseWithOneVirtualInput(const OperationPointwise* pointwise)
        {
            auto convertToFloat = [](auto&& arg) { return static_cast<float>(arg); };
            if(pointwise->getX()->isVirtual())
            {
                virtualTensor = pointwise->getX();
                virtualAlpha  = std::visit(convertToFloat, pointwise->getAlpha1());

                concreteTensor = pointwise->getB();
                concreteAlpha  = std::visit(convertToFloat, pointwise->getAlpha2());
            }
            else
            {
                concreteTensor = pointwise->getX();
                concreteAlpha  = std::visit(convertToFloat, pointwise->getAlpha1());

                virtualTensor = pointwise->getB();
                virtualAlpha  = std::visit(convertToFloat, pointwise->getAlpha2());
            }
        }
    };

    static const OpGraph& getPatternGraph()
    {
        static auto graph_gen =
            PatternGraphGenerator::Make({{"OP_CONVOLUTION_FORWARD", {"X", "W"}, {"T_C_0"}},
                                         {"OP_POINTWISE:ADD", {"T_C_0", "Z"}, {"T_A_0"}},
                                         {"OP_POINTWISE:ADD", {"T_A_0", "BIAS"}, {"T_A_1"}},
                                         {"OP_POINTWISE:RELU_FWD", {"T_A_1"}, {"Y"}}});

        return graph_gen->graph();
    }

    static bool isBiasNode(OperationPointwise* addNode)
    {
        OperationPointwiseWithOneVirtualInput add(addNode);

        auto& lengthsToCheck = add.concreteTensor->GetLengths();

        return std::count_if(lengthsToCheck.cbegin(), lengthsToCheck.cend(), [](std::size_t value) {
                   return value > size_t{1};
               }) <= 1;
    }

    static bool hasBiasNode(const OpGraph& graph)
    {
        auto* conv = dynamic_cast<OperationConvolutionForward*>(
            graph.findOutNeighByName(graph.getSourceNode(), "OP_CONVOLUTION_FORWARD"));

        auto* add1 =
            dynamic_cast<OperationPointwise*>(graph.findOutNeighByName(conv, "OP_POINTWISE:ADD"));

        auto* add2 =
            dynamic_cast<OperationPointwise*>(graph.findOutNeighByName(add1, "OP_POINTWISE:ADD"));

        return isBiasNode(add1) || isBiasNode(add2);
    }

public:
    static std::unique_ptr<GraphPatternMatcher> Make()
    {
        return std::make_unique<ConvBiasResAddActive_Fwd_Pattern>();
    }

    std::string_view name() const final
    {
        static const std::string_view n{"convbiasresaddactivation_fwd"};
        return n;
    }

    bool matches(const OpGraph* graph_ptr) const final
    {
        assert(graph_ptr);

        if(!isIsomorphic(*graph_ptr, getPatternGraph()))
        {
            return false;
        }

        return hasBiasNode(*graph_ptr);
    }

    std::vector<Engine> getEngines(OpGraph* graph_ptr) const override
    {
        assert(graph_ptr);
        assert(matches(graph_ptr));
        auto& graph = *graph_ptr;

        auto* conv = dynamic_cast<OperationConvolutionForward*>(
            graph.findOutNeighByName(graph.getSourceNode(), "OP_CONVOLUTION_FORWARD"));

        std::size_t in_c  = conv->getX()->GetLengths()[1];
        std::size_t wei_c = conv->getW()->GetLengths()[1];

        if(wei_c == std::size_t{0})
        {
            MIOPEN_THROW(
                miopenStatusBadParm,
                "invalid weight tensor provided for graph matching ConvBiasResAddActive pattern");
        }
        else if(in_c % wei_c != std::size_t{0})
        {
            MIOPEN_THROW(miopenStatusBadParm,
                         "invalid group count from input and weight tensor for graph matching "
                         "ConvBiasResAddActive pattern");
        }

        int groupCount = in_c / wei_c;

        auto* add1 =
            dynamic_cast<OperationPointwise*>(graph.findOutNeighByName(conv, "OP_POINTWISE:ADD"));

        auto* add2 =
            dynamic_cast<OperationPointwise*>(graph.findOutNeighByName(add1, "OP_POINTWISE:ADD"));

        auto* activ = dynamic_cast<OperationPointwise*>(
            graph.findOutNeighByName(add2, "OP_POINTWISE:RELU_FWD"));

        auto [addTemp, biasTemp] =
            isBiasNode(add1) ? std::make_pair(add2, add1) : std::make_pair(add1, add2);

        OperationPointwiseWithOneVirtualInput add(addTemp);
        OperationPointwiseWithOneVirtualInput bias(biasTemp);

        // The virtual tensor for the add is the result of the convolution, and combining the
        // alpha1's allow for users to specify the alpha on either, or both nodes, and have it be
        // correct.
        float alpha1 = conv->getAlpha() * add.virtualAlpha;
        float alpha2 = add.concreteAlpha;

        std::shared_ptr<GraphPatternExecutor> exec =
            std::make_shared<ConvBiasResAddActivForwardExecutor>(
                conv->getX(),
                conv->getW(),
                conv->getConvolution(),
                groupCount,
                add.concreteTensor,
                bias.concreteTensor,
                activ->getY(),
                alpha1,
                alpha2,
                std::visit([](auto&& arg) { return static_cast<float>(arg); }, activ->getAlpha1()));
        return {EngineBuilder().setGraph(graph_ptr).setExecutor(exec).setGlobalIndex(0).build()};
    }
};

class MHA_Fwd_F8_Pattern : public GraphPatternMatcher
{
    static const OpGraph& getPatternGraph()
    {
        static auto graph_gen = PatternGraphGenerator::Make({
            {"OP_MATMUL", {"Q", "K"}, {"T_BMM_0"}},
            // {"OP_POINTWISE:MUL", {"T_BMM_0", "ATTN_S"}, {"PW_S_0"}},
            // replacing the MUL node above with IDENTITY node below for now
            // because Find 2.0 mha descriptor expects attention scale as a scalar
            // parameter on host side
            {"OP_POINTWISE:IDENTITY", {"T_BMM_0"}, {"PW_S_0"}},
            {"OP_POINTWISE:MUL", {"PW_S_0", "DSCL_Q"}, {"PW_S_1"}},
            {"OP_POINTWISE:MUL", {"PW_S_1", "DSCL_K"}, {"PW_S_2"}},
            {"OP_REDUCTION:MAX", {"PW_S_2"}, {"M"}},
            {"OP_POINTWISE:SUB", {"PW_S_2", "M"}, {"T_SUB"}},
            {"OP_POINTWISE:EXP", {"T_SUB"}, {"T_EXP"}},
            {"OP_REDUCTION:ADD", {"T_EXP"}, {"T_SUM"}},
            {"OP_POINTWISE:RECIPROCAL", {"T_SUM"}, {"Z_INV"}},
            {"OP_POINTWISE:MUL", {"Z_INV", "T_EXP"}, {"T_MUL_0"}},
            {"OP_REDUCTION:MAX", {"T_MUL_0"}, {"AMAX_S"}},
            {"OP_RNG", {"SEED", "OFFSET"}, {"T_RND"}},
            {"OP_POINTWISE:MUL", {"T_RND", "T_MUL_0"}, {"T_MUL_1"}},
            {"OP_POINTWISE:MUL", {"T_MUL_1", "I_PROB"}, {"PW_S_3"}},
            {"OP_POINTWISE:MUL", {"PW_S_3", "SCL_S"}, {"PW_S_4"}},
            {"OP_MATMUL", {"PW_S_4", "V"}, {"T_BMM_1"}},
            {"OP_POINTWISE:MUL", {"T_BMM_1", "DSCL_S"}, {"PW_S_5"}},
            {"OP_POINTWISE:MUL", {"PW_S_5", "DSCL_V"}, {"PW_S_6"}},
            {"OP_POINTWISE:MUL", {"PW_S_6", "SCL_O"}, {"O"}},
            {"OP_REDUCTION:MAX", {"PW_S_6"}, {"AMAX_O"}},

        });

        return graph_gen->graph();
    }

    std::shared_ptr<TensorInfoMap> extractFind20Tensors(const OpGraph& graph,
                                                        float* attn_scale) const
    {

        assert(attn_scale);
        std::vector<std::size_t> all1s = {
            std::size_t{1}, std::size_t{1}, std::size_t{1}, std::size_t{1}};

        auto tensor_map = std::make_shared<TensorInfoMap>();

        auto add_mapping = [&](miopenTensorArgumentId_t enum_id, Tensor* tens_ptr) {
            assert(tens_ptr);
            assert(enum_id != miopenTensorArgumentIdInvalid);

            /*
             * leaving to check that correct tensors are picked
            MIOPEN_LOG_W("Tensor enum id: " << tensorEnumIdToStr(enum_id)
                                            << " tensor unique id as str: "
                                            << tensorIdAsStr(tens_ptr));
            */

            tensor_map->try_emplace(tens_ptr->getId(), TensorInfo(enum_id, tens_ptr));
        };

        for(auto [neigh, tens_ptr] : graph.getOutEdges(graph.getSourceNode()))
        {

            if(neigh->signName() == "OP_MATMUL")
            {
                auto* matmul = dynamic_cast<OperationMatmul*>(neigh);
                assert(matmul);

                if(auto* pw_prev = graph.findInNeighByName(matmul, "OP_POINTWISE:MUL");
                   pw_prev == nullptr)
                {
                    // this is the first matmul node
                    add_mapping(miopenTensorMhaQ, matmul->getA());
                    add_mapping(miopenTensorMhaK, matmul->getB());
                    /// \todo dim check on Q and K --amberhassaan May, 2024

                    /// \note  old code assuming attn_scale applies to pw_mult
                    // auto* pw_0 = dynamic_cast<OperationPointwise*>(
                    // graph.findOutNeighByName(matmul, "OP_POINTWISE:MUL"));
                    auto* pw_0 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(matmul, "OP_POINTWISE:IDENTITY"));
                    assert(pw_0);

                    float alpha1 = std::get<float>(pw_0->getAlpha1());
                    *attn_scale  = alpha1;
                    /// \note old code assuming attn_scale applies to pw_mult
                    // auto* attn_scl = pw_0->getB();
                    // assert(attn_scl->GetLengths() == all1s);
                    // add_mapping(miopenTensorMhaAttnScale, attn_scl);

                    auto* pw_1 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(pw_0, "OP_POINTWISE:MUL"));
                    assert(pw_1);
                    auto dscl_q = pw_1->getB();
                    assert(dscl_q->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaDescaleQ, dscl_q);

                    auto* pw_2 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(pw_1, "OP_POINTWISE:MUL"));
                    assert(pw_2);
                    auto* dscl_k = pw_2->getB();
                    assert(dscl_k->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaDescaleK, dscl_k);

                    auto* red = dynamic_cast<OperationReduction*>(
                        graph.findOutNeighByName(pw_2, "OP_REDUCTION:MAX"));
                    assert(red);
                    auto* m = red->getY();
                    assert(m->GetLengths()[3] == 1LL);
                    add_mapping(miopenTensorMhaM, m);
                }
                else
                {
                    // this is the second matmul node
                    add_mapping(miopenTensorMhaV, matmul->getB());

                    auto* pw_prev_cast = dynamic_cast<OperationPointwise*>(pw_prev);
                    assert(pw_prev_cast);
                    auto* scl_s = pw_prev_cast->getB();
                    assert(scl_s->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaScaleS, scl_s);

                    auto* pw_0 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(matmul, "OP_POINTWISE:MUL"));
                    assert(pw_0);

                    auto* dscl_s = pw_0->getB();
                    assert(dscl_s->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaDescaleS, dscl_s);

                    auto* pw_1 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(pw_0, "OP_POINTWISE:MUL"));
                    assert(pw_1);
                    auto* dscl_v = pw_1->getB();
                    assert(dscl_v->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaDescaleV, dscl_v);

                    auto* red = dynamic_cast<OperationReduction*>(
                        graph.findOutNeighByName(pw_1, "OP_REDUCTION:MAX"));
                    assert(red);
                    auto* amax_o = red->getY();
                    assert(amax_o->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaAmaxO, amax_o);

                    auto* pw_2 = dynamic_cast<OperationPointwise*>(
                        graph.findOutNeighByName(pw_1, "OP_POINTWISE:MUL"));
                    assert(pw_2);
                    auto* scl_o = pw_2->getB();
                    assert(scl_o->GetLengths() == all1s);
                    add_mapping(miopenTensorMhaScaleO, scl_o);

                    auto* o = pw_2->getY();
                    add_mapping(miopenTensorMhaO, o);
                }
            }
            else if(neigh->signName() == "OP_RNG")
            {
                auto* rng = dynamic_cast<OperationRng*>(neigh);
                assert(rng);
                add_mapping(miopenTensorMhaDropoutSeed, std::get<Tensor*>(rng->getSeed()));
                add_mapping(miopenTensorMhaDropoutOffset, rng->getOffset());

                auto* pw_mult_0 = dynamic_cast<OperationPointwise*>(
                    graph.findOutNeighByName(rng, "OP_POINTWISE:MUL"));
                assert(pw_mult_0);

                auto* pw_mult_1 = dynamic_cast<OperationPointwise*>(
                    graph.findOutNeighByName(pw_mult_0, "OP_POINTWISE:MUL"));
                assert(pw_mult_1);

                auto* prob = pw_mult_1->getB();
                add_mapping(miopenTensorMhaDropoutProbability, prob);
            }
        }

        { // discovering Z_INV and AMAX_S tensors
            auto* exp_node = graph.findNodeByName("OP_POINTWISE:EXP");
            assert(exp_node);

            // get exp_node's neighbor that is a Pointwise mult
            auto* pw_mult = dynamic_cast<OperationPointwise*>(
                graph.findOutNeighByName(exp_node, "OP_POINTWISE:MUL"));
            assert(pw_mult);

            auto* red = dynamic_cast<OperationReduction*>(
                graph.findOutNeighByName(pw_mult, "OP_REDUCTION:MAX"));
            assert(red);

            add_mapping(miopenTensorMhaAmaxS, red->getY());

            auto* inv_node =
                dynamic_cast<OperationPointwise*>(graph.findNodeByName("OP_POINTWISE:RECIPROCAL"));
            add_mapping(miopenTensorMhaZInv, inv_node->getY());
        }

        return tensor_map;
    }

public:
    static std::unique_ptr<GraphPatternMatcher> Make()
    {
        return std::make_unique<MHA_Fwd_F8_Pattern>();
    }

    std::string_view name() const final
    {
        static const char* n = "mha_fwd_f8";
        return n;
    }

    bool matches(const OpGraph* graph_ptr) const final
    {
        assert(graph_ptr);
        return isIsomorphic(*graph_ptr, getPatternGraph());
    }

    std::vector<Engine> getEngines(OpGraph* graph_ptr) const override
    {
        assert(graph_ptr);
        assert(matches(graph_ptr));
        auto& graph = *graph_ptr;

        miopenProblem_t mha_prob;
        MhaDescriptor mha_desc;

        float attn_scale                          = std::numeric_limits<float>::quiet_NaN();
        std::shared_ptr<TensorInfoMap> tensor_map = extractFind20Tensors(graph, &attn_scale);
        assert(attn_scale != std::numeric_limits<float>::quiet_NaN());

        mha_desc.SetParams(attn_scale);

        auto s = miopenCreateMhaProblem(&mha_prob, &mha_desc, miopenProblemDirectionForward);
        MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while creating problem for mha fwd");

        // Ensure miopenDestroyProblem() will be called even if an exception occurs
        scope_exit finallyForProblem([=]() { miopenDestroyProblem(mha_prob); });

        for(auto& [k, v] : *tensor_map)
        {
            s = miopenSetProblemTensorDescriptor(mha_prob, v.mEnumId, v.mGraphTensor);
            MIOPEN_THROW_IF(s != miopenStatusSuccess,
                            "failed while setting tensor descriptor for mha fwd");
        }

        std::vector<miopenSolution_t> solutions(10);
        size_t num_found = 0;
        s                = miopenFindSolutions(
            graph.getHandle(), mha_prob, nullptr, solutions.data(), &num_found, solutions.size());
        MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while finding solutions for mha fwd");

        solutions.resize(num_found);

        // Ensure miopenDestroySolution() will be called even if an exception occurs
        scope_exit finallyForSolutions([&]() {
            for(miopenSolution_t sol : solutions)
            {
                miopenDestroySolution(sol);
            }
        });

        std::vector<Engine> engines;
        engines.reserve(num_found);

        size_t i = 0;
        for(miopenSolution_t sol : solutions)
        {
            std::shared_ptr<GraphPatternExecutor> exec =
                std::make_shared<GraphExecutorFind20>(std::move(deref(sol)), tensor_map);

            engines.emplace_back(
                EngineBuilder().setGraph(graph_ptr).setExecutor(exec).setGlobalIndex(i).build());
            ++i;
        }

        return engines;
    }
};

class MHA_Bwd_F8_Pattern : public GraphPatternMatcher
{
    static const OpGraph& getPatternGraph()
    {
        static auto graph_gen = PatternGraphGenerator::Make({
            {"OP_RESHAPE", {"K"}, {"KT"}},
            {"OP_MATMUL", {"Q", "KT"}, {"MM0"}},
            {"OP_POINTWISE:IDENTITY", {"MM0"}, {"PWS0"}},
            // {"OP_POINTWISE:MUL", {"MM0", "ATTN_S"}, {"PWS0"}},
            // replacing the MUL node above with IDENTITY node below for now
            // because Find 2.0 mha descriptor expects attention scale as a scalar
            // parameter on host side
            {"OP_POINTWISE:MUL", {"PWS0", "DSCL_Q"}, {"PWS1"}},
            {"OP_POINTWISE:MUL", {"PWS1", "DSCL_K"}, {"PWS2"}},
            {"OP_POINTWISE:SUB", {"PWS2", "M"}, {"SUB0"}},
            {"OP_POINTWISE:EXP", {"SUB0"}, {"EXP0"}},
            {"OP_POINTWISE:MUL", {"EXP0", "ZINV"}, {"MULT0"}},
            {"OP_RNG", {"SEED", "OFFSET"}, {"RND"}},
            {"OP_POINTWISE:MUL", {"MULT0", "RND"}, {"MULT1"}},
            {"OP_POINTWISE:MUL", {"MULT1", "I_PROB"}, {"PWS3"}},
            {"OP_POINTWISE:MUL", {"PWS3", "SCL_S"}, {"PWS4"}},
            {"OP_RESHAPE", {"PWS4"}, {"PWS4T"}},
            {"OP_MATMUL", {"PWS4T", "DO"}, {"MM1"}},
            {"OP_POINTWISE:MUL", {"MM1", "DSCL_S"}, {"PWS5"}},
            {"OP_POINTWISE:MUL", {"PWS5", "DSCL_DO"}, {"PWS6"}},
            {"OP_REDUCTION:MAX", {"PWS6"}, {"AMAX_DV"}},
            {"OP_POINTWISE:MUL", {"PWS6", "SCL_DV"}, {"DV"}},

            {"OP_RESHAPE", {"V"}, {"VT"}},
            {"OP_MATMUL", {"DO", "VT"}, {"MM2"}},
            {"OP_POINTWISE:MUL", {"MM2", "DSCL_DO"}, {"PWS7"}},
            {"OP_POINTWISE:MUL", {"PWS7", "DSCL_V"}, {"PWS8"}},
            {"OP_POINTWISE:MUL", {"PWS8", "RND"}, {"PWS9"}},
            {"OP_POINTWISE:MUL", {"PWS9", "PROB"}, {"PWS10"}},

            {"OP_POINTWISE:MUL", {"DO", "DSCL_DO"}, {"PWS11"}},
            {"OP_POINTWISE:MUL", {"O", "DSCL_O"}, {"PWS12"}},
            {"OP_POINTWISE:MUL", {"PWS11", "PWS12"}, {"MULT2"}},
            {"OP_POINTWISE:MUL", {"MULT2", "PROB"}, {"PWS13"}},
            {"OP_REDUCTION:ADD", {"PWS13"}, {"SUM0"}},

            {"OP_POINTWISE:SUB", {"PWS10", "SUM0"}, {"SUB1"}},
            {"OP_POINTWISE:IDENTITY", {"SUB1"}, {"PWS14"}},
            // {"OP_POINTWISE:MUL", {"SUB1", "ATTN_S"}, {"PWS0"}},
            // replacing the MUL node above with IDENTITY node below for now
            // because Find 2.0 mha descriptor expects attention scale as a scalar
            // parameter on host side
            {"OP_POINTWISE:MUL", {"PWS14", "PWS3"}, {"MULT3"}},
            {"OP_REDUCTION:MAX", {"MULT3"}, {"AMAX_DS"}},
            {"OP_POINTWISE:MUL", {"MULT3", "SCL_DS"}, {"PWS15"}},

            {"OP_MATMUL", {"PWS15", "K"}, {"MM3"}},
            {"OP_POINTWISE:MUL", {"MM3", "DSCL_DS"}, {"PWS16"}},
            {"OP_POINTWISE:MUL", {"PWS16", "DSCL_K"}, {"PWS17"}},
            {"OP_REDUCTION:MAX", {"PWS17"}, {"AMAX_DQ"}},
            {"OP_POINTWISE:MUL", {"PWS17", "SCL_DQ"}, {"DQ"}},

            {"OP_RESHAPE", {"PWS15"}, {"PWS15T"}},
            {"OP_MATMUL", {"PWS15T", "Q"}, {"MM4"}},
            {"OP_POINTWISE:MUL", {"MM4", "DSCL_DS"}, {"PWS18"}},
            {"OP_POINTWISE:MUL", {"PWS18", "DSCL_Q"}, {"PWS19"}},
            {"OP_REDUCTION:MAX", {"PWS19"}, {"AMAX_DK"}},
            {"OP_POINTWISE:MUL", {"PWS19", "SCL_DK"}, {"DK"}},

        });

        return graph_gen->graph();
    }

    std::shared_ptr<TensorInfoMap> extractFind20Tensors(const OpGraph& graph,
                                                        float* attnScale) const
    {
        assert(attnScale);

        auto tensorMap = std::make_shared<TensorInfoMap>();

        auto addMapping = [&](miopenTensorArgumentId_t enumId, Tensor* tensPtr) {
            assert(tensPtr);
            assert(enumId != miopenTensorArgumentIdInvalid);
            tensorMap->try_emplace(tensPtr->getId(), TensorInfo(enumId, tensPtr));
        };

        const auto& starts = graph.getOutEdges(graph.getSourceNode());

        // Find top left and center parts by `transpose` nodes

        auto isTranspose = [](const OpNode::Edge& edge) -> bool {
            auto [node, tensor] = edge;
            return node->signName() == "OP_RESHAPE";
        };

        auto leftOrCenterStartIt1 = std::find_if(starts.cbegin(), starts.cend(), isTranspose);
        assert(leftOrCenterStartIt1 != starts.cend());
        auto leftOrCenterStartIt2 =
            std::find_if(leftOrCenterStartIt1 + 1, starts.cend(), isTranspose);
        assert(leftOrCenterStartIt2 != starts.cend());

        // Tell left part from center one by `identity` node

        auto* leftOrCenterMatmulPtr =
            graph.findOutNeighByName(leftOrCenterStartIt1->first, "OP_MATMUL");
        assert(leftOrCenterMatmulPtr);

        auto* leftIdentityPtr =
            graph.findOutNeighByName(leftOrCenterMatmulPtr, "OP_POINTWISE:IDENTITY");

        OpNode* leftHead   = nullptr;
        OpNode* centerHead = nullptr;

        if(leftIdentityPtr != nullptr)
        {
            leftHead   = leftOrCenterStartIt1->first;
            centerHead = leftOrCenterStartIt2->first;
        }
        else
        {
            leftHead   = leftOrCenterStartIt2->first;
            centerHead = leftOrCenterStartIt1->first;
        }

        // Walk the left part

        OpNode* currentNode = leftHead;
        auto& leftReshape   = dynamic_cast<OperationReshape&>(*currentNode);
        addMapping(miopenTensorMhaK, leftReshape.getX());
        Tensor* prevOutputTensor = leftReshape.getY();

        currentNode = graph.findOutNeighByName(currentNode, "OP_MATMUL");
        assert(currentNode);
        auto& leftMatmul0 = dynamic_cast<OperationMatmul&>(*currentNode);
        if(leftMatmul0.getA() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaQ, leftMatmul0.getA());
        }
        else
        {
            addMapping(miopenTensorMhaQ, leftMatmul0.getB());
        }

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:IDENTITY");
        assert(currentNode);
        auto& leftIdentity = dynamic_cast<OperationPointwise&>(*currentNode);
        *attnScale         = std::get<float>(leftIdentity.getAlpha1());
        prevOutputTensor   = leftIdentity.getY();

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& leftMul0 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(leftMul0.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaDescaleQ, leftMul0.getX());
        }
        else
        {
            addMapping(miopenTensorMhaDescaleQ, leftMul0.getB());
        }
        prevOutputTensor = leftMul0.getY();

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& leftMul1 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(leftMul1.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaDescaleK, leftMul1.getX());
        }
        else
        {
            addMapping(miopenTensorMhaDescaleK, leftMul1.getB());
        }

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:SUB");
        assert(currentNode);
        addMapping(miopenTensorMhaM, dynamic_cast<OperationPointwise&>(*currentNode).getB());

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:EXP");
        assert(currentNode);
        prevOutputTensor = dynamic_cast<OperationPointwise&>(*currentNode).getY();

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& leftMul2 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(leftMul2.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaZInv, leftMul2.getX());
        }
        else
        {
            addMapping(miopenTensorMhaZInv, leftMul2.getB());
        }

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);

        // Stop at diversion point and remember it

        OpNode* leftDiversionPoint0 = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(leftDiversionPoint0);

        // Take a look at RNG node

        currentNode = graph.findInNeighByName(currentNode, "OP_RNG");
        assert(currentNode);
        auto& rng = dynamic_cast<OperationRng&>(*currentNode);
        addMapping(miopenTensorMhaDropoutSeed, std::get<Tensor*>(rng.getSeed()));
        addMapping(miopenTensorMhaDropoutOffset, rng.getOffset());

        // Walk the center part

        currentNode         = centerHead;
        auto& centerReshape = dynamic_cast<OperationReshape&>(*currentNode);
        addMapping(miopenTensorMhaV, centerReshape.getX());
        prevOutputTensor = centerReshape.getY();

        currentNode = graph.findOutNeighByName(currentNode, "OP_MATMUL");
        assert(currentNode);
        auto& centerMatmul0 = dynamic_cast<OperationMatmul&>(*currentNode);
        int64_t doId        = 0; // save id of DO tensor
        if(centerMatmul0.getA() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaDO, centerMatmul0.getA());
            doId = centerMatmul0.getA()->getId();
        }
        else
        {
            addMapping(miopenTensorMhaDO, centerMatmul0.getB());
            doId = centerMatmul0.getB()->getId();
        }
        prevOutputTensor = centerMatmul0.getC();

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& centerMul0    = dynamic_cast<OperationPointwise&>(*currentNode);
        int64_t descaleDoId = 0; // save id of DescaleDO tensor
        if(centerMul0.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaDescaleDO, centerMul0.getX());
            descaleDoId = centerMul0.getX()->getId();
        }
        else
        {
            addMapping(miopenTensorMhaDescaleDO, centerMul0.getB());
            descaleDoId = centerMul0.getB()->getId();
        }
        prevOutputTensor = centerMul0.getY();

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& centerMul1 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(centerMul1.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaDescaleV, centerMul1.getX());
        }
        else
        {
            addMapping(miopenTensorMhaDescaleV, centerMul1.getB());
        }

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        prevOutputTensor = dynamic_cast<OperationPointwise&>(*currentNode).getY();

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& centerMul3 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(centerMul3.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaDropoutProbability, centerMul3.getX());
        }
        else
        {
            addMapping(miopenTensorMhaDropoutProbability, centerMul3.getB());
        }

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:SUB");
        assert(currentNode);

        // save the tail of the right part
        OpNode* rightTail = graph.findInNeighByName(currentNode, "OP_REDUCTION:ADD");
        assert(rightTail);

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:IDENTITY");
        assert(currentNode);

        // save the neighbor of the left part's diversion point
        // it also happened to be the first diversion point of the center part
        OpNode* centerDiversionPoint0 = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(centerDiversionPoint0);

        currentNode = graph.findOutNeighByName(centerDiversionPoint0, "OP_REDUCTION:MAX");
        assert(currentNode);
        addMapping(miopenTensorMhaAmaxDS, dynamic_cast<OperationReduction&>(*currentNode).getY());

        currentNode = graph.findOutNeighByName(centerDiversionPoint0, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& centerMul5 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(centerMul5.getX() != dynamic_cast<OperationPointwise&>(*centerDiversionPoint0).getY())
        {
            addMapping(miopenTensorMhaScaleDS, centerMul5.getX());
        }
        else
        {
            addMapping(miopenTensorMhaScaleDS, centerMul5.getB());
        }

        // save the head of the right bottom part
        OpNode* rightBottomHead = graph.findOutNeighByName(currentNode, "OP_RESHAPE");
        assert(rightBottomHead);

        currentNode = graph.findOutNeighByName(currentNode, "OP_MATMUL");
        assert(currentNode);
        prevOutputTensor = dynamic_cast<OperationMatmul&>(*currentNode).getC();

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& centerMul6 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(centerMul6.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaDescaleDS, centerMul6.getX());
        }
        else
        {
            addMapping(miopenTensorMhaDescaleDS, centerMul6.getB());
        }

        // save the second center part's diversion point
        OpNode* centerDiversionPoint1 = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(centerDiversionPoint1);

        currentNode = graph.findOutNeighByName(centerDiversionPoint1, "OP_REDUCTION:MAX");
        assert(currentNode);
        addMapping(miopenTensorMhaAmaxDQ, dynamic_cast<OperationReduction&>(*currentNode).getY());

        currentNode = graph.findOutNeighByName(centerDiversionPoint1, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& centerMul8 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(centerMul8.getX() != dynamic_cast<OperationPointwise&>(*centerDiversionPoint1).getY())
        {
            addMapping(miopenTensorMhaScaleDQ, centerMul8.getX());
        }
        else
        {
            addMapping(miopenTensorMhaScaleDQ, centerMul8.getB());
        }
        addMapping(miopenTensorMhaDQ, centerMul8.getY());

        // Walk the right bottom part

        currentNode = graph.findOutNeighByName(rightBottomHead, "OP_MATMUL");
        assert(currentNode);

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);

        // save right bottom part's diversion point
        OpNode* rightBottomDiversionPoint =
            graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(rightBottomDiversionPoint);

        currentNode = graph.findOutNeighByName(rightBottomDiversionPoint, "OP_REDUCTION:MAX");
        assert(currentNode);
        addMapping(miopenTensorMhaAmaxDK, dynamic_cast<OperationReduction&>(*currentNode).getY());

        currentNode = graph.findOutNeighByName(rightBottomDiversionPoint, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& rightBottomMul2 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(rightBottomMul2.getX() !=
           dynamic_cast<OperationPointwise&>(*rightBottomDiversionPoint).getY())
        {
            addMapping(miopenTensorMhaScaleDK, rightBottomMul2.getX());
        }
        else
        {
            addMapping(miopenTensorMhaScaleDK, rightBottomMul2.getB());
        }
        addMapping(miopenTensorMhaDK, rightBottomMul2.getY());

        // Walk the right part backwards

        currentNode = graph.findInNeighByName(rightTail, "OP_POINTWISE:MUL");
        assert(currentNode);

        currentNode = graph.findInNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);

        // tell right part's heads from each other by ids of
        // DO and DescaleDO tensors which were saved above
        const auto& rightHeads = graph.getInEdges(currentNode);
        auto headWithO         = std::find_if(
            rightHeads.cbegin(), rightHeads.cend(), [=](const OpNode::Edge& edge) -> bool {
                if(edge.first->signName() == "OP_POINTWISE:MUL")
                {
                    auto& operation = dynamic_cast<OperationPointwise&>(*edge.first);
                    auto xId        = operation.getX()->getId();
                    auto bId        = operation.getB()->getId();
                    return xId != doId && xId != descaleDoId && bId != doId && bId != descaleDoId;
                }
                else
                {
                    return false;
                }
            });
        assert(headWithO != rightHeads.cend());
        auto& rightMulWithO = dynamic_cast<OperationPointwise&>(*headWithO->first);

        // we cannot distinguish O from DescaleO but it doesn't matter
        addMapping(miopenTensorMhaO, rightMulWithO.getX());
        addMapping(miopenTensorMhaDescaleO, rightMulWithO.getB());

        // Continue walking the remaining left part after the first diversion point

        // Tell between pointwise:mul descendants
        const auto& leftDivPointOutEdges = graph.getOutEdges(leftDiversionPoint0);
        auto remLeftPtHeadIt             = std::find_if(leftDivPointOutEdges.cbegin(),
                                            leftDivPointOutEdges.cend(),
                                            [=](const OpNode::Edge& edge) -> bool {
                                                return edge.first != centerDiversionPoint0 &&
                                                       edge.first->signName() == "OP_POINTWISE:MUL";
                                            });
        assert(remLeftPtHeadIt != leftDivPointOutEdges.cend());

        currentNode      = remLeftPtHeadIt->first;
        prevOutputTensor = dynamic_cast<OperationPointwise&>(*leftDiversionPoint0).getY();
        auto& leftMul5   = dynamic_cast<OperationPointwise&>(*currentNode);
        if(leftMul5.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaScaleS, leftMul5.getX());
        }
        else
        {
            addMapping(miopenTensorMhaScaleS, leftMul5.getB());
        }

        currentNode = graph.findOutNeighByName(currentNode, "OP_RESHAPE");
        assert(currentNode);

        currentNode = graph.findOutNeighByName(currentNode, "OP_MATMUL");
        assert(currentNode);
        prevOutputTensor = dynamic_cast<OperationMatmul&>(*currentNode).getC();

        currentNode = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& leftMul6 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(leftMul6.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaDescaleS, leftMul6.getX());
        }
        else
        {
            addMapping(miopenTensorMhaDescaleS, leftMul6.getB());
        }

        OpNode* leftDiversionPoint1 = graph.findOutNeighByName(currentNode, "OP_POINTWISE:MUL");
        assert(leftDiversionPoint1);
        prevOutputTensor = dynamic_cast<OperationPointwise&>(*leftDiversionPoint1).getY();

        currentNode = graph.findOutNeighByName(leftDiversionPoint1, "OP_REDUCTION:MAX");
        assert(currentNode);
        addMapping(miopenTensorMhaAmaxDV, dynamic_cast<OperationReduction&>(*currentNode).getY());

        currentNode = graph.findOutNeighByName(leftDiversionPoint1, "OP_POINTWISE:MUL");
        assert(currentNode);
        auto& leftMul7 = dynamic_cast<OperationPointwise&>(*currentNode);
        if(leftMul7.getX() != prevOutputTensor)
        {
            addMapping(miopenTensorMhaScaleDV, leftMul7.getX());
        }
        else
        {
            addMapping(miopenTensorMhaScaleDV, leftMul7.getB());
        }
        addMapping(miopenTensorMhaDV, leftMul7.getY());

        return tensorMap;
    }

public:
    static std::unique_ptr<GraphPatternMatcher> Make()
    {
        return std::make_unique<MHA_Bwd_F8_Pattern>();
    }

    std::string_view name() const final
    {
        static const char* n = "mha_bwd_f8";
        return n;
    }

    bool matches(const OpGraph* graph_ptr) const final
    {
        assert(graph_ptr);
        return isIsomorphic(*graph_ptr, getPatternGraph());
    }

    std::vector<Engine> getEngines(OpGraph* graphPtr) const override
    {
        assert(graphPtr);
        assert(matches(graphPtr));
        auto& graph = *graphPtr;

        MhaDescriptor mhaDesc;

        float attnScale                          = std::numeric_limits<float>::quiet_NaN();
        std::shared_ptr<TensorInfoMap> tensorMap = extractFind20Tensors(graph, &attnScale);
        assert(attnScale != std::numeric_limits<float>::quiet_NaN());

        mhaDesc.SetParams(attnScale);

        miopenProblem_t mhaProblem;

        auto s = miopenCreateMhaProblem(&mhaProblem, &mhaDesc, miopenProblemDirectionBackward);
        MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while creating problem for mha bwd");

        // Ensure miopenDestroyProblem() will be called even if an exception occurs
        scope_exit finallyForProblem([=]() { miopenDestroyProblem(mhaProblem); });

        for(auto& [k, v] : *tensorMap)
        {
            s = miopenSetProblemTensorDescriptor(mhaProblem, v.mEnumId, v.mGraphTensor);
            MIOPEN_THROW_IF(s != miopenStatusSuccess,
                            "failed while setting tensor descriptor for mha bwd");
        }

        std::vector<miopenSolution_t> solutions(10);
        size_t numFound = 0;
        s               = miopenFindSolutions(
            graph.getHandle(), mhaProblem, nullptr, solutions.data(), &numFound, solutions.size());
        MIOPEN_THROW_IF(s != miopenStatusSuccess, "failed while finding solutions for mha bwd");

        solutions.resize(numFound);

        // Ensure miopenDestroySolution() will be called even if an exception occurs
        scope_exit finallyForSolutions([&]() {
            for(miopenSolution_t sol : solutions)
            {
                miopenDestroySolution(sol);
            }
        });

        std::vector<Engine> engines;
        engines.reserve(numFound);

        size_t i = 0;
        std::transform(solutions.cbegin(),
                       solutions.cend(),
                       std::back_inserter(engines),
                       [&i, tensorMap, graphPtr](miopenSolution_t sol) -> Engine {
                           return EngineBuilder()
                               .setGraph(graphPtr)
                               .setExecutor(std::make_shared<GraphExecutorFind20>(
                                   std::move(deref(sol)), tensorMap))
                               .setGlobalIndex(i++)
                               .build();
                       });

        return engines;
    }
};

std::vector<Engine> findEngines(OpGraph* graph)
{
    assert(graph);

    std::vector<std::unique_ptr<GraphPatternMatcher>> patterns;
    patterns.emplace_back(MHA_Fwd_F8_Pattern::Make());
    patterns.emplace_back(MHA_Bwd_F8_Pattern::Make());
    patterns.emplace_back(ConvBiasResAddActive_Fwd_Pattern::Make());

    for(const auto& p : patterns)
    {
        if(p->matches(graph))
        {
            MIOPEN_LOG_I2("Matched against pattern: " << p->name());
            return p->getEngines(graph);
        }
    }

    return {};
}

} // end namespace graphapi
} // end namespace miopen
