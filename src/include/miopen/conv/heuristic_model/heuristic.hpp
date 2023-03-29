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

#ifndef GAURD_MIOPEN_HEURISTIC_HPP_
#define GAURD_MIOPEN_HEURISTIC_HPP_

#if MIOPEN_ENABLE_AI_HEUR
#include <fdeep/fdeep.hpp>
#endif
#include <miopen/conv/context.hpp>
#include <miopen/solver.hpp>
#include <unordered_map>
#include <queue>
#include <typeinfo>
#include <string>

namespace miopen {
namespace ai {
namespace tn {
inline nlohmann::json GetMetadata(const std::string& arch)
{
    if (arch == "gfx908")
    {
        static const std::string file_path = GetSystemDbPath() + "/" + arch + "_metadata.tn.model";
        static const nlohmann::json metadata = nlohmann::json::parse(std::ifstream(file_path));
        return metadata;
    }
    throw std::invalid_argument("TunaNet not supported for " + arch);
}

inline const std::vector<std::string> GetFeatureNames(const nlohmann::json& metadata)
{
    static const std::vector<std::string> feature_names = metadata["conv_params_used_as_features"];
    return feature_names;
}

inline const std::unordered_map<size_t, std::string> GetSolverMap(const nlohmann::json& metadata)
{
    static std::unordered_map<size_t, std::string> solver_map{};
    if(solver_map.empty())
    {
        std::unordered_map<std::string, size_t> solver_map_rev = metadata["encodings"]["solver"];
        for(auto& it : solver_map_rev)
            solver_map.emplace(make_pair(it.second, it.first));
    }
    return solver_map;
}

inline const size_t GetNumSolvers(const nlohmann::json& metadata)
{
    static const size_t num_solvers = metadata["num_solvers"];
    return num_solvers;
}

inline const size_t GetDirectionCode(const miopen::conv::Direction dir, const nlohmann::json& metadata)
{
    if(dir == conv::Direction::BackwardWeights)
        return metadata["encodings"]["Direction"]["W"];
    else if(dir == conv::Direction::BackwardData)
        return metadata["encodings"]["Direction"]["B"];
    else if(dir == conv::Direction::Forward)
        return metadata["encodings"]["Direction"]["F"];
    else
        throw std::invalid_argument("Invalid direction");
}

inline const size_t GetPrecisionCode(const miopenDataType_t data_type, const nlohmann::json& metadata)
{
    if(data_type == miopenBFloat16)
        return metadata["encodings"]["Precision"]["BF16"];
    else if(data_type == miopenHalf)
        return metadata["encodings"]["Precision"]["FP16"];
    else if(data_type == miopenFloat)
        return metadata["encodings"]["Precision"]["FP32"];
    else
        throw std::invalid_argument("TunaNet doesn't support this precision");
}

inline const size_t GetLayoutCode(const std::string& layout, const nlohmann::json& metadata)
{
    if(layout == "NCDHW")
        return metadata["encodings"]["Layout"]["NCDHW"];
    else if(layout == "NCHW")
        return metadata["encodings"]["Layout"]["NCHW"];
    else
        throw std::invalid_argument("TunaNet doesn't support this layout");
}
} // namespace tn

namespace ktn {
inline nlohmann::json get_metadata(const std::string& arch, const std::string& solver)
{
    std::string file_path = GetSystemDbPath() + "/" + arch + "_" + solver + "_metadata.ktn.model";
    return nlohmann::json::parse(std::ifstream(file_path));
}

inline fdeep::model
get_model(const std::string& arch, const std::string& solver, const std::string& model_type)
{
    std::string file_path =
        GetSystemDbPath() + "/" + arch + "_" + solver + "_" + model_type + ".ktn.model";
    return fdeep::load_model(file_path, true, fdeep::dev_null_logger);
}

inline bool model_set_params(const fdeep::model& encoder,
                             const fdeep::model& decoder,
                             const nlohmann::json& metadata,
                             solver::PerformanceConfigConvAsm1x1U& config,
                             const ProblemDescription& problem,
                             std::vector<float>& features)
{
    MIOPEN_LOG_I("");

    int dim            = std::sqrt(features.size());
    auto input_tensor  = fdeep::tensor(fdeep::tensor_shape(dim, dim), features);
    auto hidden_states = encoder.predict({input_tensor}); // Get hidden states from Encoder LSTM

    std::vector<float> decoder_input_vector(1, 0.0);
    auto decoder_input_tensor = fdeep::tensor(fdeep::tensor_shape(1), decoder_input_vector);
    std::vector<fdeep::tensor> decoder_input = {
        decoder_input_tensor,
        hidden_states[0],
        hidden_states[1],
        hidden_states[2],
        hidden_states[3]}; // pass in SOS token and hidden states

    for(int i = 0; i < metadata["num_tuning_params"].get<int>(); i++)
    {
        auto output        = decoder.predict({decoder_input});
        auto output_vector = output[0].to_vector();
        std::priority_queue<std::pair<float, int>> pq;
        for(int j = 0; j < output_vector.size(); j++)
        {
            pq.push(std::make_pair(output_vector[j], j)); // sort by value at index
        }
        int output_token_index = -1;
        while(!pq.empty())
        {
            int token = pq.top().second;
            int value = metadata["decodings"]["tunings"][std::to_string(token)]
                            .get<int>(); // convert index to tuning value
            pq.pop();
            if(value < 0)
                return false;
            if(config.TryToken(i, value, problem))
            {
                output_token_index =
                    token; // index with largest value that is valid = predicted index
                break;
            }
        }
        decoder_input_tensor =
            fdeep::tensor(fdeep::tensor_shape(1), std::vector<float>(1, float(output_token_index)));
        decoder_input = {decoder_input_tensor,
                         output[1],
                         output[2],
                         output[3],
                         output[4]}; // index fed into decoder along with hidden states produced
                                     // from previous step
    }
    return true;
}
} // namespace ktn
} // namespace ai
} // namespace miopen
#endif
