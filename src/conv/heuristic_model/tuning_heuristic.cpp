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

#include <nlohmann/json.hpp>
#include <miopen/db_path.hpp>
#include <miopen/config.h>
#include <miopen/conv/heuristic_model/tuning_heuristic.hpp>
#include <string>
#include <fstream>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <fdeep/fdeep.hpp>

namespace miopen {
namespace ai {
namespace tuning {

struct PerfTuningModel::impl
{
    nlohmann::json GetModelMetadata()
    {
        std::string file_path = GetSystemDbPath() + "/" + arch + "_" + solver + "_metadata.model";
        return nlohmann::json::parse(std::ifstream(file_path));
    }

    static fdeep::model
    GetModel(const std::string& arch, const std::string& solver, const std::string& model_type)
    {
        std::string file_path =
            GetSystemDbPath() + "/" + arch + "_" + solver + "_" + model_type + ".model";
        return fdeep::load_model(file_path, true, fdeep::dev_null_logger);
    }

    impl(const std::string& arch_, const std::string& solver_)
        : arch(arch_),
          solver(solver_),
          encoder(GetModel(arch, solver, "encoder")),
          decoder(GetModel(arch, solver, "decoder")),
          metadata(GetModelMetadata())
    {
    }

    bool ModelSetParams(std::function<bool(int, int)> validator,
                        const std::vector<float>& features) const
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
                pq.push(std::make_pair(output_vector[j], j)); // sort by value at index

            int output_token_index = -1;
            while(!pq.empty())
            {
                int token = pq.top().second;
                int value = metadata["decodings"]["tunings"][std::to_string(token)]
                                .get<int>(); // convert index to tuning value
                pq.pop();
                if(value < 0)
                    return false;
                if(validator(i, value))
                {
                    output_token_index =
                        token; // index with largest value that is valid = predicted index
                    break;
                }
            }
            decoder_input_tensor = fdeep::tensor(fdeep::tensor_shape(1),
                                                 std::vector<float>(1, float(output_token_index)));
            decoder_input        = {decoder_input_tensor,
                             output[1],
                             output[2],
                             output[3],
                             output[4]}; // index fed into decoder along with hidden states produced
                                                // from previous step
        }
        return true;
    }
    size_t GetNumParams() const { return metadata["num_conv_params"].get<std::size_t>(); }
    std::string arch;
    std::string solver;
    fdeep::model encoder;
    fdeep::model decoder;
    nlohmann::json metadata;
};

PerfTuningModel::PerfTuningModel() : pImpl(nullptr) {}
PerfTuningModel::PerfTuningModel(const std::string& arch, const std::string& solver)
    : pImpl{std::make_unique<impl>(arch, solver)}
{
}
PerfTuningModel::~PerfTuningModel()                          = default;
PerfTuningModel::PerfTuningModel(PerfTuningModel&&) noexcept = default;
PerfTuningModel& PerfTuningModel::operator=(PerfTuningModel&&) noexcept = default;

bool PerfTuningModel::ModelSetParams(std::function<bool(int, int)> validator,
                                     const std::vector<float>& features) const
{
    return this->pImpl->ModelSetParams(validator, features);
}

size_t PerfTuningModel::GetNumParams() const { return this->pImpl->GetNumParams(); }

} // namespace tuning
} // namespace ai
} // namespace miopen
#endif
