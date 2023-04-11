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

//#include <miopen/config.h>
#include <miopen/conv/heuristic_model/tuning_heuristic.hpp>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <fdeep/fdeep.hpp>

namespace miopen {
namespace ai {
namespace tuning {

Metadata::Metadata (const std::string& arch, const std::string& solver)
{
    const nlohmann::json metadata = nlohmann::json::parse(std::ifstream(
        GetSystemDbPath() + "/" + arch + "_" + solver + "_metadata.ktn.model"));

    num_conv_params   = metadata["num_conv_params"].get<std::size_t>();
    num_tuning_params = metadata["num_tuning_params"].get<std::size_t>();
    tuning_decodings  = metadata["decodings"]["tunings"].get<std::unordered_map<std::string, int>>();

    n    = num_conv_params + 1;
}

Model::Model (const std::string& arch, const std::string& solver)
{
    const fdeep::model _encoder_ = fdeep::load_model(
        GetSystemDbPath() + "/" + arch + "_" + solver + "_encoder.ktn.model", true, fdeep::dev_null_logger);
    const fdeep::model _decoder_ = fdeep::load_model(
        GetSystemDbPath() + "/" + arch + "_" + solver + "_decoder.ktn.model", true, fdeep::dev_null_logger);
    Metadata _metadata_ = Metadata(arch, solver);

    encoder = &_encoder_;
    decoder = &_decoder_;
    metadata = &_metadata_;
}

std::unordered_map<std::string, Model*> GetModel(const std::string& arch)
{
    const static std::unordered_map<std::string, Model*> models = {
        {"ConvAsm1x1U", new Model(arch, "ConvAsm1x1U")}
    };
    return models;
}

std::unordered_map<std::string, Metadata*> GetMetadata(const std::string& arch)
{
    const static std::unordered_map<std::string, Metadata*> metadata = {
        {"ConvAsm1x1U", new Metadata(arch, "ConvAsm1x1U")}
    };
    return metadata;
}

fdeep::model LoadModel(const std::string& arch, const std::string& solver, const std::string& model_type)
{
    const std::string file_path =
        GetSystemDbPath() + "/" + arch + "_" + solver + "_" + model_type + ".ktn.model";
    return fdeep::load_model(file_path, true, fdeep::dev_null_logger);
}

std::unordered_map<std::string, fdeep::model*> GetEncoder(const std::string& arch)
{
    static fdeep::model conv_asm_1x1u_encoder = LoadModel(arch, "ConvAsm1x1U", "encoder");
    const static std::unordered_map<std::string, fdeep::model*> encoder = {
        {"ConvAsm1x1U", &conv_asm_1x1u_encoder}
    };
    return encoder;
}

std::unordered_map<std::string, fdeep::model*> GetDecoder(const std::string& arch)
{
    static fdeep::model conv_asm_1x1u_decoder = LoadModel(arch, "ConvAsm1x1U", "decoder");
    const static std::unordered_map<std::string, fdeep::model*> decoder = {
        {"ConvAsm1x1U", &conv_asm_1x1u_decoder}
    };
    return decoder;
}

std::vector<float> TransformFeatures(const std::string& arch,
                                     const std::string& solver,
                                     const ProblemDescription& problem, 
                                     std::size_t n)
{
    if(arch == "gfx908" && solver=="ConvAsm1x1U")
    {
        assert(n == 8); // n = 6 (numerical conv params) * 1 + 1 (nominal conv params) * 2(amount of
                        // values nominal param can take).
        std::vector<float> features(n * n, 0.0f);
        features[0]                   = problem.IsFp32() ? 2.0 : 1.0;
        int offset                    = (problem.direction.IsForward() ? 0 : 1) + 1;
        features[(offset)*n + offset] = 1.0;
        features[3 * n + 3] =
            float(problem.direction.IsForward() ? problem.n_inputs : problem.n_outputs);
        features[4 * n + 4] =
            float(problem.direction.IsForward() ? problem.n_outputs : problem.n_inputs);
        features[5 * n + 5] = float(problem.in_height);
        features[6 * n + 6] = float(problem.in_width);
        features[7 * n + 7] = float(problem.batch_sz);
        return features;
    }
    else
        MIOPEN_THROW("Kernel tuning heuristic doesn't support GPU and/or solver.");
}


bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    solver::PerformanceConfigConvAsm1x1U& config,
                    const ProblemDescription& problem)
{
    MIOPEN_LOG_I("");

    static auto model        = GetModel(arch);

    static auto encoder       = GetEncoder(arch);
    static auto decoder       = GetDecoder(arch);
    static auto metadata      = GetMetadata(arch);

    std::cout << "\n#######################" <<  metadata[solver]->num_tuning_params << "###############\n";
    std::cout << "\n#######################" <<  model[solver]->metadata->num_tuning_params << "###############\n";

    std::vector<float> features = TransformFeatures(arch, solver, problem, metadata[solver]->n);

    int dim            = std::sqrt(features.size());
    auto input_tensor  = fdeep::tensor(fdeep::tensor_shape(dim, dim), features);
    auto hidden_states = encoder[solver]->predict({input_tensor}); // Get hidden states from Encoder LSTM

    std::vector<float> decoder_input_vector(1, 0.0);
    auto decoder_input_tensor = fdeep::tensor(fdeep::tensor_shape(1), decoder_input_vector);
    std::vector<fdeep::tensor> decoder_input = {
        decoder_input_tensor,
        hidden_states[0],
        hidden_states[1],
        hidden_states[2],
        hidden_states[3]}; // pass in SOS token and hidden states

    for(int i = 0; i < metadata[solver]->num_tuning_params; ++i)
    {
        auto output        = decoder[solver]->predict({decoder_input});
        auto output_vector = output[0].to_vector();
        std::priority_queue<std::pair<float, int>> pq;
        for(int j = 0; j < output_vector.size(); j++)
            pq.push(std::make_pair(output_vector[j], j)); // sort by value at index

        int output_token_index = -1;
        while(!pq.empty())
        {
            int token = pq.top().second;
            int value = metadata[solver]->tuning_decodings[std::to_string(token)]; // convert index to tuning value
            pq.pop();
            if(value < 0)
                return false;
            if(config.ModelApplyToken(i, value, problem))
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

} // namespace tuning
} // namespace ai
} // namespace miopen
#endif
