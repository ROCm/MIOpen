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
    sos_token         = metadata["sos_token"].get<std::size_t>();
    tuning_decodings  = metadata["decodings"]["tunings"].get<std::unordered_map<std::string, int>>();
}

struct Model {
    public:
    Metadata metadata;
    Model(const std::string& arch, const std::string& solver)
        : metadata(Metadata(arch, solver)),
          encoder(fdeep::load_model(EncoderPath(arch, solver), true, fdeep::dev_null_logger)),
          decoder(fdeep::load_model(DecoderPath(arch, solver), true, fdeep::dev_null_logger)),
          encoder_input_dim(metadata.num_conv_params + 1),
          encoder_input_shape(fdeep::tensor_shape(encoder_input_dim, encoder_input_dim)),
          decoder_input_shape(fdeep::tensor_shape(1))
    {
    }
    fdeep::tensors Encode(const ProblemDescription& problem)
    {
        std::vector<float> features = ToFeatures(problem);
        fdeep::tensor input_tensor  = fdeep::tensor(encoder_input_shape, features);
        return encoder.predict({input_tensor}); 
    }
    fdeep::tensors Decode(float input, const fdeep::tensors& context)
    {
        return decoder.predict(
            {{fdeep::tensor(decoder_input_shape, std::vector<float>(1, input)),
              context[0],
              context[1],
              context[2],
              context[3]}}
        );
    }

    private:
    const fdeep::model encoder;
    const fdeep::model decoder;
    const std::size_t encoder_input_dim;
    const fdeep::tensor_shape encoder_input_shape;
    const fdeep::tensor_shape decoder_input_shape;
    static std::string EncoderPath(const std::string& arch, const std::string& solver)
    {
        return GetSystemDbPath() + "/" + arch + "_" + solver + "_encoder.ktn.model";
    }
    static std::string DecoderPath(const std::string& arch, const std::string& solver)
    {
        return GetSystemDbPath() + "/" + arch + "_" + solver + "_decoder.ktn.model";
    }
    std::vector<float> ToFeatures(const ProblemDescription& problem)
    {
        std::vector<float> features(encoder_input_dim * encoder_input_dim, 0.0f);
        features[0]                   = problem.IsFp32() ? 2.0 : 1.0;
        int offset                    = (problem.direction.IsForward() ? 0 : 1) + 1;
        features[(offset)*encoder_input_dim + offset] = 1.0;
        features[3 * encoder_input_dim + 3] =
            float(problem.direction.IsForward() ? problem.n_inputs : problem.n_outputs);
        features[4 * encoder_input_dim + 4] =
            float(problem.direction.IsForward() ? problem.n_outputs : problem.n_inputs);
        features[5 * encoder_input_dim + 5] = float(problem.in_height);
        features[6 * encoder_input_dim + 6] = float(problem.in_width);
        features[7 * encoder_input_dim + 7] = float(problem.batch_sz);
        return features;
    }
};

std::unordered_map<std::string, Model*> GetModel(const std::string& arch)
{
    static std::unordered_map<std::string, Model*> models = {
        {"ConvAsm1x1U", new Model(arch, "ConvAsm1x1U")}
    };
    return models;
}

bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    solver::PerformanceConfigConvAsm1x1U& config,
                    const ProblemDescription& problem)
{
    MIOPEN_LOG_I("");

    static auto model = GetModel(arch);

    float decoder_input    = float(model[solver]->metadata.sos_token);
    fdeep::tensors context = model[solver]->Encode(problem);
    for(std::size_t i = 0; i < model[solver]->metadata.num_tuning_params; ++i)
    {
        fdeep::tensors decoder_output = model[solver]->Decode(decoder_input, context);

        auto tokens_scores = decoder_output[0].to_vector();
        std::priority_queue<std::pair<float, int>> pq;
        for(int j = 0; j < tokens_scores.size(); j++)
            pq.push(std::make_pair(tokens_scores[j], j)); // sort by value at index

        int output_token_index = -1;
        while(!pq.empty())
        {
            int token = pq.top().second;
            // convert index to tuning value
            int value = model[solver]->metadata.tuning_decodings[std::to_string(token)];
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

        decoder_input = float(output_token_index);
        context = {decoder_output.begin() + 1, decoder_output.end()};
    }
    return true;
}

} // namespace tuning
} // namespace ai
} // namespace miopen
#endif
