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

#include <miopen/conv/heuristics/ai_heuristics.hpp>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <fdeep/fdeep.hpp>

namespace miopen {
namespace ai {
namespace common {
    nlohmann::json LoadJSON(const std::string& path)
    {
        return nlohmann::json::parse(std::ifstream(path));
    }
    template <typename U, typename V> 
    std::unordered_map<V, U> ReverseMap(const std::unordered_map<U, V>& map)
    {
        std::unordered_map<V, U> reversed_map = {};
        for(auto& it: map)
            reversed_map.emplace(make_pair(it.second, it.first));
        return reversed_map;
    }
    template <typename U, typename V>
    std::vector<V> LookupValues(const std::vector<U> keys, const std::unordered_map<U, V>& map)
    {
        std::vector<V> values = {};
        for(const U& key: keys)
            values.push_back(map.at(key));
        return values;
    }
}
#endif
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
namespace immed_mode {
Metadata::Metadata (const std::string& arch)
    : json(common::LoadJSON(GetSystemDbPath() + "/" + arch + "_metadata.tn.model")),
      direction_encodings(json["encodings"]["Direction"]),
      precision_encodings(json["encodings"]["Precision"]),
      layout_encodings(json["encodings"]["Layout"]),
      features(json["conv_params_used_as_features"]),
      num_inputs(json["num_inputs"]),
      num_outputs(json["num_outputs"]),
      num_solvers(json["num_solvers"]),
      solver_map(common::ReverseMap<std::string, size_t>(json["encodings"]["solver"])),
      features_mean(common::LookupValues<std::string, float>(features, json["stats"]["overall"]["features"]["mean"])),
      features_std(common::LookupValues<std::string, float>(features, json["stats"]["overall"]["features"]["std"]))
{
    json.~basic_json();
}
size_t Metadata::EncodeDirection(const miopen::conv::Direction& dir) const
{
    if(dir == conv::Direction::BackwardWeights)
        return direction_encodings.at("W");
    else if(dir == conv::Direction::BackwardData)
        return direction_encodings.at("B");
    else if(dir == conv::Direction::Forward)
        return direction_encodings.at("F");
    else
        throw std::invalid_argument("Invalid direction");
}
size_t Metadata::EncodePrecision(const miopenDataType_t& data_type) const
{
    if(data_type == miopenBFloat16)
        return precision_encodings.at("BF16");
    else if(data_type == miopenHalf)
        return precision_encodings.at("FP16");
    else if(data_type == miopenFloat)
        return precision_encodings.at("FP32");
    else
        throw std::invalid_argument("Unsupported precision");
}
size_t Metadata::EncodeLayout(const std::string& layout) const
{
    if(layout == "NCDHW")
        return layout_encodings.at("NCDHW");
    else if(layout == "NCHW")
        return layout_encodings.at("NCHW");
    else
        throw std::invalid_argument("Unsupported layout");
}

class Model {
    public:
    Metadata metadata;
    Model(const std::string& arch)
        : metadata(Metadata(arch)),
          model(fdeep::load_model(ModelPath(arch), true, fdeep::dev_null_logger)),
          input_shape(fdeep::tensor_shape(metadata.num_inputs)),
          offset(metadata.num_outputs - metadata.num_solvers)
    {
    }
    virtual ~Model() = default;
    virtual bool IsProblemSupported(const ProblemDescription& problem, const ConvolutionContext& ctx) const = 0;
    std::vector<float> Forward(const ProblemDescription& problem) const
    {
        std::vector<float> features = ToFeatures(problem);
        std::vector<fdeep::tensor> output = 
            model.predict({fdeep::tensor(input_shape, features)});
        std::vector<float> output_vector  = output.front().to_vector();
        std::vector<float> res(output_vector.begin() + offset, output_vector.end());
        return res;
    }

    protected:
    const fdeep::model model;
    const fdeep::tensor_shape input_shape;
    const size_t offset;
    static std::string ModelPath(const std::string& arch)
    {
        return GetSystemDbPath() + "/" + arch + ".tn.model";
    }
    virtual std::vector<float> ToFeatures(const ProblemDescription& problem) const = 0;
};

class Gfx908Model : public Model
{
    public:
    Gfx908Model()
        : Model("gfx908")
    {
    }
    bool IsProblemSupported(const ProblemDescription& problem, const ConvolutionContext& ctx) const override
    {
        // check if problem is of the kind TunaNet was trained to handle
        if(!problem.conv_problem.Is2d())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Problem not 2D");
            return false;
        }
        if(problem.conv_problem.GetGroupCount() != 1)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Group count not 1");
            return false;
        }
        if(problem.conv_problem.GetInLayout() != "NCHW" && problem.conv_problem.GetInLayout() != "NCDHW")
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Layout not supported");
            return false;
        }
        if(problem.conv_problem.GetWeightsHeight() != problem.conv_problem.GetWeightsWidth())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Filters must be square (fil_h == fil_w)");
            return false;
        }
        if(problem.conv_problem.GetPadH() != problem.conv_problem.GetPadW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Padding must be equal along all axes");
            return false;
        }
        if(problem.conv_problem.GetKernelStrideH() != problem.conv_problem.GetKernelStrideW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Stride must be equal along all axes");
            return false;
        }
        if(problem.conv_problem.GetDilationH() != 1 || problem.conv_problem.GetDilationW() != 1)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Dilation must be 1");
            return false;
        }
        const auto& data_type = problem.conv_problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Unsupported precision");
            return false;
        }

        // check if the context is s.t. no solver TunaNet may predict would be applicable
        size_t applicable_solvers = 0;
        for(const auto& solver_name : metadata.solver_map)
        {
            auto solver_id = solver::Id{solver_name.second};
            auto solver    = solver_id.GetSolver();
            if(solver.IsApplicable(ctx, problem))
            {
                applicable_solvers++;
                break;
            }
        }
        if(applicable_solvers == 0)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: No solver that TunaNet may predict applies");
            return false;
        }

        return true;
    }

    protected:
    std::vector<float> ToFeatures(const ProblemDescription& problem) const override
    {
        const auto conv_problem = problem.conv_problem;
        const bool isFwd = conv_problem.GetDirection() == conv::Direction::Forward;
        std::vector<float> features = {
            static_cast<float>(isFwd ? conv_problem.GetInChannels() :
                                       conv_problem.GetOutChannels()),
            static_cast<float>(isFwd ? conv_problem.GetInDepth() :
                                       conv_problem.GetOutDepth()),
            static_cast<float>(isFwd ? conv_problem.GetInHeight() :
                                        conv_problem.GetOutHeight()),
            static_cast<float>(isFwd ? conv_problem.GetInWidth() :
                                        conv_problem.GetOutWidth()),
            static_cast<float>(conv_problem.GetWeightsDepth()),
            static_cast<float>(conv_problem.GetWeightsHeight()),
            static_cast<float>(conv_problem.GetWeightsWidth()),
            static_cast<float>(isFwd ? conv_problem.GetOutChannels() :
                                       conv_problem.GetInChannels()),
            static_cast<float>(isFwd ? conv_problem.GetOutDepth() :
                                       conv_problem.GetInDepth()),
            static_cast<float>(isFwd ? conv_problem.GetOutHeight() :
                                       conv_problem.GetInHeight()),
            static_cast<float>(isFwd ? conv_problem.GetOutWidth() :
                                       conv_problem.GetInWidth()),
            static_cast<float>(conv_problem.GetOutBatchSize()),
            static_cast<float>(1), // TunaNet was trained on a dataset of 2D
                                   // problems where PadD was incorrectly set to 1 
            static_cast<float>(conv_problem.GetPadH()),
            static_cast<float>(conv_problem.GetPadW()),
            static_cast<float>(1), // TunaNet was trained on a dataset of 2D
                                   // problems where StrideD was incorrectly set to 1 
            static_cast<float>(conv_problem.GetKernelStrideH()),
            static_cast<float>(conv_problem.GetKernelStrideW()),
            static_cast<float>(conv_problem.GetDilationH()),
            static_cast<float>(conv_problem.GetDilationW()),
            static_cast<float>(metadata.EncodeLayout(conv_problem.GetInLayout())),
            static_cast<float>(metadata.EncodePrecision(conv_problem.GetInDataType())),
            static_cast<float>(metadata.EncodeDirection(conv_problem.GetDirection())),
            static_cast<float>(conv_problem.GetGroupCount())
        };

        // normalize
        for(size_t i = 0; i < features.size(); ++i)
            features[i] = (features[i] - metadata.features_mean[i]) / metadata.features_std[i];

        return features;
    }
};

std::unique_ptr<Model> GetModel(const std::string& arch)
{
    if(arch == "gfx908")
        return std::make_unique<Gfx908Model>();
    MIOPEN_LOG_I2("Immdiate-Mode AI Fallback Inapplicable: device not supported");
    return nullptr;
}

std::vector<uint64_t> PredictSolver(const ProblemDescription& problem,
                                    const ConvolutionContext& ctx,
                                    const std::string& device)
{
    const static std::unique_ptr<Model> model = GetModel(device);
    if(!model || !model->IsProblemSupported(problem, ctx))
        return {};

    std::string est_name = ":memory:" + device;
    auto& db             = AnyRamDb::GetCached(est_name);
    auto db_res          = db.FindRecord(problem.conv_problem);
    if(db_res)
    {
        MIOPEN_LOG_I2("Cached heuristic result found");
        std::vector<uint64_t> db_sol(db_res->size());
        // cast returned record to solver ids
        std::transform(db_res->begin(), db_res->end(), db_sol.begin(), [](boost::any id) {
            return boost::any_cast<uint64_t>(id);
        });
        if(miopen::IsLogging(LoggingLevel::Info2))
        {
            std::stringstream ss;
            for(auto& id : db_sol)
                ss << solver::Id{id}.ToString() << " ID:" << id << ", ";
            MIOPEN_LOG_I2("Cached solvers: " << ss.str());
        }
        return db_sol;
    }

    MIOPEN_LOG_I2("Evaluating Heuristic");

    std::vector<float> res     = model->Forward(problem);
    std::vector<std::pair<int, float>> sort_res(res.size());
    // sorts result based upon magnitude of result in vector, returned from Model,
    // paired with original index (idx). Sort magnitudes in descending order.
    // Greater magnitude = better solver. Indexes (idx), which will be used to map to solvers,
    // with greater corresponding magnitude are at front of the vector so they get priority.
    for(auto idx = 0; idx < res.size(); idx++)
        sort_res[idx] = {idx, res[idx]};
    const auto cmp = [](const std::pair<int, float>& a,
                        const std::pair<int, float>& b) -> bool { return a.second > b.second; };
    std::sort(sort_res.begin(), sort_res.end(), cmp);

    // map idx to solver id and then anysolver
    std::vector<uint64_t> sol;
    std::vector<boost::any> any_sol;
    for(const auto& kinder : sort_res)
    {
        const auto id     = kinder.first;
        const auto sol_id = solver::Id{model->metadata.solver_map.at(id)};
        if(!sol_id.IsValid())
        {
            MIOPEN_LOG_I2("Invalid solver " << model->metadata.solver_map.at(id) << " removed");
            continue;
        }
        sol.push_back(sol_id.Value());
        any_sol.push_back(sol_id.Value());
    }
    db.StoreRecord(problem.conv_problem, any_sol);
    if(miopen::IsLogging(LoggingLevel::Info2))
    {
        std::stringstream ss;
        for(auto& id : sol)
            ss << solver::Id{id}.ToString() << " ID:" << id << ", ";
        MIOPEN_LOG_I2("Heuristic Result: " << ss.str());
    }
    return sol;
}
} // namespace immed_mode
#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace kernel_tuning {

Metadata::Metadata (const std::string& arch, const std::string& solver)
{
    const nlohmann::json metadata = common::LoadJSON(
        GetSystemDbPath() + "/" + arch + "_" + solver + "_metadata.ktn.model");

    num_conv_params   = metadata["num_conv_params"].get<std::size_t>();
    num_tuning_params = metadata["num_tuning_params"].get<std::size_t>();
    sos_token         = metadata["sos_token"].get<std::size_t>();
    tuning_decodings  = metadata["decodings"]["tunings"].get<std::unordered_map<std::string, int>>();
}

class Model {
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
    virtual ~Model() = default;
    fdeep::tensors Encode(const ProblemDescription& problem) const
    {
        std::vector<float> features = ToFeatures(problem);
        fdeep::tensor input_tensor  = fdeep::tensor(encoder_input_shape, features);
        return encoder.predict({input_tensor}); 
    }
    fdeep::tensors Decode(const float input, const fdeep::tensors& context) const
    {
        return decoder.predict(
            {{fdeep::tensor(decoder_input_shape, std::vector<float>(1, input)),
              context[0],
              context[1],
              context[2],
              context[3]}}
        );
    }

    protected:
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
    virtual std::vector<float> ToFeatures(const ProblemDescription& problem) const = 0;
};

class ConvAsm1x1UModel : public Model
{
    public:
    ConvAsm1x1UModel(const std::string& arch)
        : Model(arch, "ConvAsm1x1U")
    {
    }
    protected:
    std::vector<float> ToFeatures(const ProblemDescription& problem) const override
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

std::unordered_map<std::string, Model*> GetModels(const std::string& arch)
{
    static std::unordered_map<std::string, Model*> models = {
        {"ConvAsm1x1U", new ConvAsm1x1UModel(arch)}
    };
    return models;
}

bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    solver::PerformanceConfigConvAsm1x1U& config,
                    const ProblemDescription& problem)
{
    static auto models = GetModels(arch);

    float decoder_input    = float(models[solver]->metadata.sos_token);
    fdeep::tensors context = models[solver]->Encode(problem);
    for(std::size_t i = 0; i < models[solver]->metadata.num_tuning_params; ++i)
    {
        fdeep::tensors decoder_output = models[solver]->Decode(decoder_input, context);

        auto tokens_scores = decoder_output[0].to_vector();
        std::priority_queue<std::pair<float, int>> pq;
        for(int j = 0; j < tokens_scores.size(); j++)
            pq.push(std::make_pair(tokens_scores[j], j)); // sort by value at index

        int output_token_index = -1;
        while(!pq.empty())
        {
            int token = pq.top().second;
            // convert index to token value
            int value = models[solver]->metadata.tuning_decodings[std::to_string(token)];
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

} // namespace kernel_tuning
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
} // namespace ai
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
