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
#endif
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
namespace immed_mode {
bool IsDeviceSupported(const std::string& device)
{
    // TODO look in src/kernels/ dir, and if DEVICE.tn.model and DEVICE_metadata.tn.model files
    // are present, add DEVICE to the list of supported devices
    static const std::set<std::string> supported_devices = {"gfx908"};
    static const bool is_device_supported = supported_devices.find(device) != supported_devices.end();
    if (!is_device_supported)
        MIOPEN_LOG_I2("TunaNet Inapplicable: device not supported");
    return is_device_supported;
}

nlohmann::json GetMetadata(const std::string& device)
{
    static const std::string file_path = GetSystemDbPath() + "/" + device + "_metadata.tn.model";
    static const nlohmann::json metadata = nlohmann::json::parse(std::ifstream(file_path));
    return metadata;
}

fdeep::model GetModel(const std::string& device)
{
    static const auto model_file =
        boost::filesystem::path(GetSystemDbPath() + "/" + device + ".tn.model").generic_string();
    static const fdeep::model model =
        fdeep::load_model(model_file, true, fdeep::dev_null_logger);

    return model;
}

std::vector<std::string> GetFeatureNames(const nlohmann::json& metadata)
{
    static const std::vector<std::string> feature_names = metadata["conv_params_used_as_features"];
    return feature_names;
}

std::unordered_map<size_t, std::string> GetSolverMap(const nlohmann::json& metadata)
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

size_t GetNumInputs(const nlohmann::json& metadata)
{
    static const size_t num_outputs = metadata["num_inputs"];
    return num_outputs;
}

size_t GetNumOutputs(const nlohmann::json& metadata)
{
    static const size_t num_outputs = metadata["num_outputs"];
    return num_outputs;
}

size_t GetNumSolvers(const nlohmann::json& metadata)
{
    static const size_t num_solvers = metadata["num_solvers"];
    return num_solvers;
}

size_t GetDirectionCode(const miopen::conv::Direction& dir, const nlohmann::json& metadata)
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

size_t GetPrecisionCode(const miopenDataType_t& data_type, const nlohmann::json& metadata)
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

size_t GetLayoutCode(const std::string& layout, const nlohmann::json& metadata)
{
    if(layout == "NCDHW")
        return metadata["encodings"]["Layout"]["NCDHW"];
    else if(layout == "NCHW")
        return metadata["encodings"]["Layout"]["NCHW"];
    else
        throw std::invalid_argument("TunaNet doesn't support this layout");
}

std::vector<float> GetFeaturesMean(const nlohmann::json& metadata)
{
    static const std::vector<std::string> features = GetFeatureNames(metadata);
    static std::unordered_map<std::string, float> feature_to_mean_map = 
        metadata["stats"]["overall"]["features"]["mean"];

    std::vector<float> features_mean(features.size());
    for(size_t i = 0; i < features.size(); i++)
        features_mean[i] = feature_to_mean_map[features[i]];

    return features_mean;
}

std::vector<float> GetFeaturesStd(const nlohmann::json& metadata)
{
    static const std::vector<std::string> features = GetFeatureNames(metadata);
    static std::unordered_map<std::string, float> feature_to_std_map = 
        metadata["stats"]["overall"]["features"]["std"];

    std::vector<float> features_std(features.size());
    for(size_t i = 0; i < features.size(); i++)
        features_std[i] = feature_to_std_map[features[i]];

    return features_std;
}

std::vector<float> ToFeatures(const conv::ProblemDescription& problem,
                              const nlohmann::json& metadata,
                              const bool normalize)
{
    const bool isFwd = problem.GetDirection() == conv::Direction::Forward;
    std::vector<float> features = {
        static_cast<float>(isFwd ? problem.GetInChannels() :
                                   problem.GetOutChannels()),
        static_cast<float>(isFwd ? problem.GetInDepth() :
                                   problem.GetOutDepth()),
        static_cast<float>(isFwd ? problem.GetInHeight() :
                                   problem.GetOutHeight()),
        static_cast<float>(isFwd ? problem.GetInWidth() :
                                   problem.GetOutWidth()),
        static_cast<float>(problem.GetWeightsDepth()),
        static_cast<float>(problem.GetWeightsHeight()),
        static_cast<float>(problem.GetWeightsWidth()),
        static_cast<float>(isFwd ? problem.GetOutChannels() :
                                   problem.GetInChannels()),
        static_cast<float>(isFwd ? problem.GetOutDepth() :
                                   problem.GetInDepth()),
        static_cast<float>(isFwd ? problem.GetOutHeight() :
                                   problem.GetInHeight()),
        static_cast<float>(isFwd ? problem.GetOutWidth() :
                                   problem.GetInWidth()),
        static_cast<float>(problem.GetOutBatchSize()),
        static_cast<float>(1), // TunaNet was trained on a dataset of 2D
                               // problems where PadD was incorrectly set to 1 
        static_cast<float>(problem.GetPadH()),
        static_cast<float>(problem.GetPadW()),
        static_cast<float>(1), // TunaNet was trained on a dataset of 2D
                               // problems where StrideD was incorrectly set to 1 
        static_cast<float>(problem.GetKernelStrideH()),
        static_cast<float>(problem.GetKernelStrideW()),
        static_cast<float>(problem.GetDilationH()),
        static_cast<float>(problem.GetDilationW()),
        static_cast<float>(GetLayoutCode(problem.GetInLayout(), metadata)),
        static_cast<float>(GetPrecisionCode(problem.GetInDataType(), metadata)),
        static_cast<float>(GetDirectionCode(problem.GetDirection(), metadata)),
        static_cast<float>(problem.GetGroupCount())
    };

    if (normalize == true)
    {
        static const std::vector<float> mu  = GetFeaturesMean(metadata);
        static const std::vector<float> sig = GetFeaturesStd(metadata);
        for(size_t i = 0; i < features.size(); ++i)
            features[i] = (features[i] - mu[i]) / sig[i];
    }

    return features;
}

bool AreFeaturesInDistributionL1(const std::vector<float>& features,
                                 const float threshold)
{
	for(size_t i = 0; i < features.size(); ++i) {
		if ((features[i] > threshold) || (features[i] < -threshold))
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Features are out-of-distribution w.r.t. L1 norm.");
			return false;
        }
	}
	return true;
}

bool AreFeaturesInDistributionL2(const std::vector<float>& features,
                                 const float threshold,
                                 const nlohmann::json& metadata)
{
	static const float dist_from_mean_avg = metadata["problem_distance_from_mean_avg"];
	static const float dist_from_mean_std = metadata["problem_distance_from_mean_std"];

	static const float upper_bound =  dist_from_mean_avg + threshold * dist_from_mean_std;
	static const float lower_bound =  dist_from_mean_avg - threshold * dist_from_mean_std;

	float squared_sum = 0;
	for(size_t i = 0; i < features.size(); ++i)
		squared_sum += std::pow(features[i], 2);
	const float distance = std::sqrt(squared_sum);

	if (distance > lower_bound && distance < upper_bound)
        return true;

    MIOPEN_LOG_I2("TunaNet Inapplicable: Features are out-of-distribution w.r.t. L2 norm.");
    return false;
}

bool IsProblemSupported(const ProblemDescription& problem,
                        const ConvolutionContext& ctx,
                        const nlohmann::json& metadata)
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
    const auto& solver_map    = GetSolverMap(metadata);
    size_t applicable_solvers = 0;
    for(const auto& solver_name : solver_map)
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

std::vector<float> CallModel(const fdeep::model& model,
                             const std::vector<float>& normalized_features,
                             const nlohmann::json& metadata)
{
    static const auto input_shape = fdeep::tensor_shape(GetNumInputs(metadata));
    std::vector<fdeep::tensor> output = 
        model.predict({fdeep::tensor(input_shape, normalized_features)});
    std::vector<float> output_vector  = output.front().to_vector();
    static const size_t offset = GetNumOutputs(metadata) - GetNumSolvers(metadata);
    std::vector<float> res(output_vector.begin() + offset, output_vector.end());
    return res;
}

std::vector<uint64_t> PredictSolver(const conv::ProblemDescription& problem,
                                    const std::vector<float>& normalized_features,
                                    bool& cached,
                                    const std::string& device,
                                    const nlohmann::json& metadata)
{
    std::string est_name = ":memory:" + device;
    auto& db             = AnyRamDb::GetCached(est_name);
    auto db_res          = db.FindRecord(problem);
    if(db_res)
    {
        cached = true;
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

    const static fdeep::model model = GetModel(device);
    std::vector<float> res     = CallModel(model, normalized_features, metadata);
    static const auto& solvers = GetSolverMap(metadata);

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
        const auto sol_id = solver::Id{solvers.at(id)};
        if(!sol_id.IsValid())
        {
            MIOPEN_LOG_I2("Invalid solver " << solvers.at(id) << " removed");
            continue;
        }
        sol.push_back(sol_id.Value());
        any_sol.push_back(sol_id.Value());
    }
    db.StoreRecord(problem, any_sol);
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
    const nlohmann::json metadata = nlohmann::json::parse(std::ifstream(
        GetSystemDbPath() + "/" + arch + "_" + solver + "_metadata.ktn.model"));

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
