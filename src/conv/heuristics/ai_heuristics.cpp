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
#include <miopen/filesystem.hpp>

namespace miopen {
namespace ai {
namespace common {

nlohmann::json LoadJSON(const fs::path& path)
{
    if(!fs::exists(path))
        MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
    return nlohmann::json::parse(std::ifstream(path));
}

template <typename U, typename V>
std::unordered_map<V, U> ReverseMap(const std::unordered_map<U, V>& map)
{
    std::unordered_map<V, U> reversed_map = {};
    for(const auto& it : map)
        reversed_map.emplace(make_pair(it.second, it.first));
    return reversed_map;
}

template <typename U, typename V>
std::vector<V> LookupValues(const std::vector<U>& keys, const std::unordered_map<U, V>& map)
{
    std::vector<V> values = {};
    values.reserve(keys.size());
    std::transform(keys.begin(), keys.end(), std::back_inserter(values), [&](const U& key) {
        return map.at(key);
    });
    return values;
}
} // namespace common

#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
namespace immed_mode {
Metadata::Metadata(const std::string& arch)
    : json(common::LoadJSON(GetSystemDbPath() / (arch + "_metadata.tn.model"))),
      direction_encodings(json["encodings"]["Direction"]),
      precision_encodings(json["encodings"]["Precision"]),
      layout_encodings(json["encodings"]["Layout"]),
      features(json["conv_params_used_as_features"]),
      num_inputs(json["num_inputs"]),
      num_outputs(json["num_outputs"]),
      num_solvers(json["num_solvers"]),
      solver_map(common::ReverseMap<std::string, size_t>(json["encodings"]["solver"])),
      features_mean(common::LookupValues<std::string, float>(
          features, json["stats"]["overall"]["features"]["mean"])),
      features_std(common::LookupValues<std::string, float>(
          features, json["stats"]["overall"]["features"]["std"])),
      test_features_mean(common::LookupValues<std::string, float>(
          features, json["stats"]["test"]["features"]["mean"])),
      test_features_std(common::LookupValues<std::string, float>(
          features, json["stats"]["test"]["features"]["std"]))
{
}

size_t Metadata::EncodeDirection(miopen::conv::Direction dir) const
{
    if(dir == conv::Direction::BackwardWeights)
        return direction_encodings.at("W");
    else if(dir == conv::Direction::BackwardData)
        return direction_encodings.at("B");
    else
        return direction_encodings.at("F");
}

size_t Metadata::EncodePrecision(miopenDataType_t data_type) const
{
    if(data_type == miopenBFloat16)
        return precision_encodings.at("BF16");
    else if(data_type == miopenHalf)
        return precision_encodings.at("FP16");
    else if(data_type == miopenFloat)
        return precision_encodings.at("FP32");
    MIOPEN_THROW("Unsupported data type passed through TunaNet applicability check");
}

size_t Metadata::EncodeLayout(const std::string& layout) const
{
    if(layout != "NCDHW" && layout != "NCHW")
        MIOPEN_THROW("Unsupported layout passed through TunaNet applicability check");
    return layout_encodings.at(layout);
}

class Model
{
public:
    Metadata metadata;
    Model(const std::string& arch)
        : metadata(Metadata(arch)),
          model(fdeep::load_model(ModelPath(arch), true, fdeep::dev_null_logger)),
          input_shape(fdeep::tensor_shape(metadata.num_inputs)),
          offset(metadata.num_outputs - metadata.num_solvers)
    {
    }
    virtual ~Model()                                                   = default;
    virtual bool IsProblemSupported(const conv::ProblemDescription& problem,
                                    const ExecutionContext& ctx) const = 0;
    std::vector<float> Forward(const conv::ProblemDescription& problem) const
    {
        std::vector<float> features       = ToFeatures(problem);
        std::vector<fdeep::tensor> output = model.predict({fdeep::tensor(input_shape, features)});
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
        const auto file_path = GetSystemDbPath() / (arch + ".tn.model");
        if(!fs::exists(file_path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load AI model file:" + file_path);
        return file_path.string();
    }
    virtual std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const = 0;
};

class Gfx908Model final : public Model
{
public:
    Gfx908Model() : Model("gfx908") {}
    bool IsProblemSupported(const conv::ProblemDescription& problem,
                            const ExecutionContext& ctx) const override
    {
        // check if problem is of the kind TunaNet was trained to handle
        if(!problem.Is2d())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Problem not 2D");
            return false;
        }
        if(problem.GetGroupCount() != 1)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Group count not 1");
            return false;
        }
        if(problem.GetInLayout() != "NCHW" && problem.GetInLayout() != "NCDHW")
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Layout not supported");
            return false;
        }
        if(problem.GetWeightsHeight() != problem.GetWeightsWidth())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Filters must be square (fil_h == fil_w)");
            return false;
        }
        if(problem.GetPadH() != problem.GetPadW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Padding must be equal along all axes");
            return false;
        }
        if(problem.GetKernelStrideH() != problem.GetKernelStrideW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Stride must be equal along all axes");
            return false;
        }
        if(problem.GetDilationH() != 1 || problem.GetDilationW() != 1)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Dilation must be 1");
            return false;
        }
        const auto data_type = problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Unsupported data type");
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
    std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const override
    {
        const bool isFwd            = problem.GetDirection() == conv::Direction::Forward;
        std::vector<float> features = {
            static_cast<float>(isFwd ? problem.GetInChannels() : problem.GetOutChannels()),
            static_cast<float>(isFwd ? problem.GetInDepth() : problem.GetOutDepth()),
            static_cast<float>(isFwd ? problem.GetInHeight() : problem.GetOutHeight()),
            static_cast<float>(isFwd ? problem.GetInWidth() : problem.GetOutWidth()),
            static_cast<float>(problem.GetWeightsDepth()),
            static_cast<float>(problem.GetWeightsHeight()),
            static_cast<float>(problem.GetWeightsWidth()),
            static_cast<float>(isFwd ? problem.GetOutChannels() : problem.GetInChannels()),
            static_cast<float>(isFwd ? problem.GetOutDepth() : problem.GetInDepth()),
            static_cast<float>(isFwd ? problem.GetOutHeight() : problem.GetInHeight()),
            static_cast<float>(isFwd ? problem.GetOutWidth() : problem.GetInWidth()),
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
            static_cast<float>(metadata.EncodeLayout(problem.GetInLayout())),
            static_cast<float>(metadata.EncodePrecision(problem.GetInDataType())),
            static_cast<float>(metadata.EncodeDirection(problem.GetDirection())),
            static_cast<float>(problem.GetGroupCount())};

        // normalize
        for(size_t i = 0; i < features.size(); ++i)
            features[i] = (features[i] - metadata.features_mean[i]) / metadata.features_std[i];

        return features;
    }
};

class Gfx90aModel final : public Model
{
public:
    Gfx90aModel() : Model("gfx90a") {}
    bool IsProblemSupported(const conv::ProblemDescription& problem,
                            const ExecutionContext& ctx) const override
    {
        // check if problem is of the kind TunaNet was trained to handle
        if(!problem.Is2d())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Problem not 2D");
            return false;
        }
        if(problem.GetInLayout() != "NCHW")
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Layout not supported");
            return false;
        }
        if(problem.GetKernelStrideH() != problem.GetKernelStrideW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Stride must be equal along all axes");
            return false;
        }
        if(problem.GetDilationH() != problem.GetDilationW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Dilation must be 1");
            return false;
        }
        if(problem.GetBias() != 0)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Bias must be 0");
            return false;
        }
        const auto data_type = problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Unsupported data type");
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
        MIOPEN_LOG_I2("TunaNet Applicable");
        return true;
    }

protected:
    std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const override
    {
        const bool isFwd            = problem.GetDirection() == conv::Direction::Forward;
        std::vector<float> features = {
            static_cast<float>(isFwd ? problem.GetInChannels() : problem.GetOutChannels()),
            static_cast<float>(isFwd ? problem.GetInHeight() : problem.GetOutHeight()),
            static_cast<float>(isFwd ? problem.GetInWidth() : problem.GetOutWidth()),
            static_cast<float>(isFwd ? problem.GetOutChannels() : problem.GetInChannels()),
            static_cast<float>(isFwd ? problem.GetOutHeight() : problem.GetInHeight()),
            static_cast<float>(isFwd ? problem.GetOutWidth() : problem.GetInWidth()),
            static_cast<float>(problem.GetWeightsHeight()),
            static_cast<float>(problem.GetWeightsWidth()),
            static_cast<float>(problem.GetPadH()),
            static_cast<float>(problem.GetPadW()),
            static_cast<float>(problem.GetKernelStrideH()),
            static_cast<float>(problem.GetKernelStrideW()),
            static_cast<float>(problem.GetDilationH()),
            static_cast<float>(problem.GetDilationW()),
            static_cast<float>(problem.GetOutBatchSize()),
            static_cast<float>(metadata.EncodePrecision(problem.GetInDataType())),
            static_cast<float>(metadata.EncodeDirection(problem.GetDirection())),
            static_cast<float>(problem.GetGroupCount())};

        // normalize
        for(size_t i = 0; i < features.size(); ++i)
            features[i] = (features[i] - metadata.features_mean[i]) / metadata.features_std[i];

        return features;
    }
};

class Gfx942Model final : public Model
{
public:
    Gfx942Model() : Model("gfx942") {}
    bool IsProblemSupported(const conv::ProblemDescription& problem,
                            const ExecutionContext& ctx) const override
    {
        // check if problem is of the kind TunaNet was trained to handle
        if(!problem.Is2d())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Problem not 2D");
            return false;
        }
        if(problem.GetInLayout() != "NCHW")
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Layout not supported");
            return false;
        }
        if(problem.GetKernelStrideH() != problem.GetKernelStrideW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Stride must be equal along all axes");
            return false;
        }
        if(problem.GetDilationH() != problem.GetDilationW())
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Dilation must be 1");
            return false;
        }
        if(problem.GetBias() != 0)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Bias must be 0");
            return false;
        }
        const auto data_type = problem.GetInDataType();
        if(data_type != miopenFloat && data_type != miopenHalf && data_type != miopenBFloat16)
        {
            MIOPEN_LOG_I2("TunaNet Inapplicable: Unsupported data type");
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
        MIOPEN_LOG_I2("TunaNet Applicable");
        return true;
    }

protected:
    std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const override
    {
        const bool isFwd            = problem.GetDirection() == conv::Direction::Forward;
        std::vector<float> features = {
            static_cast<float>(isFwd ? problem.GetInChannels() : problem.GetOutChannels()),
            static_cast<float>(isFwd ? problem.GetInHeight() : problem.GetOutHeight()),
            static_cast<float>(isFwd ? problem.GetInWidth() : problem.GetOutWidth()),
            static_cast<float>(isFwd ? problem.GetOutChannels() : problem.GetInChannels()),
            static_cast<float>(isFwd ? problem.GetOutHeight() : problem.GetInHeight()),
            static_cast<float>(isFwd ? problem.GetOutWidth() : problem.GetInWidth()),
            static_cast<float>(problem.GetWeightsHeight()),
            static_cast<float>(problem.GetWeightsWidth()),
            static_cast<float>(problem.GetPadH()),
            static_cast<float>(problem.GetPadW()),
            static_cast<float>(problem.GetKernelStrideH()),
            static_cast<float>(problem.GetKernelStrideW()),
            static_cast<float>(problem.GetDilationH()),
            static_cast<float>(problem.GetDilationW()),
            static_cast<float>(problem.GetOutBatchSize()),
            static_cast<float>(metadata.EncodeLayout(problem.GetInLayout())),
            static_cast<float>(metadata.EncodePrecision(problem.GetInDataType())),
            static_cast<float>(metadata.EncodeDirection(problem.GetDirection())),
            static_cast<float>(problem.GetGroupCount())};

        // normalize
        for(size_t i = 0; i < features.size(); ++i)
            features[i] =
                (features[i] - metadata.test_features_mean[i]) / metadata.test_features_std[i];

        return features;
    }
};

std::unique_ptr<Model> GetModel(const std::string& device)
{
    if(device == "gfx942")
        return std::make_unique<Gfx942Model>();
    if(device == "gfx90a")
        return std::make_unique<Gfx90aModel>();
    return std::make_unique<Gfx908Model>();
}

std::vector<uint64_t> PredictSolver(const conv::ProblemDescription& problem,
                                    const ExecutionContext& ctx,
                                    const std::string& device)
{
    const static std::unique_ptr<Model> model = GetModel(device);
    if(!model || !model->IsProblemSupported(problem, ctx))
        return {};

    std::string est_name = ":memory:" + device;
    auto& db             = AnyRamDb::GetCached(est_name);
    auto db_res          = db.FindRecord(problem);
    if(db_res)
    {
        MIOPEN_LOG_I2("Cached heuristic (TunaNet) result found");
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

    MIOPEN_LOG_I2("Evaluating TunaNet");

    std::vector<float> res = model->Forward(problem);
    std::vector<std::pair<int, float>> sort_res(res.size());
    // sorts result based upon magnitude of result in vector, returned from Model,
    // paired with original index (idx). Sort magnitudes in descending order.
    // Greater magnitude = better solver. Indexes (idx), which will be used to map to solvers,
    // with greater corresponding magnitude are at front of the vector so they get priority.
    for(auto idx = 0; idx < res.size(); idx++)
        sort_res[idx] = {idx, res[idx]};
    const auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) -> bool {
        return a.second > b.second;
    };
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
    db.StoreRecord(problem, any_sol);
    if(miopen::IsLogging(LoggingLevel::Info2))
    {
        std::stringstream ss;
        for(auto& id : sol)
            ss << solver::Id{id}.ToString() << " ID:" << id << ", ";
        MIOPEN_LOG_I2("TunaNet Result: " << ss.str());
    }
    return sol;
}
} // namespace immed_mode
#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace tuning {

Metadata::Metadata(const std::string& arch, const std::string& solver)
{
    const nlohmann::json metadata =
        common::LoadJSON(GetSystemDbPath() / (arch + "_" + solver + "_metadata.ktn.model"));
    predict_type = metadata["predict_type"].get<std::size_t>();
    num_tuning_params =
        metadata["num_tuning_params"].get<std::unordered_map<std::string, std::size_t>>();
    tuning_decodings =
        metadata["decodings"]["tunings"].get<std::unordered_map<std::string, std::string>>();
}

class Model
{
public:
    Metadata metadata;
    Model(const std::string& arch, const std::string& solver)
        : metadata(Metadata(arch, solver)),
          encoder(fdeep::load_model(EncoderPath(arch, solver), true, fdeep::dev_null_logger)),
          decoder(fdeep::load_model(DecoderPath(arch, solver), true, fdeep::dev_null_logger))
    {
    }
    virtual ~Model() = default;
    fdeep::tensors Encode(const std::vector<float>& features, std::size_t dim, bool transform) const
    {
        const auto tensor_shape_depth = transform ? dim : 1;
        fdeep::tensor input_tensor =
            fdeep::tensor(fdeep::tensor_shape(dim, tensor_shape_depth), features);
        return encoder.predict({input_tensor});
    }
    fdeep::tensors Decode(const float prev_token, const fdeep::tensors& context) const
    {
        return decoder.predict(
            {{fdeep::tensor(fdeep::tensor_shape(1), std::vector<float>(1, prev_token)),
              context[0],
              context[1],
              context[2],
              context[3]}});
    }

private:
    const fdeep::model encoder;
    const fdeep::model decoder;
    static std::string EncoderPath(const std::string& arch, const std::string& solver)
    {
        const auto path = GetSystemDbPath() / (arch + "_" + solver + "_encoder.ktn.model");
        if(!fs::exists(path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
        return path.string();
    }
    static std::string DecoderPath(const std::string& arch, const std::string& solver)
    {
        const auto path = GetSystemDbPath() / (arch + "_" + solver + "_decoder.ktn.model");
        if(!fs::exists(path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
        return path.string();
    }
};

std::shared_ptr<Model> GetModel(const std::string& arch, const std::string& solver)
{
    static std::map<std::string, std::shared_ptr<Model>> models;
    auto it = models.find(solver);
    if(it == models.end())
    {
        std::shared_ptr<Model> model = std::make_shared<Model>(arch, solver);
        models[solver]               = model;
        return model;
    }
    else
    {
        return it->second;
    }
}

bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    miopen::conv::Direction direction,
                    const std::vector<float>& features,
                    bool transform_features,
                    std::function<bool(std::size_t, std::string)> validator)
{
    auto model = GetModel(arch, solver);
    int dim    = 0;
    if(transform_features)
        dim = std::sqrt(features.size());
    else
        dim = features.size();
    auto start             = std::chrono::high_resolution_clock::now();
    fdeep::tensors context = model->Encode(features, dim, transform_features);
    float decoder_input    = 0.0;
    std::string dir;
    switch(direction)
    {
    case miopen::conv::Direction::Forward: dir = "fwd"; break;
    case miopen::conv::Direction::BackwardData: dir = "bwd"; break;
    case miopen::conv::Direction::BackwardWeights: dir = "wrw"; break;
    default: return false;
    }

    for(size_t i = 0, num_tuning_params = 1; i < num_tuning_params; ++i)
    {

        if(i == 0 && (model->metadata.predict_type == 0u))
            num_tuning_params = model->metadata.num_tuning_params[dir];
        fdeep::tensors decoder_output = model->Decode(decoder_input, context);

        auto token_scores = decoder_output[0].to_vector();
        std::priority_queue<std::pair<float, int>> pq;
        for(int j = 0; j < token_scores.size(); j++)
            pq.push(std::make_pair(token_scores[j], j)); // sort by value at index

        int output_token_index = -1;
        while(!pq.empty())
        {
            int token = pq.top().second;
            // convert index to token value
            std::string value = model->metadata.tuning_decodings[std::to_string(token)];
            pq.pop();
            if(value == "-1")
            {
                auto stop     = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                MIOPEN_LOG_I2("Model ran for " << duration.count() << " micro-seconds");
                return false;
            }
            if(validator(i, value))
            {
                output_token_index =
                    token; // index with largest value that is valid = predicted index
                if(i == 0 && model->metadata.predict_type != 0u)
                    num_tuning_params = model->metadata.num_tuning_params[value];
                break;
            }
        }
        decoder_input = float(output_token_index);
        context       = {decoder_output.begin() + 1, decoder_output.end()};
    }

    auto stop     = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    MIOPEN_LOG_I2("Model ran for " << duration.count() << " micro-seconds");
    return true;
}

} // namespace tuning
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
} // namespace ai
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
