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
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <filesystem>
#include <vector>
#include <memory>
#include <string>

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
    MIOPEN_THROW("Unsupported data type passed to TunaNet");
}

size_t Metadata::EncodeLayout(const std::string& layout) const
{
    return layout_encodings.at(layout);
}

/** `Model` encapuslates the machinery required to run inference on a TunaNet model
 *
 * The `Model` class encapuslates all the machinery needed to run inference on a
 * TunaNet model, including loading the TunaNet model, formatting a problem so that it
 * can be fed into TunaNet for inference, and getting TunaNet's predictions etc.
 *
 * @param arch Architecture
 */
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
    virtual ~Model() = default;
    /** Is given problem supported by TunaNet?
     *
     * A TunaNet model can only work with problems "similar" to the problems it was trained on.
     * Since our training data has changed over time, a TunaNet model trained for an earlier
     * GPU might not support the same set of problems as a TunaNet model trained for a later
     * GPU might. Thus, each subclass of `Model`, specializing `Model` to a specific GPU, must
     * implement its own `IsProblemSupported` function.
     *
     * @param problem Problem
     * @param ctx Execution context
     */
    virtual bool IsProblemSupported(const conv::ProblemDescription& problem,
                                    const ExecutionContext& ctx) const = 0;
    /** Forward (i.e., run inference on) problem through TunaNet
     *
     * This function takes in a problem, converts it to a numeric vector and feeds it TunaNet
     * for inference. Its output is a numeric vector that represents a probability distribution.
     * Each index in this vector represents a solver (as given in metadata.solver_map) and the
     * value at each index represents the probability that that solver is the fastest for given
     * convolution problem.
     *
     * @param problem Problem
     */
    std::vector<float> Forward(const conv::ProblemDescription& problem) const
    {
        std::vector<float> features       = ToFeatures(problem);
        std::vector<fdeep::tensor> output = model.predict({fdeep::tensor(input_shape, features)});
        std::vector<float> output_vector  = output.front().to_vector();
        std::vector<float> res(output_vector.begin() + offset, output_vector.end());
        return res;
    }

protected:
    const fdeep::model model;              // TunaNet model
    const fdeep::tensor_shape input_shape; // Shape of input tensor required by TunaNet
    const size_t offset; // Some TunaNet models output some "fluff" before they output kernel
                         // probabilites. This offset tells how many indexes of fluff need to
                         // be skipped in order to get to kernel probabilities.
    /** Path to model file for given GPU
     *
     * The model files for each GPU are identified by the GPU architecture. This function takes
     * in a GPU architecture and returns the path to its TunaNet model.
     *
     * @param arch Architecture
     */
    static std::string ModelPath(const std::string& arch)
    {
        const auto file_path = GetSystemDbPath() / (arch + ".tn.model");
        if(!fs::exists(file_path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load AI model file:" + file_path);
        return file_path.string();
    }
    /** Convert given problem to a numeric vector
     *
     * TunaNet takes in a numeric vector representing the given problem. The exact details
     * of this vector vary from one TunaNet model to another, and thus this function, which
     * converts a problem into a numeric vector that can be fed to TunaNet, must be implemented
     * by each sub-class of `Model` on its own.
     *
     * @param problem Problem
     */
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
        if(problem.GetInLayout() != "NCHW" && problem.GetInLayout() != "NHWC")
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
    return std::make_unique<Gfx908Model>(); // default model if GPU-specific model is not available
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
    std::vector<float> res = model->Forward(problem); // res[i] gives the probability that the
                                                      // i-th solver is the fastest for given
                                                      // problem. ( The exact name of the i-th
                                                      // solver may be obtained as follows:
                                                      // model->metadata.solver_map.at(i) )

    // sort solvers in order of their probabilities
    std::vector<std::pair<int, float>> sort_res(res.size());
    for(auto idx = 0; idx < res.size(); idx++)
        sort_res[idx] = {idx, res[idx]};
    const auto cmp = [](const std::pair<int, float>& a, const std::pair<int, float>& b) -> bool {
        return a.second > b.second;
    };
    std::sort(sort_res.begin(), sort_res.end(), cmp);

    // map solver idx to solver id and then to anysolver
    std::vector<uint64_t> sol;
    std::vector<boost::any> any_sol;
    for(const auto& kinder : sort_res)
    {
        const auto id     = kinder.first; // index of solver in probability vector
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

fs::path GetKtnModelsPath()
{
    const char* env_path = std::getenv("MIOPEN_KTN_MODELS_PATH");
    fs::path models_path = GetSystemDbPath();
    if(env_path != nullptr)
    {
        fs::path path(env_path);
        if(fs::exists(path))
        {
            models_path = path; 
        }
    }
    return models_path;
}

Metadata::Metadata(const std::string& arch, const std::string& solver)
{
    const nlohmann::json metadata =
        common::LoadJSON(GetKtnModelsPath() / (arch + "_" + solver + "_metadata.ktn.model"));
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
    Model(const std::string& arch, const std::string& solver) : metadata(Metadata(arch, solver))
    {}
};

class FrugalModel : public Model
{
public:
    FrugalModel(const std::string& arch, const std::string& solver)
        : Model(arch, solver),
          encoder(fdeep::load_model(EncoderPath(arch, solver), true, fdeep::dev_null_logger)),
          decoder(fdeep::load_model(DecoderPath(arch, solver), true, fdeep::dev_null_logger))
    {
    }
    virtual ~FrugalModel() = default;
    /**
     * Encode the input features into a "context" tensor
     *
     * @param features Input features
     * @param dim Dimension (must be equal to len(features) if transform
     *            is True and sqrt(len(features)) otherwise)
     * @param transform Reshape input features into a square matrix?
     */
    fdeep::tensors Encode(const std::vector<float>& features, std::size_t dim, bool transform) const
    {
        // if transform==True, reshape input features into a matrix of `dim x dim` dimensions.
        // otherwise, have them as a vector of size `dim`.
        const auto tensor_shape_depth = transform ? dim : 1;
        fdeep::tensor input_tensor =
            fdeep::tensor(fdeep::tensor_shape(dim, tensor_shape_depth), features);

        return encoder.predict({input_tensor});
    }
    /**
     * Decode the next token based on the previous token and the encoded context.
     *
     * Decoder predicts the next token based on the previous token and the context predicted
     * by the Encoder. A token is a representation of a kernel parameter, i.e., each unique
     * token maps to a unique kernel parameter, with the only exception being the token '-1'
     * which signals the end of the decoding process (i.e., all kernel parameters have been
     * obtained).
     *
     * @param prev_token Previous token
     * @param context Context vector obtained from encoder
     */
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
        const auto base_path = GetKtnModelsPath();
        const auto path = base_path / (arch + "_" + solver + "_encoder.ktn.model");
        MIOPEN_LOG_I2("KTN Encoder model path" << path);
        if(!fs::exists(path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
        return path.string();
    }
    static std::string DecoderPath(const std::string& arch, const std::string& solver)
    {
        const auto base_path = GetKtnModelsPath();
        const auto path = base_path / (arch + "_" + solver + "_decoder.ktn.model");
        MIOPEN_LOG_I2("KTN Decoder model path" << path);
        if(!fs::exists(path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
        return path.string();
    }
};

class OnnxTransformerModel : public Model
{
public:
    OnnxTransformerModel(const std::string& arch, const std::string& solver)
        : Model(arch, solver),
          env_(ORT_LOGGING_LEVEL_WARNING, "MIOpenAI"),
          encoder_options_(CreateSessionOptions()),
          decoder_options_(CreateSessionOptions())
    {
        encoder_session_ = std::make_unique<Ort::Session>(
            env_,
            EncoderPath(arch, solver).c_str(),
            encoder_options_);
            
        decoder_session_ = std::make_unique<Ort::Session>(
            env_,
            DecoderPath(arch, solver).c_str(),
            decoder_options_);
            
        // Initialize memory info for CPU execution
        memory_info_ = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator,
            OrtMemType::OrtMemTypeDefault);
              
        // Get encoder input names
        encoder_input_count_ = encoder_session_->GetInputCount();
        const auto& enc_input_names = encoder_session_->GetInputNames();
        encoder_input_names_.reserve(enc_input_names.size());
        for (const auto& name : enc_input_names) 
        {   
            encoder_input_names_.push_back(std::string(name));
            encoder_input_name_ptrs_.push_back(encoder_input_names_.back().c_str());
        }
        
        // Get encoder output names
        encoder_output_count_ = encoder_session_->GetOutputCount();
        const auto& enc_output_names = encoder_session_->GetOutputNames();
        encoder_output_names_.reserve(enc_output_names.size());
        for (const auto& name : enc_output_names) 
        {
            encoder_output_names_.push_back(std::string(name));
            encoder_output_name_ptrs_.push_back(encoder_output_names_.back().c_str());
        }
            
        // Get decoder input names
        decoder_input_count_ = decoder_session_->GetInputCount();
        const auto& dec_input_names = decoder_session_->GetInputNames();
        decoder_input_names_.reserve(dec_input_names.size());
        for (const auto& name : dec_input_names) 
        {
            decoder_input_names_.push_back(std::string(name));
            decoder_input_name_ptrs_.push_back(decoder_input_names_.back().c_str());
        }
        
        // Get decoder output names
        decoder_output_count_ = decoder_session_->GetOutputCount();
        const auto& dec_output_names = decoder_session_->GetOutputNames();
        decoder_output_names_.reserve(dec_output_names.size());
        for (const auto& name : dec_output_names) 
        {
            decoder_output_names_.push_back(std::string(name));
            decoder_output_name_ptrs_.push_back(decoder_output_names_.back().c_str());
        }
    }
    
    std::vector<Ort::Value> Encode(const std::vector<float>& features, 
                                   std::size_t dim, 
                                   bool transform_features) const
    {
        std::vector<int64_t> input_shape;
        if (transform_features) 
        {
            // Reshape to square matrix if transform is true
            input_shape = {1, static_cast<int64_t>(dim), static_cast<int64_t>(dim)};
        } 
        else 
        {
            // Keep as flat vector if transform is false
            input_shape = {1, static_cast<int64_t>(features.size())};
        }
        
        // Create input tensor for encoder
        Ort::Value encoder_input_tensor = Ort::Value::CreateTensor<float>(
            memory_info_,
            const_cast<float*>(features.data()), 
            features.size(),
            input_shape.data(), 
            input_shape.size());
        
        // Run encoder and return encoder outputs
        return encoder_session_->Run(
            Ort::RunOptions{nullptr},
            encoder_input_name_ptrs_.data(),
            &encoder_input_tensor,
            encoder_input_count_, 
            encoder_output_name_ptrs_.data(),
            encoder_output_count_);
    }
    
    std::vector<float> Decode(const std::vector<int32_t>& sequence, 
                  const std::vector<Ort::Value>& encoder_outputs) const
    {
        // Create tensor for current sequence
        std::vector<int64_t> sequence_shape = {1, static_cast<int64_t>(sequence.size())};
        Ort::Value sequence_tensor = Ort::Value::CreateTensor<int32_t>(
            memory_info_,
            const_cast<int32_t*>(sequence.data()),
            sequence.size(),
            sequence_shape.data(),
            sequence_shape.size());
        
        // Prepare inputs for decoder
        std::vector<Ort::Value> decoder_inputs;
        decoder_inputs.push_back(std::move(sequence_tensor));
        
        // Add encoder outputs to decoder inputs
        for (size_t i = 0; i < encoder_outputs.size(); i++) {
            // We need to clone encoder outputs for each decode step
            auto tensor_info = encoder_outputs[i].GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            auto element_type = tensor_info.GetElementType();
            
            if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) 
            {
                auto* data = encoder_outputs[i].GetTensorData<float>();
                auto size = tensor_info.GetElementCount();
                
                Ort::Value cloned = Ort::Value::CreateTensor<float>(
                    memory_info_,
                    const_cast<float*>(data),
                    size,
                    shape.data(),
                    shape.size());
                
                decoder_inputs.push_back(std::move(cloned));
            }
            else   
            {
                MIOPEN_THROW(miopenStatusInternalError, "Unsupported tensor type");
            }
        }

        // Run decoder
        std::vector<Ort::Value> decoder_outputs = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            decoder_input_name_ptrs_.data(),
            decoder_inputs.data(),
            decoder_input_count_,
            decoder_output_name_ptrs_.data(),
            decoder_output_count_);
        
        // Get token scores from first output
        auto* output_data = decoder_outputs[0].GetTensorData<float>();
        auto output_info = decoder_outputs[0].GetTensorTypeAndShapeInfo();
        auto output_shape = output_info.GetShape();
        
        // Convert output to vector of scores
        size_t vocab_size = output_shape[1];
        std::vector<float> token_scores(output_data, output_data + vocab_size);
        
        return token_scores;
    }

private:
    Ort::Env env_;
    Ort::SessionOptions encoder_options_;
    Ort::SessionOptions decoder_options_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    Ort::MemoryInfo memory_info_{nullptr};
    
    // Input/output names and counts for running the encoder and decoder
    // sessions.
    size_t encoder_input_count_ = 0;
    size_t encoder_output_count_ = 0;
    size_t decoder_input_count_ = 0;
    size_t decoder_output_count_ = 0;
    std::vector<std::string> encoder_input_names_;
    std::vector<std::string> encoder_output_names_;
    std::vector<std::string> decoder_input_names_;
    std::vector<std::string> decoder_output_names_;
    std::vector<const char*> encoder_input_name_ptrs_;
    std::vector<const char*> encoder_output_name_ptrs_;
    std::vector<const char*> decoder_input_name_ptrs_;
    std::vector<const char*> decoder_output_name_ptrs_;
    
    // Helper method to create session options
    Ort::SessionOptions CreateSessionOptions() const
    {
        Ort::SessionOptions options;
        const int max_num_threads = std::thread::hardware_concurrency();
        options.SetIntraOpNumThreads(max_num_threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return options;
    }  
    
    fs::path EncoderPath(const std::string& arch, const std::string& solver) const
    {
        const auto base_path = GetKtnModelsPath();
        const auto path = base_path / (arch + "_" + solver + "_encoder.onnx");
        MIOPEN_LOG_I2("KTN Encoder model path: " << path);
        if(!fs::exists(path))
        {
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
        } 
        return path;
    }
    
    fs::path DecoderPath(const std::string& arch, const std::string& solver) const
    {
        const auto base_path = GetKtnModelsPath();
        const auto path = base_path / (arch + "_" + solver + "_decoder.onnx");
        MIOPEN_LOG_I2("KTN Decoder model path: " << path);
        if(!fs::exists(path))
        {
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path);
        }
        return path;
    }
};

/**
 * Return the KernelTuningNet model for given architecture and solver
 *
 * KernelTuningNet models are specific to each solver and are fine-tuned for each
 * GPU skew. This function constructs the KernelTuningNet model for the given
 * architecture and solver and stores it in a static map, so that the next time
 * the same model is required it doesn't have to be constructed anew.
 *
 * @param arch GPU Architecture
 * @param solver Solver
 */
std::shared_ptr<FrugalModel> GetFrugalModel(const std::string& arch, const std::string& solver)
{
    static std::map<std::string, std::shared_ptr<FrugalModel>> frugally_models;
    auto frugal_it = frugally_models.find(solver);
    if(frugal_it == frugally_models.end()) 
    {
        std::shared_ptr<FrugalModel> model = std::make_shared<FrugalModel>(arch, solver);
        frugally_models[solver] = model;
        return model;
    } 
    return frugal_it->second;
}

std::shared_ptr<OnnxTransformerModel> GetOnnxModel(const std::string& arch, const std::string& solver)
{
    static std::map<std::string, std::shared_ptr<OnnxTransformerModel>> onnx_models;
    auto onnx_it = onnx_models.find(solver);
    if (onnx_it == onnx_models.end()) 
    {
        std::shared_ptr<OnnxTransformerModel> model = 
                std::make_shared<OnnxTransformerModel>(arch, solver);
        onnx_models[solver] = model;
        return model;
    } 
    return onnx_it->second;
}

/**
 * Set kernel parameters for given solver
 *
 * @param arch GPU Architecture
 * @param solver Solver
 * @param direction Convolution Direction
 * @param features Input features for KernelTuningNet model
 * @param transform_features Whether or not to reshape features into a square
 *                           matrix before feeding them to KernelTuningNet
 * @param validator A boolean function that accepts an index `i` and a string `v`, and returns
 *                  True iff `v` is a valid kernel parameter value at index `i`
 *                  Note: This also applies the predicted token to kernel parameters, i.e., uses the prediction.
 */
bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    miopen::conv::Direction direction,
                    const std::vector<float>& features,
                    bool transform_features,
                    std::function<bool(std::size_t, std::string)> validator)
{
    // Set direction string
    std::string dir;
    switch(direction)
    {
    case miopen::conv::Direction::Forward: dir = "fwd"; break;
    case miopen::conv::Direction::BackwardData: dir = "bwd"; break;
    case miopen::conv::Direction::BackwardWeights: dir = "wrw"; break;
    default: return false;
    }
    
    // Check if we should use ONNX model
    const char* use_onnx_env = std::getenv("MIOPEN_USE_ONNX_KTN");
    bool use_onnx = (use_onnx_env != nullptr && std::string(use_onnx_env) == "1");
    
    auto start = std::chrono::high_resolution_clock::now();
    bool success = false;
    
    // Try ONNX implementation if requested
    if (use_onnx) {
        try {
            auto onnx_model = GetOnnxModel(arch, solver);
            
            // Get dimension for features
            int dim = transform_features ? std::sqrt(features.size()) : features.size();
            
            // Run encoder to get context
            MIOPEN_LOG_I2("Running ONNX KTN encoder");
            const auto& context = onnx_model->Encode(features, dim, transform_features);
            
            // Initialize with start token
            std::vector<int32_t> sequence = {0};
            size_t num_tuning_params = 1;
            bool valid_token_found = true;
            
            MIOPEN_LOG_I2("Running ONNX KTN decoder");
            // run decoder to set kernel parameters
            for(size_t i = 0; i < num_tuning_params && valid_token_found; ++i)
            {
                if(i == 0 && (onnx_model->metadata.predict_type == 0u))
                {
                    num_tuning_params = onnx_model->metadata.num_tuning_params[dir];
                }

                // Get token scores from decoder
                std::vector<float> token_scores = onnx_model->Decode(sequence, context);

                // Order tokens by score
                std::priority_queue<std::pair<float, int>> pq;
                for(int j = 0; j < token_scores.size(); j++) {
                    pq.push(std::make_pair(token_scores[j], j));
                }

                // Find the best valid token
                int output_token_index = -1;
                valid_token_found = false;

                while(!pq.empty() && !valid_token_found)
                {
                    int token = pq.top().second;
                    std::string value = onnx_model->metadata.tuning_decodings[std::to_string(token)];
                    pq.pop();

                    if(value == "-1") 
                    { 
                        // End of decoding
                        MIOPEN_LOG_I2("ONNX KTN ended at -1");
                        break;
                    }
                    
                    if(validator(i, value)) 
                    {
                        output_token_index = token;
                        valid_token_found = true;
                        
                        if(i == 0 && onnx_model->metadata.predict_type != 0u)
                        {
                            num_tuning_params = onnx_model->metadata.num_tuning_params[value];
                        }  
                    }
                }

                if(valid_token_found) 
                {
                    sequence.push_back(output_token_index);
                }
                else 
                {
                    MIOPEN_LOG_I2("ONNX KTN could not find valid token for position " << i);
                }
            }
            
            success = valid_token_found;
        }
        catch(const miopen::Exception& ex)
        {
            MIOPEN_LOG_I2("[Warning] Could not use ONNX model: (" << ex.what() << ")");
            success = false;
        }
    }
    
    // Frugally-deep implementation
    if (!use_onnx) {
        try
        {
            auto model = GetFrugalModel(arch, solver);
            
            // Get context
            int dim = transform_features ? std::sqrt(features.size()) : features.size();
            fdeep::tensors context = model->Encode(features, dim, transform_features);
            float decoder_input = 0.0;
            bool valid_token_found = true;

            // Run decoder to set kernel parameters
            for(size_t i = 0, num_tuning_params = 1; i < num_tuning_params && valid_token_found; ++i)
            {
                if(i == 0 && (model->metadata.predict_type == 0u))
                {
                    num_tuning_params = model->metadata.num_tuning_params[dir];
                }
                    
                fdeep::tensors decoder_output = model->Decode(decoder_input, context);
                auto token_scores = decoder_output[0].to_vector();
                
                // Order tokens by score
                std::priority_queue<std::pair<float, int>> pq;
                for(int j = 0; j < token_scores.size(); j++) {
                    pq.push(std::make_pair(token_scores[j], j));
                }

                int output_token_index = -1;
                valid_token_found = false;
                
                while(!pq.empty() && !valid_token_found)
                {
                    int token = pq.top().second;
                    std::string value = model->metadata.tuning_decodings[std::to_string(token)];
                    pq.pop();

                    if(value == "-1") 
                    { 
                        // End of decoding
                        MIOPEN_LOG_I2("KTN ended at -1");
                        break;
                    }
                    
                    if(validator(i, value)) 
                    {
                        output_token_index = token;
                        valid_token_found = true;
                        
                        if(i == 0 && model->metadata.predict_type != 0u)
                        {
                            num_tuning_params = model->metadata.num_tuning_params[value];
                        }
                    }
                }
                
                if(valid_token_found) {
                    decoder_input = float(output_token_index);
                    context = {decoder_output.begin() + 1, decoder_output.end()};
                }
            }
            
            success = valid_token_found;
        }
        catch(const miopen::Exception& ex)
        {
            MIOPEN_LOG_I2("[Warning] Could not retrieve model: (" << ex.what() << ")");
            success = false;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    MIOPEN_LOG_I2("KTN ran for " << duration.count() << " micro-seconds");
    return success;
}


} // namespace tuning
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
} // namespace ai
} // namespace miopen
#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
