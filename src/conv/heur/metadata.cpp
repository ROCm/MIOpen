/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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

#include <miopen/conv/heur/metadata.hpp>
#include <cstring>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <miopen/conv/heur/heur.hpp>

namespace miopen {

const std::vector<std::string>& GetFeatureNames(const std::string& arch)
{
    static const auto& metadata = ConvHeur::GetMetadata(arch);
    static const std::vector<std::string> feature_names = metadata["conv_params_used_as_features"];
    return feature_names;
}

const std::unordered_map<size_t, std::string>& GetSolverMap(const std::string& arch)
{
    static const auto& metadata = ConvHeur::GetMetadata(arch);
    static std::unordered_map<size_t, std::string> solver_map{};
    if(solver_map.empty())
    {
        std::unordered_map<std::string, size_t> solver_map_rev = metadata["encodings"]["solver"];
        for(auto& it : solver_map_rev)
        {
            solver_map.emplace(make_pair(it.second, it.first));
        }
    }
    // returns map of model output vector index to solver
    return solver_map;
}

size_t GetDirectionMap(const miopen::conv::Direction dir, const std::string& arch)
{
    static const auto& metadata = ConvHeur::GetMetadata(arch);

    if(dir == conv::Direction::BackwardWeights)
        return metadata["encodings"]["Direction"]["W"];
    else if(dir == conv::Direction::BackwardData)
        return metadata["encodings"]["Direction"]["B"];
    else
        return metadata["encodings"]["Direction"]["F"];
}

size_t GetPrecisionMap(const miopenDataType_t data_type, const std::string& arch)
{
    static const auto& metadata = ConvHeur::GetMetadata(arch);

    if(data_type == miopenBFloat16)
        return metadata["encodings"]["Precision"]["BF16"];
    else if(data_type == miopenHalf)
        return metadata["encodings"]["Precision"]["FP16"];
    else
        return metadata["encodings"]["Precision"]["FP32"];
}

size_t GetLayoutMap(const std::string& layout, const std::string& arch)
{
    static const auto& metadata = ConvHeur::GetMetadata(arch);

    if(layout == "NCDHW")
        return metadata["encodings"]["Layout"]["NCDHW"];
    else
        return metadata["encodings"]["Layout"]["NCHW"];
}

std::vector<float> GetStat(const std::string& stat, const std::string& arch)
{
    static const auto& metadata = ConvHeur::GetMetadata(arch);

    std::unordered_map<std::string, float> stat_map =
        metadata["stats"]["overall"]["features"][stat];
    // holds feature names in proper order for input vector to the model
    std::vector<std::string> features = GetFeatureNames(arch);
    std::vector<float> stats(features.size());
    for(size_t idx = 0; idx < stats.size(); idx++)
    {
        // get stat (mean or std) based upon feature name
        stats[idx] = stat_map[features[idx]];
    }
    return stats;
}

void TransformFeatures(std::vector<float>& features, const std::string& arch)
{
    static std::vector<float> mu  = GetStat("mean", arch);
    static std::vector<float> sig = GetStat("std", arch);
    for(size_t idx = 0; idx < features.size(); ++idx)
    {
        // standardize all features
        features[idx] = (features[idx] - mu[idx]) / sig[idx];
    }
}

size_t GetOffset(const std::string& arch)
{
    static const auto& metadata = ConvHeur::GetMetadata(arch);
    size_t offset               = metadata["num_algos"];
    return offset + 1;
}

bool IsProblemInDistributionL1(const std::vector<float>& features,
                               const float threshold)
{
	for(size_t i = 0; i < features.size(); ++i) {
		if ((features[i] > threshold) || (features[i] < -threshold))
			return false;
	}
	return true;
}

bool IsProblemInDistributionL2(const std::vector<float>& features,
                               const std::string& arch,
                               const float threshold)
{
    static const auto& metadata = ConvHeur::GetMetadata(arch);
	static const float dist_from_mean_avg = metadata["problem_distance_from_mean_avg"];
	static const float dist_from_mean_std = metadata["problem_distance_from_mean_std"];

	static const float upper_bound =  dist_from_mean_avg + threshold * dist_from_mean_std;
	static const float lower_bound =  dist_from_mean_avg - threshold * dist_from_mean_std;

	float squared_sum = 0;
	for(size_t i = 0; i < features.size(); ++i)
		squared_sum += std::pow(features[i], 2);
	const float distance = std::sqrt(squared_sum);

	return distance > lower_bound && distance < upper_bound;
}

std::vector<float> CallModel(std::vector<float>& features, const std::string& arch)
{
    static boost::filesystem::path model_file =
        boost::filesystem::path(GetSystemDbPath() + "/" + arch + ".model");
    static const fdeep::model model =
        fdeep::load_model(model_file.generic_string(), true, fdeep::dev_null_logger);
    auto input = fdeep::tensor(fdeep::tensor_shape(features.size()), features);
    std::vector<fdeep::tensor> output = model.predict({input});
    std::vector<float> output_vector  = output.front().to_vector();
    // offset the first couple indexes since they are unimportant to predicting optimal solvers
    std::vector<float> res(output_vector.begin() + GetOffset(arch), output_vector.end());
    return res;
}
} // namespace miopen
