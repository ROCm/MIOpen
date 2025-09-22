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

#ifndef GAURD_MIOPEN_AI_HEURISTICS_HPP_
#define GAURD_MIOPEN_AI_HEURISTICS_HPP_

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK || MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <unordered_map>
#include <typeinfo>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <fstream>
#include <optional>
#include <miopen/miopen.h>
#include <nlohmann/json.hpp>
#include <miopen/db_path.hpp>
#include <miopen/any_solver.hpp>
#include <miopen/filesystem.hpp>
#include <miopen/anyramdb.hpp>

namespace miopen {
namespace ai {

// Common utility functions for AI heuristics (2D, 3D, and KTN)
namespace common {

/**
 * @brief Load JSON from file path
 * @param path File system path to JSON file
 * @return Parsed JSON object
 * @throws miopenStatusInternalError if file doesn't exist or can't be parsed
 */
inline nlohmann::json LoadJSON(const fs::path& path)
{
    if(!fs::exists(path))
        MIOPEN_THROW(miopenStatusInternalError, "Unable to load file: " + path.string());
    return nlohmann::json::parse(std::ifstream(path));
}

/**
 * @brief Reverse a map (swap keys and values)
 * @param map Original map to reverse
 * @return Reversed map with keys and values swapped
 */
template <typename U, typename V>
std::unordered_map<V, U> ReverseMap(const std::unordered_map<U, V>& map)
{
    std::unordered_map<V, U> reversed_map = {};
    for(const auto& it : map)
        reversed_map.emplace(std::make_pair(it.second, it.first));
    return reversed_map;
}

/**
 * @brief Lookup values from map using keys vector
 * @param keys Vector of keys to lookup
 * @param map Map to lookup values from
 * @return Vector of values corresponding to keys
 */
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
struct Metadata
{
private:
    nlohmann::json json;
    const std::unordered_map<std::string, int> direction_encodings;
    const std::unordered_map<std::string, int> precision_encodings;
    const std::unordered_map<std::string, int> layout_encodings;

public:
    const std::vector<std::string> features;
    const size_t num_inputs;
    const size_t num_outputs;
    const size_t num_solvers;
    const std::unordered_map<size_t, std::string> solver_map;
    const std::vector<float> features_mean;
    const std::vector<float> features_std;
    const std::vector<float> test_features_mean;
    const std::vector<float> test_features_std;
    Metadata(const std::string& arch);
    size_t EncodeDirection(miopen::conv::Direction dir) const;
    size_t EncodePrecision(miopenDataType_t data_type) const;
    size_t EncodeLayout(const std::string& layout) const;
};
class Model;
MIOPEN_INTERNALS_EXPORT std::vector<uint64_t> PredictSolver(const conv::ProblemDescription& problem,
                                                            const ExecutionContext& ctx,
                                                            const std::string& device);
} // namespace immed_mode

/**
 * @brief 3D convolution AI heuristics namespace
 *
 * This namespace contains classes and functions for 3D convolution AI heuristics
 * using TunaNet3D neural networks to predict optimal solvers for 3D convolution
 * operations (NCDHW layout).
 */
namespace conv3d {

/**
 * @brief 3D-specific metadata handler for TunaNet3D models
 *
 * This class provides a simple interface for accessing 3D convolution metadata.
 * All data is loaded during construction with proper error handling.
 * Design matches 2D Metadata pattern for consistency.
 */
class Metadata3D
{
private:
    const std::string arch_name;
    bool is_valid; // Error handling flag

    // Loaded data (const members like 2D pattern)
    std::vector<std::string> features;
    size_t num_inputs;
    size_t num_outputs;
    size_t num_solvers;
    std::unordered_map<size_t, std::string> solver_map;
    std::vector<float> features_mean;
    std::vector<float> features_std;

    // Encoding maps
    std::unordered_map<std::string, int> direction_encodings_3d;
    std::unordered_map<std::string, int> precision_encodings_3d;
    std::unordered_map<std::string, int> in_layout_encodings;
    std::unordered_map<std::string, int> fil_layout_encodings;
    std::unordered_map<std::string, int> out_layout_encodings;

    // Helper functions for construction
    static std::optional<std::vector<std::string>> LoadFeatures(const std::string& arch);
    static std::optional<size_t> LoadNumInputs(const std::string& arch);
    static std::optional<size_t> LoadNumOutputs(const std::string& arch);
    static std::optional<size_t> LoadNumSolvers(const std::string& arch);
    static std::optional<std::unordered_map<size_t, std::string>>
    LoadSolverMap(const std::string& arch);
    static std::optional<std::vector<float>> LoadFeaturesMean(const std::string& arch,
                                                              size_t num_inputs);
    static std::optional<std::vector<float>> LoadFeaturesStd(const std::string& arch,
                                                             size_t num_inputs);
    static std::optional<std::unordered_map<std::string, int>>
    LoadDirectionEncodings(const std::string& arch);
    static std::optional<std::unordered_map<std::string, int>>
    LoadPrecisionEncodings(const std::string& arch);
    static std::optional<std::unordered_map<std::string, int>>
    LoadInLayoutEncodings(const std::string& arch);
    static std::optional<std::unordered_map<std::string, int>>
    LoadFilLayoutEncodings(const std::string& arch);
    static std::optional<std::unordered_map<std::string, int>>
    LoadOutLayoutEncodings(const std::string& arch);

public:
    /**
     * @brief Constructor - loads all metadata immediately with error handling
     * @param arch Architecture name (e.g., "gfx942_3d")
     * @note Does not throw - use IsValid() to check for errors
     */
    MIOPEN_INTERNALS_EXPORT explicit Metadata3D(const std::string& arch);

    /**
     * @brief Check if metadata was loaded successfully
     * @return true if all data loaded correctly, false if any errors occurred
     */
    bool IsValid() const { return is_valid; }

    /**
     * @brief Get architecture name
     * @return Architecture name used during construction
     */
    const std::string& GetArchName() const { return arch_name; }

    /**
     * @brief Get list of feature names used by 3D model
     * @return Reference to feature names vector
     * @note Call IsValid() first to ensure data is available
     */
    const std::vector<std::string>& GetFeatures() const { return features; }

    /**
     * @brief Get number of input features
     * @return Number of inputs
     */
    size_t GetNumInputs() const { return num_inputs; }

    /**
     * @brief Get number of output features
     * @return Number of outputs
     */
    size_t GetNumOutputs() const { return num_outputs; }

    /**
     * @brief Get number of solvers
     * @return Number of solvers
     */
    size_t GetNumSolvers() const { return num_solvers; }

    /**
     * @brief Get solver mapping (index to name)
     * @return Reference to solver map
     */
    const std::unordered_map<size_t, std::string>& GetSolverMap() const { return solver_map; }

    /**
     * @brief Get feature mean values for normalization
     * @return Reference to features mean vector
     */
    const std::vector<float>& GetFeaturesMean() const { return features_mean; }

    /**
     * @brief Get feature standard deviation values for normalization
     * @return Reference to features std vector
     */
    const std::vector<float>& GetFeaturesStd() const { return features_std; }

    /**
     * @brief Encode convolution direction to integer
     * @param dir Convolution direction (Forward/BackwardData/BackwardWeights)
     * @return Encoded direction value, or 0 if direction not supported
     */
    MIOPEN_INTERNALS_EXPORT size_t EncodeDirection(miopen::conv::Direction dir) const;

    /**
     * @brief Encode data type to integer
     * @param data_type Data type (FP32/FP16/BF16)
     * @return Encoded precision value, or 0 if type not supported
     */
    MIOPEN_INTERNALS_EXPORT size_t EncodePrecision(miopenDataType_t data_type) const;

    /**
     * @brief Encode layout string to integer (generic)
     * @param layout Layout string
     * @return Encoded layout value, or 0 if layout not supported
     */
    MIOPEN_INTERNALS_EXPORT size_t EncodeLayout(const std::string& layout) const;

    /**
     * @brief Encode input layout string to integer
     * @param layout Input layout string
     * @return Encoded input layout value, or 0 if layout not supported
     */
    MIOPEN_INTERNALS_EXPORT size_t EncodeInLayout(const std::string& layout) const;

    /**
     * @brief Encode filter layout string to integer
     * @param layout Filter layout string
     * @return Encoded filter layout value, or 0 if layout not supported
     */
    MIOPEN_INTERNALS_EXPORT size_t EncodeFilLayout(const std::string& layout) const;

    /**
     * @brief Encode output layout string to integer
     * @param layout Output layout string
     * @return Encoded output layout value, or 0 if layout not supported
     */
    MIOPEN_INTERNALS_EXPORT size_t EncodeOutLayout(const std::string& layout) const;
};

/**
 * @brief Abstract base class for 3D AI heuristics models
 *
 * This class defines the interface for 3D convolution AI heuristics models.
 * Implementations should provide device-specific TunaNet3D inference
 * for predicting optimal 3D convolution solvers.
 */
class Model3D
{
public:
    virtual ~Model3D() = default;

    /**
     * @brief Check if a 3D convolution problem is supported by this model
     * @param problem 3D convolution problem description
     * @param ctx Execution context
     * @return true if problem is supported, false otherwise
     */
    virtual bool IsProblemSupported(const conv::ProblemDescription& problem,
                                    const ExecutionContext& ctx) const = 0;

    /**
     * @brief Run TunaNet3D inference on the given 3D problem
     * @param problem 3D convolution problem description
     * @return Vector of solver probabilities (one per solver)
     */
    virtual std::vector<float> Forward(const conv::ProblemDescription& problem) const = 0;

    /**
     * @brief Get solver index to name mapping
     * @return Map from solver indices to solver names for result interpretation
     */
    virtual const std::unordered_map<size_t, std::string>& GetSolverMap() const = 0;

protected:
    /**
     * @brief Extract numerical features from 3D convolution problem
     * @param problem 3D convolution problem description
     * @return Feature vector for TunaNet3D input (23 features)
     */
    virtual std::vector<float> ToFeatures(const conv::ProblemDescription& problem) const = 0;
};

/**
 * @brief Factory function to create 3D AI heuristics model for given device
 * @param device GPU device name (e.g., "gfx942", "gfx90a")
 * @return Device-specific 3D model instance, or nullptr if unsupported
 */
MIOPEN_INTERNALS_EXPORT std::unique_ptr<Model3D> Get3DModel(const std::string& device);
} // namespace conv3d

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace tuning {
struct Metadata
{
    std::size_t predict_type;
    std::unordered_map<std::string, std::size_t> num_tuning_params;
    std::unordered_map<std::string, std::string> tuning_decodings;
    Metadata(const std::string& arch, const std::string& solver);
};

bool ModelSetParams(const std::string& arch,
                    const std::string& solver,
                    conv::Direction direction,
                    const std::vector<float>& features,
                    bool transform_features,
                    std::function<bool(std::size_t, std::string)> validator);
} // namespace tuning
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
} // namespace ai
} // namespace miopen
#endif
#endif // GAURD_MIOPEN_AI_HEURISTICS_HPP_
