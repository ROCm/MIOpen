#include <nlohmann/json.hpp>
#include <miopen/miopen.h>
#include <miopen/conv/context.hpp>
#include <miopen/solver.hpp>
#include <unordered_map>
#include <queue>
#include <typeinfo>
#include <string>
#include <iostream>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
namespace miopen {
namespace ai {
namespace tuning {
nlohmann::json GetModelMetadata(const std::string& arch, const std::string& solver)
{
    std::string file_path = GetSystemDbPath() + "/" + arch + "_" + solver + "_metadata.model";
    return nlohmann::json::parse(std::ifstream(file_path));
}
} // namespace tuning
} // namespace ai
} // namespace miopen
#endif
