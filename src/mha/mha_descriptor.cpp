#include <miopen/mha/mha_descriptor.hpp>
#include <miopen/logger.hpp>

#include <nlohmann/json.hpp>

namespace miopen {

extern "C" miopenStatus_t miopenCreateMHADescriptor(miopenMHADescriptor_t* mhaDesc)
{
    MIOPEN_LOG_FUNCTION(mhaDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(mhaDesc);
        desc       = new miopen::MHADescriptor();
    });
}

extern "C" miopenStatus_t miopenSetMHADescriptor(miopenMHADescriptor_t mhaDesc, float scale)
{
    MIOPEN_LOG_FUNCTION(mhaDesc, scale);
    return miopen::try_([&] { miopen::deref(mhaDesc).SetParams(scale); });
}

extern "C" miopenStatus_t miopenGetMHADescriptor(miopenMHADescriptor_t mhaDesc, float* scale)
{
    MIOPEN_LOG_FUNCTION(mhaDesc);
    return miopen::try_([&] { *scale = miopen::deref(mhaDesc).GetScale(); });
}

std::ostream& operator<<(std::ostream& stream, const MHADescriptor& x)
{
    stream << "mha,"
           << "scale" << x.GetScale() << ",";

    return stream;
}

void to_json(nlohmann::json& json, const MHADescriptor& descriptor)
{
    json = nlohmann::json{
        {"scale", descriptor.GetScale()},
    };
}

void from_json(const nlohmann::json& json, MHADescriptor& descriptor)
{
    json.at("scale").get_to(descriptor.scale);
}

} // namespace miopen
