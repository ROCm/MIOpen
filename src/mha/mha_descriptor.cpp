#include <miopen/mha/mha_descriptor.hpp>

#include <nlohmann/json.hpp>

namespace miopen {

extern "C" miopenStatus_t miopenCreateMHADescriptor(miopenSoftmaxDescriptor_t* mhaDesc)
{
    MIOPEN_LOG_FUNCTION(mhaDesc);
    return miopen::try_([&] {
        auto& desc = miopen::deref(mhaDesc);
        desc       = new miopen::MHADescriptor();
    });
}

extern "C" miopenStatus_t miopenSetMHADescriptor(miopenSoftmaxDescriptor_t mhaDesc,
                                                     float scale,
                                                     float dropoutProbability)
{
    MIOPEN_LOG_FUNCTION(mhaDesc, scale, dropoutProbability);
    return miopen::try_([&] { miopen::deref(activDesc).SetParams(scale, dropoutProbability); });
}

std::ostream& operator<<(std::ostream& stream, const MHADescriptor& x)
{
    stream << "mha," << "scale" << x.GetScale() << ",dropoutProbability" << x.GetDropoutProbability() << ",";

    return stream;
}

void to_json(nlohmann::json& json, const MHADescriptor& descriptor)
{
    json = nlohmann::json{
        {"scale", descriptor.GetScale()},
        {"dropoutProbability", descriptor.GetDropoutProbability()},
    };
}

void from_json(const nlohmann::json& json, SoftmaxDescriptor& descriptor)
{
    json.at("scale").get_to(descriptor.scale);
    json.at("dropoutProbability").get_to(descriptor.dropoutProbability);
}