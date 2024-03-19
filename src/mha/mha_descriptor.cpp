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

extern "C" miopenStatus_t miopenSetMHADescriptor(miopenMHADescriptor_t mhaDesc,
                                                 float scale,
                                                 float dropoutProbability,
                                                 uint64_t dropoutSeed,
                                                 uint64_t dropoutOffset)
{
    MIOPEN_LOG_FUNCTION(mhaDesc, scale, dropoutProbability, dropoutSeed, dropoutOffset);
    return miopen::try_([&] {
        miopen::deref(mhaDesc).SetParams(scale, dropoutProbability, dropoutSeed, dropoutOffset);
    });
}

extern "C" miopenStatus_t miopenGetMHADescriptor(miopenMHADescriptor_t mhaDesc,
                                                 float* scale,
                                                 float* dropoutProbability,
                                                 uint64_t* dropoutSeed,
                                                 uint64_t* dropoutOffset)
{
    MIOPEN_LOG_FUNCTION(mhaDesc);
    return miopen::try_([&] {
        *scale              = miopen::deref(mhaDesc).GetScale();
        *dropoutProbability = miopen::deref(mhaDesc).GetDropoutProbability();
        *dropoutSeed        = miopen::deref(mhaDesc).GetDropoutSeed();
        *dropoutOffset      = miopen::deref(mhaDesc).GetDropoutOffset();
    });
}

std::ostream& operator<<(std::ostream& stream, const MHADescriptor& x)
{
    stream << "mha,"
           << "scale" << x.GetScale() << ",dropoutProbability" << x.GetDropoutProbability()
           << ",dropoutSeed" << x.dropoutSeed << ",dropoutOffset" << x.dropoutOffset << ",";

    return stream;
}

void to_json(nlohmann::json& json, const MHADescriptor& descriptor)
{
    json = nlohmann::json{
        {"scale", descriptor.GetScale()},
        {"dropoutProbability", descriptor.GetDropoutProbability()},
        {"dropoutSeed", descriptor.GetDropoutSeed()},
        {"dropoutOffset", descriptor.GetDropoutOffset()},
    };
}

void from_json(const nlohmann::json& json, MHADescriptor& descriptor)
{
    json.at("scale").get_to(descriptor.scale);
    json.at("dropoutProbability").get_to(descriptor.dropoutProbability);
    json.at("dropoutSeed").get_to(descriptor.dropoutSeed);
    json.at("dropoutOffset").get_to(descriptor.dropoutOffset);
}

} // namespace miopen
