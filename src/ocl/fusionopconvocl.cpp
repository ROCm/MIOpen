#include <miopen/fusion.hpp>

namespace miopen {

ProblemDescription ConvForwardOpDescriptor::GetConvProblem()
{
    TensorDescriptor o_desc;
    GetOutputDesc(o_desc);

    conv::ProblemDescription conv_problem(
        input_desc, filter_desc, o_desc, base_desc, miopen::conv::Direction::Forward);

    return conv_problem;
}

miopenStatus_t ConvForwardOpDescriptor::GetNetworkConfig(std::stringstream& network_config,
                                                         Handle& handle)
{
    std::ignore = handle;

    ProblemDescription conv_problem = GetConvProblem();

    std::string conv_config;
    conv_problem.BuildConfKey(conv_config);
    network_config << conv_config;
    return miopenStatusSuccess;
}

} // namespace miopen
