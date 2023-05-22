#include <miopen/problem_description.hpp>

#include <miopen/convolution.hpp>

#include <functional>
#include <sstream>
#include <tuple>

namespace miopen {

std::ostream& operator<<(std::ostream& stream, std::function<void(std::ostream&)>&& manipulator)
{
    manipulator(stream);
    return stream;
}

int ProblemDescription::mloBuildConf_Key(std::string& conf_key) const
{
    conv_problem.BuildConfKey(conf_key);
    return (0);
}

bool ProblemDescription::IsLayoutDefault() const { return conv_problem.IsLayoutDefault(); }

bool ProblemDescription::IsLayoutNHWC() const { return conv_problem.IsLayoutNHWC(); }

bool ProblemDescription::IsLayoutNCHWc() const { return conv_problem.IsLayoutNCHWc(); }

void ProblemDescription::Serialize(std::ostream& stream) const
{
    return conv_problem.Serialize(stream);
}

ProblemDescription::ProblemDescription(const TensorDescriptor& in,
                                       const TensorDescriptor& weights,
                                       const TensorDescriptor& out,
                                       const ConvolutionDescriptor& conv,
                                       conv::Direction dir,
                                       int bias_)
    : ProblemDescription(dir == conv::Direction::Forward
                             ? conv::ProblemDescription{in, weights, out, conv, dir, bias_}
                             : conv::ProblemDescription{out, weights, in, conv, dir, bias_})
{
}

ProblemDescription::ProblemDescription(conv::ProblemDescription desc)
    : conv_problem(std::move(desc)), direction(conv_problem.GetDirection())
{
}

} // namespace miopen
