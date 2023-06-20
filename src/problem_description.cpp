#include <miopen/problem_description.hpp>

#include <miopen/convolution.hpp>

#include <functional>
#include <sstream>
#include <tuple>

namespace miopen {

void ProblemDescription::BuildConfKey(std::string& conf_key) const
{
    conv_problem.BuildConfKey(conf_key);
}

bool ProblemDescription::IsLayoutDefault() const { return conv_problem.IsLayoutDefault(); }

bool ProblemDescription::IsLayoutNHWC() const { return conv_problem.IsLayoutNHWC(); }

bool ProblemDescription::IsLayoutNCHWc() const { return conv_problem.IsLayoutNCHWc(); }

void ProblemDescription::Serialize(std::ostream& stream) const
{
    return conv_problem.Serialize(stream);
}

ProblemDescription::ProblemDescription(conv::ProblemDescription desc)
    : conv_problem(std::move(desc)), direction(conv_problem.GetDirection())
{
}

} // namespace miopen
