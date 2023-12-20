#include <miopen/problem_description.hpp>

namespace miopen {

ProblemDescription::ProblemDescription(conv::ProblemDescription desc)
    : conv::ProblemDescription(std::move(desc)), direction(GetDirection())
{
#if FIN_OLD_PROBLEM_DESCRIPTION_COMPAT
    conv_problem.p = this;
#endif
}

} // namespace miopen
