#include <miopen/problem_description.hpp>

namespace miopen {

ProblemDescription::ProblemDescription(conv::ProblemDescription desc)
    : conv::ProblemDescription(std::move(desc)), direction(GetDirection())
{
}

} // namespace miopen
