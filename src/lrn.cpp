#include <miopen/lrn.hpp>
#include <miopen/logger.hpp>
#include <cassert>

namespace miopen {

LRNDescriptor::LRNDescriptor() {}

LRNDescriptor::LRNDescriptor(miopenLRNMode_t m,
		const unsigned int pn, 
		const double	*pparms) : lrnN(pn), parms(pparms, pparms+3), mode(m) {}

miopenLRNMode_t LRNDescriptor::GetMode() const
{
	return this->mode;
}

unsigned int LRNDescriptor::GetN() const
{
	return this->lrnN;
}

double LRNDescriptor::GetAlpha() const
{
	return this->parms[0];
}

double LRNDescriptor::GetBeta() const
{
	return this->parms[1];
}

double LRNDescriptor::GetK() const
{
	return this->parms[2];
}
std::ostream& operator<< (std::ostream& stream, const LRNDescriptor& x)
{
    MIOPEN_LOG_ENUM(stream, x.mode, miopenLRNWithinChannel, miopenLRNCrossChannel) << ", ";
    stream << x.lrnN << ", ";
    LogRange(stream, x.parms, ", ") << ", ";
    return stream;
}
}  // namespace miopen
