#include<mlopen/lrn.hpp>
#include <cassert>

namespace mlopen {

LRNDescriptor::LRNDescriptor() {}

LRNDescriptor::LRNDescriptor(mlopenLRNMode_t m,
		const unsigned int pn, 
		const double	*pparms) : lrnN(pn), parms(pparms, pparms+3), mode(m) {}

mlopenLRNMode_t LRNDescriptor::GetMode() const
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
}  // namespace mlopen
