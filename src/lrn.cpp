#include<lrn.hpp>
#include <cassert>

mlopenLRNDescriptor::mlopenLRNDescriptor() {}

mlopenLRNDescriptor::mlopenLRNDescriptor(mlopenLRNMode_t m,
		const unsigned int pn, 
		const double	*pparms) : mode(m), lrnN(pn), parms(pparms, pparms+3) {}

mlopenLRNMode_t mlopenLRNDescriptor::GetMode() const
{
	return this->mode;
}

unsigned int mlopenLRNDescriptor::GetN() const
{
	return this->lrnN;
}

double mlopenLRNDescriptor::GetAlpha() const
{
	return this->parms[0];
}

double mlopenLRNDescriptor::GetBeta() const
{
	return this->parms[1];
}

double mlopenLRNDescriptor::GetK() const
{
	return this->parms[2];
}

