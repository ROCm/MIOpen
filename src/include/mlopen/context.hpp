#ifndef GUARD_MLOPEN_CONTEXT_HPP_
#define GUARD_MLOPEN_CONTEXT_HPP_

#include <mlopen.h>
#include <mlopen/object.hpp>
#include <vector>
#include <cstdio>
#include <cstring>
#include <memory>

namespace mlopen {

struct ContextImpl;

struct Context : mlopenHandle {
	
	Context();
	Context(int numStreams, mlopenAcceleratorQueue_t *streams);
	~Context();

	mlopenAcceleratorQueue_t GetStream();

	// Deprecated
	int GetStream (mlopenAcceleratorQueue_t *stream, ...)
	{
		*stream = this->GetStream();
		return 0;
	}

	std::unique_ptr<ContextImpl> impl;
	
};
}
MLOPEN_DEFINE_OBJECT(mlopenHandle, mlopen::Context);


#endif // GUARD_MLOPEN_CONTEXT_HPP_
