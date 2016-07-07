#include "Handle.hpp"

#if MLOpen_BACKEND_HIP

template<>
mlopenStatus_t mlopenContext::CreateDefaultStream<hipStream_t>() {
	hipStream_t stream;
	hipStreamCreate(&stream);

	SetStream(stream);
	return mlopenStatusSuccess;
}
#endif // OpenCL vs HIP


