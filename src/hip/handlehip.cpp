#include <mlopen/context.hpp>

template<>
mlopenStatus_t mlopenContext::CreateDefaultStream<hipStream_t>() {
	hipStream_t stream;
	hipStreamCreate(&stream);

	SetStream(stream);
	return mlopenStatusSuccess;
}


