#include "Handle.hpp"

mlopenContext::mlopenContext (int numStreams, 
		mlopenStream_t *streams) {

	for(int i = 0; i < numStreams; i++) {
		_streams.push_back(streams[i]);
	}
}

mlopenStatus_t mlopenContext::SetStream (int numStreams,
		mlopenStream_t *streams) {
	printf("In Internal SetStream\n");

	if(numStreams == 0 && streams == NULL) {
		return mlopenStatusBadParm;
	}

	for(int i = 0; i < numStreams; i++) {
		_streams.push_back(streams[i]);
	}

	return mlopenStatusSuccess;
}

mlopenStatus_t mlopenContext::GetStream (mlopenStream_t *stream,
		int numStream) const {
	printf("In Internal GetStream\n");

	if(numStream >= _streams.size()) {
		return mlopenStatusBadParm;
	}

	// Using the default stream or the user defined stream.
	// Neglecting numStream parameter for now.
	*stream = _streams.back();

	return mlopenStatusSuccess;
}
