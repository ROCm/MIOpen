#include "Handle.hpp"
#include <cstdio>

extern "C" 
mlopenStatus_t mlopenCreate(mlopenHandle_t *handle,
		int numStreams,
		mlopenAcceleratorQueue_t *streams ) {
		
	// if handle not valid
	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}

	try {
		if(numStreams != 0) {
			*handle = new mlopenContext(numStreams, streams);
		}
		else {
			*handle = new mlopenContext();
		}
	} catch (mlopenStatus_t status) {
		return status;
	}
	
	return mlopenStatusSuccess;
}

// TODO: Stream size should be a spearate parameter
extern "C"
mlopenStatus_t mlopenGetStream(mlopenHandle_t handle,
		mlopenAcceleratorQueue_t *streamID,
		int numStream) {
	*streamID = handle->GetStream();
	return mlopenStatusSuccess;
}

extern "C"
mlopenStatus_t mlopenDestroy(mlopenHandle_t handle) {
	try {
		delete handle;
	} catch (mlopenStatus_t status) {
		return status;
	}
	return mlopenStatusSuccess;
}
