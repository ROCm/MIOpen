#include "Handle.hpp"
#include <cstdio>

extern "C" 
mlopenStatus_t mlopenCreate(mlopenHandle_t *handle,
		int numStreams,
		mlopenStream_t *streams ) {
		
	printf("In mlopenCreate\n");
	// if handle not valid
	if(handle == nullptr) {
		return mlopenStatusBadParm;
	}

#if MLOpen_BACKEND_OPENCL
	try {
		if(numStreams != 0) {
			*handle = new mlopenContext(numStreams, streams);
		}
		else {
			*handle = new mlopenContext();
			(*handle)->CreateDefaultStream<cl_command_queue>();
		}
	} catch (mlopenStatus_t status) {
		return status;
	}
	
#elif MLOpen_BACKEND_HIP
	try {
		if(numStreams != 0) {
			*handle = new mlopenContext(numStreams, streams);
		}
		else {
			*handle = new mlopenContext();
			(*handle)->CreateDefaultStream<hipStream_t>();
		}
	} catch (mlopenStatus_t status) {
		return status;
	}
#endif
	
	return mlopenStatusSuccess;
}

extern "C" 
mlopenStatus_t mlopenSetStream(mlopenHandle_t handle,
		int numStreams,
		mlopenStream_t *streamIDs) {
	printf("In mlopenSetStream\n");
	return handle->SetStream(numStreams, streamIDs);
}

extern "C"
mlopenStatus_t mlopenGetStream(mlopenHandle_t handle,
		mlopenStream_t *streamID,
		int numStream) {
	return handle->GetStream(streamID, numStream);
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
