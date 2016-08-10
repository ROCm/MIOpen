#ifndef GUARD_MLOPEN_CONTEXT_HPP_
#define GUARD_MLOPEN_CONTEXT_HPP_

#include <mlopen.h>
#include <mlopen/object.hpp>
#include <mlopen/common.hpp>
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

	ManageDataPtr Create(int sz);
	ManageDataPtr& WriteTo(const void* data, ManageDataPtr& ddata, int sz);
    void ReadTo(void* data, const ManageDataPtr& ddata, int sz);

	template<class T>
	ManageDataPtr Create(int sz)
	{
		return this->Create(sz*sizeof(T));
	}

	template<class Container>
    ManageDataPtr Write(const Container& c)
    {
    	typedef typename Container::value_type type;
    	auto buf = this->Create<type>(c.size());
    	return std::move(this->WriteTo(reinterpret_cast<const void*>(c.data()), buf, c.size()*sizeof(type)));
    }

    template<class T>
    std::vector<T> Read(const ManageDataPtr& ddata, int sz)
    {
    	std::vector<T> result(sz);
    	this->ReadTo(result.data(), ddata, sz*sizeof(T));
    	return result;
    }

	std::unique_ptr<ContextImpl> impl;
	
};
}
MLOPEN_DEFINE_OBJECT(mlopenHandle, mlopen::Context);


#endif // GUARD_MLOPEN_CONTEXT_HPP_
