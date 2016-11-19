#ifndef GUARD_MLOPEN_CONTEXT_HPP_
#define GUARD_MLOPEN_CONTEXT_HPP_

#include <mlopen.h>
#include <mlopen/object.hpp>
#include <mlopen/common.hpp>
#include <mlopen/kernel.hpp>
#include <vector>
#include <cstdio>
#include <cstring>
#include <memory>

namespace mlopen {

struct HandleImpl;

struct Handle : mlopenHandle {
	
	Handle();
	Handle(int numStreams, mlopenAcceleratorQueue_t *streams);
    Handle(Handle&&) noexcept;
	~Handle();

	mlopenAcceleratorQueue_t GetStream() const;

    void EnableProfiling(bool enable=true);

    float GetKernelTime() const;
#if MLOPEN_BACKEND_OPENCL
    KernelInvoke GetKernel(
            const std::string& algorithm,
            const std::string& network_config,
            const std::string& program_name,
            const std::string& kernel_name,
            const std::vector<size_t>& vld,
            const std::vector<size_t>& vgd,
            const std::string& params);

    KernelInvoke GetKernel(
        const std::string& algorithm,
        const std::string& network_config);

    void Finish() const;
    void Flush() const;
#endif

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
    	using type = typename Container::value_type;
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

	std::unique_ptr<HandleImpl> impl;
	
};
} // namespace mlopen
MLOPEN_DEFINE_OBJECT(mlopenHandle, mlopen::Handle);


#endif // GUARD_MLOPEN_CONTEXT_HPP_
