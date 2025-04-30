#ifndef GUARD_MLOPEN_ALLOCATOR_HPP
#define GUARD_MLOPEN_ALLOCATOR_HPP

#include <cassert>

#include <miopen/common.hpp>
#include <miopen/errors.hpp>
#include <miopen/manage_ptr.hpp>
#include <miopen/miopen.h>

namespace miopen {

struct AllocatorDeleter
{
    miopenDeallocatorFunction deallocator;
    void* context;

    template <class T>
    void operator()(T* x) const
    {
        assert(deallocator != nullptr);
        if(x != nullptr)
        {
            deallocator(context, x);
        }
    }
};
struct Allocator
{
    miopenAllocatorFunction allocator;
    miopenDeallocatorFunction deallocator;
    void* context;

    using ManageDataPtr =
        std::unique_ptr<typename std::remove_pointer<Data_t>::type, AllocatorDeleter>;

    ManageDataPtr operator()(std::size_t n) const
    {
        assert(allocator != nullptr);
        assert(deallocator != nullptr);
        auto result = allocator(context, n);
        if(result == nullptr && n != 0)
        {
            MIOPEN_THROW("Custom allocator failed to allocate memory for buffer size " +
                         std::to_string(n) + ": ");
        }
        return ManageDataPtr{DataCast(result), AllocatorDeleter{deallocator, context}};
    }
};

} // namespace miopen

#endif
