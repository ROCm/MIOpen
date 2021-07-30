#ifndef CK_DATA_TYPE_ENUM_HPP
#define CK_DATA_TYPE_ENUM_HPP

namespace ck {

// this enumerate should be synchronized with include/miopen.h
typedef enum
{
    Half     = 0,
    Float    = 1,
    Int32    = 2,
    Int8     = 3,
    Int8x4   = 4,
    BFloat16 = 5,
    Double   = 6,
    Unknown  = 100,
} DataTypeEnum_t;

} // namespace ck
#endif
