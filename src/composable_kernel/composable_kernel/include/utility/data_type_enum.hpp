#ifndef CK_DATA_TYPE_ENUM_HPP
#define CK_DATA_TYPE_ENUM_HPP

namespace ck {

enum DataTypeEnum_t
{
    Half     = 0,
    Float    = 1,
    Int32    = 2,
    Int8     = 3,
    Int8x4   = 4,
    BFloat16 = 5,
    Double   = 6,
    Unknown  = 100,
};

} // namespace ck
#endif
