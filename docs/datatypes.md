
# Datatypes


MIOpen contains several datatypes at different levels of support. The enumerated datatypes are shown below:

```
typedef enum {
    miopenHalf     = 0,
    miopenFloat    = 1,
    miopenInt32    = 2,
    miopenInt8     = 3,
    miopenInt8x4   = 4,
    miopenBFloat16 = 5,
} miopenDataType_t;
```

Of these types only `miopenFloat` and `miopenHalf` are fully supported across all layers in MIOpen. Please see the individual layers in API reference section for specific datatype support and limitations.

Type descriptions:
 * `miopenHalf` - 16-bit floating point
 * `miopenFloat` - 32-bit floating point
 * `miopenInt32` - 32-bit integer, used primarily for `int8` convolution outputs
 * `miopenInt8` - 8-bit integer, currently only supported by `int8` convolution forward path, tensor set, tensor copy, tensor cast, tensor transform, tensor transpose, and im2col.
 * `miopenInt8x4` - 8-bit 4 element vector type used primarily with `int8` convolutions forward path.
 * `miopenBFloat16` - brain float fp-16 (8-bit exponent, 7-bit fraction), currently only supported by convolutions, tensor set, and tensor copy.


Note: In addition to the standard datatypes above, pooling contains its own indexing datatypes:
```
typedef enum {
    miopenIndexUint8  = 0,
    miopenIndexUint16 = 1,
    miopenIndexUint32 = 2,
    miopenIndexUint64 = 3,
} miopenIndexType_t;
```

