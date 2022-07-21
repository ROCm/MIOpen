
Dropout Layer
=============

The dropout layer API documentation


miopenRNGType_t
---------------

.. doxygenenum::  miopenRNGType_t

miopenCreateDropoutDescriptor
-----------------------------

.. doxygenfunction::  miopenCreateDropoutDescriptor

miopenGetDropoutDescriptor
--------------------------

.. doxygenfunction::  miopenGetDropoutDescriptor

miopenRestoreDropoutDescriptor
------------------------------

.. doxygenfunction::  miopenRestoreDropoutDescriptor

miopenDestroyDropoutDescriptor
------------------------------

.. doxygenfunction::  miopenDestroyDropoutDescriptor

miopenSetDropoutDescriptor
--------------------------

.. doxygenfunction::  miopenSetDropoutDescriptor

miopenDropoutGetReserveSpaceSize
--------------------------------

.. doxygenfunction::  miopenDropoutGetReserveSpaceSize

miopenDropoutGetStatesSize
--------------------------

.. doxygenfunction::  miopenDropoutGetStatesSize

miopenDropoutForward
--------------------

.. doxygenfunction::  miopenDropoutForward

**Return value description:**

* `miopenStatusSuccess` - No errors.

* `miopenStatusBadParm` - Incorrect parameter detected. Check if the following conditions are met:

  - Input/Output dimension/element size/datatype does not match.

  - Tensor dimension is not in the range of 1D to 5D.

  - Noise shape is unsupported.

  - Dropout rate  is not in the range of (0, 1].

  - Insufficient state size for parallel PRNG.

  - Insufficient reservespace size.

  - Memory required by dropout forward configs exceeds GPU memory range.

miopenDropoutBackward
---------------------

.. doxygenfunction::  miopenDropoutBackward

**Return value description:**

* `miopenStatusSuccess` - No errors.

* `miopenStatusBadParm` - Incorrect parameter detected. Check if the following conditions are met:

  - Input/Output dimension/element size/datatype does not match.

  - Tensor dimension is not in the range of 1D to 5D.

  - Dropout rate  is not in the range of (0, 1].

  - Insufficient reservespace size.

  - Memory required by dropout backward configs exceeds GPU memory range.
