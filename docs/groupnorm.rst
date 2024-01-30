
GroupNorm Layer(experimental)
=============================

The groupnorm types and functions.
It splits input channels into num_group groups and do normalize for each group.

To enable this, define MIOPEN_BETA_API before including miopen.h.


miopenNormMode_t
-----------------------

.. doxygenenum::  miopenNormMode_t

miopenGroupNormForward
----------------------------------

.. doxygenfunction::  miopenGroupNormForward

