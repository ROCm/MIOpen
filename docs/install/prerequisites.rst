.. meta::
  :description: MIOpen prerequisites
  :keywords: MIOpen, ROCm, API, documentation, prerequisites, install

********************************************************************
MIOpen prerequisites
********************************************************************

To install MIOpen, you must first install these prerequisites. These prerequisites apply to
all types of MIOpen installations.

* A :doc:`ROCm <rocm:index>`-enabled platform
* A base software stack that includes either:

  * :doc:`HIP <hip:index>` (HIP and HCC libraries and header files)
  * OpenCL (OpenCL libraries and header files) (Using MIOpen with OpenCL is now deprecated.)

* `ROCm CMake <https://github.com/ROCm/rocm-cmake>`_: CMake modules for common
  build tasks needed for the ROCm software stack
* `Half <http://half.sourceforge.net/>`_: An IEEE 754-based, half-precision floating-point library
* `Boost <http://www.boost.org/>`_: Version 1.79 is recommended, because older versions might need patches
  to work on newer systems

  * MIOpen uses the ``boost-system`` and ``boost-filesystem`` packages to enable persistent
    :doc:`kernel cache <../conceptual/cache>`

* `SQLite3 <https://sqlite.org/index.html>`_: A read-write performance database
* lbzip2: A multi-threaded compression and decompression utility
* :doc:`rocBLAS <rocblas:index>`: AMD's library for Basic Linear Algebra Subprograms (BLAS) on the
  ROCm platform.

  * Minimum version for pre-ROCm 3.5
    `master-rocm-2.10 <https://github.com/ROCm/rocBLAS/tree/master-rocm-2.10>`_
  * Minimum version for post-ROCm 3.5
    `master-rocm-3.5 <https://github.com/ROCm/rocBLAS/tree/master-rocm-3.5>`_

* `Multi-Level Intermediate Representation (MLIR) <https://github.com/ROCm/rocMLIR>`_, with an
  MIOpen dialect to support and complement kernel development
* :doc:`Composable Kernel <composable_kernel:index>`: A C++ templated device library for
  GEMM-like and reduction-like operators.
