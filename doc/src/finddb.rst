Find-Db Database
================

Prior to MIOpen 2.0, users utilized calls such as ``miopenFindConvolution*Algorithm()`` to gather a set of convolution algorithms in the form of an array of ``miopenConvSolution_t`` structs. This process is time consuming because it requires online benchmarking of competing algorithms. In MIOpen 2.0 an :doc:`immediate mode <find_and_immediate>` is introduced.

Immediate mode is based on a database which contains the results of calls to the legacy Find() stage. This database is called ``Find-Db``. It consists of two parts:

* **System Find-Db**, a system-wide storage which holds the pre-run values for the most applicable configurations,
* **User Find-Db**, a per-user storage which is intended to hold results for arbitrary user-run configurations. It also performs double duty as a cache for the Find() stage.

The User Find-Db **always takes precedence** over System Find-Db.

By default, System Find-Db resides within MIOpen's install location, while User Find-Db resides in the user's home directory. See `Setting up locations <install.html#setting-up-locations>`_ for more information.

* The System Find-Db is *not* modified upon installation of MIOpen.
* There are separate Find databases for HIP and OpenCL backends.

Populating the User Find-Db
---------------------------

MIOpen collects Find-db information during the following MIOpen API calls:

* ``miopenFindConvolutionForwardAlgorithm()``
* ``miopenFindConvolutionBackwardDataAlgorithm()``
* ``miopenFindConvolutionBackwardWeightsAlgorithm()``

During the call, find data entries are collected for one *problem configuration* (implicitly defined by the tensor descriptors and convolution descriptor passed to API function).

Updating MIOpen and the User Find-Db
------------------------------------

When the user installs a new version of MIOpen, the new version of MIOpen will *ignore* old **User find-db*** files. Thus, the user is *not required* to move or delete their old User find-db files. However, the user may wish to re-collect the information into their brand new **User find-db**. This should be done in the same way as it was done with the previous version of the library -- *if* it was done. This would keep Immediate mode optimized.

Disabling Find-Db
-----------------

By default MIOpen will use the Find-Db. Users can disable the Find-Db by setting the environmental variable ``MIOPEN_DEBUG_DISABLE_FIND_DB`` to 1:

.. code-block:: console

   export MIOPEN_DEBUG_DISABLE_FIND_DB=1

.. note::

   The System Find-Db has the ability to be cached into memory and may increase performance dramatically. To disable this option use the cmake configuration flag:

   .. code-block::

      -DMIOPEN_DEBUG_FIND_DB_CACHING=Off
