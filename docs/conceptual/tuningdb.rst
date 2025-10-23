.. meta::
  :description: Using the MIOpen performance database
  :keywords: MIOpen, ROCm, API, documentation, performance database

************************************************************************************************
Tuning performance databases
************************************************************************************************

A key element for ensuring the best performance is to do tuning on the shapes used by your model.
MIOpen uses the following to decide upon the best solver to be used for a requested convolution:

* User DB:  This stores the results of previous tunings and by default is found in "~/.config/miopen", although this can be changed.
* System DB:  This is part of the MIOpen install files and has specific shapes that the MIOpen team has saved as tuning choices.
* Heuristics: This is a model trained on the shapes and solvers so that it picks the best solver and parameters.
  The goal is that this is within 90% of what can be achieved by specific tuning for the given shape,
  but manual tuning is expected to often give somewhat better results.

Manual tuning can either be incremental or exhaustive as detailed in the next sections.

Incremental tuning
==========================================================

This method conducts tuning when MIOpen encounters a new shape that doesn't exist in the user or system DBs.
Tuning is conducted once for each missing shape and the results are saved to the user DB.
Use this mode to add missing tuning information to the user database. This method defaults
to heuristic find results when FDB entries are missing.

Enable this feature using these commands:

.. code:: bash

  export MIOPEN_FIND_ENFORCE=3
  export MIOPEN_USER_DB_PATH="/user/specified/directory"

Exhaustive tuning
----------------------------------------------------------------------------------------------------------

Method 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Exhaustive tuning can be executed similarly to incremental tuning by ignoring the system DB.
Redirecting the system DB path causes MIOpen to miss system entries and heuristic models,
forcing a user DB entry to be generated. Each unique configuration is tuned exactly once,
with results aggregated in the user DB files.

Enable this feature using these commands:

.. code:: bash

  export MIOPEN_FIND_ENFORCE=3
  export MIOPEN_USER_DB_PATH="/user/specified/directory"
  export MIOPEN_SYSTEM_DB_PATH="$MIOPEN_USER_DB_PATH"


The default location for the MIOpen user database is "$HOME/.config/miopen"

Method 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To exhaustively tune all shapes explicitly, use ``MIOPEN_ENABLE_LOGGING_CMD=1`` with the target application to pull out all ``MIOpenDriver`` commands.
Then enable exhaustive tuning and run the unique commands.

.. code:: bash

  export MIOPEN_FIND_ENFORCE=4
  export MIOPEN_USER_DB_PATH="/user/specified/directory"


Kernel cache
==========================================================

``MIOPEN_CUSTOM_CACHE_DIR`` can be used to set the location of the user kernel cache, which is where compiled
versions of kernels needed on your specific GPU will be saved. The presence of entries in this cache
reduces first time run delay by reducing the compile time.

The cache directory defaults to "$HOME/.cache/miopen".

Post tuning
==========================================================
Unset ``MIOPEN_FIND_MODE`` and ``MIOPEN_FIND_ENFORCE`` to return to the default behavior. The behavior when these are set is expected
to be the same as the default when all shapes are tuned, but some first run delay has been observed.

If ``MIOPEN_SYSTEM_DB_PATH`` was set for exhaustive tuning, it can now be unset to revert to the default system
database install directory. The tuning entries created in the user database directory are preferred over the system entries.

The user DB at ``MIOPEN_USER_DB_PATH`` now contains all of the shapes that have been tuned.
This path must continue to point to the newly created/updated user database files.

Unset these variables using these commands:

.. code:: bash

  unset MIOPEN_FIND_MODE
  unset MIOPEN_FIND_ENFORCE
  unset MIOPEN_SYSTEM_DB_PATH
