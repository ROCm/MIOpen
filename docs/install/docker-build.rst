.. meta::
  :description: Build MIOpen using Docker
  :keywords: MIOpen, ROCm, API, documentation, Docker

********************************************************************
Build MIOpen using Docker
********************************************************************

You can build MIOpen using Docker by either downloading a prebuilt image or creating your own.

.. note::

   For ease of use, the prebuilt Docker image is recommended.

*  Downloading a prebuilt image

   You can find prebuilt Docker images at `ROCm Docker Hub <https://hub.docker.com/r/rocm/miopen/tags>`_.

*  Building your own image

   #. To build the Docker image, use ``docker build``:

      .. code-block:: bash

         docker build -t miopen-image .

   #. To enter the development environment, use ``docker run``, for example:

      .. code-block:: bash

         docker run -it -v $HOME:/data --privileged --rm --device=/dev/kfd --device /dev/dri:/dev/dri:rw
         --volume /dev/dri:/dev/dri:rw -v /var/lib/docker/:/var/lib/docker --group-add video
         --cap-add=SYS_PTRACE --security-opt seccomp=unconfined miopen-image

   #. Enter the Docker environment and run ``git clone MIOpen``. You can now build MIOpen using
      CMake. For instructions on how to build MIOpen from source, see :doc:`building MIOpen <./build-source>`.
