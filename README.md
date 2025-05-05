# About
A C++ library to perform discrete Fourier transformations up to 3D.
It performs Fourier series expansion for a (periodic) complex function defined in 0 < x, y, z < 1.
The input function can be discrete data points (with even spacing in x, y and z) or function pointer.
One can use OpenMP or MPI to accelerate calculations.

# Build
This library can be built with **cmake**. \
In a **Linux/UNIX** system, one can build at a subdirectory with the following commands. \
&ensp;$ mkdir [subdirectory name] \
&ensp;$ cd [subdirectory name] \
&ensp;$ cmake [directory for the LibFFourier local repository] \
&ensp;$ cmake --build .
