# About
A C++ library to perform discrete Fourier transformations up to 3D.
It performs Fourier series expansion for a (periodic) complex function defined in 0 < _x_, _y_, _z_ < 1.
The input function can be discrete data points (with even spacing in _x_, _y_ and _z_) or function pointer.
One can use `OpenMP` and/or `MPI` to accelerate calculations.

# Build
This library can be built with **cmake**. \
In a **Linux/UNIX** system, one can build at a subdirectory with the following commands.
```
$ mkdir [subdirectory name]
$ cd [subdirectory name]
$ cmake [directory for the LibFFourier local repository]
$ cmake --build .
```
