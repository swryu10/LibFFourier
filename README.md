# About
A C++ library to perform discrete Fourier transformations up to 3D.
It performs Fourier series expansion for a (periodic) complex function defined in 0 < _x_, _y_, _z_ < 1.
The input function can be discrete data points (with even spacing in _x_, _y_ and _z_) or function pointer.

# Parallel computing
One can use `OpenMP` and/or `MPI` to accelerate calculations.
If `OpenMP` and/or `MPI` are installed, they will be automatially linked in the process of building.
If neither is found, it will be a single-threaded application.
* **OpenMP** \
An environmental variable `OMP_NUM_THREADS` can be used to specify the number of threads in `OpenMP`.
`export` and `setenv` commands can be used to set environmental variables in **bash/zsh** and **tcsh/csh** shells, respectively.
For example, one can have 4 `OpenMP` threads with one of the following commands.
```
$ export OMP_NUM_THREADS=4
$ setenv OMP_NUM_THREADS 4
```
* **MPI** \
In the case of `MPI`, `mpiexec` (or `mpirun`) command need to be used to run applications on multiple processors.
Number of processors to be used can be specified with `-n` (or `-np`) option.
If one wants to use 4 processors, for example, the following command can be run.
```
$ mpiexec -n 4 [name of executable]
```

# Build
This library can be built with **cmake**. \
In a **Linux/UNIX** system, one can build at a subdirectory with the following commands.
```
$ mkdir [subdirectory name]
$ cd [subdirectory name]
$ cmake [directory for the LibFFourier local repository]
$ cmake --build .
```
