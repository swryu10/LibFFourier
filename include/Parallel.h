#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#ifdef _OPENMP
#include<omp.h>
#endif
#ifdef _MPI
#include<mpi.h>
#endif

namespace ParallelMPI {

// number of processors in MPI
extern int size_;
/* processor id in MPI
 * which can be 0 ... size_ - 1 */
extern int rank_;

/* initialize MPI
 * which can be called at the beginning
 * of main function */
void func_ini(int argc, char *argv[]);
/* finalize MPI
 * which can be called at the end
 * of main function */
void func_fin();

} // end namespace ParallelMPI
 
#endif
