#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#ifdef _MPI
#include<mpi.h>
#endif

namespace ParallelMPI {

extern int size_;
extern int rank_;

void func_ini(int argc, char *argv[]);
void func_fin();

} // end namespace ParallelMPI
 
#endif
