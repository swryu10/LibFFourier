#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#ifdef _OPENMP
#include<omp.h>
#endif
#ifdef _MPI
#include<mpi.h>
#endif

class ParallelMPI {
  private:

    // number of processors in MPI
    static int size_;
    /* processor id in MPI
     * which can be 0 ... size_ - 1 */
    static int rank_;

    static bool flag_init_;

  public:

    /* initialize MPI
     * which can be called at the beginning
     * of main function */
    static void func_ini(int argc, char *argv[]);
    /* finalize MPI
     * which can be called at the end
     * of main function */
    static void func_fin();

    static int size() {return size_;}
    static int rank() {return rank_;}
};
 
#endif
