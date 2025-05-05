#include"Parallel.h"

namespace ParallelMPI {

int size_ = 1;
int rank_ = 0;

void func_ini(int argc, char *argv[]) {
    #ifdef _MPI
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    #endif

    return;
}

void func_fin() {
    #ifdef _MPI
    MPI_Finalize();
    #endif

    return;
}

} // end namespace ParallelMPI
