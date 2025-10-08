#include"Parallel.h"

int ParallelMPI::size_ = 1;
int ParallelMPI::rank_ = 0;

bool ParallelMPI::flag_init_ = false;

void ParallelMPI::func_ini(int argc, char *argv[]) {
    if (flag_init_) {
        return;
    }

    #ifdef _MPI
    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    #endif

    flag_init_ = true;

    return;
}

void ParallelMPI::func_fin() {
    if (!flag_init_) {
        return;
    }

    #ifdef _MPI
    MPI_Finalize();
    #endif

    flag_init_ = false;

    return;
}
