#include<stdio.h>
#include<math.h>
#include"FTransformer1D.h"

namespace FFourier {

void Transformer1D::init(int num_in_mesh,
                         CNumber *mesh_in_func_x) {
    reset();

    if (num_in_mesh < 2) {
        return;
    }

    num_mesh_ = num_in_mesh;
    factor_inv_ =
        1. / static_cast<double>(num_mesh_);

    list_num_mesh_pr_ = new int[ParallelMPI::size_];
    for (int ipr = 0; ipr < ParallelMPI::size_; ipr++) {
        list_num_mesh_pr_[ipr] = 0;
    }

    for (int ix = 0; ix < num_mesh_; ix++) {
        int ipr = ix % ParallelMPI::size_;
        list_num_mesh_pr_[ipr] += 1;
    }

    if (ParallelMPI::rank_ == 0) {
        mesh_func_x_ = new CNumber[num_mesh_];
        mesh_func_k_ = new CNumber[num_mesh_];

        for (int ix = 0; ix < num_mesh_; ix++) {
            mesh_func_x_[ix] = mesh_in_func_x[ix];
        }
    }

    initialized_ = true;
    make();

    return;
}

void Transformer1D::init(int num_in_mesh,
                         CNumber (*ptr_in_func_x)(double)) {
    reset();

    if (num_in_mesh < 2) {
        return;
    }

    num_mesh_ = num_in_mesh;
    factor_inv_ =
        1. / static_cast<double>(num_mesh_);

    list_num_mesh_pr_ = new int[ParallelMPI::size_];
    for (int ipr = 0; ipr < ParallelMPI::size_; ipr++) {
        list_num_mesh_pr_[ipr] = 0;
    }

    for (int ix = 0; ix < num_mesh_; ix++) {
        int ipr = ix % ParallelMPI::size_;
        list_num_mesh_pr_[ipr] += 1;
    }

    if (ParallelMPI::rank_ == 0) {
        mesh_func_x_ = new CNumber[num_mesh_];
        mesh_func_k_ = new CNumber[num_mesh_];

        for (int ix = 0; ix < num_mesh_; ix++) {
            double x_now =
                static_cast<double>(ix) /
                static_cast<double>(num_mesh_);
            mesh_func_x_[ix] = (*ptr_in_func_x)(x_now);
        }
    }

    initialized_ = true;
    make();

    return;
}

void Transformer1D::make() {
    if (!initialized_) {
        return;
    }

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    int num_mesh_pr = list_num_mesh_pr_[ParallelMPI::rank_];

    CNumber *mesh_fn_x = new CNumber[num_mesh_];
    CNumber *mesh_fn_k;
    if (num_mesh_pr > 0) {
        mesh_fn_k = new CNumber[num_mesh_pr];
    }

    #ifdef _MPI
    double *set_fn_x = new double[2 * num_mesh_];
    #endif

    if (ParallelMPI::rank_ == 0) {
        for (int ix = 0; ix < num_mesh_; ix++) {
            mesh_fn_x[ix] = mesh_func_x_[ix];

            #ifdef _MPI
            set_fn_x[2 * ix] = mesh_func_x_[ix][0];
            set_fn_x[2 * ix + 1] = mesh_func_x_[ix][1];
            #endif
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            tag = ipr * 100 + 1;
            MPI_Send(set_fn_x, 2 * num_mesh_, MPI_DOUBLE,
                     ipr, tag, MPI_COMM_WORLD);
        }
        #endif
    } else {
        #ifdef _MPI
        tag = ParallelMPI::rank_ * 100 + 1;
        MPI_Recv(set_fn_x, 2 * num_mesh_, MPI_DOUBLE,
                 0, tag, MPI_COMM_WORLD, &status);

        for (int ix = 0; ix < num_mesh_; ix++) {
            mesh_fn_x[ix][0] = set_fn_x[2 * ix];
            mesh_fn_x[ix][1] = set_fn_x[2 * ix + 1];
        }
        #endif
    }

    #ifdef _MPI
    delete [] set_fn_x;
    #endif

    #ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    #ifdef _OPENMP
    #pragma omp parallel
    {  // parallel code begins
    #endif
        #ifdef _OPENMP
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        /*
        fprintf(stdout, "Transformer1D:make\n");
        fprintf(stdout, "  OPENMP : n_thread = %d, tid = %d\n",
                n_thread, tid);
        */
        #endif

        for (int ikpr = 0; ikpr < num_mesh_pr; ikpr++) {
            #ifdef _OPENMP
            if (ikpr % n_thread != tid) {
                continue;
            }
            #endif

            int ik =
                ParallelMPI::rank_ +
                ParallelMPI::size_ * ikpr;
            mesh_fn_k[ikpr] = next(ik, num_mesh_,
                                   mesh_fn_x);
        }
    #ifdef _OPENMP
    }  // parallel code ends
    #endif

    #ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    if (ParallelMPI::rank_ == 0) {
        for (int ikpr = 0; ikpr < num_mesh_pr; ikpr++) {
            int ik =
                ParallelMPI::rank_ +
                ParallelMPI::size_ * ikpr;

            mesh_func_k_[ik] = mesh_fn_k[ikpr];
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            if (list_num_mesh_pr_[ipr] <= 0) {
                continue;
            }

            double *set_fn_k =
                new double[2 * list_num_mesh_pr_[ipr]];

            tag = ipr * 100 + 2;
            MPI_Recv(set_fn_k, 2 * list_num_mesh_pr_[ipr], MPI_DOUBLE,
                     ipr, tag, MPI_COMM_WORLD, &status);

            for (int jkpr = 0; jkpr < list_num_mesh_pr_[ipr]; jkpr++) {
                int jk = ParallelMPI::size_ * jkpr + ipr;

                mesh_func_k_[jk][0] = set_fn_k[2 * jkpr];
                mesh_func_k_[jk][1] = set_fn_k[2 * jkpr + 1];
            }

            delete [] set_fn_k;
        }
        #endif
    } else {
        #ifdef _MPI
        if (num_mesh_pr > 0) {
            double *set_fn_k =
                new double[2 * num_mesh_pr];

            for (int ikpr = 0; ikpr < num_mesh_pr; ikpr++) {
                set_fn_k[2 * ikpr] = mesh_fn_k[ikpr][0];
                set_fn_k[2 * ikpr + 1] = mesh_fn_k[ikpr][1];
            }

            tag = ParallelMPI::rank_ * 100 + 2;
            MPI_Send(set_fn_k, 2 * num_mesh_pr, MPI_DOUBLE,
                     0, tag, MPI_COMM_WORLD);

            delete [] set_fn_k;
        }
        #endif
    }

    delete [] mesh_fn_x;
    if (num_mesh_pr > 0) {
        delete [] mesh_fn_k;
    }

    return;
}

void Transformer1D::export_func_r(std::string name_file,
                                  int num_in_pt_x,
                                  CNumber (*ptr_in_func_x)(double)) {
    if (!initialized_) {
        return;
    }

    if (ParallelMPI::rank_ != 0) {
        return;
    }

    FILE *ptr_fout;
    ptr_fout = fopen(name_file.c_str(), "w");

    if (ptr_fout == NULL) {
        return;
    }

    fprintf(ptr_fout, "# num_point = %d\n", num_in_pt_x);

    for (int ix = 0; ix < num_in_pt_x; ix++) {
        double x_now = static_cast<double>(ix) /
                       static_cast<double>(num_in_pt_x);
        fprintf(ptr_fout, "    %e", x_now);
        CNumber cnum_func_dft = get_func_r(x_now);
        fprintf(ptr_fout, "    %e    %e",
                cnum_func_dft[0], cnum_func_dft[1]);
        if (ptr_in_func_x != NULL) {
            CNumber cnum_func_ini = (*ptr_in_func_x)(x_now);
            fprintf(ptr_fout, "    %e    %e",
                cnum_func_ini[0], cnum_func_ini[1]);
        }
        fprintf(ptr_fout, "\n");
    }

    fclose(ptr_fout);

    return;
}

void Transformer1D::reset() {
    if (!initialized_) {
        return;
    }

    if (ParallelMPI::rank_ == 0) {
        delete [] mesh_func_x_;
        delete [] mesh_func_k_;
    }

    delete [] list_num_mesh_pr_;

    num_mesh_ = 0;

    initialized_ = false;

    return;
}

CNumber Transformer1D::next(int ik, int num_in_mesh,
                            CNumber *mesh_in_func_x) {
    int num1 = 1;
    int num2 = num_in_mesh;

    for (int i = 2; i < num_in_mesh; i++) {
        if (num_in_mesh % i == 0) {
            num1 = i;
            num2 = num_in_mesh / num1;

            break;
        }
    }

    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    if (num1 == 1) {
        if (num_in_mesh == 1) {
            cnum_ret = mesh_in_func_x[0];
        } else {
            CNumber z_in_unit;
            z_in_unit[0] = cos(2. * M_PI *
                               static_cast<double>(ik) /
                               static_cast<double>(num_in_mesh));
            z_in_unit[1] = sin(2. * M_PI *
                               static_cast<double>(ik) /
                               static_cast<double>(num_in_mesh));
            for (int ix = 0; ix < num_in_mesh; ix++) {
                cnum_ret = cnum_ret +
                    (mesh_in_func_x[ix] / (z_in_unit ^ ix));
            }
        }
    } else {
        CNumber z_in_unit;
        z_in_unit[0] = cos(2. * M_PI *
                           static_cast<double>(ik) /
                           static_cast<double>(num_in_mesh));
        z_in_unit[1] = sin(2. * M_PI *
                           static_cast<double>(ik) /
                           static_cast<double>(num_in_mesh));

        CNumber *mesh1_func_x = new CNumber[num1];
        for (int ix = 0; ix < num1; ix++) {
            mesh1_func_x[ix][0] = 0.;
            mesh1_func_x[ix][1] = 0.;

            for (int jx = 0; jx < num2; jx++) {
                mesh1_func_x[ix] = mesh1_func_x[ix] +
                    (mesh_in_func_x[num2 * ix + jx] / (z_in_unit ^ jx));
            }
        }

        cnum_ret = next(ik, num1,
                        mesh1_func_x);

        delete [] mesh1_func_x;
    }

    return cnum_ret;
}

CNumber Transformer1D::get_func_r(double x_in) {
    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    if (ParallelMPI::rank_ != 0) {
        return cnum_ret;
    }

    CNumber z_in_unit;
    z_in_unit[0] = cos(2. * M_PI * x_in);
    z_in_unit[1] = sin(2. * M_PI * x_in);

    for (int ik = 0; ik < num_mesh_; ik++) {
        int jk = ik;
        if (2 * ik >= num_mesh_) {
            jk = ik - num_mesh_;
        }
        cnum_ret = cnum_ret +
            (mesh_func_k_[ik] * (z_in_unit ^ jk));
    }

    return factor_inv_ * cnum_ret;
}

} // end namespace FFourier
