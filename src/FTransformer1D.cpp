#include<stdio.h>
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

    z_unit_[0] = cos(2. * M_PI /
                     static_cast<double>(num_mesh_));
    z_unit_[1] = sin(2. * M_PI /
                     static_cast<double>(num_mesh_));

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

        for (int ix = 0; ix < num_mesh_; ix++) {
            mesh_func_x_[ix] = mesh_in_func_x[ix];
        }
    }

    int num_mesh_pr =
        list_num_mesh_pr_[ParallelMPI::rank_];
    if (num_mesh_pr > 0) {
        mesh_func_k_pr_ = new CNumber[num_mesh_pr];
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

    z_unit_[0] = cos(2. * M_PI /
                     static_cast<double>(num_mesh_));
    z_unit_[1] = sin(2. * M_PI /
                     static_cast<double>(num_mesh_));

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

        for (int ix = 0; ix < num_mesh_; ix++) {
            double x_now =
                static_cast<double>(ix) /
                static_cast<double>(num_mesh_);
            mesh_func_x_[ix] = (*ptr_in_func_x)(x_now);
        }
    }

    int num_mesh_pr =
        list_num_mesh_pr_[ParallelMPI::rank_];
    if (num_mesh_pr > 0) {
        mesh_func_k_pr_ = new CNumber[num_mesh_pr];
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
            mesh_func_k_pr_[ikpr] = next(ik, num_mesh_,
                                         mesh_fn_x);
        }
    #ifdef _OPENMP
    }  // parallel code ends
    #endif

    #ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif

    delete [] mesh_fn_x;

    return;
}

void Transformer1D::export_func_r(std::string name_file,
                                  int num_in_pt_x,
                                  CNumber (*ptr_in_func_x)(double)) {
    if (!initialized_) {
        return;
    }

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    int flag_file = 0;
    FILE *ptr_fout;
    if (ParallelMPI::rank_ == 0) {
        ptr_fout = fopen(name_file.c_str(), "w");
        if (ptr_fout != NULL) {
            flag_file = 1;
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            tag = ipr * 100 + 99;
            MPI_Send(&flag_file, 1, MPI_INT,
                     ipr, tag, MPI_COMM_WORLD);
        }
        #endif
    } else {
        #ifdef _MPI
        tag = ParallelMPI::rank_ * 100 + 99;
        MPI_Recv(&flag_file, 1, MPI_INT,
                 0, tag, MPI_COMM_WORLD, &status);
        #endif
    }

    if (flag_file == 0) {
        return;
    }

    if (ParallelMPI::rank_ == 0) {
        fprintf(ptr_fout, "# num_point = %d\n", num_in_pt_x);
        fprintf(ptr_fout, "# x    f.re    f.im    df_dx.re    df_dx.im");
        fprintf(ptr_fout, "    f_ini.re    f_ini.im\n");
    }

    for (int ix = 0; ix < num_in_pt_x; ix++) {
        double x_now = static_cast<double>(ix) /
                       static_cast<double>(num_in_pt_x);
        if (ParallelMPI::rank_ == 0) {
            fprintf(ptr_fout, "    %e", x_now);
        }
        CNumber cnum_df_dx_dft;
        CNumber cnum_func_dft =
            get_func_r(x_now, &cnum_df_dx_dft);
        if (ParallelMPI::rank_ == 0) {
            fprintf(ptr_fout, "    %e    %e    %e    %e",
                    cnum_func_dft[0], cnum_func_dft[1],
                    cnum_df_dx_dft[0], cnum_df_dx_dft[1]);
            if (ptr_in_func_x != NULL) {
                CNumber cnum_func_ini = (*ptr_in_func_x)(x_now);
                fprintf(ptr_fout, "    %e    %e",
                    cnum_func_ini[0], cnum_func_ini[1]);
            }
            fprintf(ptr_fout, "\n");
        }
    }

    if (ParallelMPI::rank_ == 0) {
        fclose(ptr_fout);
    }

    return;
}

void Transformer1D::reset() {
    if (!initialized_) {
        return;
    }

    if (ParallelMPI::rank_ == 0) {
        delete [] mesh_func_x_;
    }

    if (list_num_mesh_pr_[ParallelMPI::rank_] > 0) {
        delete [] mesh_func_k_pr_;;
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

    int mul_n_mesh = num_mesh_ / num_in_mesh;

    if (num1 == 1) {
        if (num_in_mesh == 1) {
            cnum_ret = mesh_in_func_x[0];
        } else {
            for (int ix = 0; ix < num_in_mesh; ix++) {
                cnum_ret = cnum_ret +
                    (mesh_in_func_x[ix] /
                     (z_unit_ ^ (ik * ix * mul_n_mesh)));
            }
        }
    } else {
        CNumber *mesh1_func_x = new CNumber[num1];
        for (int ix = 0; ix < num1; ix++) {
            mesh1_func_x[ix][0] = 0.;
            mesh1_func_x[ix][1] = 0.;

            for (int jx = 0; jx < num2; jx++) {
                mesh1_func_x[ix] = mesh1_func_x[ix] +
                    (mesh_in_func_x[num2 * ix + jx] /
                     (z_unit_ ^ (ik * jx * mul_n_mesh)));
            }
        }

        cnum_ret = next(ik, num1,
                        mesh1_func_x);

        delete [] mesh1_func_x;
    }

    return cnum_ret;
}

CNumber Transformer1D::get_func_r(double x_in,
                                  CNumber *ptr_df_dx) {
    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    CNumber z_in_unit;
    z_in_unit[0] = cos(2. * M_PI * x_in);
    z_in_unit[1] = sin(2. * M_PI * x_in);

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    int num_mesh_pr =
        list_num_mesh_pr_[ParallelMPI::rank_];

    #ifdef _OPENMP
    CNumber *list_c_func =
        new CNumber[num_mesh_pr];

    #pragma omp parallel
    {  // parallel code begins
    #endif
        #ifdef _OPENMP
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        /*
        fprintf(stdout, "Transformer1D:get_func_r\n");
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
            int jk = ik;
            if (2 * ik >= num_mesh_) {
                jk = ik - num_mesh_;
            }

            #ifdef _OPENMP
            list_c_func[ikpr] =
                (mesh_func_k_pr_[ikpr] * (z_in_unit ^ jk));
            #else
            cnum_ret = cnum_ret +
                (mesh_func_k_pr_[ikpr] * (z_in_unit ^ jk));
            #endif
        }
    #ifdef _OPENMP
    }  // parallel code ends

    for (int ikpr = 0; ikpr < num_mesh_pr; ikpr++) {
        cnum_ret = cnum_ret + list_c_func[ikpr];
    }

    delete [] list_c_func;
    #endif

    #ifdef _MPI
    MPI_Barrier(MPI_COMM_WORLD);

    if (ParallelMPI::rank_ == 0) {
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            double *set_func = new double[2];

            tag = ipr * 100 + 12;
            MPI_Recv(set_func, 2, MPI_DOUBLE,
                     ipr, tag, MPI_COMM_WORLD, &status);

            cnum_ret[0] += set_func[0];
            cnum_ret[1] += set_func[1];

            delete [] set_func;
        }
    } else {
        double *set_func = new double[2];
        set_func[0] = cnum_ret[0];
        set_func[1] = cnum_ret[1];

        tag = ParallelMPI::rank_ * 100 + 12;
        MPI_Send(set_func, 2, MPI_DOUBLE,
                 0, tag, MPI_COMM_WORLD);

        cnum_ret[0] = 0.;
        cnum_ret[1] = 0.;

        delete [] set_func;
    }
    #endif

    int flag_df_dx = 0;
    if (ParallelMPI::rank_ == 0) {
        if (ptr_df_dx != NULL) {
            flag_df_dx = 1;
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            tag = ipr * 100 + 10;
            MPI_Send(&flag_df_dx, 1, MPI_INT,
                     ipr, tag, MPI_COMM_WORLD);
        }
        #endif
    } else {
        #ifdef _MPI
        tag = ParallelMPI::rank_ * 100 + 10;
        MPI_Recv(&flag_df_dx, 1, MPI_INT,
                 0, tag, MPI_COMM_WORLD, &status);
        #endif
    }

    if (flag_df_dx != 0) {
        CNumber cnum_df_dx;
        cnum_df_dx[0] = 0.;
        cnum_df_dx[1] = 0.;

        #ifdef _OPENMP
        CNumber *list_c_df_dx =
            new CNumber[num_mesh_pr];

        #pragma omp parallel
        {  // parallel code begins
        #endif
            #ifdef _OPENMP
            int n_thread = omp_get_num_threads();
            int tid = omp_get_thread_num();
            /*
            fprintf(stdout, "Transformer1D:get_func_r\n");
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
                int jk = ik;
                if (2 * ik >= num_mesh_) {
                    jk = ik - num_mesh_;
                }

                if (jk == 0) {
                    continue;
                }

                CNumber fac_deriv;
                fac_deriv[0] = 0.;
                fac_deriv[1] =
                    2. * M_PI * static_cast<double>(jk);

                #ifdef _OPENMP
                list_c_df_dx[ikpr] = fac_deriv *
                    (mesh_func_k_pr_[ikpr] * (z_in_unit ^ jk));
                #else
                cnum_df_dx = cnum_df_dx + fac_deriv *
                    (mesh_func_k_pr_[ikpr] * (z_in_unit ^ jk));
                #endif
            }
        #ifdef _OPENMP
        }  // parallel code ends

        for (int ikpr = 0; ikpr < num_mesh_pr; ikpr++) {
            cnum_df_dx = cnum_df_dx + list_c_df_dx[ikpr];
        }

        delete [] list_c_df_dx;
        #endif

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);

        if (ParallelMPI::rank_ == 0) {
            for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
                double *set_df_dx = new double[2];

                tag = ipr * 100 + 13;
                MPI_Recv(set_df_dx, 2, MPI_DOUBLE,
                         ipr, tag, MPI_COMM_WORLD, &status);

                cnum_df_dx[0] += set_df_dx[0];
                cnum_df_dx[1] += set_df_dx[1];

                delete [] set_df_dx;
            }
        } else {
            double *set_df_dx = new double[2];
            set_df_dx[0] = cnum_df_dx[0];
            set_df_dx[1] = cnum_df_dx[1];

            tag = ParallelMPI::rank_ * 100 + 13;
            MPI_Send(set_df_dx, 2, MPI_DOUBLE,
                     0, tag, MPI_COMM_WORLD);

            cnum_df_dx[0] = 0.;
            cnum_df_dx[1] = 0.;

            delete [] set_df_dx;
        }
        #endif

        if (ParallelMPI::rank_ == 0) {
            *ptr_df_dx = factor_inv_ * cnum_df_dx;
        }
    }

    return factor_inv_ * cnum_ret;
}

CNumber Transformer1D::get_func_r(int ix,
                                  CNumber *ptr_df_dx) {
    int jx = (ix + num_mesh_) % num_mesh_;

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    int flag_df_dx = 0;
    if (ParallelMPI::rank_ == 0) {
        if (ptr_df_dx != NULL) {
            flag_df_dx = 1;
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            tag = ipr * 100 + 20;
            MPI_Send(&flag_df_dx, 1, MPI_INT,
                     ipr, tag, MPI_COMM_WORLD);
        }
        #endif
    } else {
        #ifdef _MPI
        tag = ParallelMPI::rank_ * 100 + 20;
        MPI_Recv(&flag_df_dx, 1, MPI_INT,
                 0, tag, MPI_COMM_WORLD, &status);
        #endif
    }

    if (flag_df_dx != 0) {
        CNumber cnum_df_dx;
        cnum_df_dx[0] = 0.;
        cnum_df_dx[1] = 0.;

        int num_mesh_pr =
            list_num_mesh_pr_[ParallelMPI::rank_];

        #ifdef _OPENMP
        CNumber *list_c_df_dx =
            new CNumber[num_mesh_pr];

        #pragma omp parallel
        {  // parallel code begins
        #endif
            #ifdef _OPENMP
            int n_thread = omp_get_num_threads();
            int tid = omp_get_thread_num();
            /*
            fprintf(stdout, "Transformer1D:get_func_r\n");
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
                int jk = ik;
                if (2 * ik >= num_mesh_) {
                    jk = ik - num_mesh_;
                }

                if (jk == 0) {
                    continue;
                }

                CNumber fac_deriv;
                fac_deriv[0] = 0.;
                fac_deriv[1] =
                    2. * M_PI * static_cast<double>(jk);

                #ifdef _OPENMP
                list_c_df_dx[ikpr] = fac_deriv *
                    (mesh_func_k_pr_[ikpr] * (z_unit_ ^ (jk * jx)));
                #else
                cnum_df_dx = cnum_df_dx + fac_deriv *
                    (mesh_func_k_pr_[ikpr] * (z_unit_ ^ (jk * jx)));
                #endif
            }
        #ifdef _OPENMP
        }  // parallel code ends

        for (int ikpr = 0; ikpr < num_mesh_pr; ikpr++) {
            cnum_df_dx = cnum_df_dx + list_c_df_dx[ikpr];
        }

        delete [] list_c_df_dx;
        #endif

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);

        if (ParallelMPI::rank_ == 0) {
            for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
                double *set_df_dx = new double[2];

                tag = ipr * 100 + 23;
                MPI_Recv(set_df_dx, 2, MPI_DOUBLE,
                         ipr, tag, MPI_COMM_WORLD, &status);

                cnum_df_dx[0] += set_df_dx[0];
                cnum_df_dx[1] += set_df_dx[1];

                delete [] set_df_dx;
            }
        } else {
            double *set_df_dx = new double[2];
            set_df_dx[0] = cnum_df_dx[0];
            set_df_dx[1] = cnum_df_dx[1];

            tag = ParallelMPI::rank_ * 100 + 23;
            MPI_Send(set_df_dx, 2, MPI_DOUBLE,
                     0, tag, MPI_COMM_WORLD);

            cnum_df_dx[0] = 0.;
            cnum_df_dx[1] = 0.;

            delete [] set_df_dx;
        }
        #endif

        if (ParallelMPI::rank_ == 0) {
            *ptr_df_dx = factor_inv_ * cnum_df_dx;
        }
    }

    if (ParallelMPI::rank_ == 0) {
        return mesh_func_x_[jx];
    } else {
        CNumber cnum_ret;
        cnum_ret[0] = 0.;
        cnum_ret[1] = 0.;

        return cnum_ret;
    }
}

CNumber Transformer1D::get_func_k(int ik) {
    int jk = (ik + num_mesh_) % num_mesh_;

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    int ipr_src = jk % ParallelMPI::size_;

    if (ParallelMPI::rank_ == 0) {
        if (ParallelMPI::rank_ == ipr_src) {
            int jkpr =
                (jk - ParallelMPI::rank_) /
                ParallelMPI::size_;

            cnum_ret = mesh_func_k_pr_[jkpr];
        } else {
            #ifdef _MPI
            double *set_func = new double[2];

            tag = ipr_src * 100 + 31;
            MPI_Recv(set_func, 2, MPI_DOUBLE,
                     ipr_src, tag, MPI_COMM_WORLD, &status);

            cnum_ret[0] = set_func[0];
            cnum_ret[1] = set_func[1];

            delete [] set_func;
            #endif
        }
    } else {
        if (ParallelMPI::rank_ == ipr_src) {
            #ifdef _MPI
            int jkpr =
                (jk - ParallelMPI::rank_) /
                ParallelMPI::size_;

            double *set_func = new double[2];
            set_func[0] = mesh_func_k_pr_[jkpr][0];
            set_func[1] = mesh_func_k_pr_[jkpr][1];

            tag = ipr_src * 100 + 31;
            MPI_Send(set_func, 2, MPI_DOUBLE,
                     0, tag, MPI_COMM_WORLD);

            delete [] set_func;
            #endif
        }
    }

    return cnum_ret;
}

} // end namespace FFourier
