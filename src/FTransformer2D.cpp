#include<stdio.h>
#include<math.h>
#include"FTransformer1D.h"
#include"FTransformer2D.h"

namespace FFourier {

void Transformer2D::init(int num_in_mesh_x,
                         int num_in_mesh_y,
                         CNumber **mesh_in_func_r) {
    reset();

    if (num_in_mesh_x < 2 ||
        num_in_mesh_y < 2) {
        return;
    }

    num_mesh_x_ = num_in_mesh_x;
    num_mesh_y_ = num_in_mesh_y;
    factor_inv_ =
        1. / static_cast<double>(num_mesh_x_ * num_mesh_y_);

    z_unit_x_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_x_));
    z_unit_x_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_x_));

    z_unit_y_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_y_));
    z_unit_y_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_y_));

    num_mmid_x_ = (num_mesh_x_ + (num_mesh_x_ % 2)) / 2;
    num_mmid_y_ = (num_mesh_y_ + (num_mesh_y_ % 2)) / 2;

    list_num_mesh_x_pr_ = new int[ParallelMPI::size_];
    for (int ipr = 0; ipr < ParallelMPI::size_; ipr++) {
        list_num_mesh_x_pr_[ipr] = 0;
    }

    for (int irx = 0; irx < num_mesh_x_; irx++) {
        int ipr = irx % ParallelMPI::size_;
        list_num_mesh_x_pr_[ipr] += 1;
    }

    if (ParallelMPI::rank_ == 0) {
        mesh_func_r_ = new CNumber *[num_mesh_x_];
        for (int irx = 0; irx < num_mesh_x_; irx++) {
            mesh_func_r_[irx] = new CNumber[num_mesh_y_];

            for (int iry = 0; iry < num_mesh_y_; iry++) {
                mesh_func_r_[irx][iry] =
                    mesh_in_func_r[irx][iry];
            }
        }
    }

    int num_mesh_x_pr =
        list_num_mesh_x_pr_[ParallelMPI::rank_];
    if (num_mesh_x_pr > 0) {
        mesh_func_k_pr_ = new CNumber *[num_mesh_x_pr];
        for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
            mesh_func_k_pr_[ikxpr] = new CNumber[num_mesh_y_];
        }
    }

    initialized_ = true;
    make();

    return;
}

void Transformer2D::init(int num_in_mesh_x,
                         int num_in_mesh_y,
                         CNumber (*ptr_in_func_r)(double,
                                                  double)) {
    reset();

    if (num_in_mesh_x < 2 ||
        num_in_mesh_y < 2) {
        return;
    }

    num_mesh_x_ = num_in_mesh_x;
    num_mesh_y_ = num_in_mesh_y;
    factor_inv_ =
        1. / static_cast<double>(num_mesh_x_ * num_mesh_y_);

    z_unit_x_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_x_));
    z_unit_x_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_x_));

    z_unit_y_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_y_));
    z_unit_y_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_y_));

    num_mmid_x_ = (num_mesh_x_ + (num_mesh_x_ % 2)) / 2;
    num_mmid_y_ = (num_mesh_y_ + (num_mesh_y_ % 2)) / 2;

    double nd_mesh_x = static_cast<double>(num_mesh_x_);
    double nd_mesh_y = static_cast<double>(num_mesh_y_);

    list_num_mesh_x_pr_ = new int[ParallelMPI::size_];
    for (int ipr = 0; ipr < ParallelMPI::size_; ipr++) {
        list_num_mesh_x_pr_[ipr] = 0;
    }

    for (int irx = 0; irx < num_mesh_x_; irx++) {
        int ipr = irx % ParallelMPI::size_;
        list_num_mesh_x_pr_[ipr] += 1;
    }

    if (ParallelMPI::rank_ == 0) {
        mesh_func_r_ = new CNumber *[num_mesh_x_];
        for (int irx = 0; irx < num_mesh_x_; irx++) {
            mesh_func_r_[irx] = new CNumber[num_mesh_y_];

            double x_now =
                static_cast<double>(irx) / nd_mesh_x;

            for (int iry = 0; iry < num_mesh_y_; iry++) {
                double y_now =
                    static_cast<double>(iry) / nd_mesh_y;

                mesh_func_r_[irx][iry] =
                    (*ptr_in_func_r)(x_now, y_now);
            }
        }
    }

    int num_mesh_x_pr =
        list_num_mesh_x_pr_[ParallelMPI::rank_];
    if (num_mesh_x_pr > 0) {
        mesh_func_k_pr_ = new CNumber *[num_mesh_x_pr];
        for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
            mesh_func_k_pr_[ikxpr] = new CNumber[num_mesh_y_];
        }
    }

    initialized_ = true;
    make();

    return;
}

void Transformer2D::make() {
    if (!initialized_) {
        return;
    }

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    Transformer1D *ptr_dft_x =
        new Transformer1D [num_mesh_x_]();

    for (int irx = 0; irx < num_mesh_x_; irx++) {
        CNumber *ptr_mesh_fn_y = NULL;
        if (ParallelMPI::rank_ == 0) {
            ptr_mesh_fn_y = mesh_func_r_[irx];
        }

        ptr_dft_x[irx].init(num_mesh_y_,
                            ptr_mesh_fn_y);

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
    }

    Transformer1D dft_all;
    CNumber *mesh_func_x;
    if (ParallelMPI::rank_ == 0) {
        mesh_func_x = new CNumber [num_mesh_x_];
    }

    for (int iky = 0; iky < num_mesh_y_; iky++) {
        for (int irx = 0; irx < num_mesh_x_; irx++) {
            CNumber cnum_func_x =
                ptr_dft_x[irx].get_func_k(iky);

            if (ParallelMPI::rank_ == 0) {
                mesh_func_x[irx] = cnum_func_x;
            }
        }

        dft_all.init(num_mesh_x_,
                     mesh_func_x);

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);
        #endif

        CNumber *list_func_k;
        if (ParallelMPI::rank_ == 0) {
            list_func_k = new CNumber[num_mesh_x_];
        }

        for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
            CNumber cnum_func_k =
                dft_all.get_func_k(ikx);

            if (ParallelMPI::rank_ == 0) {
                list_func_k[ikx] = cnum_func_k;
            }
        }

        if (ParallelMPI::rank_ == 0) {
            for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
                int ipr_tar = ikx % ParallelMPI::size_;
                if (ipr_tar == 0) {
                    int ikxpr =
                        (ikx - ParallelMPI::rank_) /
                        ParallelMPI::size_;

                    mesh_func_k_pr_[ikxpr][iky] = list_func_k[ikx];
                } else {
                    double *set_func_k = new double[2];
                    set_func_k[0] = list_func_k[ikx][0];
                    set_func_k[1] = list_func_k[ikx][1];

                    tag = (num_mesh_x_ * iky + ikx) *
                          ParallelMPI::size_ +
                          ipr_tar;
                    MPI_Send(set_func_k, 2, MPI_DOUBLE,
                             ipr_tar, tag, MPI_COMM_WORLD);

                    delete [] set_func_k;
                }
            }

            delete [] list_func_k;
        } else {
            for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
                int ipr_tar = ikx % ParallelMPI::size_;
                if (ParallelMPI::rank_ != ipr_tar) {
                    continue;
                }

                double *set_func_k = new double[2];

                tag = (num_mesh_x_ * iky + ikx) *
                      ParallelMPI::size_ +
                      ipr_tar;
                MPI_Recv(set_func_k, 2, MPI_DOUBLE,
                         0, tag, MPI_COMM_WORLD, &status);

                int ikxpr =
                    (ikx - ParallelMPI::rank_) /
                    ParallelMPI::size_;
                mesh_func_k_pr_[ikxpr][iky][0] = set_func_k[0];
                mesh_func_k_pr_[ikxpr][iky][1] = set_func_k[1];

                delete [] set_func_k;
            }
        }
    }

    if (ParallelMPI::rank_ == 0) {
        delete [] mesh_func_x;
    }

    delete [] ptr_dft_x;

    return;
}

void Transformer2D::export_func_r(std::string name_file,
                                  int num_in_pt_x, int num_in_pt_y,
                                  CNumber (*ptr_in_func_r)(double,
                                                           double)) {
    if (!initialized_) {
        return;
    }

    if (num_in_pt_x < 2 ||
        num_in_pt_y < 2) {
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

    double nd_pt_x = static_cast<double>(num_in_pt_x);
    double nd_pt_y = static_cast<double>(num_in_pt_y);

    CNumber **tab_func_r_dft;
    CNumber **tab_func_r_ini;
    if (ParallelMPI::rank_ == 0) {
        tab_func_r_dft = new CNumber *[num_in_pt_x];
        tab_func_r_ini = new CNumber *[num_in_pt_x];

        for (int ix = 0; ix < num_in_pt_x; ix++) {
            tab_func_r_dft[ix] = new CNumber[num_in_pt_y];
            tab_func_r_ini[ix] = new CNumber[num_in_pt_y];
        }
    }

    for (int ix = 0; ix < num_in_pt_x; ix++) {
        double x_now = static_cast<double>(ix) / nd_pt_x;

        for (int iy = 0; iy < num_in_pt_y; iy++) {
            double y_now = static_cast<double>(iy) / nd_pt_y;

            CNumber cnum_df_dx_dft;
            CNumber cnum_df_dy_dft;
            CNumber cnum_func_dft = get_func_r(x_now, y_now,
                                               &cnum_df_dx_dft,
                                               &cnum_df_dy_dft);
            if (ParallelMPI::rank_ == 0) {
                tab_func_r_dft[ix][iy] = cnum_func_dft;
                if (ptr_in_func_r != NULL) {
                    tab_func_r_ini[ix][iy] =
                        (*ptr_in_func_r)(x_now, y_now);
                }
            }
        }
    }

    if (ParallelMPI::rank_ == 0) {
        fprintf(ptr_fout, "# num_pt_x = %d, num_pt_y = %d\n",
                num_in_pt_x, num_in_pt_y);

        for (int ix = 0; ix < num_in_pt_x; ix++) {
            double x_now = static_cast<double>(ix) / nd_pt_x;
            for (int iy = 0; iy < num_in_pt_y; iy++) {
                double y_now = static_cast<double>(iy) / nd_pt_y;

                fprintf(ptr_fout, "    %e    %e",
                        x_now, y_now);

                fprintf(ptr_fout, "    %e    %e",
                        tab_func_r_dft[ix][iy][0],
                        tab_func_r_dft[ix][iy][1]);

                if (ptr_in_func_r != NULL) {
                    fprintf(ptr_fout, "    %e    %e",
                            tab_func_r_ini[ix][iy][0],
                            tab_func_r_ini[ix][iy][1]);
                }

                fprintf(ptr_fout, "\n");
            }

            fprintf(ptr_fout, "\n");
        }

        fclose(ptr_fout);

        for (int ix = 0; ix < num_in_pt_x; ix++) {
            delete [] tab_func_r_dft[ix];
            delete [] tab_func_r_ini[ix];
        }
        delete [] tab_func_r_dft;
        delete [] tab_func_r_ini;
    }

    return;
}

void Transformer2D::reset() {
    if (!initialized_) {
        return;
    }

    if (ParallelMPI::rank_ == 0) {
        for (int irx = 0; irx < num_mesh_x_; irx++) {
            delete [] mesh_func_r_[irx];
        }

        delete [] mesh_func_r_;
    }

    int num_mesh_x_pr =
        list_num_mesh_x_pr_[ParallelMPI::rank_];
    if (num_mesh_x_pr > 0) {
        for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
            delete [] mesh_func_k_pr_[ikxpr];
        }

        delete [] mesh_func_k_pr_;
    }

    delete [] list_num_mesh_x_pr_;

    num_mesh_x_ = 0;
    num_mesh_y_ = 0;

    num_mmid_x_ = 0;
    num_mmid_y_ = 0;

    initialized_ = false;

    return;
}

CNumber Transformer2D::get_func_r(double x_in,
                                  double y_in,
                                  CNumber *ptr_df_dx,
                                  CNumber *ptr_df_dy) {
    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    CNumber zx_in_unit;
    zx_in_unit[0] = cos(2. * M_PI * x_in);
    zx_in_unit[1] = sin(2. * M_PI * x_in);
    CNumber *list_zx_unit = new CNumber[num_mesh_x_];
    list_zx_unit[0][0] = 1.;
    list_zx_unit[0][1] = 0.;
    for (int ikx = 1; ikx < num_mmid_x_; ikx++) {
        int ikx_prev = ikx - 1;
        list_zx_unit[ikx] =
            list_zx_unit[ikx_prev] * zx_in_unit;
    }
    for (int ikx = num_mesh_x_ - 1; ikx >= num_mmid_x_; ikx--) {
        int ikx_prev = (ikx + 1) % num_mesh_x_;
        list_zx_unit[ikx] =
            list_zx_unit[ikx_prev] / zx_in_unit;
    }

    CNumber zy_in_unit;
    zy_in_unit[0] = cos(2. * M_PI * y_in);
    zy_in_unit[1] = sin(2. * M_PI * y_in);
    CNumber *list_zy_unit = new CNumber[num_mesh_y_];
    list_zy_unit[0][0] = 1.;
    list_zy_unit[0][1] = 0.;
    for (int iky = 1; iky < num_mmid_y_; iky++) {
        int iky_prev = iky - 1;
        list_zy_unit[iky] =
            list_zy_unit[iky_prev] * zy_in_unit;
    }
    for (int iky = num_mesh_y_ - 1; iky >= num_mmid_y_; iky--) {
        int iky_prev = (iky + 1) % num_mesh_y_;
        list_zy_unit[iky] =
            list_zy_unit[iky_prev] / zy_in_unit;
    }

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    int num_mesh_x_pr =
        list_num_mesh_x_pr_[ParallelMPI::rank_];

    #ifdef _OPENMP
    CNumber *list_c_func =
        new CNumber[num_mesh_x_pr];

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

        for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
            #ifdef _OPENMP
            if (ikxpr % n_thread != tid) {
                continue;
            }
            #endif

            int ikx =
                ParallelMPI::rank_ +
                ParallelMPI::size_ * ikxpr;

            #ifdef _OPENMP
            list_c_func[ikxpr][0] = 0.;
            list_c_func[ikxpr][1] = 0.;
            #endif
            for (int iky = 0; iky < num_mesh_y_; iky++) {
                CNumber func_now =
                    mesh_func_k_pr_[ikxpr][iky] *
                    list_zx_unit[ikx] * list_zy_unit[iky];

                #ifdef _OPENMP
                list_c_func[ikxpr] =
                    list_c_func[ikxpr] + func_now;
                #else
                cnum_ret = cnum_ret + func_now;
                #endif
            }
        }
    #ifdef _OPENMP
    }  // parallel code ends

    for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
        cnum_ret = cnum_ret + list_c_func[ikxpr];
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

    int *flag_df_dr = new int[2];
    flag_df_dr[0] = 0;
    flag_df_dr[1] = 0;
    if (ParallelMPI::rank_ == 0) {
        if (ptr_df_dx != NULL) {
            flag_df_dr[0] = 1;
        }

        if (ptr_df_dy != NULL) {
            flag_df_dr[1] = 1;
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            tag = ipr * 100 + 10;
            MPI_Send(flag_df_dr, 2, MPI_INT,
                     ipr, tag, MPI_COMM_WORLD);
        }
        #endif
    } else {
        #ifdef _MPI
        tag = ParallelMPI::rank_ * 100 + 10;
        MPI_Recv(flag_df_dr, 2, MPI_INT,
                 0, tag, MPI_COMM_WORLD, &status);
        #endif
    }

    if (flag_df_dr[0] != 1 ||
        flag_df_dr[1] != 1) {
        CNumber cnum_df_dx;
        cnum_df_dx[0] = 0.;
        cnum_df_dx[1] = 0.;

        CNumber cnum_df_dy;
        cnum_df_dy[0] = 0.;
        cnum_df_dy[1] = 0.;

        #ifdef _OPENMP
        CNumber *list_c_df_dx =
            new CNumber[num_mesh_x_pr];
        CNumber *list_c_df_dy =
            new CNumber[num_mesh_x_pr];

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

            for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
                #ifdef _OPENMP
                if (ikxpr % n_thread != tid) {
                    continue;
                }
                #endif

                int ikx =
                    ParallelMPI::rank_ +
                    ParallelMPI::size_ * ikxpr;

                int jkx = ikx;
                if (ikx >= num_mmid_x_) {
                    jkx = ikx - num_mesh_x_;
                }

                CNumber fac_deriv_x;
                fac_deriv_x[0] = 0.;
                fac_deriv_x[1] =
                    2. * M_PI * static_cast<double>(jkx);

                #ifdef _OPENMP
                list_c_df_dx[ikxpr][0] = 0.;
                list_c_df_dx[ikxpr][1] = 0.;

                list_c_df_dy[ikxpr][0] = 0.;
                list_c_df_dy[ikxpr][1] = 0.;
                #endif
                for (int iky = 0; iky < num_mesh_y_; iky++) {
                    int jky = iky;
                    if (iky >= num_mmid_y_) {
                        jky = iky - num_mesh_y_;
                    }

                    CNumber fac_deriv_y;
                    fac_deriv_y[0] = 0.;
                    fac_deriv_y[1] =
                        2. * M_PI * static_cast<double>(jky);

                    CNumber func_now =
                        mesh_func_k_pr_[ikxpr][iky] *
                        list_zx_unit[ikx] * list_zy_unit[iky];

                    #ifdef _OPENMP
                    list_c_df_dx[ikxpr] =
                        list_c_df_dx[ikxpr] +
                        (fac_deriv_x * func_now);

                    list_c_df_dy[ikxpr] =
                        list_c_df_dy[ikxpr] +
                        (fac_deriv_y * func_now);
                    #else
                    cnum_df_dx = cnum_df_dx +
                        (fac_deriv_x * func_now);

                    cnum_df_dy = cnum_df_dy +
                        (fac_deriv_y * func_now);
                    #endif
                }
            }
        #ifdef _OPENMP
        }  // parallel code ends

        for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
            cnum_df_dx = cnum_df_dx + list_c_df_dx[ikxpr];
            cnum_df_dy = cnum_df_dy + list_c_df_dy[ikxpr];
        }

        delete [] list_c_df_dx;
        delete [] list_c_df_dy;
        #endif

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);

        if (ParallelMPI::rank_ == 0) {
            for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
                double *set_df_dr = new double[4];

                tag = ipr * 100 + 13;
                MPI_Recv(set_df_dr, 4, MPI_DOUBLE,
                         ipr, tag, MPI_COMM_WORLD, &status);

                cnum_df_dx[0] += set_df_dr[0];
                cnum_df_dx[1] += set_df_dr[1];
                cnum_df_dy[0] += set_df_dr[2];
                cnum_df_dy[1] += set_df_dr[3];

                delete [] set_df_dr;
            }
        } else {
            double *set_df_dr = new double[4];
            set_df_dr[0] = cnum_df_dx[0];
            set_df_dr[1] = cnum_df_dx[1];
            set_df_dr[2] = cnum_df_dy[0];
            set_df_dr[3] = cnum_df_dy[1];

            tag = ParallelMPI::rank_ * 100 + 13;
            MPI_Send(set_df_dr, 4, MPI_DOUBLE,
                     0, tag, MPI_COMM_WORLD);

            cnum_df_dx[0] = 0.;
            cnum_df_dx[1] = 0.;

            cnum_df_dy[0] = 0.;
            cnum_df_dy[1] = 0.;

            delete [] set_df_dr;
        }
        #endif

        if (ParallelMPI::rank_ == 0) {
            if (flag_df_dr[0] != 1) {
                *ptr_df_dx = factor_inv_ * cnum_df_dx;
            }

            if (flag_df_dr[1] != 1) {
                *ptr_df_dy = factor_inv_ * cnum_df_dy;
            }
        }
    }

    delete [] flag_df_dr;

    delete [] list_zx_unit;
    delete [] list_zy_unit;

    return factor_inv_ * cnum_ret;
}

CNumber Transformer2D::get_func_r(int irx, int iry,
                                  CNumber *ptr_df_dx,
                                  CNumber *ptr_df_dy) {
    int jrx = (irx + num_mesh_x_) % num_mesh_x_;
    int jry = (iry + num_mesh_y_) % num_mesh_y_;

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    int num_mesh_x_pr =
        list_num_mesh_x_pr_[ParallelMPI::rank_];

    int *flag_df_dr = new int[2];
    flag_df_dr[0] = 0;
    flag_df_dr[1] = 0;
    if (ParallelMPI::rank_ == 0) {
        if (ptr_df_dx != NULL) {
            flag_df_dr[0] = 1;
        }

        if (ptr_df_dy != NULL) {
            flag_df_dr[1] = 1;
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            tag = ipr * 100 + 20;
            MPI_Send(flag_df_dr, 2, MPI_INT,
                     ipr, tag, MPI_COMM_WORLD);
        }
        #endif
    } else {
        #ifdef _MPI
        tag = ParallelMPI::rank_ * 100 + 20;
        MPI_Recv(flag_df_dr, 2, MPI_INT,
                 0, tag, MPI_COMM_WORLD, &status);
        #endif
    }

    if (flag_df_dr[0] != 1 ||
        flag_df_dr[1] != 1) {
        CNumber cnum_df_dx;
        cnum_df_dx[0] = 0.;
        cnum_df_dx[1] = 0.;

        CNumber cnum_df_dy;
        cnum_df_dy[0] = 0.;
        cnum_df_dy[1] = 0.;

        #ifdef _OPENMP
        CNumber *list_c_df_dx =
            new CNumber[num_mesh_x_pr];
        CNumber *list_c_df_dy =
            new CNumber[num_mesh_x_pr];

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

            for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
                #ifdef _OPENMP
                if (ikxpr % n_thread != tid) {
                    continue;
                }
                #endif

                int ikx =
                    ParallelMPI::rank_ +
                    ParallelMPI::size_ * ikxpr;

                int jkx = ikx;
                if (ikx >= num_mmid_x_) {
                    jkx = ikx - num_mesh_x_;
                }

                CNumber fac_deriv_x;
                fac_deriv_x[0] = 0.;
                fac_deriv_x[1] =
                    2. * M_PI * static_cast<double>(jkx);

                #ifdef _OPENMP
                list_c_df_dx[ikxpr][0] = 0.;
                list_c_df_dx[ikxpr][1] = 0.;

                list_c_df_dy[ikxpr][0] = 0.;
                list_c_df_dy[ikxpr][1] = 0.;
                #endif
                for (int iky = 0; iky < num_mesh_y_; iky++) {
                    int jky = iky;
                    if (iky >= num_mmid_y_) {
                        jky = iky - num_mesh_y_;
                    }

                    CNumber fac_deriv_y;
                    fac_deriv_y[0] = 0.;
                    fac_deriv_y[1] =
                        2. * M_PI * static_cast<double>(jky);

                    CNumber func_now =
                        mesh_func_k_pr_[ikxpr][iky] *
                        (z_unit_x_ ^ (jkx * jrx)) *
                        (z_unit_y_ ^ (jky * jry));

                    #ifdef _OPENMP
                    list_c_df_dx[ikxpr] =
                        list_c_df_dx[ikxpr] +
                        (fac_deriv_x * func_now);

                    list_c_df_dy[ikxpr] =
                        list_c_df_dy[ikxpr] +
                        (fac_deriv_y * func_now);
                    #else
                    cnum_df_dx = cnum_df_dx +
                        (fac_deriv_x * func_now);

                    cnum_df_dy = cnum_df_dy +
                        (fac_deriv_y * func_now);
                    #endif
                }
            }
        #ifdef _OPENMP
        }  // parallel code ends

        for (int ikxpr = 0; ikxpr < num_mesh_x_pr; ikxpr++) {
            cnum_df_dx = cnum_df_dx + list_c_df_dx[ikxpr];
            cnum_df_dy = cnum_df_dy + list_c_df_dy[ikxpr];
        }

        delete [] list_c_df_dx;
        delete [] list_c_df_dy;
        #endif

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);

        if (ParallelMPI::rank_ == 0) {
            for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
                double *set_df_dr = new double[4];

                tag = ipr * 100 + 23;
                MPI_Recv(set_df_dr, 4, MPI_DOUBLE,
                         ipr, tag, MPI_COMM_WORLD, &status);

                cnum_df_dx[0] += set_df_dr[0];
                cnum_df_dx[1] += set_df_dr[1];
                cnum_df_dy[0] += set_df_dr[2];
                cnum_df_dy[1] += set_df_dr[3];

                delete [] set_df_dr;
            }
        } else {
            double *set_df_dr = new double[4];
            set_df_dr[0] = cnum_df_dx[0];
            set_df_dr[1] = cnum_df_dx[1];
            set_df_dr[2] = cnum_df_dy[0];
            set_df_dr[3] = cnum_df_dy[1];

            tag = ParallelMPI::rank_ * 100 + 23;
            MPI_Send(set_df_dr, 4, MPI_DOUBLE,
                     0, tag, MPI_COMM_WORLD);

            cnum_df_dx[0] = 0.;
            cnum_df_dx[1] = 0.;

            cnum_df_dy[0] = 0.;
            cnum_df_dy[1] = 0.;

            delete [] set_df_dr;
        }
        #endif

        if (ParallelMPI::rank_ == 0) {
            if (flag_df_dr[0] != 1) {
                *ptr_df_dx = factor_inv_ * cnum_df_dx;
            }

            if (flag_df_dr[1] != 1) {
                *ptr_df_dy = factor_inv_ * cnum_df_dy;
            }
        }
    }

    delete [] flag_df_dr;

    if (ParallelMPI::rank_ == 0) {
        return mesh_func_r_[jrx][jry];
    } else {
        CNumber cnum_ret;
        cnum_ret[0] = 0.;
        cnum_ret[1] = 0.;

        return cnum_ret;
    }
}

CNumber Transformer2D::get_func_k(int ikx, int iky) {
    int jkx = (ikx + num_mesh_x_) % num_mesh_x_;
    int jky = (iky + num_mesh_y_) % num_mesh_y_;

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    int ipr_src = jkx % ParallelMPI::size_;

    if (ParallelMPI::rank_ == 0) {
        if (ParallelMPI::rank_ == ipr_src) {
            int jkxpr =
                (jkx - ParallelMPI::rank_) /
                ParallelMPI::size_;

            cnum_ret = mesh_func_k_pr_[jkxpr][jky];
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
            int jkxpr =
                (jkx - ParallelMPI::rank_) /
                ParallelMPI::size_;

            double *set_func = new double[2];
            set_func[0] = mesh_func_k_pr_[jkxpr][jky][0];
            set_func[1] = mesh_func_k_pr_[jkxpr][jky][1];

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
