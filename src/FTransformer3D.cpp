#include<stdio.h>
#include<math.h>
#include"FTransformer1D.h"
#include"FTransformer2D.h"
#include"FTransformer3D.h"

namespace FFourier {

void Transformer3D::init(int num_in_mesh_z,
                         int num_in_mesh_x,
                         int num_in_mesh_y,
                         CNumber ***mesh_in_func_r) {
    reset();

    if (num_in_mesh_z < 2 ||
        num_in_mesh_x < 2 ||
        num_in_mesh_y < 2) {
        return;
    }

    num_mesh_z_ = num_in_mesh_z;
    num_mesh_x_ = num_in_mesh_x;
    num_mesh_y_ = num_in_mesh_y;
    factor_inv_ =
        1. / static_cast<double>(num_mesh_z_ *
                                 num_mesh_x_ *
                                 num_mesh_y_);

    z_unit_z_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_z_));
    z_unit_z_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_z_));

    z_unit_x_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_x_));
    z_unit_x_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_x_));

    z_unit_y_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_y_));
    z_unit_y_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_y_));

    num_mmid_z_ = (num_mesh_z_ + (num_mesh_z_ % 2)) / 2;
    num_mmid_x_ = (num_mesh_x_ + (num_mesh_x_ % 2)) / 2;
    num_mmid_y_ = (num_mesh_y_ + (num_mesh_y_ % 2)) / 2;

    list_num_mesh_z_pr_ = new int[ParallelMPI::size_];
    for (int ipr = 0; ipr < ParallelMPI::size_; ipr++) {
        list_num_mesh_z_pr_[ipr] = 0;
    }

    for (int irz = 0; irz < num_mesh_z_; irz++) {
        int ipr = irz % ParallelMPI::size_;
        list_num_mesh_z_pr_[ipr] += 1;
    }

    if (ParallelMPI::rank_ == 0) {
        mesh_func_r_ = new CNumber **[num_mesh_z_];
        for (int irz = 0; irz < num_mesh_z_; irz++) {
            mesh_func_r_[irz] = new CNumber *[num_mesh_x_];
            for (int irx = 0; irx < num_mesh_x_; irx++) {
                mesh_func_r_[irz][irx] = new CNumber[num_mesh_y_];

                for (int iry = 0; iry < num_mesh_y_; iry++) {
                    mesh_func_r_[irz][irx][iry] =
                        mesh_in_func_r[irz][irx][iry];
                }
            }
        }
    }

    int num_mesh_z_pr =
        list_num_mesh_z_pr_[ParallelMPI::rank_];
    if (num_mesh_z_pr > 0) {
        mesh_func_k_pr_ = new CNumber **[num_mesh_z_pr];
        for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
            mesh_func_k_pr_[ikzpr] =
                new CNumber *[num_mesh_x_];
            for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
                mesh_func_k_pr_[ikzpr][ikx] =
                    new CNumber[num_mesh_y_];
            }
        }
    }

    initialized_ = true;
    make();

    return;
}

void Transformer3D::init(int num_in_mesh_z,
                         int num_in_mesh_x,
                         int num_in_mesh_y,
                         CNumber (*ptr_in_func_r)(double,
                                                  double,
                                                  double)) {
    reset();

    if (num_in_mesh_z < 2 ||
        num_in_mesh_x < 2 ||
        num_in_mesh_y < 2) {
        return;
    }

    num_mesh_z_ = num_in_mesh_z;
    num_mesh_x_ = num_in_mesh_x;
    num_mesh_y_ = num_in_mesh_y;
    factor_inv_ =
        1. / static_cast<double>(num_mesh_z_ *
                                 num_mesh_x_ *
                                 num_mesh_y_);

    z_unit_z_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_z_));
    z_unit_z_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_z_));

    z_unit_x_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_x_));
    z_unit_x_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_x_));

    z_unit_y_[0] = cos(2. * M_PI /
                       static_cast<double>(num_mesh_y_));
    z_unit_y_[1] = sin(2. * M_PI /
                       static_cast<double>(num_mesh_y_));

    num_mmid_z_ = (num_mesh_z_ + (num_mesh_z_ % 2)) / 2;
    num_mmid_x_ = (num_mesh_x_ + (num_mesh_x_ % 2)) / 2;
    num_mmid_y_ = (num_mesh_y_ + (num_mesh_y_ % 2)) / 2;

    list_num_mesh_z_pr_ = new int[ParallelMPI::size_];
    for (int ipr = 0; ipr < ParallelMPI::size_; ipr++) {
        list_num_mesh_z_pr_[ipr] = 0;
    }

    for (int irz = 0; irz < num_mesh_z_; irz++) {
        int ipr = irz % ParallelMPI::size_;
        list_num_mesh_z_pr_[ipr] += 1;
    }

    double nd_mesh_z = static_cast<double>(num_mesh_z_);
    double nd_mesh_x = static_cast<double>(num_mesh_x_);
    double nd_mesh_y = static_cast<double>(num_mesh_y_);

    if (ParallelMPI::rank_ == 0) {
        mesh_func_r_ = new CNumber **[num_mesh_z_];
        for (int irz = 0; irz < num_mesh_z_; irz++) {
            mesh_func_r_[irz] = new CNumber *[num_mesh_x_];

            double z_now =
                static_cast<double>(irz) / nd_mesh_z;

            for (int irx = 0; irx < num_mesh_x_; irx++) {
                mesh_func_r_[irz][irx] = new CNumber[num_mesh_y_];

                double x_now =
                    static_cast<double>(irx) / nd_mesh_x;

                for (int iry = 0; iry < num_mesh_y_; iry++) {
                    double y_now =
                        static_cast<double>(iry) / nd_mesh_y;

                    mesh_func_r_[irz][irx][iry] =
                        (*ptr_in_func_r)(z_now, x_now, y_now);
                }
            }
        }
    }

    int num_mesh_z_pr =
        list_num_mesh_z_pr_[ParallelMPI::rank_];
    if (num_mesh_z_pr > 0) {
        mesh_func_k_pr_ = new CNumber **[num_mesh_z_pr];
        for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
            mesh_func_k_pr_[ikzpr] =
                new CNumber *[num_mesh_x_];
            for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
                mesh_func_k_pr_[ikzpr][ikx] =
                    new CNumber[num_mesh_y_];
            }
        }
    }

    initialized_ = true;
    make();

    return;
}

void Transformer3D::make() {
    if (!initialized_) {
        return;
    }

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    Transformer2D *ptr_dft_xy =
        new Transformer2D [num_mesh_z_]();

    for (int irz = 0; irz < num_mesh_z_; irz++) {
        CNumber **ptr_mesh_fn_xy = NULL;
        if (ParallelMPI::rank_ == 0) {
            ptr_mesh_fn_xy = mesh_func_r_[irz];
        }

        ptr_dft_xy[irz].init(num_mesh_x_,
                             num_mesh_y_,
                             ptr_mesh_fn_xy);

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
    }

    Transformer1D dft_all;
    CNumber *mesh_func_z;
    if (ParallelMPI::rank_ == 0) {
        mesh_func_z = new CNumber [num_mesh_z_];
    }

    int num_mesh_xy =
        num_mesh_x_ * num_mesh_y_;

    for (int ikxy = 0; ikxy < num_mesh_xy; ikxy++) {
        int ikx = ikxy % num_mesh_x_;
        int iky = (ikxy - ikx) / num_mesh_x_;

        for (int irz = 0; irz < num_mesh_z_; irz++) {
            CNumber cnum_func_z =
                ptr_dft_xy[irz].get_func_k(ikx, iky);

            if (ParallelMPI::rank_ == 0) {
                mesh_func_z[irz] = cnum_func_z;
            }
        }

        dft_all.init(num_mesh_z_,
                     mesh_func_z);

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);
        #endif

        CNumber *list_func_k;
        if (ParallelMPI::rank_ == 0) {
            list_func_k = new CNumber[num_mesh_z_];
        }

        for (int ikz = 0; ikz < num_mesh_z_; ikz++) {
            CNumber cnum_func_k =
                dft_all.get_func_k(ikz);

            if (ParallelMPI::rank_ == 0) {
                list_func_k[ikz] = cnum_func_k;
            }
        }

        if (ParallelMPI::rank_ == 0) {
            for (int ikz = 0; ikz < num_mesh_z_; ikz++) {
                int ipr_tar = ikz % ParallelMPI::size_;
                if (ipr_tar == 0) {
                    int ikzpr =
                        (ikz - ParallelMPI::rank_) /
                        ParallelMPI::size_;

                    mesh_func_k_pr_[ikzpr][ikx][iky] =
                        list_func_k[ikz];
                } else {
                    #ifdef _MPI
                    double *set_func_k = new double[2];
                    set_func_k[0] = list_func_k[ikz][0];
                    set_func_k[1] = list_func_k[ikz][1];

                    tag = (num_mesh_z_ * ikxy + ikz) *
                          ParallelMPI::size_ +
                          ipr_tar;
                    MPI_Send(set_func_k, 2, MPI_DOUBLE,
                             ipr_tar, tag, MPI_COMM_WORLD);

                    delete [] set_func_k;
                    #endif
                }
            }

            delete [] list_func_k;
        } else {
            #ifdef _MPI
            for (int ikz = 0; ikz < num_mesh_z_; ikz++) {
                int ipr_tar = ikz % ParallelMPI::size_;
                if (ParallelMPI::rank_ != ipr_tar) {
                    continue;
                }

                double *set_func_k = new double[2];

                tag = (num_mesh_z_ * ikxy + ikz) *
                      ParallelMPI::size_ +
                      ipr_tar;
                MPI_Recv(set_func_k, 2, MPI_DOUBLE,
                         0, tag, MPI_COMM_WORLD, &status);

                int ikzpr =
                    (ikz - ParallelMPI::rank_) /
                    ParallelMPI::size_;
                mesh_func_k_pr_[ikzpr][ikx][iky][0] =
                    set_func_k[0];
                mesh_func_k_pr_[ikzpr][ikx][iky][1] =
                    set_func_k[1];

                delete [] set_func_k;
            }
            #endif
        }
    }

    if (ParallelMPI::rank_ == 0) {
        delete [] mesh_func_z;
    }

    delete [] ptr_dft_xy;

    return;
}

void Transformer3D::export_func_r(std::string name_file,
                                  int num_in_pt_x,
                                  int num_in_pt_y,
                                  int axis_z,
                                  double z_plane,
                                  CNumber (*ptr_in_func_r)(double,
                                                           double,
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

    double *xvec_now = new double[3];

    double z_proj = z_plane;

    for (int ix = 0; ix < num_in_pt_x; ix++) {
        double x_proj = static_cast<double>(ix) / nd_pt_x;

        for (int iy = 0; iy < num_in_pt_y; iy++) {
            double y_proj = static_cast<double>(iy) / nd_pt_y;

            int iiz = axis_z % 3;
            int iix = (1 + axis_z) % 3;
            int iiy = (2 + axis_z) % 3;

            xvec_now[iiz] = z_proj;
            xvec_now[iix] = x_proj;
            xvec_now[iiy] = y_proj;

            CNumber cnum_df_dz_dft;
            CNumber cnum_df_dx_dft;
            CNumber cnum_df_dy_dft;
            CNumber cnum_func_dft =
                get_func_r(xvec_now[0],
                           xvec_now[1],
                           xvec_now[2],
                           &cnum_df_dz_dft,
                           &cnum_df_dx_dft,
                           &cnum_df_dy_dft);

            if (ParallelMPI::rank_ == 0) {
                tab_func_r_dft[ix][iy] = cnum_func_dft;
                if (ptr_in_func_r != NULL) {
                    tab_func_r_ini[ix][iy] =
                        (*ptr_in_func_r)(xvec_now[0],
                                         xvec_now[1],
                                         xvec_now[2]);
                }
            }
        }
    }

    delete [] xvec_now;

    if (ParallelMPI::rank_ == 0) {
        if (axis_z % 3 == 1) {
            fprintf(ptr_fout, "# x = %e\n", z_plane);
            fprintf(ptr_fout, "# num_pt_y = %d, num_pt_z = %d\n",
                    num_in_pt_x, num_in_pt_y);
        } else if (axis_z % 3 == 2) {
            fprintf(ptr_fout, "# y = %e\n", z_plane);
            fprintf(ptr_fout, "# num_pt_z = %d, num_pt_x = %d\n",
                    num_in_pt_x, num_in_pt_y);
        } else {
            fprintf(ptr_fout, "# z = %e\n", z_plane);
            fprintf(ptr_fout, "# num_pt_x = %d, num_pt_y = %d\n",
                    num_in_pt_x, num_in_pt_y);
        }

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

void Transformer3D::reset() {
    if (!initialized_) {
        return;
    }

    if (ParallelMPI::rank_ == 0) {
        for (int irz = 0; irz < num_mesh_z_; irz++) {
            for (int irx = 0; irx < num_mesh_x_; irx++) {
                delete [] mesh_func_r_[irz][irx];
            }

            delete [] mesh_func_r_[irz];
        }

        delete [] mesh_func_r_;
    }

    int num_mesh_z_pr =
        list_num_mesh_z_pr_[ParallelMPI::rank_];
    if (num_mesh_z_pr > 0) {
        for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
            for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
                delete [] mesh_func_k_pr_[ikzpr][ikx];
            }

            delete [] mesh_func_k_pr_[ikzpr];
        }

        delete [] mesh_func_k_pr_;
    }

    delete [] list_num_mesh_z_pr_;

    num_mesh_z_ = 0;
    num_mesh_x_ = 0;
    num_mesh_y_ = 0;

    num_mmid_z_ = 0;
    num_mmid_x_ = 0;
    num_mmid_y_ = 0;

    initialized_ = false;

    return;
}

CNumber Transformer3D::get_func_r(double z_in,
                                  double x_in,
                                  double y_in,
                                  CNumber *ptr_df_dz,
                                  CNumber *ptr_df_dx,
                                  CNumber *ptr_df_dy) {
    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    CNumber zz_in_unit;
    zz_in_unit[0] = cos(2. * M_PI * z_in);
    zz_in_unit[1] = sin(2. * M_PI * z_in);
    CNumber *list_zz_unit = new CNumber[num_mesh_z_];
    list_zz_unit[0][0] = 1.;
    list_zz_unit[0][1] = 0.;
    for (int ikz = 1; ikz < num_mmid_z_; ikz++) {
        int ikz_prev = ikz - 1;
        list_zz_unit[ikz] =
            list_zz_unit[ikz_prev] * zz_in_unit;
    }
    for (int ikz = num_mesh_z_ - 1; ikz >= num_mmid_z_; ikz--) {
        int ikz_prev = (ikz + 1) % num_mesh_z_;
        list_zz_unit[ikz] =
            list_zz_unit[ikz_prev] / zz_in_unit;
    }

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

    int num_mesh_z_pr =
        list_num_mesh_z_pr_[ParallelMPI::rank_];
    int num_mesh_xy =
        num_mesh_x_ * num_mesh_y_;

    #ifdef _OPENMP
    CNumber *list_c_func =
        new CNumber[num_mesh_z_pr];

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

        for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
            #ifdef _OPENMP
            if (ikzpr % n_thread != tid) {
                continue;
            }
            #endif

            int ikz =
                ParallelMPI::rank_ +
                ParallelMPI::size_ * ikzpr;

            #ifdef _OPENMP
            list_c_func[ikzpr][0] = 0.;
            list_c_func[ikzpr][1] = 0.;
            #endif
            for (int ikxy = 0; ikxy < num_mesh_xy; ikxy++) {
                int ikx = ikxy % num_mesh_x_;
                int iky = (ikxy - ikx) / num_mesh_x_;

                CNumber func_now =
                    mesh_func_k_pr_[ikzpr][ikx][iky] *
                    list_zz_unit[ikz] *
                    list_zx_unit[ikx] *
                    list_zy_unit[iky];

                #ifdef _OPENMP
                list_c_func[ikzpr] =
                    list_c_func[ikzpr] + func_now;
                #else
                cnum_ret = cnum_ret + func_now;
                #endif
            }
        }
    #ifdef _OPENMP
    }  // parallel code ends

    for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
        cnum_ret = cnum_ret + list_c_func[ikzpr];
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

    int *flag_df_dr = new int[3];
    flag_df_dr[0] = 0;
    flag_df_dr[1] = 0;
    flag_df_dr[2] = 0;
    if (ParallelMPI::rank_ == 0) {
        if (ptr_df_dz != NULL) {
            flag_df_dr[0] = 1;
        }

        if (ptr_df_dx != NULL) {
            flag_df_dr[1] = 1;
        }

        if (ptr_df_dy != NULL) {
            flag_df_dr[2] = 1;
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            tag = ipr * 100 + 10;
            MPI_Send(flag_df_dr, 3, MPI_INT,
                     ipr, tag, MPI_COMM_WORLD);
        }
        #endif
    } else {
        #ifdef _MPI
        tag = ParallelMPI::rank_ * 100 + 10;
        MPI_Recv(flag_df_dr, 3, MPI_INT,
                 0, tag, MPI_COMM_WORLD, &status);
        #endif
    }

    if (flag_df_dr[0] != 0 ||
        flag_df_dr[1] != 0 ||
        flag_df_dr[2] != 0) {
        CNumber cnum_df_dz;
        cnum_df_dz[0] = 0.;
        cnum_df_dz[1] = 0.;

        CNumber cnum_df_dx;
        cnum_df_dx[0] = 0.;
        cnum_df_dx[1] = 0.;

        CNumber cnum_df_dy;
        cnum_df_dy[0] = 0.;
        cnum_df_dy[1] = 0.;

        #ifdef _OPENMP
        CNumber *list_c_df_dz =
            new CNumber[num_mesh_z_pr];
        CNumber *list_c_df_dx =
            new CNumber[num_mesh_z_pr];
        CNumber *list_c_df_dy =
            new CNumber[num_mesh_z_pr];

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

            for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
                #ifdef _OPENMP
                if (ikzpr % n_thread != tid) {
                    continue;
                }
                #endif

                int ikz =
                    ParallelMPI::rank_ +
                    ParallelMPI::size_ * ikzpr;

                int jkz = ikz;
                if (ikz >= num_mmid_z_) {
                    jkz = ikz - num_mesh_z_;
                }

                CNumber fac_deriv_z;
                fac_deriv_z[0] = 0.;
                fac_deriv_z[1] =
                    2. * M_PI * static_cast<double>(jkz);

                #ifdef _OPENMP
                list_c_df_dz[ikzpr][0] = 0.;
                list_c_df_dz[ikzpr][1] = 0.;

                list_c_df_dx[ikzpr][0] = 0.;
                list_c_df_dx[ikzpr][1] = 0.;

                list_c_df_dy[ikzpr][0] = 0.;
                list_c_df_dy[ikzpr][1] = 0.;
                #endif
                for (int ikxy = 0; ikxy < num_mesh_xy; ikxy++) {
                    int ikx = ikxy % num_mesh_x_;
                    int iky = (ikxy - ikx) / num_mesh_x_;

                    int jkx = ikx;
                    if (ikx >= num_mmid_x_) {
                        jkx = ikx - num_mesh_x_;
                    }

                    int jky = iky;
                    if (iky >= num_mmid_y_) {
                        jky = iky - num_mesh_y_;
                    }

                    CNumber fac_deriv_x;
                    fac_deriv_x[0] = 0.;
                    fac_deriv_x[1] =
                        2. * M_PI * static_cast<double>(jkx);

                    CNumber fac_deriv_y;
                    fac_deriv_y[0] = 0.;
                    fac_deriv_y[1] =
                        2. * M_PI * static_cast<double>(jky);

                    CNumber func_now =
                        mesh_func_k_pr_[ikzpr][ikx][iky] *
                        list_zz_unit[ikz] *
                        list_zx_unit[ikx] *
                        list_zy_unit[iky];

                    #ifdef _OPENMP
                    list_c_df_dz[ikzpr] =
                        list_c_df_dz[ikzpr] +
                        (fac_deriv_z * func_now);

                    list_c_df_dx[ikzpr] =
                        list_c_df_dx[ikzpr] +
                        (fac_deriv_x * func_now);

                    list_c_df_dy[ikzpr] =
                        list_c_df_dy[ikzpr] +
                        (fac_deriv_y * func_now);
                    #else
                    cnum_df_dz = cnum_df_dz +
                        (fac_deriv_z * func_now);

                    cnum_df_dx = cnum_df_dx +
                        (fac_deriv_x * func_now);

                    cnum_df_dy = cnum_df_dy +
                        (fac_deriv_y * func_now);
                    #endif
                }
            }
        #ifdef _OPENMP
        }  // parallel code ends

        for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
            cnum_df_dz = cnum_df_dz + list_c_df_dz[ikzpr];
            cnum_df_dx = cnum_df_dx + list_c_df_dx[ikzpr];
            cnum_df_dy = cnum_df_dy + list_c_df_dy[ikzpr];
        }

        delete [] list_c_df_dz;
        delete [] list_c_df_dx;
        delete [] list_c_df_dy;
        #endif

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);

        if (ParallelMPI::rank_ == 0) {
            for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
                double *set_df_dr = new double[6];

                tag = ipr * 100 + 13;
                MPI_Recv(set_df_dr, 6, MPI_DOUBLE,
                         ipr, tag, MPI_COMM_WORLD, &status);

                cnum_df_dz[0] += set_df_dr[0];
                cnum_df_dz[1] += set_df_dr[1];
                cnum_df_dx[0] += set_df_dr[2];
                cnum_df_dx[1] += set_df_dr[3];
                cnum_df_dy[0] += set_df_dr[4];
                cnum_df_dy[1] += set_df_dr[5];

                delete [] set_df_dr;
            }
        } else {
            double *set_df_dr = new double[6];
            set_df_dr[0] = cnum_df_dz[0];
            set_df_dr[1] = cnum_df_dz[1];
            set_df_dr[2] = cnum_df_dx[0];
            set_df_dr[3] = cnum_df_dx[1];
            set_df_dr[4] = cnum_df_dy[0];
            set_df_dr[5] = cnum_df_dy[1];

            tag = ParallelMPI::rank_ * 100 + 13;
            MPI_Send(set_df_dr, 6, MPI_DOUBLE,
                     0, tag, MPI_COMM_WORLD);

            cnum_df_dz[0] = 0.;
            cnum_df_dz[1] = 0.;

            cnum_df_dx[0] = 0.;
            cnum_df_dx[1] = 0.;

            cnum_df_dy[0] = 0.;
            cnum_df_dy[1] = 0.;

            delete [] set_df_dr;
        }
        #endif

        if (ParallelMPI::rank_ == 0) {
            if (flag_df_dr[0] != 0) {
                *ptr_df_dz = factor_inv_ * cnum_df_dz;
            }

            if (flag_df_dr[1] != 0) {
                *ptr_df_dx = factor_inv_ * cnum_df_dx;
            }

            if (flag_df_dr[2] != 0) {
                *ptr_df_dy = factor_inv_ * cnum_df_dy;
            }
        }
    }

    delete [] flag_df_dr;

    delete [] list_zz_unit;
    delete [] list_zx_unit;
    delete [] list_zy_unit;

    return factor_inv_ * cnum_ret;
}

CNumber Transformer3D::get_func_r(int irz, int irx, int iry,
                                  CNumber *ptr_df_dz,
                                  CNumber *ptr_df_dx,
                                  CNumber *ptr_df_dy) {
    int jrz = (irz + num_mesh_z_) % num_mesh_z_;
    int jrx = (irx + num_mesh_x_) % num_mesh_x_;
    int jry = (iry + num_mesh_y_) % num_mesh_y_;

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    int *flag_df_dr = new int[3];
    flag_df_dr[0] = 0;
    flag_df_dr[1] = 0;
    flag_df_dr[2] = 0;
    if (ParallelMPI::rank_ == 0) {
        if (ptr_df_dz != NULL) {
            flag_df_dr[0] = 1;
        }

        if (ptr_df_dx != NULL) {
            flag_df_dr[1] = 1;
        }

        if (ptr_df_dy != NULL) {
            flag_df_dr[2] = 1;
        }

        #ifdef _MPI
        for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
            tag = ipr * 100 + 20;
            MPI_Send(flag_df_dr, 3, MPI_INT,
                     ipr, tag, MPI_COMM_WORLD);
        }
        #endif
    } else {
        #ifdef _MPI
        tag = ParallelMPI::rank_ * 100 + 20;
        MPI_Recv(flag_df_dr, 3, MPI_INT,
                 0, tag, MPI_COMM_WORLD, &status);
        #endif
    }

    if (flag_df_dr[0] != 0 ||
        flag_df_dr[1] != 0 ||
        flag_df_dr[2] != 0) {
        CNumber cnum_df_dz;
        cnum_df_dz[0] = 0.;
        cnum_df_dz[1] = 0.;

        CNumber cnum_df_dx;
        cnum_df_dx[0] = 0.;
        cnum_df_dx[1] = 0.;

        CNumber cnum_df_dy;
        cnum_df_dy[0] = 0.;
        cnum_df_dy[1] = 0.;

        int num_mesh_z_pr =
            list_num_mesh_z_pr_[ParallelMPI::rank_];
        int num_mesh_xy =
            num_mesh_x_ * num_mesh_y_;

        #ifdef _OPENMP
        CNumber *list_c_df_dz =
            new CNumber[num_mesh_z_pr];
        CNumber *list_c_df_dx =
            new CNumber[num_mesh_z_pr];
        CNumber *list_c_df_dy =
            new CNumber[num_mesh_z_pr];

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

            for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
                #ifdef _OPENMP
                if (ikzpr % n_thread != tid) {
                    continue;
                }
                #endif

                int ikz =
                    ParallelMPI::rank_ +
                    ParallelMPI::size_ * ikzpr;

                int jkz = ikz;
                if (ikz >= num_mmid_z_) {
                    jkz = ikz - num_mesh_z_;
                }

                CNumber fac_deriv_z;
                fac_deriv_z[0] = 0.;
                fac_deriv_z[1] =
                    2. * M_PI * static_cast<double>(jkz);

                #ifdef _OPENMP
                list_c_df_dz[ikzpr][0] = 0.;
                list_c_df_dz[ikzpr][1] = 0.;

                list_c_df_dx[ikzpr][0] = 0.;
                list_c_df_dx[ikzpr][1] = 0.;

                list_c_df_dy[ikzpr][0] = 0.;
                list_c_df_dy[ikzpr][1] = 0.;
                #endif
                for (int ikxy = 0; ikxy < num_mesh_xy; ikxy++) {
                    int ikx = ikxy % num_mesh_x_;
                    int iky = (ikxy - ikx) / num_mesh_x_;

                    int jkx = ikx;
                    if (ikx >= num_mmid_x_) {
                        jkx = ikx - num_mesh_x_;
                    }

                    int jky = iky;
                    if (iky >= num_mmid_y_) {
                        jky = iky - num_mesh_y_;
                    }

                    CNumber fac_deriv_x;
                    fac_deriv_x[0] = 0.;
                    fac_deriv_x[1] =
                        2. * M_PI * static_cast<double>(jkx);

                    CNumber fac_deriv_y;
                    fac_deriv_y[0] = 0.;
                    fac_deriv_y[1] =
                        2. * M_PI * static_cast<double>(jky);

                    CNumber func_now =
                        mesh_func_k_pr_[ikzpr][ikx][iky] *
                        (z_unit_z_ ^ (jkz * jrz)) *
                        (z_unit_x_ ^ (jkx * jrx)) *
                        (z_unit_y_ ^ (jky * jry));

                    #ifdef _OPENMP
                    list_c_df_dz[ikzpr] =
                        list_c_df_dz[ikzpr] +
                        (fac_deriv_z * func_now);

                    list_c_df_dx[ikzpr] =
                        list_c_df_dx[ikzpr] +
                        (fac_deriv_x * func_now);

                    list_c_df_dy[ikzpr] =
                        list_c_df_dy[ikzpr] +
                        (fac_deriv_y * func_now);
                    #else
                    cnum_df_dz = cnum_df_dz +
                        (fac_deriv_z * func_now);

                    cnum_df_dx = cnum_df_dx +
                        (fac_deriv_x * func_now);

                    cnum_df_dy = cnum_df_dy +
                        (fac_deriv_y * func_now);
                    #endif
                }
            }
        #ifdef _OPENMP
        }  // parallel code ends

        for (int ikzpr = 0; ikzpr < num_mesh_z_pr; ikzpr++) {
            cnum_df_dz = cnum_df_dz + list_c_df_dz[ikzpr];
            cnum_df_dx = cnum_df_dx + list_c_df_dx[ikzpr];
            cnum_df_dy = cnum_df_dy + list_c_df_dy[ikzpr];
        }

        delete [] list_c_df_dz;
        delete [] list_c_df_dx;
        delete [] list_c_df_dy;
        #endif

        #ifdef _MPI
        MPI_Barrier(MPI_COMM_WORLD);

        if (ParallelMPI::rank_ == 0) {
            for (int ipr = 1; ipr < ParallelMPI::size_; ipr++) {
                double *set_df_dr = new double[6];

                tag = ipr * 100 + 23;
                MPI_Recv(set_df_dr, 6, MPI_DOUBLE,
                         ipr, tag, MPI_COMM_WORLD, &status);

                cnum_df_dz[0] += set_df_dr[0];
                cnum_df_dz[1] += set_df_dr[1];
                cnum_df_dx[0] += set_df_dr[2];
                cnum_df_dx[1] += set_df_dr[3];
                cnum_df_dy[0] += set_df_dr[4];
                cnum_df_dy[1] += set_df_dr[5];

                delete [] set_df_dr;
            }
        } else {
            double *set_df_dr = new double[6];
            set_df_dr[0] = cnum_df_dz[0];
            set_df_dr[1] = cnum_df_dz[1];
            set_df_dr[2] = cnum_df_dx[0];
            set_df_dr[3] = cnum_df_dx[1];
            set_df_dr[4] = cnum_df_dy[0];
            set_df_dr[5] = cnum_df_dy[1];

            tag = ParallelMPI::rank_ * 100 + 23;
            MPI_Send(set_df_dr, 6, MPI_DOUBLE,
                     0, tag, MPI_COMM_WORLD);

            cnum_df_dz[0] = 0.;
            cnum_df_dz[1] = 0.;

            cnum_df_dx[0] = 0.;
            cnum_df_dx[1] = 0.;

            cnum_df_dy[0] = 0.;
            cnum_df_dy[1] = 0.;

            delete [] set_df_dr;
        }
        #endif

        if (ParallelMPI::rank_ == 0) {
            if (flag_df_dr[0] != 0) {
                *ptr_df_dz = factor_inv_ * cnum_df_dz;
            }

            if (flag_df_dr[1] != 0) {
                *ptr_df_dx = factor_inv_ * cnum_df_dx;
            }

            if (flag_df_dr[2] != 0) {
                *ptr_df_dy = factor_inv_ * cnum_df_dy;
            }
        }
    }

    delete [] flag_df_dr;

    if (ParallelMPI::rank_ == 0) {
        return mesh_func_r_[jrz][jrx][jry];
    } else {
        CNumber cnum_ret;
        cnum_ret[0] = 0.;
        cnum_ret[1] = 0.;

        return cnum_ret;
    }
}

CNumber Transformer3D::get_func_k(int ikz, int ikx, int iky) {
    int jkz = (ikz + num_mesh_z_) % num_mesh_z_;
    int jkx = (ikx + num_mesh_x_) % num_mesh_x_;
    int jky = (iky + num_mesh_y_) % num_mesh_y_;

    #ifdef _MPI
    int tag;
    MPI_Status status;
    #endif

    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    int ipr_src = jkz % ParallelMPI::size_;

    if (ParallelMPI::rank_ == 0) {
        if (ParallelMPI::rank_ == ipr_src) {
            int jkzpr =
                (jkz - ParallelMPI::rank_) /
                ParallelMPI::size_;

            cnum_ret = mesh_func_k_pr_[jkzpr][jkx][jky];
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
            int jkzpr =
                (jkz - ParallelMPI::rank_) /
                ParallelMPI::size_;

            double *set_func = new double[2];
            set_func[0] =
                mesh_func_k_pr_[jkzpr][jkx][jky][0];
            set_func[1] =
                mesh_func_k_pr_[jkzpr][jkx][jky][1];

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
