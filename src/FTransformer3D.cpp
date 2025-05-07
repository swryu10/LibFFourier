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

    num_mmid_z_ = (num_mesh_z_ + (num_mesh_z_ % 2)) / 2;
    num_mmid_x_ = (num_mesh_x_ + (num_mesh_x_ % 2)) / 2;
    num_mmid_y_ = (num_mesh_y_ + (num_mesh_y_ % 2)) / 2;

    if (ParallelMPI::rank_ == 0) {
        mesh_func_r_ = new CNumber **[num_mesh_z_];
        mesh_func_k_ = new CNumber **[num_mesh_z_];
        for (int irz = 0; irz < num_mesh_z_; irz++) {
            mesh_func_r_[irz] = new CNumber *[num_mesh_x_];
            mesh_func_k_[irz] = new CNumber *[num_mesh_x_];
            for (int irx = 0; irx < num_mesh_x_; irx++) {
                mesh_func_r_[irz][irx] = new CNumber[num_mesh_y_];
                mesh_func_k_[irz][irx] = new CNumber[num_mesh_y_];

                for (int iry = 0; iry < num_mesh_y_; iry++) {
                    mesh_func_r_[irz][irx][iry] =
                        mesh_in_func_r[irz][irx][iry];
                }
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

    num_mmid_z_ = (num_mesh_z_ + (num_mesh_z_ % 2)) / 2;
    num_mmid_x_ = (num_mesh_x_ + (num_mesh_x_ % 2)) / 2;
    num_mmid_y_ = (num_mesh_y_ + (num_mesh_y_ % 2)) / 2;

    double nd_mesh_z = static_cast<double>(num_mesh_z_);
    double nd_mesh_x = static_cast<double>(num_mesh_x_);
    double nd_mesh_y = static_cast<double>(num_mesh_y_);

    if (ParallelMPI::rank_ == 0) {
        mesh_func_r_ = new CNumber **[num_mesh_z_];
        mesh_func_k_ = new CNumber **[num_mesh_z_];
        for (int irz = 0; irz < num_mesh_z_; irz++) {
            mesh_func_r_[irz] = new CNumber *[num_mesh_x_];
            mesh_func_k_[irz] = new CNumber *[num_mesh_x_];

            double z_now =
                static_cast<double>(irz) / nd_mesh_z;

            for (int irx = 0; irx < num_mesh_x_; irx++) {
                mesh_func_r_[irz][irx] = new CNumber[num_mesh_y_];
                mesh_func_k_[irz][irx] = new CNumber[num_mesh_y_];

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

    initialized_ = true;
    make();

    return;
}

void Transformer3D::make() {
    if (!initialized_) {
        return;
    }

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
    CNumber *mesh_func_z = new CNumber [num_mesh_z_];

    for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
        for (int iky = 0; iky < num_mesh_y_; iky++) {
            if (ParallelMPI::rank_ == 0) {
                for (int irz = 0; irz < num_mesh_z_; irz++) {
                    mesh_func_z[irz] =
                        ptr_dft_xy[irz].get_func_k(ikx, iky);
                }
            }

            dft_all.init(num_mesh_z_,
                         mesh_func_z);

            #ifdef _MPI
            MPI_Barrier(MPI_COMM_WORLD);
            #endif

            if (ParallelMPI::rank_ == 0) {
                for (int ikz = 0; ikz < num_mesh_z_; ikz++) {
                    mesh_func_k_[ikz][ikx][iky] =
                        dft_all.get_func_k(ikz);
                }
            }
        }
    }

    delete [] mesh_func_z;

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

    if (ParallelMPI::rank_ != 0) {
        return;
    }

    if (num_in_pt_x < 2 ||
        num_in_pt_y < 2) {
        return;
    }

    FILE *ptr_fout;
    ptr_fout = fopen(name_file.c_str(), "w");

    if (ptr_fout == NULL) {
        return;
    }

    double nd_pt_x = static_cast<double>(num_in_pt_x);
    double nd_pt_y = static_cast<double>(num_in_pt_y);

    CNumber **tab_func_r_dft = new CNumber *[num_in_pt_x];
    CNumber **tab_func_r_ini = new CNumber *[num_in_pt_x];
    for (int ix = 0; ix < num_in_pt_x; ix++) {
        tab_func_r_dft[ix] = new CNumber[num_in_pt_y];
        tab_func_r_ini[ix] = new CNumber[num_in_pt_y];
    }

    #ifdef _OPENMP
    #pragma omp parallel
    {  // parallel code begins
    #endif
        #ifdef _OPENMP
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        #endif

        double *xvec_now = new double[3];

        double z_proj = z_plane;

        for (int ix = 0; ix < num_in_pt_x; ix++) {
            #ifdef _OPENMP
            if (ix % n_thread != tid) {
                continue;
            }
            #endif

            double x_proj = static_cast<double>(ix) / nd_pt_x;

            for (int iy = 0; iy < num_in_pt_y; iy++) {
                double y_proj = static_cast<double>(iy) / nd_pt_y;

                int iiz = axis_z % 3;
                int iix = (1 + axis_z) % 3;
                int iiy = (2 + axis_z) % 3;

                xvec_now[iiz] = z_proj;
                xvec_now[iix] = x_proj;
                xvec_now[iiy] = y_proj;

                tab_func_r_dft[ix][iy] =
                    get_func_r(xvec_now[0],
                               xvec_now[1],
                               xvec_now[2]);
                if (ptr_in_func_r != NULL) {
                    tab_func_r_ini[ix][iy] =
                        (*ptr_in_func_r)(xvec_now[0],
                                         xvec_now[1],
                                         xvec_now[2]);
                }
            }
        }

        delete [] xvec_now;
    #ifdef _OPENMP
    }  // parallel code ends
    #endif

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
                delete [] mesh_func_k_[irz][irx];
            }

            delete [] mesh_func_r_[irz];
            delete [] mesh_func_k_[irz];
        }
        delete [] mesh_func_r_;
        delete [] mesh_func_k_;
    }

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
                                  double y_in) {
    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    if (ParallelMPI::rank_ != 0) {
        return cnum_ret;
    }

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

    for (int ikz = 0; ikz < num_mesh_z_; ikz++) {
        for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
            for (int iky = 0; iky < num_mesh_y_; iky++) {
                cnum_ret = cnum_ret +
                    (mesh_func_k_[ikz][ikx][iky] *
                     list_zz_unit[ikz] *
                     list_zx_unit[ikx] *
                     list_zy_unit[iky]);
            }
        }
    }

    delete [] list_zz_unit;
    delete [] list_zx_unit;
    delete [] list_zy_unit;

    return factor_inv_ * cnum_ret;
}

} // end namespace FFourier
