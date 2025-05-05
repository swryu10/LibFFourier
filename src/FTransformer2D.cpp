#include<stdio.h>
#include<math.h>
#ifdef _OPENMP
#include<omp.h>
#endif
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

    num_mmid_x_ = (num_mesh_x_ + (num_mesh_x_ % 2)) / 2;
    num_mmid_y_ = (num_mesh_y_ + (num_mesh_y_ % 2)) / 2;

    mesh_func_r_ = new CNumber *[num_mesh_x_];
    mesh_func_k_ = new CNumber *[num_mesh_x_];
    for (int irx = 0; irx < num_mesh_x_; irx++) {
        mesh_func_r_[irx] = new CNumber[num_mesh_y_];
        mesh_func_k_[irx] = new CNumber[num_mesh_y_];

        for (int iry = 0; iry < num_mesh_y_; iry++) {
            mesh_func_r_[irx][iry] =
                mesh_in_func_r[irx][iry];
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

    num_mmid_x_ = (num_mesh_x_ + (num_mesh_x_ % 2)) / 2;
    num_mmid_y_ = (num_mesh_y_ + (num_mesh_y_ % 2)) / 2;

    double nd_mesh_x = static_cast<double>(num_mesh_x_);
    double nd_mesh_y = static_cast<double>(num_mesh_y_);

    mesh_func_r_ = new CNumber *[num_mesh_x_];
    mesh_func_k_ = new CNumber *[num_mesh_x_];
    for (int irx = 0; irx < num_mesh_x_; irx++) {
        mesh_func_r_[irx] = new CNumber[num_mesh_y_];
        mesh_func_k_[irx] = new CNumber[num_mesh_y_];

        double x_now =
            static_cast<double>(irx) / nd_mesh_x;

        for (int iry = 0; iry < num_mesh_y_; iry++) {
            double y_now =
                static_cast<double>(iry) / nd_mesh_y;

            mesh_func_r_[irx][iry] =
                (*ptr_in_func_r)(x_now, y_now);
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

    Transformer1D *ptr_dft_x =
        new Transformer1D [num_mesh_x_]();

    for (int irx = 0; irx < num_mesh_x_; irx++) {
        ptr_dft_x[irx].init(num_mesh_y_,
                            mesh_func_r_[irx]);
    }

    Transformer1D dft_all;
    CNumber *mesh_func_x = new CNumber [num_mesh_x_];

    for (int iky = 0; iky < num_mesh_y_; iky++) {
        for (int irx = 0; irx < num_mesh_x_; irx++) {
            mesh_func_x[irx] =
                ptr_dft_x[irx].get_func_k(iky);
        }

        dft_all.init(num_mesh_x_,
                     mesh_func_x);

        for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
            mesh_func_k_[ikx][iky] =
                dft_all.get_func_k(ikx);
        }
    }

    delete [] mesh_func_x;

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

        for (int ix = 0; ix < num_in_pt_x; ix++) {
            #ifdef _OPENMP
            if (ix % n_thread != tid) {
                continue;
            }
            #endif

            double x_now = static_cast<double>(ix) / nd_pt_x;

            for (int iy = 0; iy < num_in_pt_y; iy++) {
                double y_now = static_cast<double>(iy) / nd_pt_y;

                tab_func_r_dft[ix][iy] =
                    get_func_r(x_now, y_now);
                if (ptr_in_func_r != NULL) {
                    tab_func_r_ini[ix][iy] =
                        (*ptr_in_func_r)(x_now, y_now);
                }
            }
        }
    #ifdef _OPENMP
    }  // parallel code ends
    #endif

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

    return;
}

void Transformer2D::reset() {
    if (!initialized_) {
        return;
    }

    for (int irx = 0; irx < num_mesh_x_; irx++) {
        delete [] mesh_func_r_[irx];
        delete [] mesh_func_k_[irx];
    }

    delete [] mesh_func_r_;
    delete [] mesh_func_k_;

    num_mesh_x_ = 0;
    num_mesh_y_ = 0;

    num_mmid_x_ = 0;
    num_mmid_y_ = 0;

    initialized_ = false;

    return;
}

CNumber Transformer2D::get_func_r(double x_in,
                                  double y_in) {
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

    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
        for (int iky = 0; iky < num_mesh_y_; iky++) {
            cnum_ret = cnum_ret +
                (mesh_func_k_[ikx][iky] *
                 list_zx_unit[ikx] * list_zy_unit[iky]);
        }
    }

    delete [] list_zx_unit;
    delete [] list_zy_unit;

    return factor_inv_ * cnum_ret;
}

} // end namespace FFourier
