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
                         CNumber **mesh_in_func_rx_ry) {
    reset();

    if (num_in_mesh_x < 2 ||
        num_in_mesh_y < 2) {
        return;
    }

    num_mesh_x_ = num_in_mesh_x;
    num_mesh_y_ = num_in_mesh_y;
    factor_inv_ =
        1. / static_cast<double>(num_mesh_x_ * num_mesh_y_);

    mesh_func_rx_ry_ = new CNumber *[num_mesh_x_];
    mesh_func_kx_ky_ = new CNumber *[num_mesh_x_];
    for (int irx = 0; irx < num_mesh_x_; irx++) {
        mesh_func_rx_ry_[irx] = new CNumber[num_mesh_y_];
        mesh_func_kx_ky_[irx] = new CNumber[num_mesh_y_];

        for (int iry = 0; iry < num_mesh_y_; iry++) {
            mesh_func_rx_ry_[irx][iry] =
                mesh_in_func_rx_ry[irx][iry];
        }
    }

    initialized_ = true;
    make();

    return;
}

void Transformer2D::init(int num_in_mesh_x,
                         int num_in_mesh_y,
                         CNumber (*ptr_in_func_rx_ry)(double,
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

    mesh_func_rx_ry_ = new CNumber *[num_mesh_x_];
    mesh_func_kx_ky_ = new CNumber *[num_mesh_x_];
    for (int irx = 0; irx < num_mesh_x_; irx++) {
        mesh_func_rx_ry_[irx] = new CNumber[num_mesh_y_];
        mesh_func_kx_ky_[irx] = new CNumber[num_mesh_y_];

        double x_now =
            static_cast<double>(irx) /
            static_cast<double>(num_mesh_x_);

        for (int iry = 0; iry < num_mesh_y_; iry++) {
            double y_now =
                static_cast<double>(iry) /
                static_cast<double>(num_mesh_y_);

            mesh_func_rx_ry_[irx][iry] =
                (*ptr_in_func_rx_ry)(x_now, y_now);
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
                            mesh_func_rx_ry_[irx]);
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
            mesh_func_kx_ky_[ikx][iky] =
                dft_all.get_func_k(ikx);
        }
    }

    delete [] mesh_func_x;

    delete [] ptr_dft_x;

    return;
}

void Transformer2D::reset() {
    if (!initialized_) {
        return;
    }

    for (int irx = 0; irx < num_mesh_x_; irx++) {
        delete [] mesh_func_rx_ry_[irx];
        delete [] mesh_func_kx_ky_[irx];
    }

    delete [] mesh_func_rx_ry_;
    delete [] mesh_func_kx_ky_;

    num_mesh_x_ = 0;
    num_mesh_y_ = 0;

    initialized_ = false;

    return;
}

CNumber Transformer2D::get_func_rx_ry(double x_in,
                                      double y_in) {
    CNumber zx_in_unit;
    zx_in_unit[0] = cos(2. * M_PI * x_in);
    zx_in_unit[1] = sin(2. * M_PI * x_in);
    CNumber *list_zx_unit = new CNumber[num_mesh_x_];
    list_zx_unit[0][0] = 1.;
    list_zx_unit[0][1] = 0.;
    for (int ikx = 1; ikx < num_mesh_x_; ikx++) {
        list_zx_unit[ikx] =
            list_zx_unit[ikx - 1] * zx_in_unit;
    }

    CNumber zy_in_unit;
    zy_in_unit[0] = cos(2. * M_PI * y_in);
    zy_in_unit[1] = sin(2. * M_PI * y_in);
    CNumber *list_zy_unit = new CNumber[num_mesh_y_];
    list_zy_unit[0][0] = 1.;
    list_zy_unit[0][1] = 0.;
    for (int iky = 1; iky < num_mesh_y_; iky++) {
        list_zy_unit[iky] =
            list_zy_unit[iky - 1] * zy_in_unit;
    }

    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    for (int ikx = 0; ikx < num_mesh_x_; ikx++) {
        for (int iky = 0; iky < num_mesh_y_; iky++) {
            cnum_ret = cnum_ret +
                (mesh_func_kx_ky_[ikx][iky] *
                 list_zx_unit[ikx] * list_zy_unit[iky]);
        }
    }

    delete [] list_zx_unit;
    delete [] list_zy_unit;

    return factor_inv_ * cnum_ret;
}

} // end namespace FFourier
