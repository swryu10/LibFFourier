#include<math.h>
#ifdef _OPENMP
#include<omp.h>
#endif
#include"FTransformer1D.h"

namespace FFourier {

void Transformer1D::init(int num_in_mesh,
                         CNumber *mesh_in_func_x) {
    reset();

    if (num_in_mesh < 2) {
        return;
    }

    num_mesh_ = num_in_mesh;

    mesh_func_x_ = new CNumber[num_mesh_];
    mesh_func_k_ = new CNumber[num_mesh_];

    for (int ix = 0; ix < num_mesh_; ix++) {
        mesh_func_x_[ix] = mesh_in_func_x[ix];
    }

    z_unit_[0] = cos(2. * M_PI /
                     static_cast<double>(num_mesh_));
    z_unit_[1] = sin(2. * M_PI /
                     static_cast<double>(num_mesh_));

    #ifdef _OPENMP
    #pragma omp parallel
    {  // parallel code begins
    #endif
        #ifdef _OPENMP
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        #else
        int n_thread = 1;
        int tid = 0;
        #endif

        for (int ik = 0; ik < num_mesh_; ik++) {
            if (ik % n_thread != 0) {
                continue;
            }

            mesh_func_k_[ik] = next(ik, num_mesh_,
                                    mesh_func_x_);
        }
    #ifdef _OPENMP
    }  // parallel code ends
    #endif

    initialized_ = true;

    return;
}

void Transformer1D::reset() {
    if (!initialized_) {
        return;
    }

    num_mesh_ = 0;

    delete [] mesh_func_x_;
    delete [] mesh_func_k_;

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

CNumber Transformer1D::get_func_x(double x_in) {
    CNumber z_in_unit;
    z_in_unit[0] = cos(2. * M_PI * x_in);
    z_in_unit[1] = sin(2. * M_PI * x_in);

    CNumber cnum_ret;
    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;

    for (int ik = 0; ik < num_mesh_; ik++) {
        cnum_ret = cnum_ret +
            (mesh_func_k_[ik] * (z_in_unit ^ ik));
    }

    return cnum_ret / static_cast<double>(num_mesh_);
}

} // end namespace FFourier
