#include<stdio.h>
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

    mesh_func_x_ = new CNumber[num_mesh_];
    mesh_func_k_ = new CNumber[num_mesh_];

    for (int ix = 0; ix < num_mesh_; ix++) {
        double x_now =
            static_cast<double>(ix) /
            static_cast<double>(num_mesh_);
        mesh_func_x_[ix] = (*ptr_in_func_x)(x_now);
    }

    initialized_ = true;
    make();

    return;
}

void Transformer1D::make() {
    if (!initialized_) {
        return;
    }

    #ifdef _OPENMP
    #pragma omp parallel
    {  // parallel code begins
    #endif
        #ifdef _OPENMP
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        #endif

        for (int ik = 0; ik < num_mesh_; ik++) {
            #ifdef _OPENMP
            if (ik % n_thread != tid) {
                continue;
            }
            #endif

            mesh_func_k_[ik] = next(ik, num_mesh_,
                                    mesh_func_x_);
        }
    #ifdef _OPENMP
    }  // parallel code ends
    #endif

    return;
}

void Transformer1D::shift(double dx) {
    if (!initialized_) {
        return;
    }

    CNumber z_d_unit;
    z_d_unit[0] = cos(2. * M_PI * dx);
    z_d_unit[1] = sin(2. * M_PI * dx);

    #ifdef _OPENMP
    #pragma omp parallel
    {  // parallel code begins
    #endif
        #ifdef _OPENMP
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        #endif

        for (int ik = 0; ik < num_mesh_; ik++) {
            #ifdef _OPENMP
            if (ik % n_thread != tid) {
                continue;
            }
            #endif

            mesh_func_k_[ik] =
                mesh_func_k_[ik] / (z_d_unit ^ ik);
        }

        #ifdef _OPENMP
        // syncronize threads
        #pragma omp barrier
        #endif

        for (int ix = 0; ix < num_mesh_; ix++) {
            #ifdef _OPENMP
            if (ix % n_thread != tid) {
                continue;
            }
            #endif

            double x_now =
                static_cast<double>(ix) /
                static_cast<double>(num_mesh_);
            mesh_func_x_[ix] = get_func_x(x_now);
        }
    #ifdef _OPENMP
    }  // parallel code ends
    #endif

    return;
}

void Transformer1D::amplify(double fac) {
    if (!initialized_) {
        return;
    }

    for (int ix = 0; ix < num_mesh_; ix++) {
        mesh_func_x_[ix] = fac * mesh_func_x_[ix];
        mesh_func_k_[ix] = fac * mesh_func_k_[ix];
    }

    return;
}

void Transformer1D::export_file(std::string name_file) {
    if (!initialized_) {
        return;
    }

    FILE *ptr_fout;
    ptr_fout = fopen(name_file.c_str(), "w");

    if (ptr_fout == NULL) {
        return;
    }

    fprintf(ptr_fout, "# num_mesh_ = %d\n", num_mesh_);

    for (int ik = 0; ik < num_mesh_; ik++) {
        double x_now = static_cast<double>(ik) /
                       static_cast<double>(num_mesh_);
        fprintf(ptr_fout, "    %d    %e    %e", ik,
                mesh_func_k_[ik][0], mesh_func_k_[ik][1]);
        fprintf(ptr_fout, "    %e    %e    %e", x_now,
                mesh_func_x_[ik][0], mesh_func_x_[ik][1]);
        fprintf(ptr_fout, "\n");
    }

    fclose(ptr_fout);

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
