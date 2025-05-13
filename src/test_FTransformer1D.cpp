#include<stdio.h>
#include<string>
#include"FTransformer1D.h"

int n_mesh = 16;
double width_signal = 0.2;
CNumber signal_rectangle(double x);
CNumber signal_gaussians(double x);
CNumber signal_multi_tri(double x);

int n_pt = 256;

int main(int argc, char *argv[]) {
    ParallelMPI::func_ini(argc, argv);
    #ifdef _MPI
    fprintf(stdout, "MPI : size = %d, rank = %d\n",
            ParallelMPI::size_, ParallelMPI::rank_);
    #endif

    CNumber (*ptr_func_x)(double);
    FFourier::Transformer1D dft;

    std::string name_rectangle =
        "tab1D_signal_rectangle.txt";
    ptr_func_x = &signal_rectangle;
    dft.init(n_mesh, ptr_func_x);
    dft.export_func_r(name_rectangle, n_pt,
                      ptr_func_x);

    std::string name_gaussians =
        "tab1D_signal_gaussians.txt";
    ptr_func_x = &signal_gaussians;
    dft.init(n_mesh, ptr_func_x);
    dft.export_func_r(name_gaussians, n_pt,
                      ptr_func_x);

    ptr_func_x = &signal_multi_tri;
    dft.init(n_mesh, ptr_func_x);

    if (ParallelMPI::rank_ == 0) {
        fprintf(stdout, "# n_mesh = %d\n", n_mesh);
    }
    for (int ik = 0; ik < n_mesh; ik++) {
        CNumber func_k =
            dft.get_func_k(ik);
        if (ParallelMPI::rank_ == 0) {
            fprintf(stdout, "    %d    %e    %e\n",
                    ik, func_k[0], func_k[1]);
        }
    }

    ParallelMPI::func_fin();

    return 0;
}

CNumber signal_rectangle(double x) {
    CNumber cnum_ret;

    if (x < width_signal) {
        cnum_ret[0] = 1.;
        cnum_ret[1] = 0.;
    } else {
        cnum_ret[0] = 0.;
        cnum_ret[1] = 1.;
    }

    return cnum_ret;
}

CNumber signal_gaussians(double x) {
    CNumber cnum_ret;
    cnum_ret[1] = 0.;

    cnum_ret[0] =
        (exp(-0.5 * x * x / (width_signal * width_signal)) +
         exp(-0.5 * (1. - x) * (1. - x) /
                    (width_signal * width_signal)));

    return cnum_ret;
}

CNumber signal_multi_tri(double x) {
    CNumber cnum_ret;

    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;
    for (int i = 0; i < 5; i++) {
        double iid = static_cast<double>(i + 1);
        cnum_ret[0] +=
            iid * cos(2. * M_PI * 2. * iid * x);
        cnum_ret[1] +=
            iid * sin(2. * M_PI * 2. * iid * x);
    }

    return cnum_ret;
}
