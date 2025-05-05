#include<stdio.h>
#include<string>
#ifdef _OPENMP
#include<omp.h>
#endif
#include"FTransformer3D.h"

int n_mesh_z = 16;
int n_mesh_x = 16;
int n_mesh_y = 16;
double width_signal_z = 0.3;
double width_signal_x = 0.2;
double width_signal_y = 0.25;
CNumber signal_rectangle(double z, double x, double y);
CNumber signal_gaussians(double z, double x, double y);
CNumber signal_multi_tri(double z, double x, double y);

int n_pt_x = 64;
int n_pt_y = 64;

int main(int argc, char *argv[]) {
    ParallelMPI::func_ini(argc, argv);
    fprintf(stdout, "MPI : size = %d, rank = %d\n",
            ParallelMPI::size_, ParallelMPI::rank_);

    CNumber (*ptr_func_r)(double, double, double);
    FFourier::Transformer3D dft;

    fprintf(stdout, "  signal_rectangle\n");
    ptr_func_r = &signal_rectangle;
    dft.init(n_mesh_z, n_mesh_x, n_mesh_y,
             ptr_func_r);
    std::string name_rectangle =
        "tab3D_signal_rectangle.txt";
    dft.export_func_r(name_rectangle,
                      n_pt_x, n_pt_y,
                      0, 0.15,
                      ptr_func_r);

    fprintf(stdout, "  signal_gaussians\n");
    ptr_func_r = &signal_gaussians;
    dft.init(n_mesh_z, n_mesh_x, n_mesh_y,
             ptr_func_r);
    std::string name_gaussians =
        "tab3D_signal_gaussians.txt";
    dft.export_func_r(name_gaussians,
                      n_pt_x, n_pt_y,
                      0, 0.15,
                      ptr_func_r);

    fprintf(stdout, "  signal_multi_tri\n");
    ptr_func_r = &signal_multi_tri;
    dft.init(n_mesh_z, n_mesh_x, n_mesh_y,
             ptr_func_r);
    for (int ikz = 0; ikz < n_mesh_z; ikz++) {
        for (int ikx = 0; ikx < n_mesh_x; ikx++) {
            for (int iky = 0; iky < n_mesh_y; iky++) {
                CNumber func_k =
                    dft.get_func_k(ikz, ikx, iky);

                if (func_k.get_abs() > 1.0e-2) {
                    fprintf(stdout,
                            "    ikx = %d, iky = %d\n",
                            ikx, iky);
                    fprintf(stdout,
                            "      func_k = (%e, %e)\n",
                            func_k[0], func_k[1]);
                }
            }
        }
    }

    ParallelMPI::func_fin();

    return 0;
}

CNumber signal_rectangle(double z, double x, double y) {
    CNumber cnum_ret;

    if (z < width_signal_z &&
        x < width_signal_x &&
        y < width_signal_y) {
        cnum_ret[0] = 1.;
        cnum_ret[1] = 0.;
    } else {
        cnum_ret[0] = 0.;
        cnum_ret[1] = 1.;
    }

    return cnum_ret;
}

CNumber signal_gaussians(double z, double x, double y) {
    CNumber cnum_ret;
    cnum_ret[1] = 0.;

    double *func_gauss_z = new double[2];
    func_gauss_z[0] =
        exp(-0.5 * z * z / (width_signal_z * width_signal_z));
    func_gauss_z[1] =
        exp(-0.5 * (1. - z) * (1. - z) /
                   (width_signal_z * width_signal_z));

    double *func_gauss_x = new double[2];
    func_gauss_x[0] =
        exp(-0.5 * x * x / (width_signal_x * width_signal_x));
    func_gauss_x[1] =
        exp(-0.5 * (1. - x) * (1. - x) /
                   (width_signal_x * width_signal_x));

    double *func_gauss_y = new double[2];
    func_gauss_y[0] =
        exp(-0.5 * y * y / (width_signal_y * width_signal_y));
    func_gauss_y[1] =
        exp(-0.5 * (1. - y) * (1. - y) /
                   (width_signal_y * width_signal_y));

    cnum_ret[0] = 1.;
    for (int j = 0; j < 8; j++) {
        int jz = j % 2;
        int jxy = (j - jz) / 2;
        int jx = jxy % 2;
        int jy = (jxy - jx) / 2;

        cnum_ret[0] -=
            func_gauss_z[jz] * func_gauss_x[jx] * func_gauss_y[jy];
    }

    return cnum_ret;
}

CNumber signal_multi_tri(double z, double x, double y) {
    CNumber cnum_ret;

    cnum_ret[0] = 0.;
    cnum_ret[1] = 0.;
    for (int i = 0; i < 5; i++) {
        double iidx = static_cast<double>(i + 1);
        double iidy = 6. - iidx;
        cnum_ret[0] +=
            cos(2. * M_PI * 4. *
                (iidx * x + iidy * y));
        cnum_ret[1] +=
            sin(2. * M_PI * 4. *
                (iidx * x + iidy * y));
    }

    return cnum_ret;
}
