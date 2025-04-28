#include<stdio.h>
#include<string>
#ifdef _OPENMP
#include<omp.h>
#endif
#include"FTransformer2D.h"

int n_mesh_x = 32;
int n_mesh_y = 32;
double width_signal_x = 0.2;
double width_signal_y = 0.25;
CNumber signal_rectangle(double x, double y);
CNumber signal_gaussians(double x, double y);
CNumber signal_multi_tri(double x, double y);

int n_bin_x = 32;
int n_bin_y = 32;

int main(int argc, char *argv[]) {
    CNumber (*ptr_func_rx_ry)(double, double);
    FFourier::Transformer2D dft;
    CNumber *bin_func_rx_ry_ini =
        new CNumber[n_bin_x * n_bin_y];
    CNumber *bin_func_rx_ry_fin =
        new CNumber[n_bin_x * n_bin_y];

    fprintf(stdout, "  signal_rectangle\n");
    ptr_func_rx_ry = &signal_rectangle;
    dft.init(n_mesh_x, n_mesh_y,
             ptr_func_rx_ry);
    #ifdef _OPENMP
    #pragma omp parallel
    {  // parallel code begins
    #endif
        #ifdef _OPENMP
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        #endif

        for (int irx = 0; irx < n_bin_x; irx++) {
            #ifdef _OPENMP
            if (irx % n_thread != tid) {
                continue;
            }
            #endif

            double x_now =
                static_cast<double>(irx) /
                static_cast<double>(n_bin_x);
            for (int iry = 0; iry < n_bin_y; iry++) {
                double y_now =
                    static_cast<double>(iry) /
                    static_cast<double>(n_bin_y);
                bin_func_rx_ry_ini[n_bin_y * irx + iry] =
                    signal_rectangle(x_now, y_now);
                bin_func_rx_ry_fin[n_bin_y * irx + iry] =
                    dft.get_func_rx_ry(x_now, y_now);
            }
        }
    #ifdef _OPENMP
    }  // parallel code ends
    #endif
    FILE *ptr_fout_rectangle =
        fopen("tab2D_signal_rectangle.txt", "w");
    for (int irx = 0; irx < n_bin_x; irx++) {
        double x_now =
            static_cast<double>(irx) /
            static_cast<double>(n_bin_x);
        for (int iry = 0; iry < n_bin_y; iry++) {
            double y_now =
                static_cast<double>(iry) /
                static_cast<double>(n_bin_y);

            CNumber diff_func_rx_ry =
                bin_func_rx_ry_fin[n_bin_y * irx + iry] -
                bin_func_rx_ry_ini[n_bin_y * irx + iry];

            fprintf(ptr_fout_rectangle,
                    "  %e  %e  %e  %e  %e  %e\n",
                    x_now, y_now,
                    bin_func_rx_ry_fin[n_bin_y * irx + iry][0],
                    bin_func_rx_ry_fin[n_bin_y * irx + iry][1],
                    diff_func_rx_ry[0],
                    diff_func_rx_ry[1]);
        }

        fprintf(ptr_fout_rectangle, "\n");
    }
    fclose(ptr_fout_rectangle);

    fprintf(stdout, "  signal_gaussians\n");
    ptr_func_rx_ry = &signal_gaussians;
    dft.init(n_mesh_x, n_mesh_y,
             ptr_func_rx_ry);
    #ifdef _OPENMP
    #pragma omp parallel
    {  // parallel code begins
    #endif
        #ifdef _OPENMP
        int n_thread = omp_get_num_threads();
        int tid = omp_get_thread_num();
        #endif

        for (int irx = 0; irx < n_bin_x; irx++) {
            #ifdef _OPENMP
            if (irx % n_thread != tid) {
                continue;
            }
            #endif

            double x_now =
                static_cast<double>(irx) /
                static_cast<double>(n_bin_x);
            for (int iry = 0; iry < n_bin_y; iry++) {
                double y_now =
                    static_cast<double>(iry) /
                    static_cast<double>(n_bin_y);
                bin_func_rx_ry_ini[n_bin_y * irx + iry] =
                    signal_gaussians(x_now, y_now);
                bin_func_rx_ry_fin[n_bin_y * irx + iry] =
                    dft.get_func_rx_ry(x_now, y_now);
            }
        }
    #ifdef _OPENMP
    }  // parallel code ends
    #endif
    FILE *ptr_fout_gaussians =
        fopen("tab2D_signal_gaussians.txt", "w");
    for (int irx = 0; irx < n_bin_x; irx++) {
        double x_now =
            static_cast<double>(irx) /
            static_cast<double>(n_bin_x);
        for (int iry = 0; iry < n_bin_y; iry++) {
            double y_now =
                static_cast<double>(iry) /
                static_cast<double>(n_bin_y);

            CNumber diff_func_rx_ry =
                bin_func_rx_ry_fin[n_bin_y * irx + iry] -
                bin_func_rx_ry_ini[n_bin_y * irx + iry];

            fprintf(ptr_fout_gaussians,
                    "  %e  %e  %e  %e  %e  %e\n",
                    x_now, y_now,
                    bin_func_rx_ry_fin[n_bin_y * irx + iry][0],
                    bin_func_rx_ry_fin[n_bin_y * irx + iry][1],
                    diff_func_rx_ry[0],
                    diff_func_rx_ry[1]);
        }

        fprintf(ptr_fout_gaussians, "\n");
    }
    fclose(ptr_fout_gaussians);

    fprintf(stdout, "  signal_multi_tri\n");
    ptr_func_rx_ry = &signal_multi_tri;
    dft.init(n_mesh_x, n_mesh_y, ptr_func_rx_ry);
    for (int ikx = 0; ikx < n_mesh_x; ikx++) {
        for (int iky = 0; iky < n_mesh_y; iky++) {
            CNumber func_kx_ky =
                dft.get_func_kx_ky(ikx, iky);

            if (func_kx_ky.get_abs() > 1.0e-2) {
                fprintf(stdout,
                        "    ikx = %d, iky = %d\n",
                        ikx, iky);
                fprintf(stdout,
                        "      func_kx_ky = (%e, %e)\n",
                        func_kx_ky[0], func_kx_ky[1]);
            }
        }
    }

    delete [] bin_func_rx_ry_ini;
    delete [] bin_func_rx_ry_fin;

    return 0;
}

CNumber signal_rectangle(double x, double y) {
    CNumber cnum_ret;

    if (x < width_signal_x &&
        y < width_signal_y) {
        cnum_ret[0] = 1.;
        cnum_ret[1] = 0.;
    } else {
        cnum_ret[0] = 0.;
        cnum_ret[1] = 1.;
    }

    return cnum_ret;
}

CNumber signal_gaussians(double x, double y) {
    CNumber cnum_ret;
    cnum_ret[1] = 0.;

    cnum_ret[0] =
        exp(-0.5 * x * x / (width_signal_x * width_signal_x)
            -0.5 * y * y / (width_signal_y * width_signal_y)) +
        exp(-0.5 * (1. - x) * (1. - x) /
                   (width_signal_x * width_signal_x)
            -0.5 * y * y / (width_signal_y * width_signal_y)) +
        exp(-0.5 * x * x /
                   (width_signal_x * width_signal_x)
            -0.5 * (1. - y) * (1. - y) /
                   (width_signal_y * width_signal_y)) +
        exp(-0.5 * (1. - x) * (1. - x) /
                   (width_signal_x * width_signal_x)
            -0.5 * (1. - y) * (1. - y) /
                   (width_signal_y * width_signal_y));

    return cnum_ret;
}

CNumber signal_multi_tri(double x, double y) {
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
