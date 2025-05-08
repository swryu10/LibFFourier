#ifndef _FTRANSFORMER2D_H_
#define _FTRANSFORMER2D_H_

#include<string>
#include"Parallel.h"
#include"CNumber.h"

namespace FFourier {

class Transformer2D {
  private :

    /* number of bins
     *   num_mesh_x_ in x
     *   num_mesh_y_ in y
     * same for spatial (time) domain and
     * wavevector (frequency) domain */
    int num_mesh_x_;
    int num_mesh_y_;

    /* midpoints in x and y
     * used to internal calculations
     *   num_mmid_x_ = num_mesh_x_ / 2
     *   num_mmid_y_ = num_mesh_y_ / 2 */
    int num_mmid_x_;
    int num_mmid_y_;

    /* function in the sptial (time) domain
     * mesh_func_r_[irx][iry] = f(x, y)
     *   at x = irx / num_mesh_x_
     *      y = iry / num_mesh_y_
     *   where irx = 0 ... num_mesh_x_ - 1
     *         iry = 0 ... num_mesh_y_ - 1 */
    CNumber **mesh_func_r_;
    /* wavenumber (frequency) component
     * mesh_func_k_[ikx][iky]
     *     = the ikx/iky-th component
     *   where ikx = 0 ... num_mesh_x_ - 1
     *         iky = 0 ... num_mesh_y_ - 1 */
    CNumber **mesh_func_k_;

    double factor_inv_;

    bool initialized_;

  public :

    Transformer2D() {
        initialized_ = false;

        return;
    }

    ~Transformer2D() {
        reset();

        return;
    }

    void init(int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber **mesh_in_func_r);
    void init(int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber (*ptr_in_func_r)(double,
                                       double));

    void make();

    void export_func_r(std::string name_file,
                       int num_in_pt_x, int num_in_pt_y,
                       CNumber (*ptr_in_func_r)(double,
                                                double) = NULL);

    void reset();

    CNumber get_func_r(double x_in,
                       double y_in);
    CNumber get_func_r(int irx, int iry) {
        if (ParallelMPI::rank_ != 0) {
            CNumber cnum_ret;
            cnum_ret[0] = 0.;
            cnum_ret[1] = 0.;

            return cnum_ret;
        }

        int jrx = (irx + num_mesh_x_) % num_mesh_x_;
        int jry = (iry + num_mesh_y_) % num_mesh_y_;
        return mesh_func_r_[jrx][jry];
    }

    CNumber get_func_k(int ikx, int iky) {
        if (ParallelMPI::rank_ != 0) {
            CNumber cnum_ret;
            cnum_ret[0] = 0.;
            cnum_ret[1] = 0.;

            return cnum_ret;
        }

        int jkx = (ikx + num_mesh_x_) % num_mesh_x_;
        int jky = (iky + num_mesh_y_) % num_mesh_y_;
        return mesh_func_k_[jkx][jky];
    }
};

} // end namespace FFourier

#endif
