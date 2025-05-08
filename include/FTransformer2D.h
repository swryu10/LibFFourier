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

    /* function to perform DFT
     * which is called in init function */
    void make();

  public :

    // constructor
    Transformer2D() {
        initialized_ = false;

        return;
    }

    // destructor
    ~Transformer2D() {
        reset();

        return;
    }

    /* initialize and perform DFT
     *
     * num_in_mesh_x : number of mesh bins in x
     * num_in_mesh_y : number of mesh bins in y
     *   num_mesh_x_ = num_in_mesh_x
     *   num_mesh_y_ = num_in_mesh_y
     * mesh_in_func_r : array for the tabulated function
     * For irx = 0 ... num_mesh_x_ - 1 and
     *     iry = 0 ... num_mesh_y_ - 1,
     *   mesh_func_r_[irx][iry] = mesh_in_func_r[irx][iry] */
    void init(int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber **mesh_in_func_r);
    /* ptr_in_func_x : pointer to the function
     *                 to be Fourier-transformed
     * For irx = 0 ... num_mesh_x_ - 1 and
     *     iry = 0 ... num_mesh_y_ - 1,
     *   mesh_func_r_[irx][iry] = (*ptr_in_func_x)(x, y)
     *                            at x = irx / num_mesh_x_ and
     *                               y = iry / num_mesh_y_ */
    void init(int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber (*ptr_in_func_r)(double,
                                       double));

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
