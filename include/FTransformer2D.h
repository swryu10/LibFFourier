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

    /* number of bins in each MPI processor
     * used to parallelize with MPI */
    int *list_num_mesh_x_pr_;

    /* function in the sptial (time) domain
     * mesh_func_r_[irx][iry] = f(x, y)
     *   at x = irx / num_mesh_x_
     *      y = iry / num_mesh_y_
     *   where irx = 0 ... num_mesh_x_ - 1
     *         iry = 0 ... num_mesh_y_ - 1 */
    CNumber **mesh_func_r_;
    /* wavenumber (frequency) component in each MPI processor
     * mesh_func_k_[ikxpr][iky]
     *     = the ikxpr/iky-th component
     *   where ikxpr = 0 ... list_num_mesh_x_pr_[rank] - 1
     *         iky = 0 ... num_mesh_y_ - 1 */
    CNumber **mesh_func_k_pr_;

    double factor_inv_;

    CNumber z_unit_x_;
    CNumber z_unit_y_;

    bool have_mesh_func_;
    bool initialized_;

    /* function to perform DFT
     * which is called in init function */
    void make();

  public :

    // constructor
    Transformer2D() {
        have_mesh_func_ = false;
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

    void alloc_mesh_func(int num_in_mesh_x,
                         int num_in_mesh_y);

    void export_func_r(std::string name_file,
                       int num_in_pt_x, int num_in_pt_y,
                       CNumber (*ptr_in_func_r)(double,
                                                double) = NULL);

    void reset();

    CNumber get_func_r(double x_in,
                       double y_in,
                       CNumber *ptr_df_dx = NULL,
                       CNumber *ptr_df_dy = NULL);
    CNumber get_func_r(int irx, int iry,
                       CNumber *ptr_df_dx = NULL,
                       CNumber *ptr_df_dy = NULL);

    CNumber get_func_k(int ikx, int iky);
};

} // end namespace FFourier

#endif
