#ifndef _FTRANSFORMER3D_H_
#define _FTRANSFORMER3D_H_

#include<string>
#include"CNumber.h"

namespace FFourier {

class Transformer3D {
  private:

    /* number of bins
     *   num_mesh_z_ in z
     *   num_mesh_x_ in x
     *   num_mesh_y_ in y
     * same for spatial (time) domain and
     * wavevector (frequency) domain */
    int num_mesh_z_;
    int num_mesh_x_;
    int num_mesh_y_;

    /* midpoints in z, x and y
     * used to internal calculations
     *   num_mmid_z_ = num_mesh_z_ / 2
     *   num_mmid_x_ = num_mesh_x_ / 2
     *   num_mmid_y_ = num_mesh_y_ / 2 */
    int num_mmid_z_;
    int num_mmid_x_;
    int num_mmid_y_;

    /* number of bins in each MPI processor
     * used to parallelize with MPI */
    int *list_num_mesh_z_pr_;

    /* function in the sptial (time) domain
     * mesh_func_r_[irz][irx][iry] = f(z, x, y)
     *   at z = irz / num_mesh_z_
     *      x = irx / num_mesh_x_
     *      y = iry / num_mesh_y_
     *   where irz = 0 ... num_mesh_z_ - 1
     *         irx = 0 ... num_mesh_x_ - 1
     *         iry = 0 ... num_mesh_y_ - 1 */
    CNumber ***mesh_func_r_;
    /* wavenumber (frequency) component in each MPI processor
     * mesh_func_k_[ikzpr][ikx][iky]
     *     = the ikzpr/ikx/iky-th component
     *   where ikzpr = 0 ... list_num_mesh_z_pr_[rank] - 1
     *         ikx = 0 ... num_mesh_x_ - 1
     *         iky = 0 ... num_mesh_y_ - 1 */
    CNumber ***mesh_func_k_pr_;

    double factor_inv_;

    CNumber z_unit_z_;
    CNumber z_unit_x_;
    CNumber z_unit_y_;

    bool have_mesh_func_;
    bool initialized_;

    /* function to perform DFT
     * which is called in init function */
    void make();

  public:

    // constructor
    Transformer3D() {
        have_mesh_func_ = false;
        initialized_ = false;

        return;
    }

    // destructor
    ~Transformer3D() {
        reset();

        return;
    }

    /* initialize and perform DFT
     *
     * num_in_mesh_z : number of mesh bins in z
     * num_in_mesh_x : number of mesh bins in x
     * num_in_mesh_y : number of mesh bins in y
     *   num_mesh_z_ = num_in_mesh_z
     *   num_mesh_x_ = num_in_mesh_x
     *   num_mesh_y_ = num_in_mesh_y
     * mesh_in_func_r : array for the tabulated function
     * For irz = 0 ... num_mesh_z_ - 1 and
     *     irx = 0 ... num_mesh_x_ - 1 and
     *     iry = 0 ... num_mesh_y_ - 1,
     *   mesh_func_r_[irz][irx][iry]
     *       = mesh_in_func_r[irz][irx][iry] */
    void init(int num_in_mesh_z,
              int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber ***mesh_in_func_r);
    /* ptr_in_func_x : pointer to the function
     *                 to be Fourier-transformed
     * For irz = 0 ... num_mesh_z_ - 1 and
     *     irx = 0 ... num_mesh_x_ - 1 and
     *     iry = 0 ... num_mesh_y_ - 1,
     *   mesh_func_r_[irz][irx][iry] = (*ptr_in_func_x)(z, x, y)
     *                                 at z = irz / num_mesh_z_ and
     *                                    x = irx / num_mesh_x_ and
     *                                    y = iry / num_mesh_y_ */
    void init(int num_in_mesh_z,
              int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber (*ptr_in_func_r)(double,
                                       double,
                                       double));

    void alloc_mesh_func(int num_in_mesh_z,
                         int num_in_mesh_x,
                         int num_in_mesh_y);

    void export_func_r(std::string name_file,
                       int num_in_pt_x,
                       int num_in_pt_y,
                       int axis_z = 0,
                       double z_plane = 0.5,
                       CNumber (*ptr_in_func_r)(double,
                                                double,
                                                double) = NULL);

    void reset();

    CNumber get_func_r(double z_in,
                       double x_in,
                       double y_in,
                       CNumber *ptr_df_dz = NULL,
                       CNumber *ptr_df_dx = NULL,
                       CNumber *ptr_df_dy = NULL);
    CNumber get_func_r(int irz, int irx, int iry,
                       CNumber *ptr_df_dz = NULL,
                       CNumber *ptr_df_dx = NULL,
                       CNumber *ptr_df_dy = NULL);

    CNumber get_func_k(int ikz, int ikx, int iky);
};

} // end namespace FFourier

#endif
