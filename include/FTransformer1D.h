#ifndef _FTRANSFORMER1D_H_
#define _FTRANSFORMER1D_H_

#include<math.h>
#include<string>
#include"Parallel.h"
#include"CNumber.h"

namespace FFourier {

class Transformer1D {
  private :

    /* number of bins
     * same for spatial (time) domain and
     * wavevector (frequency) domain */
    int num_mesh_;
    /* number of bins in each MPI processor
     * used to parallelize with MPI */
    int *list_num_mesh_pr_;

    /* function in the sptial (time) domain
     * mesh_func_x_[ix] = f(x)
     *   at x = ix / num_mesh_
     *   where ix = 0 ... num_mesh_ - 1 */
    CNumber *mesh_func_x_;
    /* wavenumber (frequency) component in each MPI processor
     * mesh_func_k_pr_[ikpr] = the ikpr-th component
     *   where ikpr = 0 ... list_num_mesh_pr_[rank] - 1 */
    CNumber *mesh_func_k_pr_;

    double factor_inv_;

    CNumber z_unit_;

    bool initialized_;

    /* function to perform DFT
     * which is called in init function */
    void make();

  public :

    // constructor
    Transformer1D() {
        initialized_ = false;

        return;
    }

    // destructor
    ~Transformer1D() {
        reset();

        return;
    }

    /* initialize and perform DFT
     *
     * num_in_mesh : number of mesh bins
     *   num_mesh_ = num_in_mesh
     * mesh_in_func_x : array for the tabulated function
     * For ix = 0 ... num_mesh_ - 1,
     *   mesh_func_x_[ix] = mesh_in_func_x[ix] */
    void init(int num_in_mesh,
              CNumber *mesh_in_func_x);
    /* ptr_in_func_x : pointer to the function
     *                 to be Fourier-transformed
     * For ix = 0 ... num_mesh_ - 1,
     *   mesh_func_x_[ix] = (*ptr_in_func_x)(x)
     *                      at x = ix / num_mesh_ */
    void init(int num_in_mesh,
              CNumber (*ptr_in_func_x)(double));

    void export_func_r(std::string name_file,
                       int num_in_pt,
                       CNumber (*ptr_in_func_x)(double) = NULL);

    void reset();

    CNumber next(int ik, int num_in_mesh,
                 CNumber *mesh_in_func_x);

    CNumber get_func_r(double x_in,
                       CNumber *ptr_df_dx = NULL);
    CNumber get_func_r(int ix,
                       CNumber *ptr_df_dx = NULL);

    CNumber get_func_k(int ik);
};

} // end namespace FFourier

#endif
