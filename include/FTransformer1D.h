#ifndef _FTRANSFORMER1D_H_
#define _FTRANSFORMER1D_H_

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
    /* wavenumber (frequency) component
     * mesh_func_k_[ik] = the ik-th component
     *   where ik = 0 ... num_mesh_ - 1 */
    CNumber *mesh_func_k_;

    double factor_inv_;

    bool initialized_;

  public :

    Transformer1D() {
        initialized_ = false;

        return;
    }

    ~Transformer1D() {
        reset();

        return;
    }

    void init(int num_in_mesh,
              CNumber *mesh_in_func_x);
    void init(int num_in_mesh,
              CNumber (*ptr_in_func_x)(double));

    void make();

    void export_func_r(std::string name_file,
                       int num_in_pt,
                       CNumber (*ptr_in_func_x)(double) = NULL);

    void reset();

    CNumber next(int ik, int num_in_mesh,
                 CNumber *mesh_in_func_x);

    CNumber get_func_r(double x_in);
    CNumber get_func_r(int ix) {
        if (ParallelMPI::rank_ != 0) {
            CNumber cnum_ret;
            cnum_ret[0] = 0.;
            cnum_ret[1] = 0.;

            return cnum_ret;
        }

        int jx = (ix + num_mesh_) % num_mesh_;
        return mesh_func_x_[jx];
    }

    CNumber get_func_k(int ik) {
        if (ParallelMPI::rank_ != 0) {
            CNumber cnum_ret;
            cnum_ret[0] = 0.;
            cnum_ret[1] = 0.;

            return cnum_ret;
        }

        int jk = (ik + num_mesh_) % num_mesh_;
        return mesh_func_k_[jk];
    }
};

} // end namespace FFourier

#endif
