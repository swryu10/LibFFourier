#ifndef _FTRANSFORMER1D_H_
#define _FTRANSFORMER1D_H_

#include<string>
#include"Parallel.h"
#include"CNumber.h"

namespace FFourier {

class Transformer1D {
  private :

    int num_mesh_;
    int *list_num_mesh_pr_;

    CNumber *mesh_func_x_;
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
