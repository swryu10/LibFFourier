#ifndef _FTRANSFORMER1D_H_
#define _FTRANSFORMER1D_H_

#include<string>
#include"CNumber.h"

namespace FFourier {

class Transformer1D {
  private :

    int num_mesh_;

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
        int jx = (ix + num_mesh_) % num_mesh_;
        return mesh_func_x_[jx];
    }

    CNumber get_func_k(int ik) {
        int jk = (ik + num_mesh_) % num_mesh_;
        return mesh_func_k_[jk];
    }
};

} // end namespace FFourier

#endif
