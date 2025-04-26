#ifndef _FTRANSFORMER1D_H_
#define _FTRANSFORMER1D_H_

#include"CNumber.h"

namespace FFourier {

class Transformer1D {
  private :

    int num_mesh_;

    CNumber *mesh_func_x_;
    CNumber *mesh_func_k_;

    CNumber z_unit_;

    bool initialized_;

  public :

    Transformer1D() {
        initialized_ = false;
    }

    ~Transformer1D() {
        reset();
    }

    void init(int num_in_mesh,
              CNumber *mesh_in_func_x);

    void reset();

    CNumber next(int ik, int num_in_mesh,
                 CNumber *mesh_in_func_x);

    CNumber get_func_x(double x_in);
    CNumber get_func_k(int ik) {return mesh_func_k_[ik];}
};

} // end namespace FFourier

#endif
