#ifndef _FTRANSFORMER2D_H_
#define _FTRANSFORMER2D_H_

#include<string>
#include"CNumber.h"

namespace FFourier {

class Transformer2D {
  private :

    int num_mesh_x_;
    int num_mesh_y_;

    CNumber **mesh_func_rx_ry_;
    CNumber **mesh_func_kx_ky_;

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
              CNumber **mesh_in_func_rx_ry);
    void init(int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber (*ptr_in_func_rx_ry)(double,
                                           double));

    void make();

    void reset();

    CNumber get_func_rx_ry(double x_in,
                           double y_in);
    CNumber get_func_rx_ry(int irx, int iry) {
        return mesh_func_rx_ry_[irx][iry];
    }

    CNumber get_func_kx_ky(int ikx, int iky) {
        return mesh_func_kx_ky_[ikx][iky];
    }
};

} // end namespace FFourier

#endif
