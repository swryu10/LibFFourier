#ifndef _FTRANSFORMER2D_H_
#define _FTRANSFORMER2D_H_

#include<string>
#include"CNumber.h"

namespace FFourier {

class Transformer2D {
  private :

    int num_mesh_x_;
    int num_mesh_y_;

    int num_mmid_x_;
    int num_mmid_y_;

    CNumber **mesh_func_r_;
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
        int jrx = (irx + num_mesh_x_) % num_mesh_x_;
        int jry = (iry + num_mesh_y_) % num_mesh_y_;
        return mesh_func_r_[jrx][jry];
    }

    CNumber get_func_k(int ikx, int iky) {
        int jkx = (ikx + num_mesh_x_) % num_mesh_x_;
        int jky = (iky + num_mesh_y_) % num_mesh_y_;
        return mesh_func_k_[jkx][jky];
    }
};

} // end namespace FFourier

#endif
