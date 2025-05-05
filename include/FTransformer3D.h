#ifndef _FTRANSFORMER3D_H_
#define _FTRANSFORMER3D_H_

#include<string>
#include"Parallel.h"
#include"CNumber.h"

namespace FFourier {

class Transformer3D {
  private :

    int num_mesh_z_;
    int num_mesh_x_;
    int num_mesh_y_;

    int num_mmid_z_;
    int num_mmid_x_;
    int num_mmid_y_;

    CNumber ***mesh_func_r_;
    CNumber ***mesh_func_k_;

    double factor_inv_;

    bool initialized_;

  public :

    Transformer3D() {
        initialized_ = false;

        return;
    }

    ~Transformer3D() {
        reset();

        return;
    }

    void init(int num_in_mesh_z,
              int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber ***mesh_in_func_r);
    void init(int num_in_mesh_z,
              int num_in_mesh_x,
              int num_in_mesh_y,
              CNumber (*ptr_in_func_r)(double,
                                       double,
                                       double));

    void make();

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
                       double y_in);
    CNumber get_func_r(int irz, int irx, int iry) {
        int jrz = (irz + num_mesh_z_) % num_mesh_z_;
        int jrx = (irx + num_mesh_x_) % num_mesh_x_;
        int jry = (iry + num_mesh_y_) % num_mesh_y_;
        return mesh_func_r_[jrz][jrx][jry];
    }

    CNumber get_func_k(int ikz, int ikx, int iky) {
        int jkz = (ikz + num_mesh_z_) % num_mesh_z_;
        int jkx = (ikx + num_mesh_x_) % num_mesh_x_;
        int jky = (iky + num_mesh_y_) % num_mesh_y_;
        return mesh_func_k_[jkz][jkx][jky];
    }
};

} // end namespace FFourier

#endif
