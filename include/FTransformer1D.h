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

    void shift(double dx);

    void amplify(double fac);

    void export_file(std::string name_file);

    void reset();

    CNumber next(int ik, int num_in_mesh,
                 CNumber *mesh_in_func_x);

    CNumber get_func_x(double x_in);
    CNumber get_func_x(int ix) {return mesh_func_x_[ix];}

    CNumber get_func_k(int ik) {return mesh_func_k_[ik];}
};

} // end namespace FFourier

#endif
