#ifndef _CNUMBER_H_
#define _CNUMBER_H_

#include<math.h>

class CNumber {
  private :

    double *z_;

  public :

    /* constructor
     * (default) */
    CNumber() {
        z_ = new double[2];

        z_[0] = 0.;
        z_[1] = 0.;

        return;
    }

    /* constructor
     * (copy) */
    CNumber(const CNumber &cnum_src) {
        z_ = new double[2];

        z_[0] = cnum_src.z_[0];
        z_[1] = cnum_src.z_[1];

        return;
    }

    // destructor
    ~CNumber() {
        delete [] z_;

        return;
    }

    CNumber &operator=(const CNumber &cnum_src) {
        z_[0] = cnum_src.z_[0];
        z_[1] = cnum_src.z_[1];

        return *this;
    }

    double &operator[](int i) {return z_[i];}

    void set(double *z_in) {
        z_[0] = z_in[0];
        z_[1] = z_in[1];

        return;
    }

    void set(double z_in_re, double z_in_im) {
        z_[0] = z_in_re;
        z_[1] = z_in_im;

        return;
    }

    void set_re(double z_in_re) {z_[0] = z_in_re;}
    void set_im(double z_in_im) {z_[1] = z_in_im;}

    double get_re() {return z_[0];}
    double get_im() {return z_[1];}

    double get_abs() {
        double abs =
            sqrt(z_[0] * z_[0] + z_[1] * z_[1]);

        return abs;
    }

    double get_abs2() {
        double abs2 =
            z_[0] * z_[0] + z_[1] * z_[1];

        return abs2;
    }

    double get_ang_azi() {
        double abs = get_abs();
        double sign_azi = 1.;
        if (z_[1] < 0.) {
            sign_azi = -1.;
        }

        return sign_azi * acos(z_[0] / abs);
    }

    CNumber get_conjugate() {
        CNumber cnum_ret;
        cnum_ret.z_[0] = z_[0];
        cnum_ret.z_[1] = -z_[1];

        return cnum_ret;
    }

    CNumber get_inverse() {
        double abs2 = get_abs2();

        CNumber cnum_ret;
        cnum_ret.z_[0] = z_[0] / abs2;
        cnum_ret.z_[1] = -z_[1] / abs2;

        return cnum_ret;
    }

    friend CNumber operator+(double num1, CNumber cnum2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = num1 + cnum2.z_[0];
        cnum_ret.z_[1] = cnum2.z_[1];

        return cnum_ret;
    }

    friend CNumber operator+(CNumber cnum1, double num2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = cnum1.z_[0] + num2;
        cnum_ret.z_[1] = cnum1.z_[1];

        return cnum_ret;
    }

    friend CNumber operator+(CNumber cnum1, CNumber cnum2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = cnum1.z_[0] + cnum2.z_[0];
        cnum_ret.z_[1] = cnum1.z_[1] + cnum2.z_[1];

        return cnum_ret;
    }

    friend CNumber operator-(double num1, CNumber cnum2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = num1 - cnum2.z_[0];
        cnum_ret.z_[1] = -cnum2.z_[1];

        return cnum_ret;
    }

    friend CNumber operator-(CNumber cnum1, double num2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = cnum1.z_[0] - num2;
        cnum_ret.z_[1] = cnum1.z_[1];

        return cnum_ret;
    }

    friend CNumber operator-(CNumber cnum1, CNumber cnum2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = cnum1.z_[0] - cnum2.z_[0];
        cnum_ret.z_[1] = cnum1.z_[1] - cnum2.z_[1];

        return cnum_ret;
    }

    friend CNumber operator*(double num1, CNumber cnum2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = num1 * cnum2.z_[0];
        cnum_ret.z_[1] = num1 * cnum2.z_[1];

        return cnum_ret;
    }

    friend CNumber operator*(CNumber cnum1, double num2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = cnum1.z_[0] * num2;
        cnum_ret.z_[1] = cnum1.z_[1] * num2;

        return cnum_ret;
    }

    friend CNumber operator*(CNumber cnum1, CNumber cnum2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] =
            cnum1.z_[0] * cnum2.z_[0] -
            cnum1.z_[1] * cnum2.z_[1];
        cnum_ret.z_[1] =
            cnum1.z_[1] * cnum2.z_[0] +
            cnum1.z_[0] * cnum2.z_[1];

        return cnum_ret;
    }

    friend CNumber operator/(double num1, CNumber cnum2) {
        CNumber cnum_ret;

        double abs2_cnum2 = cnum2.get_abs2();

        cnum_ret.z_[0] =
            num1 * cnum2.z_[0] / abs2_cnum2;
        cnum_ret.z_[1] =
            -num1 * cnum2.z_[1] / abs2_cnum2;

        return cnum_ret;
    }

    friend CNumber operator/(CNumber cnum1, double num2) {
        CNumber cnum_ret;

        cnum_ret.z_[0] = cnum1.z_[0] / num2;
        cnum_ret.z_[1] = cnum1.z_[1] / num2;

        return cnum_ret;
    }

    friend CNumber operator/(CNumber cnum1, CNumber cnum2) {
        CNumber cnum_ret;

        double abs2_cnum2 = cnum2.get_abs2();

        cnum_ret.z_[0] =
            (cnum1.z_[0] * cnum2.z_[0] +
             cnum1.z_[1] * cnum2.z_[1]) / abs2_cnum2;
        cnum_ret.z_[1] =
            (cnum1.z_[1] * cnum2.z_[0] -
             cnum1.z_[0] * cnum2.z_[1]) / abs2_cnum2;

        return cnum_ret;
    }

    // integer power
    friend CNumber operator^(CNumber cnum1, int n) {
        int sign = n > 0 ? 1 : -1;
        int abs_n = sign * n;

        CNumber cnum_ret;
        cnum_ret.set(1., 0.);

        CNumber cnum_prod;
        if (sign > 0) {
            cnum_prod = cnum1;
        } else {
            cnum_prod = cnum1.get_inverse();
        }

        for (int i = 0; i < abs_n; i++) {
            cnum_ret = cnum_ret * cnum_prod;
        }

        return cnum_ret;
    }
};

#endif
