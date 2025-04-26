#include"CNumber.h"

CNumber operator+(double num1, CNumber cnum2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = num1 + cnum2.z_[0];
    cnum_ret.z_[1] = cnum2.z_[1];

    return cnum_ret;
}

CNumber operator+(CNumber cnum1, double num2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = cnum1.z_[0] + num2;
    cnum_ret.z_[1] = cnum1.z_[1];

    return cnum_ret;
}

CNumber operator+(CNumber cnum1, CNumber cnum2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = cnum1.z_[0] + cnum2.z_[0];
    cnum_ret.z_[1] = cnum1.z_[1] + cnum2.z_[1];

    return cnum_ret;
}

CNumber operator-(double num1, CNumber cnum2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = num1 - cnum2.z_[0];
    cnum_ret.z_[1] = -cnum2.z_[1];

    return cnum_ret;
}

CNumber operator-(CNumber cnum1, double num2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = cnum1.z_[0] - num2;
    cnum_ret.z_[1] = cnum1.z_[1];

    return cnum_ret;
}

CNumber operator-(CNumber cnum1, CNumber cnum2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = cnum1.z_[0] - cnum2.z_[0];
    cnum_ret.z_[1] = cnum1.z_[1] - cnum2.z_[1];

    return cnum_ret;
}

CNumber operator*(double num1, CNumber cnum2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = num1 * cnum2.z_[0];
    cnum_ret.z_[1] = num1 * cnum2.z_[1];

    return cnum_ret;
}

CNumber operator*(CNumber cnum1, double num2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = cnum1.z_[0] * num2;
    cnum_ret.z_[1] = cnum1.z_[1] * num2;

    return cnum_ret;
}

CNumber operator*(CNumber cnum1, CNumber cnum2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] =
        cnum1.z_[0] * cnum2.z_[0] -
        cnum1.z_[1] * cnum2.z_[1];
    cnum_ret.z_[1] =
        cnum1.z_[1] * cnum2.z_[0] +
        cnum1.z_[0] * cnum2.z_[1];

    return cnum_ret;
}

CNumber operator/(double num1, CNumber cnum2) {
    CNumber cnum_ret;

    double abs2_cnum2 = cnum2.get_abs2();

    cnum_ret.z_[0] =
        num1 * cnum2.z_[0] / abs2_cnum2;
    cnum_ret.z_[1] =
        -num1 * cnum2.z_[1] / abs2_cnum2;

    return cnum_ret;
}

CNumber operator/(CNumber cnum1, double num2) {
    CNumber cnum_ret;

    cnum_ret.z_[0] = cnum1.z_[0] / num2;
    cnum_ret.z_[1] = cnum1.z_[1] / num2;

    return cnum_ret;
}

CNumber operator/(CNumber cnum1, CNumber cnum2) {
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

CNumber operator^(CNumber cnum1, int n) {
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
