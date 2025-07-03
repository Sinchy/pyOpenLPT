//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// chaccum.cpp
//
// Code generation for function 'chaccum'
//

// Include files
#include "chaccum.h"
#include "colon.h"
#include "conv2AXPYValidCMP.h"
#include "find.h"
#include "imfilter.h"
#include "minOrMax.h"
#include "multithresh.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>

// Function Declarations
static float rt_hypotf_snf(float u0, float u1);

// Function Definitions
static float rt_hypotf_snf(float u0, float u1)
{
    float a;
    float y;
    a = std::abs(u0);
    y = std::abs(u1);
    if (a < y) {
        a /= y;
        y *= std::sqrt(a * a + 1.0F);
    }
    else if (a > y) {
        y /= a;
        y = a * std::sqrt(y * y + 1.0F);
    }
    else if (!std::isnan(y)) {
        y = a * 1.41421354F;
    }
    return y;
}

namespace coder {
    void chaccum(const ::coder::array<unsigned char, 2U>& varargin_1,
        const double varargin_2_data[], const int varargin_2_size[2],
        ::coder::array<creal_T, 2U>& accumMatrix,
        ::coder::array<float, 2U>& gradientImg)
    {
        array<creal_T, 2U> Opca;
        array<creal_T, 2U> w;
        array<creal_T, 2U> x;
        array<creal_T, 1U> wkeep;
        array<double, 2U> b_a;
        array<double, 2U> lnR;
        array<double, 2U> r;
        array<double, 2U> radiusRange;
        array<float, 2U> A;
        array<float, 2U> Gx;
        array<float, 2U> Gy;
        array<float, 2U> a;
        array<float, 1U> b_gradientImg;
        array<int, 1U> Ex;
        array<int, 1U> Ey;
        array<int, 1U> idxE;
        array<int, 1U> idxE_chunk;
        array<int, 1U> xckeep;
        array<int, 1U> yckeep;
        array<bool, 2U> inside;
        array<bool, 1U> rows_to_keep;
        double pad[2];
        double radiusRangeIn_data[2];
        int csz_idx_0;
        int csz_idx_1;
        int i;
        int i1;
        int i2;
        int loop_ub;
        int radiusRangeIn_size_idx_1;
        bool exitg1;
        bool y;
        if (varargin_2_size[1] == 2) {
            if (varargin_2_data[0] == varargin_2_data[1]) {
                radiusRangeIn_size_idx_1 = 1;
                radiusRangeIn_data[0] = varargin_2_data[0];
            }
            else {
                radiusRangeIn_size_idx_1 = 2;
                for (i = 0; i < 2; i++) {
                    radiusRangeIn_data[i] = varargin_2_data[i];
                }
            }
        }
        else {
            radiusRangeIn_size_idx_1 = 1;
            radiusRangeIn_data[0] = varargin_2_data[0];
        }
        rows_to_keep.set_size(varargin_1.size(0) * varargin_1.size(1));
        csz_idx_1 = varargin_1.size(0) * varargin_1.size(1);
        for (i = 0; i < csz_idx_1; i++) {
            rows_to_keep[i] = (varargin_1[i] == varargin_1[0]);
        }
        y = true;
        csz_idx_0 = 1;
        exitg1 = false;
        while ((!exitg1) && (csz_idx_0 <= rows_to_keep.size(0))) {
            if (!rows_to_keep[csz_idx_0 - 1]) {
                y = false;
                exitg1 = true;
            }
            else {
                csz_idx_0++;
            }
        }
        if (y) {
            accumMatrix.set_size(varargin_1.size(0), varargin_1.size(1));
            csz_idx_1 = varargin_1.size(0) * varargin_1.size(1);
            for (i = 0; i < csz_idx_1; i++) {
                accumMatrix[i].re = 0.0;
                accumMatrix[i].im = 0.0;
            }
            gradientImg.set_size(varargin_1.size(0), varargin_1.size(1));
            csz_idx_1 = varargin_1.size(0) * varargin_1.size(1);
            for (i = 0; i < csz_idx_1; i++) {
                gradientImg[i] = 0.0F;
            }
        }
        else {
            double apnd;
            double b_r;
            double cdiff;
            double ndbl;
            double xcStep;
            float Gmax;
            float edgeThresh;
            int M;
            int N;
            int k;
            A.set_size(varargin_1.size(0), varargin_1.size(1));
            csz_idx_1 = varargin_1.size(0) * varargin_1.size(1);
            for (i = 0; i < csz_idx_1; i++) {
                A[i] = static_cast<float>(varargin_1[i]) / 255.0F;
            }
            pad[0] = 1.0;
            pad[1] = 1.0;
            if ((A.size(0) == 0) || (A.size(1) == 0)) {
                Gx.set_size(A.size(0), A.size(1));
                csz_idx_1 = A.size(0) * A.size(1);
                for (i = 0; i < csz_idx_1; i++) {
                    Gx[i] = A[i];
                }
            }
            else {
                padImage_outSize(A, pad, a);
                b_a.set_size(a.size(0), a.size(1));
                csz_idx_1 = a.size(0) * a.size(1);
                for (i = 0; i < csz_idx_1; i++) {
                    b_a[i] = a[i];
                }
                internal::conv2AXPYValidCMP(b_a, r);
                Gx.set_size(r.size(0), r.size(1));
                csz_idx_1 = r.size(0) * r.size(1);
                for (i = 0; i < csz_idx_1; i++) {
                    Gx[i] = static_cast<float>(r[i]);
                }
            }
            pad[0] = 1.0;
            pad[1] = 1.0;
            if ((A.size(0) == 0) || (A.size(1) == 0)) {
                Gy.set_size(A.size(0), A.size(1));
                csz_idx_1 = A.size(0) * A.size(1);
                for (i = 0; i < csz_idx_1; i++) {
                    Gy[i] = A[i];
                }
            }
            else {
                padImage_outSize(A, pad, a);
                b_a.set_size(a.size(0), a.size(1));
                csz_idx_1 = a.size(0) * a.size(1);
                for (i = 0; i < csz_idx_1; i++) {
                    b_a[i] = a[i];
                }
                internal::b_conv2AXPYValidCMP(b_a, r);
                Gy.set_size(r.size(0), r.size(1));
                csz_idx_1 = r.size(0) * r.size(1);
                for (i = 0; i < csz_idx_1; i++) {
                    Gy[i] = static_cast<float>(r[i]);
                }
            }
            if (Gx.size(0) <= Gy.size(0)) {
                csz_idx_0 = Gx.size(0);
            }
            else {
                csz_idx_0 = Gy.size(0);
            }
            if (Gx.size(1) <= Gy.size(1)) {
                csz_idx_1 = Gx.size(1);
            }
            else {
                csz_idx_1 = Gy.size(1);
            }
            gradientImg.set_size(csz_idx_0, csz_idx_1);
            csz_idx_0 *= csz_idx_1;
            for (k = 0; k < csz_idx_0; k++) {
                gradientImg[k] = rt_hypotf_snf(Gx[k], Gy[k]);
            }
            csz_idx_0 = gradientImg.size(0) * gradientImg.size(1);
            b_gradientImg = gradientImg.reshape(csz_idx_0);
            Gmax = internal::maximum(b_gradientImg);
            a.set_size(gradientImg.size(0), gradientImg.size(1));
            csz_idx_1 = gradientImg.size(0) * gradientImg.size(1);
            for (i = 0; i < csz_idx_1; i++) {
                a[i] = gradientImg[i] / Gmax;
            }
            edgeThresh = multithresh(a);
            Gmax *= edgeThresh;
            inside.set_size(gradientImg.size(0), gradientImg.size(1));
            csz_idx_1 = gradientImg.size(0) * gradientImg.size(1);
            for (i = 0; i < csz_idx_1; i++) {
                inside[i] = (gradientImg[i] > Gmax);
            }
            eml_find(inside, xckeep, yckeep);
            Ey.set_size(xckeep.size(0));
            csz_idx_1 = xckeep.size(0);
            for (i = 0; i < csz_idx_1; i++) {
                Ey[i] = xckeep[i];
            }
            Ex.set_size(yckeep.size(0));
            csz_idx_1 = yckeep.size(0);
            for (i = 0; i < csz_idx_1; i++) {
                Ex[i] = yckeep[i];
            }
            csz_idx_0 = gradientImg.size(0);
            idxE.set_size(Ey.size(0));
            csz_idx_1 = Ey.size(0);
            for (i = 0; i < csz_idx_1; i++) {
                idxE[i] = Ey[i] + csz_idx_0 * (Ex[i] - 1);
            }
            if (radiusRangeIn_size_idx_1 > 1) {
                if (std::isnan(radiusRangeIn_data[0]) ||
                    std::isnan(radiusRangeIn_data[1])) {
                    radiusRange.set_size(1, 1);
                    radiusRange[0] = rtNaN;
                }
                else if (radiusRangeIn_data[1] < radiusRangeIn_data[0]) {
                    radiusRange.set_size(1, 0);
                }
                else if ((std::isinf(radiusRangeIn_data[0]) ||
                    std::isinf(radiusRangeIn_data[1])) &&
                    (radiusRangeIn_data[0] == radiusRangeIn_data[1])) {
                    radiusRange.set_size(1, 1);
                    radiusRange[0] = rtNaN;
                }
                else {
                    eml_float_colon(radiusRangeIn_data[0], radiusRangeIn_data[1],
                        radiusRange);
                }
            }
            else {
                radiusRange.set_size(1, 1);
                radiusRange[0] = radiusRangeIn_data[0];
            }
            if (radiusRange.size(1) > 1) {
                lnR.set_size(1, radiusRange.size(1));
                csz_idx_1 = radiusRange.size(1);
                for (i = 0; i < csz_idx_1; i++) {
                    lnR[i] = radiusRange[i];
                }
                csz_idx_0 = radiusRange.size(1);
                for (k = 0; k < csz_idx_0; k++) {
                    lnR[k] = std::log(lnR[k]);
                }
                b_r = lnR[0];
                ndbl = lnR[lnR.size(1) - 1] - lnR[0];
                csz_idx_1 = lnR.size(1) - 1;
                lnR.set_size(1, lnR.size(1));
                for (i = 0; i <= csz_idx_1; i++) {
                    lnR[i] = (lnR[i] - b_r) / ndbl * 2.0 * 3.1415926535897931 -
                        3.1415926535897931;
                }
            }
            else {
                lnR.set_size(1, 1);
                lnR[0] = 0.0;
            }
            pad[0] = 1.0;
            pad[1] = lnR.size(1);
            Opca.set_size(1, lnR.size(1));
            x.set_size(1, lnR.size(1));
            csz_idx_1 = lnR.size(1);
            for (i = 0; i < csz_idx_1; i++) {
                x[i].re = lnR[i] * 0.0;
                x[i].im = lnR[i];
            }
            csz_idx_0 = x.size(1);
            for (k = 0; k < csz_idx_0; k++) {
                if (x[k].im == 0.0) {
                    x[k].re = std::exp(x[k].re);
                    x[k].im = 0.0;
                }
                else if (std::isinf(x[k].im) && std::isinf(x[k].re) &&
                    (x[k].re < 0.0)) {
                    x[k].re = 0.0;
                    x[k].im = 0.0;
                }
                else {
                    b_r = std::exp(x[k].re / 2.0);
                    x[k].re = b_r * (b_r * std::cos(x[k].im));
                    x[k].im = b_r * (b_r * std::sin(x[k].im));
                }
            }
            csz_idx_1 = x.size(1);
            for (i = 0; i < csz_idx_1; i++) {
                Opca[i] = x[i];
            }
            Opca.set_size(1, Opca.size(1));
            csz_idx_1 = Opca.size(1) - 1;
            for (i = 0; i <= csz_idx_1; i++) {
                b_r = Opca[i].re;
                apnd = Opca[i].im;
                cdiff = 6.2831853071795862 * radiusRange[i];
                if (apnd == 0.0) {
                    ndbl = b_r / cdiff;
                    b_r = 0.0;
                }
                else if (b_r == 0.0) {
                    ndbl = 0.0;
                    b_r = apnd / cdiff;
                }
                else {
                    ndbl = b_r / cdiff;
                    b_r = apnd / cdiff;
                }
                Opca[i].re = ndbl;
                Opca[i].im = b_r;
            }
            xcStep = std::floor(1.0E+6 / static_cast<double>(radiusRange.size(1)));
            N = A.size(1);
            M = A.size(0);
            accumMatrix.set_size(A.size(0), A.size(1));
            csz_idx_1 = A.size(0) * A.size(1);
            for (i = 0; i < csz_idx_1; i++) {
                accumMatrix[i].re = 0.0;
                accumMatrix[i].im = 0.0;
            }
            i = static_cast<int>((static_cast<double>(Ex.size(0)) + (xcStep - 1.0)) /
                xcStep);
            if (0 <= i - 1) {
                i1 = radiusRange.size(1);
                i2 = radiusRange.size(1);
                pad[0] = static_cast<unsigned int>(A.size(0));
                pad[1] = static_cast<unsigned int>(A.size(1));
                loop_ub = A.size(0) * A.size(1);
            }
            for (int b_i{ 0 }; b_i < i; b_i++) {
                b_r = static_cast<double>(b_i) * xcStep + 1.0;
                csz_idx_0 = static_cast<int>(
                    std::fmin((b_r + xcStep) - 1.0, static_cast<double>(Ex.size(0))));
                if (std::isnan(b_r)) {
                    csz_idx_1 = 1;
                }
                else if (csz_idx_0 < b_r) {
                    csz_idx_1 = 0;
                }
                else if (b_r == b_r) {
                    csz_idx_1 = static_cast<int>(static_cast<double>(csz_idx_0) - b_r) + 1;
                }
                else {
                    ndbl = std::floor((static_cast<double>(csz_idx_0) - b_r) + 0.5);
                    apnd = b_r + ndbl;
                    cdiff = apnd - static_cast<double>(csz_idx_0);
                    if (b_r > csz_idx_0) {
                        radiusRangeIn_size_idx_1 = static_cast<int>(b_r);
                    }
                    else {
                        radiusRangeIn_size_idx_1 = csz_idx_0;
                    }
                    if (std::abs(cdiff) <
                        4.4408920985006262E-16 *
                        static_cast<double>(radiusRangeIn_size_idx_1)) {
                        ndbl++;
                        apnd = csz_idx_0;
                    }
                    else if (cdiff > 0.0) {
                        apnd = b_r + (ndbl - 1.0);
                    }
                    else {
                        ndbl++;
                    }
                    if (ndbl >= 0.0) {
                        csz_idx_0 = static_cast<int>(ndbl);
                    }
                    else {
                        csz_idx_0 = 0;
                    }
                    lnR.set_size(1, csz_idx_0);
                    if ((csz_idx_0 > 0) && (csz_idx_0 > 1)) {
                        lnR[csz_idx_0 - 1] = apnd;
                        csz_idx_1 = (csz_idx_0 - 1) / 2;
                        for (k = 0; k <= csz_idx_1 - 2; k++) {
                            lnR[k + 1] = b_r + (static_cast<double>(k) + 1.0);
                            lnR[(csz_idx_0 - k) - 2] = apnd - (static_cast<double>(k) + 1.0);
                        }
                        if (csz_idx_1 << 1 == csz_idx_0 - 1) {
                            lnR[csz_idx_1] = (b_r + apnd) / 2.0;
                        }
                        else {
                            lnR[csz_idx_1] = b_r + static_cast<double>(csz_idx_1);
                            lnR[csz_idx_1 + 1] = apnd - static_cast<double>(csz_idx_1);
                        }
                    }
                    csz_idx_1 = lnR.size(1);
                }
                xckeep.set_size(csz_idx_1);
                yckeep.set_size(csz_idx_1);
                idxE_chunk.set_size(csz_idx_1);
                for (radiusRangeIn_size_idx_1 = 0; radiusRangeIn_size_idx_1 < csz_idx_1;
                    radiusRangeIn_size_idx_1++) {
                    csz_idx_0 = (static_cast<int>(b_r) + radiusRangeIn_size_idx_1) - 1;
                    xckeep[radiusRangeIn_size_idx_1] = Ex[csz_idx_0];
                    yckeep[radiusRangeIn_size_idx_1] = Ey[csz_idx_0];
                    idxE_chunk[radiusRangeIn_size_idx_1] = idxE[csz_idx_0];
                }
                a.set_size(idxE_chunk.size(0), radiusRange.size(1));
                A.set_size(idxE_chunk.size(0), radiusRange.size(1));
                w.set_size(idxE_chunk.size(0), radiusRange.size(1));
                inside.set_size(idxE_chunk.size(0), radiusRange.size(1));
                rows_to_keep.set_size(idxE_chunk.size(0));
                csz_idx_1 = idxE_chunk.size(0);
                for (k = 0; k < csz_idx_1; k++) {
                    rows_to_keep[k] = false;
                }
                for (radiusRangeIn_size_idx_1 = 0; radiusRangeIn_size_idx_1 < i1;
                    radiusRangeIn_size_idx_1++) {
                    k = idxE_chunk.size(0);
                    for (csz_idx_1 = 0; csz_idx_1 < k; csz_idx_1++) {
                        b_r = radiusRange[radiusRangeIn_size_idx_1];
                        Gmax = static_cast<float>(xckeep[csz_idx_1]) +
                            static_cast<float>(-b_r) *
                            (Gx[idxE_chunk[csz_idx_1] - 1] /
                                gradientImg[idxE_chunk[csz_idx_1] - 1]);
                        if (Gmax > 0.0F) {
                            edgeThresh = Gmax + 0.5F;
                        }
                        else if (Gmax < 0.0F) {
                            edgeThresh = Gmax - 0.5F;
                        }
                        else {
                            edgeThresh = 0.0F;
                        }
                        a[csz_idx_1 + a.size(0) * radiusRangeIn_size_idx_1] = edgeThresh;
                        Gmax = static_cast<float>(yckeep[csz_idx_1]) +
                            static_cast<float>(-b_r) *
                            (Gy[idxE_chunk[csz_idx_1] - 1] /
                                gradientImg[idxE_chunk[csz_idx_1] - 1]);
                        if (Gmax > 0.0F) {
                            Gmax += 0.5F;
                        }
                        else if (Gmax < 0.0F) {
                            Gmax -= 0.5F;
                        }
                        else {
                            Gmax = 0.0F;
                        }
                        A[csz_idx_1 + A.size(0) * radiusRangeIn_size_idx_1] = Gmax;
                        w[csz_idx_1 + w.size(0) * radiusRangeIn_size_idx_1] =
                            Opca[radiusRangeIn_size_idx_1];
                        y = ((edgeThresh >= 1.0F) && (static_cast<double>(edgeThresh) <= N) &&
                            (Gmax >= 1.0F) && (static_cast<double>(Gmax) < M));
                        inside[csz_idx_1 + inside.size(0) * radiusRangeIn_size_idx_1] = y;
                        if (y) {
                            rows_to_keep[csz_idx_1] = true;
                        }
                    }
                }
                xckeep.set_size(a.size(0) * a.size(1));
                yckeep.set_size(A.size(0) * A.size(1));
                wkeep.set_size(w.size(0) * w.size(1));
                csz_idx_0 = -1;
                for (radiusRangeIn_size_idx_1 = 0; radiusRangeIn_size_idx_1 < i2;
                    radiusRangeIn_size_idx_1++) {
                    k = idxE_chunk.size(0);
                    for (csz_idx_1 = 0; csz_idx_1 < k; csz_idx_1++) {
                        if (rows_to_keep[csz_idx_1] &&
                            inside[csz_idx_1 + inside.size(0) * radiusRangeIn_size_idx_1]) {
                            csz_idx_0++;
                            xckeep[csz_idx_0] = static_cast<int>(
                                a[csz_idx_1 + a.size(0) * radiusRangeIn_size_idx_1]);
                            yckeep[csz_idx_0] = static_cast<int>(
                                A[csz_idx_1 + A.size(0) * radiusRangeIn_size_idx_1]);
                            wkeep[csz_idx_0] =
                                w[csz_idx_1 + w.size(0) * radiusRangeIn_size_idx_1];
                        }
                    }
                }
                w.set_size(static_cast<int>(pad[0]), static_cast<int>(pad[1]));
                for (k = 0; k < loop_ub; k++) {
                    w[k].re = 0.0;
                    w[k].im = 0.0;
                }
                for (radiusRangeIn_size_idx_1 = 0; radiusRangeIn_size_idx_1 <= csz_idx_0;
                    radiusRangeIn_size_idx_1++) {
                    w[(yckeep[radiusRangeIn_size_idx_1] +
                        w.size(0) * (xckeep[radiusRangeIn_size_idx_1] - 1)) -
                        1]
                        .re = w[(yckeep[radiusRangeIn_size_idx_1] +
                            w.size(0) * (xckeep[radiusRangeIn_size_idx_1] - 1)) -
                        1]
                        .re +
                        wkeep[radiusRangeIn_size_idx_1].re;
                    w[(yckeep[radiusRangeIn_size_idx_1] +
                        w.size(0) * (xckeep[radiusRangeIn_size_idx_1] - 1)) -
                        1]
                        .im = w[(yckeep[radiusRangeIn_size_idx_1] +
                            w.size(0) * (xckeep[radiusRangeIn_size_idx_1] - 1)) -
                        1]
                        .im +
                        wkeep[radiusRangeIn_size_idx_1].im;
                }
                csz_idx_1 = accumMatrix.size(0) * accumMatrix.size(1);
                for (k = 0; k < csz_idx_1; k++) {
                    accumMatrix[k].re = accumMatrix[k].re + w[k].re;
                    accumMatrix[k].im = accumMatrix[k].im + w[k].im;
                }
            }
        }
    }

} // namespace coder

// End of code generation (chaccum.cpp)
