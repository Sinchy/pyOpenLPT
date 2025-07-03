//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// imfilter.cpp
//
// Code generation for function 'imfilter'
//

// Include files
#include "imfilter.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Definitions
namespace coder {
    void padImage_outSize(const ::coder::array<float, 2U>& a_tmp,
        const double pad[2], ::coder::array<float, 2U>& a)
    {
        array<double, 2U> y;
        array<int, 2U> idxA;
        array<unsigned int, 2U> idxDir;
        if ((a_tmp.size(0) == 0) || (a_tmp.size(1) == 0)) {
            double sizeA_idx_0;
            double sizeA_idx_1;
            int loop_ub;
            sizeA_idx_0 = static_cast<double>(a_tmp.size(0)) + 2.0 * pad[0];
            sizeA_idx_1 = static_cast<double>(a_tmp.size(1)) + 2.0 * pad[1];
            a.set_size(static_cast<int>(sizeA_idx_0), static_cast<int>(sizeA_idx_1));
            loop_ub = static_cast<int>(sizeA_idx_0) * static_cast<int>(sizeA_idx_1);
            for (int i{ 0 }; i < loop_ub; i++) {
                a[i] = 0.0F;
            }
        }
        else {
            int i;
            int loop_ub;
            unsigned int u;
            if (a_tmp.size(0) + 2U < a_tmp.size(1) + 2U) {
                u = a_tmp.size(1) + 2U;
            }
            else {
                u = a_tmp.size(0) + 2U;
            }
            idxA.set_size(static_cast<int>(u), 2);
            y.set_size(
                1, static_cast<int>(
                    static_cast<double>(static_cast<unsigned int>(a_tmp.size(0))) -
                    1.0) +
                1);
            loop_ub = static_cast<int>(
                static_cast<double>(static_cast<unsigned int>(a_tmp.size(0))) - 1.0);
            for (i = 0; i <= loop_ub; i++) {
                y[i] = static_cast<double>(i) + 1.0;
            }
            idxDir.set_size(1, y.size(1) + 2);
            idxDir[0] = 1U;
            loop_ub = y.size(1);
            for (i = 0; i < loop_ub; i++) {
                idxDir[i + 1] = static_cast<unsigned int>(y[i]);
            }
            idxDir[y.size(1) + 1] = static_cast<unsigned int>(a_tmp.size(0));
            loop_ub = idxDir.size(1);
            for (i = 0; i < loop_ub; i++) {
                idxA[i] = static_cast<int>(idxDir[i]);
            }
            y.set_size(
                1, static_cast<int>(
                    static_cast<double>(static_cast<unsigned int>(a_tmp.size(1))) -
                    1.0) +
                1);
            loop_ub = static_cast<int>(
                static_cast<double>(static_cast<unsigned int>(a_tmp.size(1))) - 1.0);
            for (i = 0; i <= loop_ub; i++) {
                y[i] = static_cast<double>(i) + 1.0;
            }
            idxDir.set_size(1, y.size(1) + 2);
            idxDir[0] = 1U;
            loop_ub = y.size(1);
            for (i = 0; i < loop_ub; i++) {
                idxDir[i + 1] = static_cast<unsigned int>(y[i]);
            }
            idxDir[y.size(1) + 1] = static_cast<unsigned int>(a_tmp.size(1));
            loop_ub = idxDir.size(1);
            for (i = 0; i < loop_ub; i++) {
                idxA[i + idxA.size(0)] = static_cast<int>(idxDir[i]);
            }
            i = static_cast<int>(static_cast<double>(a_tmp.size(1)) + 2.0 * pad[1]);
            a.set_size(
                static_cast<int>(static_cast<double>(a_tmp.size(0)) + 2.0 * pad[0]),
                static_cast<int>(static_cast<double>(a_tmp.size(1)) + 2.0 * pad[1]));
            for (int j{ 0 }; j < i; j++) {
                loop_ub = a.size(0);
                for (int b_i{ 0 }; b_i < loop_ub; b_i++) {
                    a[b_i + a.size(0) * j] =
                        a_tmp[(idxA[b_i] + a_tmp.size(0) * (idxA[j + idxA.size(0)] - 1)) -
                        1];
                }
            }
        }
    }

} // namespace coder

// End of code generation (imfilter.cpp)
