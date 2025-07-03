//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// conv2AXPYValidCMP.cpp
//
// Code generation for function 'conv2AXPYValidCMP'
//

// Include files
#include "conv2AXPYValidCMP.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Definitions
namespace coder {
    namespace internal {
        void b_conv2AXPYValidCMP(const ::coder::array<double, 2U>& a,
            ::coder::array<double, 2U>& c)
        {
            static const signed char b[9]{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
            array<double, 1U> cj;
            int bij;
            int i;
            int ib;
            int jb;
            int mc;
            mc = a.size(0) - 2;
            if ((a.size(0) - 2 == 0) || (a.size(1) - 2 == 0)) {
                c.set_size(a.size(0) - 2, a.size(1) - 2);
                mc = (a.size(0) - 2) * (a.size(1) - 2);
                for (int ub_loop{ 0 }; ub_loop < mc; ub_loop++) {
                    c[ub_loop] = 0.0;
                }
            }
            else {
                int ub_loop;
                c.set_size(a.size(0) - 2, a.size(1) - 2);
                ub_loop = a.size(1) - 3;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(cj, bij,   \
                                                                    jb, ib, i)

                for (int j = 0; j <= ub_loop; j++) {
                    cj.set_size(mc);
                    for (jb = 0; jb < mc; jb++) {
                        cj[jb] = 0.0;
                    }
                    for (jb = 0; jb < 3; jb++) {
                        for (ib = 0; ib < 3; ib++) {
                            bij = b[(3 * (2 - jb) - ib) + 2];
                            for (i = 0; i < mc; i++) {
                                cj[i] = cj[i] + static_cast<double>(bij) *
                                    a[(i + ib) + a.size(0) * (j + jb)];
                            }
                        }
                    }
                    bij = cj.size(0);
                    for (jb = 0; jb < bij; jb++) {
                        c[jb + c.size(0) * j] = cj[jb];
                    }
                }
            }
        }

        void conv2AXPYValidCMP(const ::coder::array<double, 2U>& a,
            ::coder::array<double, 2U>& c)
        {
            static const signed char b[9]{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };
            array<double, 1U> cj;
            int bij;
            int i;
            int ib;
            int jb;
            int mc;
            mc = a.size(0) - 2;
            if ((a.size(0) - 2 == 0) || (a.size(1) - 2 == 0)) {
                c.set_size(a.size(0) - 2, a.size(1) - 2);
                mc = (a.size(0) - 2) * (a.size(1) - 2);
                for (int ub_loop{ 0 }; ub_loop < mc; ub_loop++) {
                    c[ub_loop] = 0.0;
                }
            }
            else {
                int ub_loop;
                c.set_size(a.size(0) - 2, a.size(1) - 2);
                ub_loop = a.size(1) - 3;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(cj, bij,   \
                                                                    jb, ib, i)

                for (int j = 0; j <= ub_loop; j++) {
                    cj.set_size(mc);
                    for (jb = 0; jb < mc; jb++) {
                        cj[jb] = 0.0;
                    }
                    for (jb = 0; jb < 3; jb++) {
                        for (ib = 0; ib < 3; ib++) {
                            bij = b[(3 * (2 - jb) - ib) + 2];
                            for (i = 0; i < mc; i++) {
                                cj[i] = cj[i] + static_cast<double>(bij) *
                                    a[(i + ib) + a.size(0) * (j + jb)];
                            }
                        }
                    }
                    bij = cj.size(0);
                    for (jb = 0; jb < bij; jb++) {
                        c[jb + c.size(0) * j] = cj[jb];
                    }
                }
            }
        }

    } // namespace internal
} // namespace coder

// End of code generation (conv2AXPYValidCMP.cpp)
