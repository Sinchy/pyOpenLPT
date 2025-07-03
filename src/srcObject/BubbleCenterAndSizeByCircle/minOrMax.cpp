//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// minOrMax.cpp
//
// Code generation for function 'minOrMax'
//

// Include files
#include "minOrMax.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>

// Function Definitions
namespace coder {
    namespace internal {
        float maximum(const ::coder::array<float, 1U>& x)
        {
            float ex;
            int last;
            last = x.size(0);
            if (x.size(0) <= 2) {
                if (x.size(0) == 1) {
                    ex = x[0];
                }
                else if ((x[0] < x[x.size(0) - 1]) ||
                    (std::isnan(x[0]) && (!std::isnan(x[x.size(0) - 1])))) {
                    ex = x[x.size(0) - 1];
                }
                else {
                    ex = x[0];
                }
            }
            else {
                int idx;
                int k;
                if (!std::isnan(x[0])) {
                    idx = 1;
                }
                else {
                    bool exitg1;
                    idx = 0;
                    k = 2;
                    exitg1 = false;
                    while ((!exitg1) && (k <= last)) {
                        if (!std::isnan(x[k - 1])) {
                            idx = k;
                            exitg1 = true;
                        }
                        else {
                            k++;
                        }
                    }
                }
                if (idx == 0) {
                    ex = x[0];
                }
                else {
                    ex = x[idx - 1];
                    idx++;
                    for (k = idx; k <= last; k++) {
                        float f;
                        f = x[k - 1];
                        if (ex < f) {
                            ex = f;
                        }
                    }
                }
            }
            return ex;
        }

        double maximum(const double x_data[], int x_size)
        {
            double ex;
            if (x_size <= 2) {
                if (x_size == 1) {
                    ex = x_data[0];
                }
                else {
                    ex = x_data[x_size - 1];
                    if ((!(x_data[0] < ex)) && ((!std::isnan(x_data[0])) || std::isnan(ex))) {
                        ex = x_data[0];
                    }
                }
            }
            else {
                int idx;
                int k;
                if (!std::isnan(x_data[0])) {
                    idx = 1;
                }
                else {
                    bool exitg1;
                    idx = 0;
                    k = 2;
                    exitg1 = false;
                    while ((!exitg1) && (k <= x_size)) {
                        if (!std::isnan(x_data[k - 1])) {
                            idx = k;
                            exitg1 = true;
                        }
                        else {
                            k++;
                        }
                    }
                }
                if (idx == 0) {
                    ex = x_data[0];
                }
                else {
                    ex = x_data[idx - 1];
                    idx++;
                    for (k = idx; k <= x_size; k++) {
                        double d;
                        d = x_data[k - 1];
                        if (ex < d) {
                            ex = d;
                        }
                    }
                }
            }
            return ex;
        }

        double maximum(const double x_data[], const int x_size[2])
        {
            double ex;
            int last;
            last = x_size[1];
            if (x_size[1] <= 2) {
                if (x_size[1] == 1) {
                    ex = x_data[0];
                }
                else {
                    ex = x_data[x_size[1] - 1];
                    if ((!(x_data[0] < ex)) && ((!std::isnan(x_data[0])) || std::isnan(ex))) {
                        ex = x_data[0];
                    }
                }
            }
            else {
                int idx;
                int k;
                if (!std::isnan(x_data[0])) {
                    idx = 1;
                }
                else {
                    bool exitg1;
                    idx = 0;
                    k = 2;
                    exitg1 = false;
                    while ((!exitg1) && (k <= last)) {
                        if (!std::isnan(x_data[k - 1])) {
                            idx = k;
                            exitg1 = true;
                        }
                        else {
                            k++;
                        }
                    }
                }
                if (idx == 0) {
                    ex = x_data[0];
                }
                else {
                    ex = x_data[idx - 1];
                    idx++;
                    for (k = idx; k <= last; k++) {
                        double d;
                        d = x_data[k - 1];
                        if (ex < d) {
                            ex = d;
                        }
                    }
                }
            }
            return ex;
        }

    } // namespace internal
} // namespace coder

// End of code generation (minOrMax.cpp)
