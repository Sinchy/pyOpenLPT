//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// nullAssignment.cpp
//
// Code generation for function 'nullAssignment'
//

// Include files
#include "nullAssignment.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Definitions
namespace coder {
    namespace internal {
        void nullAssignment(::coder::array<float, 2U>& x,
            const ::coder::array<bool, 2U>& idx)
        {
            int k;
            int k0;
            int nxin;
            int nxout;
            nxin = x.size(1);
            nxout = 0;
            k0 = idx.size(1);
            for (k = 0; k < k0; k++) {
                nxout += idx[k];
            }
            nxout = x.size(1) - nxout;
            k0 = -1;
            for (k = 0; k < nxin; k++) {
                if ((k + 1 > idx.size(1)) || (!idx[k])) {
                    k0++;
                    x[k0] = x[k];
                }
            }
            if (1 > nxout) {
                nxout = 0;
            }
            x.set_size(x.size(0), nxout);
        }

    } // namespace internal
} // namespace coder

// End of code generation (nullAssignment.cpp)
