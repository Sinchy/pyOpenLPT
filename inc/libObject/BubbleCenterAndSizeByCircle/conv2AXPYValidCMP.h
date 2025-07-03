//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// conv2AXPYValidCMP.h
//
// Code generation for function 'conv2AXPYValidCMP'
//

#ifndef CONV2AXPYVALIDCMP_H
#define CONV2AXPYVALIDCMP_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "omp.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
    namespace internal {
        void b_conv2AXPYValidCMP(const ::coder::array<double, 2U>& a,
            ::coder::array<double, 2U>& c);

        void conv2AXPYValidCMP(const ::coder::array<double, 2U>& a,
            ::coder::array<double, 2U>& c);

    } // namespace internal
} // namespace coder

#endif
// End of code generation (conv2AXPYValidCMP.h)
