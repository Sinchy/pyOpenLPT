//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// medfilt2.h
//
// Code generation for function 'medfilt2'
//

#ifndef MEDFILT2_H
#define MEDFILT2_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "omp.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
    namespace images {
        namespace internal {
            namespace coder {
                namespace optimized {
                    void medfilt2(::coder::array<double, 2U>& inImg,
                        ::coder::array<double, 2U>& outImg);

                }
            } // namespace coder
        } // namespace internal
    } // namespace images
} // namespace coder

#endif
// End of code generation (medfilt2.h)
