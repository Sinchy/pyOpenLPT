//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// imfilter.h
//
// Code generation for function 'imfilter'
//

#ifndef IMFILTER_H
#define IMFILTER_H

// Include files
#include "coder_array.h"
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>
#include <omp.h>


// Function Declarations
namespace coder {
void padImage_outSize(const ::coder::array<float, 2U> &a_tmp,
                      const double pad[2], ::coder::array<float, 2U> &a);

}

#endif
// End of code generation (imfilter.h)
