//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// regionprops.h
//
// Code generation for function 'regionprops'
//

#ifndef REGIONPROPS_H
#define REGIONPROPS_H

// Include files
#include "coder_array.h"
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>
#include <omp.h>


// Type Declarations
struct struct_T;

// Function Declarations
namespace coder {
void regionprops(const ::coder::array<bool, 2U> &varargin_1,
                 const ::coder::array<double, 2U> &varargin_2,
                 ::coder::array<struct_T, 1U> &outstats);

}

#endif
// End of code generation (regionprops.h)
