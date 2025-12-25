//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// bwconncomp.h
//
// Code generation for function 'bwconncomp'
//

#ifndef BWCONNCOMP_H
#define BWCONNCOMP_H

// Include files
#include "coder_array.h"
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>
#include <omp.h>


// Function Declarations
namespace coder {
void bwconncomp(const ::coder::array<bool, 2U> &varargin_1,
                double *CC_Connectivity, double CC_ImageSize[2],
                double *CC_NumObjects,
                ::coder::array<double, 1U> &CC_RegionIndices,
                ::coder::array<int, 1U> &CC_RegionLengths);

}

#endif
// End of code generation (bwconncomp.h)
