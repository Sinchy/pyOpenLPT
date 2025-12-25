//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// sortIdx.h
//
// Code generation for function 'sortIdx'
//

#ifndef SORTIDX_H
#define SORTIDX_H

// Include files
#include "coder_array.h"
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>
#include <omp.h>


// Function Declarations
namespace coder {
namespace internal {
void merge_block(::coder::array<int, 2U> &idx, ::coder::array<float, 2U> &x,
                 int offset, int n, int preSortLevel,
                 ::coder::array<int, 1U> &iwork,
                 ::coder::array<float, 1U> &xwork);

void sortIdx(::coder::array<double, 2U> &x, ::coder::array<int, 2U> &idx);

} // namespace internal
} // namespace coder

#endif
// End of code generation (sortIdx.h)
