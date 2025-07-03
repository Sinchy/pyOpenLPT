//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// imresize.h
//
// Code generation for function 'imresize'
//

#ifndef IMRESIZE_H
#define IMRESIZE_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "omp.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
void b_resizeAlongDim2D(const ::coder::array<unsigned char, 2U> &in,
                        const ::coder::array<double, 2U> &weights,
                        const ::coder::array<int, 2U> &indices,
                        double out_length,
                        ::coder::array<unsigned char, 2U> &out);

void contributions(int in_length, double out_length, double scale,
                   double kernel_width, ::coder::array<double, 2U> &weights,
                   ::coder::array<int, 2U> &indices);

void resizeAlongDim2D(const ::coder::array<unsigned char, 2U> &in,
                      const ::coder::array<double, 2U> &weights,
                      const ::coder::array<int, 2U> &indices, double out_length,
                      ::coder::array<unsigned char, 2U> &out);

} // namespace coder

#endif
// End of code generation (imresize.h)
