//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// find.h
//
// Code generation for function 'find'
//

#ifndef FIND_H
#define FIND_H

// Include files
#include "coder_array.h"
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>
#include <omp.h>


// Function Declarations
namespace coder {
void eml_find(const ::coder::array<bool, 2U> &x, ::coder::array<int, 1U> &i,
              ::coder::array<int, 1U> &j);

}

#endif
// End of code generation (find.h)
