//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// unique.h
//
// Code generation for function 'unique'
//

#ifndef UNIQUE_H
#define UNIQUE_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "omp.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
    void unique_vector(const ::coder::array<float, 1U>& a,
        ::coder::array<float, 1U>& b);

}

#endif
// End of code generation (unique.h)
