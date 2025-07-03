//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// minOrMax.h
//
// Code generation for function 'minOrMax'
//

#ifndef MINORMAX_H
#define MINORMAX_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "omp.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder {
	namespace internal {
		float maximum(const ::coder::array<float, 1U>& x);

		double maximum(const double x_data[], int x_size);

		double maximum(const double x_data[], const int x_size[2]);

	} // namespace internal
} // namespace coder

#endif
// End of code generation (minOrMax.h)
