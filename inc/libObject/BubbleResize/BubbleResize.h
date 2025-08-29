//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// BubbleResize.h
//
// Code generation for function 'BubbleResize'
//

#ifndef BUBBLERESIZE_H
#define BUBBLERESIZE_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "omp.h"
#include <cstddef>
#include <cstdlib>

#include "ResizeBubble_data.h"
#include "imresize.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>

#include "Matrix.h"

#ifdef fpclassify
#undef fpclassify
#endif

// Type Definitions
class BubbleResize {
public:
  BubbleResize();
  ~BubbleResize();
  Image& ResizeBubble(Image const& b_img, int d_b, double b_img_max = 255);
};

#endif
// End of code generation (BubbleResize.h)
