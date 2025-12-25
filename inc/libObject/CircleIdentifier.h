//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: CircleIdentifier.h
//
// MATLAB Coder version            : 23.2
// C/C++ source code generated on  : 03-Jul-2025 18:08:54
//

#ifndef CIRCLEIDENTIFIER_H
#define CIRCLEIDENTIFIER_H

// Include Files
#include <cstddef>
#include <cstdlib>
#include <omp.h>


#include "BubbleCenterAndSizeByCircle_data.h"
#include "BubbleCenterAndSizeByCircle_internal_types.h"
#include "NeighborhoodProcessor.h"
#include "chaccum.h"
#include "coder_array.h"
#include "imhmax.h"
#include "medfilt2.h"
#include "regionprops.h"
#include "rt_defines.h"
#include "rt_nonfinite.h"
#include "sort.h"
#include <cmath>
#include <cstring>
#include <math.h>


#include "Matrix.h"
#include <algorithm>
#include <vector>

// Type Definitions
class CircleIdentifier {
public:
  CircleIdentifier(Image const &img_input);
  ~CircleIdentifier();
  std::vector<double> BubbleCenterAndSizeByCircle(std::vector<Pt2D> &center,
                                                  std::vector<double> &radius,
                                                  double rmin, double rmax,
                                                  double sense);

private:
  coder::array<unsigned char, 2U> img;
};

#endif
//
// File trailer for CircleIdentifier.h
//
// [EOF]
//
