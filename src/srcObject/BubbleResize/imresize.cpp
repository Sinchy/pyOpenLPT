//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// imresize.cpp
//
// Code generation for function 'imresize'
//

// Include files
#include "imresize.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>

// Function Declarations
static double rt_powd_snf(double u0, double u1);

// Function Definitions
static double rt_powd_snf(double u0, double u1)
{
  double y;
  if (std::isnan(u0) || std::isnan(u1)) {
    y = rtNaN;
  } else {
    double d;
    double d1;
    d = std::abs(u0);
    d1 = std::abs(u1);
    if (std::isinf(u1)) {
      if (d == 1.0) {
        y = 1.0;
      } else if (d > 1.0) {
        if (u1 > 0.0) {
          y = rtInf;
        } else {
          y = 0.0;
        }
      } else if (u1 > 0.0) {
        y = 0.0;
      } else {
        y = rtInf;
      }
    } else if (d1 == 0.0) {
      y = 1.0;
    } else if (d1 == 1.0) {
      if (u1 > 0.0) {
        y = u0;
      } else {
        y = 1.0 / u0;
      }
    } else if (u1 == 2.0) {
      y = u0 * u0;
    } else if ((u1 == 0.5) && (u0 >= 0.0)) {
      y = std::sqrt(u0);
    } else if ((u0 < 0.0) && (u1 > std::floor(u1))) {
      y = rtNaN;
    } else {
      y = std::pow(u0, u1);
    }
  }
  return y;
}

namespace coder {
void b_resizeAlongDim2D(const ::coder::array<unsigned char, 2U> &in,
                        const ::coder::array<double, 2U> &weights,
                        const ::coder::array<int, 2U> &indices,
                        double out_length,
                        ::coder::array<unsigned char, 2U> &out)
{
  double sumVal1;
  int i;
  int i1;
  int inRInd;
  int k;
  int linearInds;
  int outCInd;
  int pixelIndex;
  int pixelIndex_tmp;
  int ub_loop;
  unsigned char u;
  ub_loop = in.size(0) - 1;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
    pixelIndex, linearInds, sumVal1, i, outCInd, i1, k, pixelIndex_tmp, u,     \
    inRInd)

  for (inRInd = 0; inRInd <= ub_loop; inRInd++) {
    i = static_cast<int>(out_length);
    for (outCInd = 0; outCInd < i; outCInd++) {
      sumVal1 = 0.0;
      //  Core - second dimension
      i1 = weights.size(0);
      linearInds = weights.size(0) * outCInd + 1;
      for (k = 0; k < i1; k++) {
        pixelIndex_tmp = (linearInds + k) - 1;
        pixelIndex = (inRInd + (indices[pixelIndex_tmp] - 1) * in.size(0)) + 1;
        sumVal1 +=
            weights[pixelIndex_tmp] * static_cast<double>(in[pixelIndex - 1]);
      }
      sumVal1 = std::round(sumVal1);
      if (sumVal1 < 256.0) {
        if (sumVal1 >= 0.0) {
          u = static_cast<unsigned char>(sumVal1);
        } else {
          u = 0U;
        }
      } else if (sumVal1 >= 256.0) {
        u = MAX_uint8_T;
      } else {
        u = 0U;
      }
      out[inRInd + out.size(0) * outCInd] = u;
    }
  }
}

void contributions(int in_length, double out_length, double scale,
                   double kernel_width, ::coder::array<double, 2U> &weights,
                   ::coder::array<int, 2U> &indices)
{
  array<double, 2U> absx;
  array<double, 2U> absx2;
  array<double, 2U> x;
  array<double, 2U> y;
  array<double, 1U> bsum;
  array<double, 1U> u;
  array<int, 2U> aux;
  array<int, 2U> b_indices;
  array<int, 2U> r;
  array<int, 2U> r1;
  array<int, 1U> left;
  array<bool, 2U> copyCols;
  double k;
  int acoef;
  int b_k;
  int bcoef;
  int lastBlockLength;
  int nblocks;
  int nx;
  int xoffset;
  int yk;
  //  Contributions, using pixel indices
  if (scale < 1.0) {
    kernel_width = 4.0 / scale;
  }
  if (std::isnan(out_length)) {
    y.set_size(1, 1);
    y[0] = rtNaN;
  } else if (out_length < 1.0) {
    y.set_size(1, 0);
  } else if (std::isinf(out_length) && (1.0 == out_length)) {
    y.set_size(1, 1);
    y[0] = rtNaN;
  } else {
    yk = static_cast<int>(std::floor(out_length - 1.0));
    y.set_size(1, yk + 1);
    for (xoffset = 0; xoffset <= yk; xoffset++) {
      y[xoffset] = static_cast<double>(xoffset) + 1.0;
    }
  }
  k = 0.5 * (1.0 - 1.0 / scale);
  u.set_size(y.size(1));
  yk = y.size(1);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    u[xoffset] = y[xoffset] / scale + k;
  }
  k = kernel_width / 2.0;
  bsum.set_size(u.size(0));
  yk = u.size(0);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    bsum[xoffset] = u[xoffset] - k;
  }
  nx = bsum.size(0);
  for (b_k = 0; b_k < nx; b_k++) {
    bsum[b_k] = std::floor(bsum[b_k]);
  }
  left.set_size(bsum.size(0));
  yk = bsum.size(0);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    left[xoffset] = static_cast<int>(bsum[xoffset]);
  }
  nx = static_cast<int>(std::ceil(kernel_width) + 2.0);
  if (nx - 1 < 0) {
    nx = 0;
  }
  aux.set_size(1, nx);
  if (nx > 0) {
    aux[0] = 0;
    yk = 0;
    for (b_k = 2; b_k <= nx; b_k++) {
      yk++;
      aux[b_k - 1] = yk;
    }
  }
  indices.set_size(left.size(0), aux.size(1));
  if ((left.size(0) != 0) && (aux.size(1) != 0)) {
    bcoef = (aux.size(1) != 1);
    xoffset = aux.size(1) - 1;
    for (b_k = 0; b_k <= xoffset; b_k++) {
      nx = bcoef * b_k;
      acoef = (left.size(0) != 1);
      lastBlockLength = indices.size(0) - 1;
      for (nblocks = 0; nblocks <= lastBlockLength; nblocks++) {
        indices[nblocks + indices.size(0) * b_k] =
            left[acoef * nblocks] + aux[nx];
      }
    }
  }
  absx.set_size(indices.size(0), indices.size(1));
  yk = indices.size(0) * indices.size(1);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    absx[xoffset] = indices[xoffset];
  }
  nx = absx.size(0);
  yk = u.size(0);
  if (nx < yk) {
    yk = nx;
  }
  if (absx.size(0) == 1) {
    yk = u.size(0);
  } else if (u.size(0) == 1) {
    yk = absx.size(0);
  } else if (u.size(0) == absx.size(0)) {
    yk = u.size(0);
  }
  x.set_size(yk, absx.size(1));
  nx = absx.size(0);
  yk = u.size(0);
  if (nx < yk) {
    yk = nx;
  }
  if (absx.size(0) == 1) {
    yk = u.size(0);
  } else if (u.size(0) == 1) {
    yk = absx.size(0);
  } else if (u.size(0) == absx.size(0)) {
    yk = u.size(0);
  }
  if ((yk != 0) && (absx.size(1) != 0)) {
    bcoef = (absx.size(1) != 1);
    xoffset = absx.size(1) - 1;
    for (b_k = 0; b_k <= xoffset; b_k++) {
      nx = bcoef * b_k;
      acoef = (u.size(0) != 1);
      yk = (absx.size(0) != 1);
      lastBlockLength = x.size(0) - 1;
      for (nblocks = 0; nblocks <= lastBlockLength; nblocks++) {
        x[nblocks + x.size(0) * b_k] =
            u[acoef * nblocks] - absx[yk * nblocks + absx.size(0) * nx];
      }
    }
  }
  if (scale < 1.0) {
    yk = x.size(0) * x.size(1);
    for (xoffset = 0; xoffset < yk; xoffset++) {
      x[xoffset] = scale * x[xoffset];
    }
  }
  nx = x.size(0) * x.size(1);
  absx.set_size(x.size(0), x.size(1));
  for (b_k = 0; b_k < nx; b_k++) {
    absx[b_k] = std::abs(x[b_k]);
  }
  absx2.set_size(absx.size(0), absx.size(1));
  nx = absx.size(0) * absx.size(1);
  for (b_k = 0; b_k < nx; b_k++) {
    absx2[b_k] = absx[b_k] * absx[b_k];
  }
  weights.set_size(absx.size(0), absx.size(1));
  nx = absx.size(0) * absx.size(1);
  for (b_k = 0; b_k < nx; b_k++) {
    weights[b_k] = rt_powd_snf(absx[b_k], 3.0);
  }
  yk = absx2.size(0) * absx2.size(1);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    absx2[xoffset] = 2.5 * absx2[xoffset];
  }
  yk = weights.size(0) * weights.size(1);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    weights[xoffset] =
        ((1.5 * weights[xoffset] - absx2[xoffset]) + 1.0) *
            static_cast<double>(absx[xoffset] <= 1.0) +
        (((-0.5 * weights[xoffset] + absx2[xoffset]) - 4.0 * absx[xoffset]) +
         2.0) *
            static_cast<double>((1.0 < absx[xoffset]) &&
                                (absx[xoffset] <= 2.0));
  }
  if (scale < 1.0) {
    yk = weights.size(0) * weights.size(1);
    for (xoffset = 0; xoffset < yk; xoffset++) {
      weights[xoffset] = scale * weights[xoffset];
    }
  }
  if ((weights.size(0) == 0) || (weights.size(1) == 0)) {
    u.set_size(weights.size(0));
    yk = weights.size(0);
    for (xoffset = 0; xoffset < yk; xoffset++) {
      u[xoffset] = 0.0;
    }
  } else {
    int xj;
    bcoef = weights.size(0) - 1;
    acoef = weights.size(0) << 10;
    u.set_size(weights.size(0));
    bsum.set_size(weights.size(0));
    if (weights.size(1) <= 1024) {
      nx = weights.size(1);
      lastBlockLength = 0;
      nblocks = 1;
    } else {
      nx = 1024;
      nblocks = weights.size(1) / 1024;
      lastBlockLength = weights.size(1) - (nblocks << 10);
      if (lastBlockLength > 0) {
        nblocks++;
      } else {
        lastBlockLength = 1024;
      }
    }
    for (xj = 0; xj <= bcoef; xj++) {
      u[xj] = weights[xj];
      bsum[xj] = 0.0;
    }
    for (b_k = 2; b_k <= nx; b_k++) {
      xoffset = (b_k - 1) * (bcoef + 1);
      for (xj = 0; xj <= bcoef; xj++) {
        u[xj] = u[xj] + weights[xoffset + xj];
      }
    }
    for (int ib{2}; ib <= nblocks; ib++) {
      nx = (ib - 1) * acoef;
      for (xj = 0; xj <= bcoef; xj++) {
        bsum[xj] = weights[nx + xj];
      }
      if (ib == nblocks) {
        yk = lastBlockLength;
      } else {
        yk = 1024;
      }
      for (b_k = 2; b_k <= yk; b_k++) {
        xoffset = nx + (b_k - 1) * (bcoef + 1);
        for (xj = 0; xj <= bcoef; xj++) {
          bsum[xj] = bsum[xj] + weights[xoffset + xj];
        }
      }
      for (xj = 0; xj <= bcoef; xj++) {
        u[xj] = u[xj] + bsum[xj];
      }
    }
  }
  absx.set_size(weights.size(0), weights.size(1));
  yk = weights.size(0) * weights.size(1);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    absx[xoffset] = weights[xoffset];
  }
  nx = u.size(0);
  yk = weights.size(0);
  if (nx < yk) {
    yk = nx;
  }
  if (u.size(0) == 1) {
    nx = weights.size(0);
  } else if (weights.size(0) == 1) {
    nx = u.size(0);
  } else if (weights.size(0) == u.size(0)) {
    nx = weights.size(0);
  } else {
    nx = yk;
  }
  yk = weights.size(1);
  weights.set_size(nx, yk);
  if ((nx != 0) && (yk != 0)) {
    acoef = (absx.size(1) != 1);
    xoffset = yk - 1;
    for (b_k = 0; b_k <= xoffset; b_k++) {
      nx = acoef * b_k;
      yk = (absx.size(0) != 1);
      bcoef = (u.size(0) != 1);
      lastBlockLength = weights.size(0) - 1;
      for (nblocks = 0; nblocks <= lastBlockLength; nblocks++) {
        weights[nblocks + weights.size(0) * b_k] =
            absx[yk * nblocks + absx.size(0) * nx] / u[bcoef * nblocks];
      }
    }
  }
  //  Create the auxiliary matrix:
  yk = in_length << 1;
  aux.set_size(1, yk);
  aux[0] = 1;
  aux[in_length] = in_length;
  for (nblocks = 2; nblocks <= in_length; nblocks++) {
    aux[nblocks - 1] = aux[nblocks - 2] + 1;
    nx = in_length + nblocks;
    aux[nx - 1] = aux[nx - 2] - 1;
  }
  //  Mirror the out-of-bounds indices using mod:
  xoffset = indices.size(0) * indices.size(1);
  for (nblocks = 0; nblocks < xoffset; nblocks++) {
    k = static_cast<double>(indices[nblocks]) - 1.0;
    if (yk == 0) {
      if (static_cast<double>(indices[nblocks]) - 1.0 == 0.0) {
        k = 0.0;
      }
    } else if (static_cast<double>(indices[nblocks]) - 1.0 == 0.0) {
      k = 0.0;
    } else {
      k = std::fmod(static_cast<double>(indices[nblocks]) - 1.0,
                    static_cast<double>(yk));
      if (k == 0.0) {
        k = 0.0;
      } else if (static_cast<double>(indices[nblocks]) - 1.0 < 0.0) {
        k += static_cast<double>(yk);
      }
    }
    indices[nblocks] = aux[static_cast<int>(k)];
  }
  copyCols.set_size(1, weights.size(1));
  yk = weights.size(1);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    copyCols[xoffset] = false;
  }
  nx = weights.size(1);
  yk = 0;
  for (nblocks = 0; nblocks < nx; nblocks++) {
    bool exitg1;
    bcoef = yk + weights.size(0);
    acoef = yk;
    yk += weights.size(0);
    exitg1 = false;
    while ((!exitg1) && (acoef + 1 <= bcoef)) {
      if ((weights[acoef] == 0.0) || std::isnan(weights[acoef])) {
        acoef++;
      } else {
        copyCols[nblocks] = true;
        exitg1 = true;
      }
    }
  }
  yk = copyCols.size(1) - 1;
  nx = 0;
  for (nblocks = 0; nblocks <= yk; nblocks++) {
    if (copyCols[nblocks]) {
      nx++;
    }
  }
  r.set_size(1, nx);
  nx = 0;
  for (nblocks = 0; nblocks <= yk; nblocks++) {
    if (copyCols[nblocks]) {
      r[nx] = nblocks + 1;
      nx++;
    }
  }
  nx = weights.size(0) - 1;
  absx.set_size(r.size(1), weights.size(0));
  for (xoffset = 0; xoffset <= nx; xoffset++) {
    yk = r.size(1);
    for (lastBlockLength = 0; lastBlockLength < yk; lastBlockLength++) {
      absx[lastBlockLength + absx.size(0) * xoffset] =
          weights[xoffset + weights.size(0) * (r[lastBlockLength] - 1)];
    }
  }
  weights.set_size(absx.size(0), absx.size(1));
  yk = absx.size(0) * absx.size(1);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    weights[xoffset] = absx[xoffset];
  }
  yk = copyCols.size(1) - 1;
  nx = 0;
  for (nblocks = 0; nblocks <= yk; nblocks++) {
    if (copyCols[nblocks]) {
      nx++;
    }
  }
  r1.set_size(1, nx);
  nx = 0;
  for (nblocks = 0; nblocks <= yk; nblocks++) {
    if (copyCols[nblocks]) {
      r1[nx] = nblocks + 1;
      nx++;
    }
  }
  nx = indices.size(0) - 1;
  b_indices.set_size(r1.size(1), indices.size(0));
  for (xoffset = 0; xoffset <= nx; xoffset++) {
    yk = r1.size(1);
    for (lastBlockLength = 0; lastBlockLength < yk; lastBlockLength++) {
      b_indices[lastBlockLength + b_indices.size(0) * xoffset] =
          indices[xoffset + indices.size(0) * (r1[lastBlockLength] - 1)];
    }
  }
  indices.set_size(b_indices.size(0), b_indices.size(1));
  yk = b_indices.size(0) * b_indices.size(1);
  for (xoffset = 0; xoffset < yk; xoffset++) {
    indices[xoffset] = b_indices[xoffset];
  }
}

void resizeAlongDim2D(const ::coder::array<unsigned char, 2U> &in,
                      const ::coder::array<double, 2U> &weights,
                      const ::coder::array<int, 2U> &indices, double out_length,
                      ::coder::array<unsigned char, 2U> &out)
{
  double sumVal1;
  int i;
  int i1;
  int k;
  int linearInds;
  int outRInd;
  int sumVal1_tmp;
  int ub_loop;
  unsigned char u;
  ub_loop = static_cast<int>(static_cast<double>(in.size(0) * in.size(1)) /
                             static_cast<double>(in.size(0))) -
            1;
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
    linearInds, sumVal1, i, outRInd, i1, k, sumVal1_tmp, u)

  for (int inCInd = 0; inCInd <= ub_loop; inCInd++) {
    i = static_cast<int>(out_length);
    for (outRInd = 0; outRInd < i; outRInd++) {
      sumVal1 = 0.0;
      i1 = weights.size(0);
      linearInds = weights.size(0) * outRInd + 1;
      //  Core - first dimension
      for (k = 0; k < i1; k++) {
        sumVal1_tmp = (linearInds + k) - 1;
        sumVal1 += weights[sumVal1_tmp] *
                   static_cast<double>(
                       in[(indices[sumVal1_tmp] + in.size(0) * inCInd) - 1]);
      }
      sumVal1 = std::round(sumVal1);
      if (sumVal1 < 256.0) {
        if (sumVal1 >= 0.0) {
          u = static_cast<unsigned char>(sumVal1);
        } else {
          u = 0U;
        }
      } else if (sumVal1 >= 256.0) {
        u = MAX_uint8_T;
      } else {
        u = 0U;
      }
      out[outRInd + out.size(0) * inCInd] = u;
    }
  }
}

} // namespace coder

// End of code generation (imresize.cpp)
