//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// multithresh.cpp
//
// Code generation for function 'multithresh'
//

// Include files
#include "multithresh.h"
#include "BubbleCenterAndSizeByCircle_data.h"
#include "minOrMax.h"
#include "nullAssignment.h"
#include "rt_nonfinite.h"
#include "sort.h"
#include "unique.h"
#include "coder_array.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <math.h>

// Function Declarations
namespace coder {
    static void getpdfOptimized(const ::coder::array<float, 2U>& A, double p[256],
        float* minA, float* maxA, bool* emptyp);

}

// Function Definitions
namespace coder {
    static void getpdfOptimized(const ::coder::array<float, 2U>& A, double p[256],
        float* minA, float* maxA, bool* emptyp)
    {
        array<float, 1U> b_A;
        array<unsigned char, 2U> Ascaled;
        double localBins1[256];
        double localBins2[256];
        double localBins3[256];
        double nanCount;
        double nanCountPrime;
        float b_val;
        float difference;
        float minAPrime;
        float val;
        int N;
        int b_i;
        int b_k;
        int i;
        int k;
        bool exitg1;
        bool useParfor;
        *emptyp = true;
        Ascaled.set_size(A.size(0), A.size(1));
        i = A.size(0) * A.size(1);
        for (b_i = 0; b_i < i; b_i++) {
            Ascaled[b_i] = 0U;
        }
        nanCount = 0.0;
        N = A.size(0) * A.size(1) - 1;
        useParfor = (50000 < N + 1);
        *minA = rtInfF;
        *maxA = rtMinusInfF;
        if (useParfor) {
#pragma omp parallel num_threads(omp_get_max_threads()) private(               \
    nanCountPrime, val, minAPrime, b_k)
        {
            minAPrime = rtInfF;
            val = rtMinusInfF;
            nanCountPrime = 0.0;
#pragma omp for nowait
            for (b_k = 0; b_k <= N; b_k++) {
                if ((!std::isinf(A[b_k])) && (!std::isnan(A[b_k]))) {
                    minAPrime = std::fmin(A[b_k], minAPrime);
                    val = std::fmax(A[b_k], val);
                }
                else if (std::isnan(A[b_k])) {
                    nanCountPrime++;
                }
            }
            //omp_set_nest_lock(&emlrtNestLockGlobal);
#pragma omp critical 
            {

                *minA = std::fmin(minAPrime, *minA);
                *maxA = std::fmax(val, *maxA);
                nanCount += nanCountPrime;
            }
            //omp_unset_nest_lock(&emlrtNestLockGlobal);
        }
        }
        else {
            for (k = 0; k <= N; k++) {
                if ((!std::isinf(A[k])) && (!std::isnan(A[k]))) {
                    *minA = std::fmin(A[k], *minA);
                    *maxA = std::fmax(A[k], *maxA);
                }
                else if (std::isnan(A[k])) {
                    nanCount++;
                }
            }
        }
        if (!(*minA == *maxA)) {
            if (std::isinf(*minA)) {
                N = A.size(0) * A.size(1);
                if (A.size(0) * A.size(1) <= 2) {
                    if (A.size(0) * A.size(1) == 1) {
                        *minA = A[0];
                    }
                    else if ((A[0] > A[A.size(0) * A.size(1) - 1]) ||
                        (std::isnan(A[0]) &&
                            (!std::isnan(A[A.size(0) * A.size(1) - 1])))) {
                        *minA = A[A.size(0) * A.size(1) - 1];
                    }
                    else {
                        *minA = A[0];
                    }
                }
                else {
                    if (!std::isnan(A[0])) {
                        i = 1;
                    }
                    else {
                        i = 0;
                        k = 2;
                        exitg1 = false;
                        while ((!exitg1) && (k <= N)) {
                            if (!std::isnan(A[k - 1])) {
                                i = k;
                                exitg1 = true;
                            }
                            else {
                                k++;
                            }
                        }
                    }
                    if (i == 0) {
                        *minA = A[0];
                    }
                    else {
                        *minA = A[i - 1];
                        b_i = i + 1;
                        for (k = b_i; k <= N; k++) {
                            if (*minA > A[k - 1]) {
                                *minA = A[k - 1];
                            }
                        }
                    }
                }
                i = A.size(0) * A.size(1);
                b_A = A.reshape(i);
                *maxA = internal::maximum(b_A);
            }
            else {
                difference = *maxA - *minA;
                if (useParfor) {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(val, b_k)

                    for (b_k = 0; b_k <= N; b_k++) {
                        val = (A[b_k] - *minA) / difference * 255.0F;
                        if (val < 0.0F) {
                            Ascaled[b_k] = 0U;
                        }
                        else if (val > 255.0F) {
                            Ascaled[b_k] = MAX_uint8_T;
                        }
                        else {
                            Ascaled[b_k] = static_cast<unsigned char>(val + 0.5F);
                        }
                    }
                }
                else {
                    for (k = 0; k <= N; k++) {
                        b_val = (A[k] - *minA) / difference * 255.0F;
                        if (b_val < 0.0F) {
                            Ascaled[k] = 0U;
                        }
                        else if (b_val > 255.0F) {
                            Ascaled[k] = MAX_uint8_T;
                        }
                        else {
                            Ascaled[k] = static_cast<unsigned char>(b_val + 0.5F);
                        }
                    }
                }
                if ((Ascaled.size(0) == 0) || (Ascaled.size(1) == 0)) {
                    std::memset(&p[0], 0, 256U * sizeof(double));
                }
                else {
                    std::memset(&p[0], 0, 256U * sizeof(double));
                    std::memset(&localBins1[0], 0, 256U * sizeof(double));
                    std::memset(&localBins2[0], 0, 256U * sizeof(double));
                    std::memset(&localBins3[0], 0, 256U * sizeof(double));
                    for (i = 0; i + 4 <= Ascaled.size(0) * Ascaled.size(1); i += 4) {
                        localBins1[Ascaled[i]]++;
                        localBins2[Ascaled[i + 1]]++;
                        localBins3[Ascaled[i + 2]]++;
                        p[Ascaled[i + 3]]++;
                    }
                    while (i + 1 <= Ascaled.size(0) * Ascaled.size(1)) {
                        p[Ascaled[i]]++;
                        i++;
                    }
                    for (i = 0; i < 256; i++) {
                        p[i] = ((p[i] + localBins1[i]) + localBins2[i]) + localBins3[i];
                    }
                }
                p[0] -= nanCount;
                nanCount = static_cast<double>(N + 1) - nanCount;
                for (b_i = 0; b_i < 256; b_i++) {
                    p[b_i] /= nanCount;
                }
                *emptyp = false;
            }
        }
    }

    float multithresh(const ::coder::array<float, 2U>& varargin_1)
    {
        array<double, 2U> uniqueVals_d;
        array<float, 2U> threshL;
        array<float, 2U> threshout;
        array<float, 1U> b;
        array<float, 1U> b_varargin_1;
        array<bool, 2U> r;
        array<bool, 2U> r1;
        array<bool, 2U> r2;
        array<bool, 1U> x;
        double mu[256];
        double omega[256];
        double sigma_b_squared[256];
        float thresh;
        float threshCandidate;
        int k;
        bool emptyp;
        if ((varargin_1.size(0) == 0) || (varargin_1.size(1) == 0)) {
            if ((varargin_1.size(0) * varargin_1.size(1) != 0) &&
                (1 - varargin_1.size(0) * varargin_1.size(1) > 0)) {
                if (varargin_1[0] > 1.0F) {
                    threshL.set_size(threshL.size(0),
                        varargin_1.size(0) * varargin_1.size(1) + 1);
                }
                else {
                    threshL.set_size(threshL.size(0),
                        varargin_1.size(0) * varargin_1.size(1));
                }
                if (1.0 - static_cast<double>(threshL.size(1)) > 0.0) {
                    int idx;
                    threshout.set_size(1, threshL.size(1) + 1);
                    threshout[threshL.size(1)] = 0.0F;
                    uniqueVals_d.set_size(uniqueVals_d.size(0),
                        varargin_1.size(0) * varargin_1.size(1));
                    threshCandidate = varargin_1[0];
                    idx = 1;
                    while (idx <= 1) {
                        double idxSum;
                        int ix;
                        bool exitg1;
                        threshCandidate++;
                        idxSum = std::abs(static_cast<double>(threshCandidate));
                        if ((!std::isinf(idxSum)) && (!std::isnan(idxSum)) &&
                            (!(idxSum <= 2.2250738585072014E-308))) {
                            frexp(idxSum, &k);
                        }
                        x.set_size(uniqueVals_d.size(1));
                        emptyp = false;
                        ix = 1;
                        exitg1 = false;
                        while ((!exitg1) && (ix <= x.size(0))) {
                            if (!x[ix - 1]) {
                                ix++;
                            }
                            else {
                                emptyp = true;
                                exitg1 = true;
                            }
                        }
                        if (!emptyp) {
                            threshout[threshL.size(1)] = threshCandidate;
                            idx = 2;
                        }
                    }
                    internal::sort(threshout);
                }
            }
            thresh = 1.0F;
        }
        else {
            getpdfOptimized(varargin_1, mu, &thresh, &threshCandidate, &emptyp);
            if (emptyp) {
                if (std::isnan(thresh)) {
                    thresh = 1.0F;
                }
            }
            else {
                double d;
                double idxNum;
                double idxSum;
                double maxval;
                int idx;
                int ix;
                std::copy(&mu[0], &mu[256], &omega[0]);
                for (k = 0; k < 255; k++) {
                    omega[k + 1] += omega[k];
                }
                for (ix = 0; ix < 256; ix++) {
                    mu[ix] *= static_cast<double>(ix) + 1.0;
                }
                for (k = 0; k < 255; k++) {
                    mu[k + 1] += mu[k];
                }
                idxSum = mu[255];
                for (k = 0; k < 256; k++) {
                    d = omega[k];
                    idxNum = idxSum * d - mu[k];
                    mu[k] = idxNum;
                    sigma_b_squared[k] = idxNum * idxNum / (d * (1.0 - d));
                }
                if (!std::isnan(sigma_b_squared[0])) {
                    idx = 1;
                }
                else {
                    bool exitg1;
                    idx = 0;
                    k = 2;
                    exitg1 = false;
                    while ((!exitg1) && (k <= 256)) {
                        if (!std::isnan(sigma_b_squared[k - 1])) {
                            idx = k;
                            exitg1 = true;
                        }
                        else {
                            k++;
                        }
                    }
                }
                if (idx == 0) {
                    maxval = sigma_b_squared[0];
                }
                else {
                    maxval = sigma_b_squared[idx - 1];
                    ix = idx + 1;
                    for (k = ix; k < 257; k++) {
                        d = sigma_b_squared[k - 1];
                        if (maxval < d) {
                            maxval = d;
                        }
                    }
                }
                if ((!std::isinf(maxval)) && (!std::isnan(maxval))) {
                    idxSum = 0.0;
                    idxNum = 0.0;
                    for (idx = 0; idx < 256; idx++) {
                        if (sigma_b_squared[idx] == maxval) {
                            idxSum += static_cast<double>(idx) + 1.0;
                            idxNum++;
                        }
                    }
                    threshout.set_size(1, 1);
                    threshout[0] = static_cast<float>(
                        thresh + (idxSum / idxNum - 1.0) / 255.0 *
                        (static_cast<double>(threshCandidate) - thresh));
                }
                else {
                    idx = varargin_1.size(0) * varargin_1.size(1);
                    b_varargin_1 = varargin_1.reshape(idx);
                    unique_vector(b_varargin_1, b);
                    r.set_size(1, b.size(0));
                    idx = b.size(0);
                    for (ix = 0; ix < idx; ix++) {
                        r[ix] = std::isinf(b[ix]);
                    }
                    r1.set_size(1, b.size(0));
                    idx = b.size(0);
                    for (ix = 0; ix < idx; ix++) {
                        r1[ix] = std::isnan(b[ix]);
                    }
                    threshL.set_size(1, b.size(0));
                    idx = b.size(0);
                    for (ix = 0; ix < idx; ix++) {
                        threshL[ix] = b[ix];
                    }
                    r2.set_size(1, r.size(1));
                    idx = r.size(1);
                    for (ix = 0; ix < idx; ix++) {
                        r2[ix] = (r[ix] || r1[ix]);
                    }
                    internal::nullAssignment(threshL, r2);
                    if (threshL.size(1) == 0) {
                        threshout.set_size(1, 1);
                        threshout[0] = 1.0F;
                    }
                    else {
                        threshout.set_size(1, threshL.size(1));
                        idx = threshL.size(1);
                        for (ix = 0; ix < idx; ix++) {
                            threshout[ix] = threshL[ix];
                        }
                    }
                    idx = varargin_1.size(0) * varargin_1.size(1);
                    b_varargin_1 = varargin_1.reshape(idx);
                    unique_vector(b_varargin_1, b);
                    r.set_size(1, b.size(0));
                    idx = b.size(0);
                    for (ix = 0; ix < idx; ix++) {
                        r[ix] = std::isinf(b[ix]);
                    }
                    r1.set_size(1, b.size(0));
                    idx = b.size(0);
                    for (ix = 0; ix < idx; ix++) {
                        r1[ix] = std::isnan(b[ix]);
                    }
                    threshL.set_size(1, b.size(0));
                    idx = b.size(0);
                    for (ix = 0; ix < idx; ix++) {
                        threshL[ix] = b[ix];
                    }
                    r2.set_size(1, r.size(1));
                    idx = r.size(1);
                    for (ix = 0; ix < idx; ix++) {
                        r2[ix] = (r[ix] || r1[ix]);
                    }
                    internal::nullAssignment(threshL, r2);
                    if (threshL.size(1) <= 1) {
                        if (threshL.size(1) == 0) {
                            threshout.set_size(1, 1);
                            threshout[0] = 1.0F;
                        }
                        else {
                            threshout.set_size(1, 1);
                            threshout[0] = threshL[0];
                        }
                    }
                    else {
                        idxSum = static_cast<double>(threshCandidate) - thresh;
                        threshout.set_size(1, threshout.size(1));
                        idx = threshout.size(1) - 1;
                        for (ix = 0; ix <= idx; ix++) {
                            threshout[ix] =
                                static_cast<float>(thresh + threshout[ix] / 255.0 * idxSum);
                        }
                    }
                }
                thresh = threshout[0];
            }
        }
        return thresh;
    }

} // namespace coder

// End of code generation (multithresh.cpp)
