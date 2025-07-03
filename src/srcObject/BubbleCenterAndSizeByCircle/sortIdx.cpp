//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// sortIdx.cpp
//
// Code generation for function 'sortIdx'
//

// Include files
#include "sortIdx.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>

// Function Declarations
namespace coder {
    namespace internal {
        static void merge(::coder::array<int, 2U>& idx, ::coder::array<float, 2U>& x,
            int offset, int np, int nq, ::coder::array<int, 1U>& iwork,
            ::coder::array<float, 1U>& xwork);

        static void merge(::coder::array<int, 1U>& idx, ::coder::array<double, 1U>& x,
            int offset, int np, int nq, ::coder::array<int, 1U>& iwork,
            ::coder::array<double, 1U>& xwork);

        static void merge_block(::coder::array<int, 1U>& idx,
            ::coder::array<double, 1U>& x, int offset, int n,
            int preSortLevel, ::coder::array<int, 1U>& iwork,
            ::coder::array<double, 1U>& xwork);

    } // namespace internal
} // namespace coder

// Function Definitions
namespace coder {
    namespace internal {
        static void merge(::coder::array<int, 2U>& idx, ::coder::array<float, 2U>& x,
            int offset, int np, int nq, ::coder::array<int, 1U>& iwork,
            ::coder::array<float, 1U>& xwork)
        {
            if (nq != 0) {
                int iout;
                int j;
                int n_tmp;
                int p;
                int q;
                n_tmp = np + nq;
                for (j = 0; j < n_tmp; j++) {
                    iout = offset + j;
                    iwork[j] = idx[iout];
                    xwork[j] = x[iout];
                }
                p = 0;
                q = np;
                iout = offset - 1;
                int exitg1;
                do {
                    exitg1 = 0;
                    iout++;
                    if (xwork[p] <= xwork[q]) {
                        idx[iout] = iwork[p];
                        x[iout] = xwork[p];
                        if (p + 1 < np) {
                            p++;
                        }
                        else {
                            exitg1 = 1;
                        }
                    }
                    else {
                        idx[iout] = iwork[q];
                        x[iout] = xwork[q];
                        if (q + 1 < n_tmp) {
                            q++;
                        }
                        else {
                            q = iout - p;
                            for (j = p + 1; j <= np; j++) {
                                iout = q + j;
                                idx[iout] = iwork[j - 1];
                                x[iout] = xwork[j - 1];
                            }
                            exitg1 = 1;
                        }
                    }
                } while (exitg1 == 0);
            }
        }

        static void merge(::coder::array<int, 1U>& idx, ::coder::array<double, 1U>& x,
            int offset, int np, int nq, ::coder::array<int, 1U>& iwork,
            ::coder::array<double, 1U>& xwork)
        {
            if (nq != 0) {
                int iout;
                int j;
                int n_tmp;
                int p;
                int q;
                n_tmp = np + nq;
                for (j = 0; j < n_tmp; j++) {
                    iout = offset + j;
                    iwork[j] = idx[iout];
                    xwork[j] = x[iout];
                }
                p = 0;
                q = np;
                iout = offset - 1;
                int exitg1;
                do {
                    exitg1 = 0;
                    iout++;
                    if (xwork[p] >= xwork[q]) {
                        idx[iout] = iwork[p];
                        x[iout] = xwork[p];
                        if (p + 1 < np) {
                            p++;
                        }
                        else {
                            exitg1 = 1;
                        }
                    }
                    else {
                        idx[iout] = iwork[q];
                        x[iout] = xwork[q];
                        if (q + 1 < n_tmp) {
                            q++;
                        }
                        else {
                            q = iout - p;
                            for (j = p + 1; j <= np; j++) {
                                iout = q + j;
                                idx[iout] = iwork[j - 1];
                                x[iout] = xwork[j - 1];
                            }
                            exitg1 = 1;
                        }
                    }
                } while (exitg1 == 0);
            }
        }

        static void merge_block(::coder::array<int, 1U>& idx,
            ::coder::array<double, 1U>& x, int offset, int n,
            int preSortLevel, ::coder::array<int, 1U>& iwork,
            ::coder::array<double, 1U>& xwork)
        {
            int bLen;
            int nPairs;
            nPairs = n >> preSortLevel;
            bLen = 1 << preSortLevel;
            while (nPairs > 1) {
                int nTail;
                int tailOffset;
                if ((nPairs & 1) != 0) {
                    nPairs--;
                    tailOffset = bLen * nPairs;
                    nTail = n - tailOffset;
                    if (nTail > bLen) {
                        merge(idx, x, offset + tailOffset, bLen, nTail - bLen, iwork, xwork);
                    }
                }
                tailOffset = bLen << 1;
                nPairs >>= 1;
                for (nTail = 0; nTail < nPairs; nTail++) {
                    merge(idx, x, offset + nTail * tailOffset, bLen, bLen, iwork, xwork);
                }
                bLen = tailOffset;
            }
            if (n > bLen) {
                merge(idx, x, offset, bLen, n - bLen, iwork, xwork);
            }
        }

        void merge_block(::coder::array<int, 2U>& idx, ::coder::array<float, 2U>& x,
            int offset, int n, int preSortLevel,
            ::coder::array<int, 1U>& iwork,
            ::coder::array<float, 1U>& xwork)
        {
            int bLen;
            int nPairs;
            nPairs = n >> preSortLevel;
            bLen = 1 << preSortLevel;
            while (nPairs > 1) {
                int nTail;
                int tailOffset;
                if ((nPairs & 1) != 0) {
                    nPairs--;
                    tailOffset = bLen * nPairs;
                    nTail = n - tailOffset;
                    if (nTail > bLen) {
                        merge(idx, x, offset + tailOffset, bLen, nTail - bLen, iwork, xwork);
                    }
                }
                tailOffset = bLen << 1;
                nPairs >>= 1;
                for (nTail = 0; nTail < nPairs; nTail++) {
                    merge(idx, x, offset + nTail * tailOffset, bLen, bLen, iwork, xwork);
                }
                bLen = tailOffset;
            }
            if (n > bLen) {
                merge(idx, x, offset, bLen, n - bLen, iwork, xwork);
            }
        }

        void sortIdx(::coder::array<double, 2U>& x, ::coder::array<int, 2U>& idx)
        {
            array<double, 1U> b_x;
            array<double, 1U> xwork;
            array<int, 1U> b_idx;
            array<int, 1U> iwork;
            double b_xwork[256];
            double x4[4];
            double d;
            double d1;
            int b_iwork[256];
            int idx4[4];
            int b;
            int bLen;
            int bLen2;
            int b_b;
            int exitg1;
            int i2;
            int i3;
            int i4;
            int ib;
            int idx_tmp;
            int k;
            int nBlocks;
            int nNaNs;
            int nNonNaN;
            int nPairs;
            int quartetOffset;
            int unnamed_idx_0;
            signed char perm[4];
            unnamed_idx_0 = x.size(0);
            idx.set_size(unnamed_idx_0, 1);
            for (int i{ 0 }; i < unnamed_idx_0; i++) {
                idx[i] = 0;
            }
            if (x.size(0) != 0) {
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
    xwork, iwork, b_idx, b_x, ib, bLen, x4, idx4, quartetOffset, nNaNs, k, i3, \
    idx_tmp, i2, i4, d, d1, perm, nNonNaN, nBlocks, b, b_b, bLen2, nPairs,     \
    b_iwork, b_xwork, exitg1)

                for (unnamed_idx_0 = 0; unnamed_idx_0 < 1; unnamed_idx_0++) {
                    ib = idx.size(0);
                    b_idx.set_size(idx.size(0));
                    for (bLen = 0; bLen < ib; bLen++) {
                        b_idx[bLen] = idx[bLen];
                    }
                    ib = x.size(0);
                    b_x.set_size(ib);
                    for (bLen = 0; bLen < ib; bLen++) {
                        b_x[bLen] = x[bLen];
                    }
                    bLen = x.size(0);
                    x4[0] = 0.0;
                    idx4[0] = 0;
                    x4[1] = 0.0;
                    idx4[1] = 0;
                    x4[2] = 0.0;
                    idx4[2] = 0;
                    x4[3] = 0.0;
                    idx4[3] = 0;
                    iwork.set_size(idx.size(0));
                    ib = idx.size(0);
                    for (quartetOffset = 0; quartetOffset < ib; quartetOffset++) {
                        iwork[quartetOffset] = 0;
                    }
                    ib = x.size(0);
                    xwork.set_size(ib);
                    for (quartetOffset = 0; quartetOffset < ib; quartetOffset++) {
                        xwork[quartetOffset] = 0.0;
                    }
                    nNaNs = 0;
                    ib = -1;
                    for (k = 0; k < bLen; k++) {
                        if (std::isnan(b_x[k])) {
                            idx_tmp = (bLen - nNaNs) - 1;
                            b_idx[idx_tmp] = k + 1;
                            xwork[idx_tmp] = b_x[k];
                            nNaNs++;
                        }
                        else {
                            ib++;
                            idx4[ib] = k + 1;
                            x4[ib] = b_x[k];
                            if (ib + 1 == 4) {
                                quartetOffset = k - nNaNs;
                                if (x4[0] >= x4[1]) {
                                    ib = 1;
                                    i2 = 2;
                                }
                                else {
                                    ib = 2;
                                    i2 = 1;
                                }
                                if (x4[2] >= x4[3]) {
                                    i3 = 3;
                                    i4 = 4;
                                }
                                else {
                                    i3 = 4;
                                    i4 = 3;
                                }
                                d = x4[ib - 1];
                                d1 = x4[i3 - 1];
                                if (d >= d1) {
                                    d = x4[i2 - 1];
                                    if (d >= d1) {
                                        perm[0] = static_cast<signed char>(ib);
                                        perm[1] = static_cast<signed char>(i2);
                                        perm[2] = static_cast<signed char>(i3);
                                        perm[3] = static_cast<signed char>(i4);
                                    }
                                    else if (d >= x4[i4 - 1]) {
                                        perm[0] = static_cast<signed char>(ib);
                                        perm[1] = static_cast<signed char>(i3);
                                        perm[2] = static_cast<signed char>(i2);
                                        perm[3] = static_cast<signed char>(i4);
                                    }
                                    else {
                                        perm[0] = static_cast<signed char>(ib);
                                        perm[1] = static_cast<signed char>(i3);
                                        perm[2] = static_cast<signed char>(i4);
                                        perm[3] = static_cast<signed char>(i2);
                                    }
                                }
                                else {
                                    d1 = x4[i4 - 1];
                                    if (d >= d1) {
                                        if (x4[i2 - 1] >= d1) {
                                            perm[0] = static_cast<signed char>(i3);
                                            perm[1] = static_cast<signed char>(ib);
                                            perm[2] = static_cast<signed char>(i2);
                                            perm[3] = static_cast<signed char>(i4);
                                        }
                                        else {
                                            perm[0] = static_cast<signed char>(i3);
                                            perm[1] = static_cast<signed char>(ib);
                                            perm[2] = static_cast<signed char>(i4);
                                            perm[3] = static_cast<signed char>(i2);
                                        }
                                    }
                                    else {
                                        perm[0] = static_cast<signed char>(i3);
                                        perm[1] = static_cast<signed char>(i4);
                                        perm[2] = static_cast<signed char>(ib);
                                        perm[3] = static_cast<signed char>(i2);
                                    }
                                }
                                b_idx[quartetOffset - 3] = idx4[perm[0] - 1];
                                b_idx[quartetOffset - 2] = idx4[perm[1] - 1];
                                b_idx[quartetOffset - 1] = idx4[perm[2] - 1];
                                b_idx[quartetOffset] = idx4[perm[3] - 1];
                                b_x[quartetOffset - 3] = x4[perm[0] - 1];
                                b_x[quartetOffset - 2] = x4[perm[1] - 1];
                                b_x[quartetOffset - 1] = x4[perm[2] - 1];
                                b_x[quartetOffset] = x4[perm[3] - 1];
                                ib = -1;
                            }
                        }
                    }
                    i3 = (x.size(0) - nNaNs) - 1;
                    if (ib + 1 > 0) {
                        perm[1] = 0;
                        perm[2] = 0;
                        perm[3] = 0;
                        if (ib + 1 == 1) {
                            perm[0] = 1;
                        }
                        else if (ib + 1 == 2) {
                            if (x4[0] >= x4[1]) {
                                perm[0] = 1;
                                perm[1] = 2;
                            }
                            else {
                                perm[0] = 2;
                                perm[1] = 1;
                            }
                        }
                        else if (x4[0] >= x4[1]) {
                            if (x4[1] >= x4[2]) {
                                perm[0] = 1;
                                perm[1] = 2;
                                perm[2] = 3;
                            }
                            else if (x4[0] >= x4[2]) {
                                perm[0] = 1;
                                perm[1] = 3;
                                perm[2] = 2;
                            }
                            else {
                                perm[0] = 3;
                                perm[1] = 1;
                                perm[2] = 2;
                            }
                        }
                        else if (x4[0] >= x4[2]) {
                            perm[0] = 2;
                            perm[1] = 1;
                            perm[2] = 3;
                        }
                        else if (x4[1] >= x4[2]) {
                            perm[0] = 2;
                            perm[1] = 3;
                            perm[2] = 1;
                        }
                        else {
                            perm[0] = 3;
                            perm[1] = 2;
                            perm[2] = 1;
                        }
                        for (k = 0; k <= ib; k++) {
                            idx_tmp = perm[k] - 1;
                            quartetOffset = (i3 - ib) + k;
                            b_idx[quartetOffset] = idx4[idx_tmp];
                            b_x[quartetOffset] = x4[idx_tmp];
                        }
                    }
                    ib = (nNaNs >> 1) + 1;
                    for (k = 0; k <= ib - 2; k++) {
                        quartetOffset = (i3 + k) + 1;
                        i2 = b_idx[quartetOffset];
                        idx_tmp = (bLen - k) - 1;
                        b_idx[quartetOffset] = b_idx[idx_tmp];
                        b_idx[idx_tmp] = i2;
                        b_x[quartetOffset] = xwork[idx_tmp];
                        b_x[idx_tmp] = xwork[quartetOffset];
                    }
                    if ((nNaNs & 1) != 0) {
                        ib += i3;
                        b_x[ib] = xwork[ib];
                    }
                    nNonNaN = x.size(0) - nNaNs;
                    ib = 2;
                    if (nNonNaN > 1) {
                        if (x.size(0) >= 256) {
                            nBlocks = nNonNaN >> 8;
                            if (nBlocks > 0) {
                                for (b = 0; b < nBlocks; b++) {
                                    i4 = (b << 8) - 1;
                                    for (b_b = 0; b_b < 6; b_b++) {
                                        bLen = 1 << (b_b + 2);
                                        bLen2 = bLen << 1;
                                        nPairs = 256 >> (b_b + 3);
                                        for (k = 0; k < nPairs; k++) {
                                            i2 = (i4 + k * bLen2) + 1;
                                            for (quartetOffset = 0; quartetOffset < bLen2;
                                                quartetOffset++) {
                                                ib = i2 + quartetOffset;
                                                b_iwork[quartetOffset] = b_idx[ib];
                                                b_xwork[quartetOffset] = b_x[ib];
                                            }
                                            i3 = 0;
                                            quartetOffset = bLen;
                                            ib = i2 - 1;
                                            do {
                                                exitg1 = 0;
                                                ib++;
                                                if (b_xwork[i3] >= b_xwork[quartetOffset]) {
                                                    b_idx[ib] = b_iwork[i3];
                                                    b_x[ib] = b_xwork[i3];
                                                    if (i3 + 1 < bLen) {
                                                        i3++;
                                                    }
                                                    else {
                                                        exitg1 = 1;
                                                    }
                                                }
                                                else {
                                                    b_idx[ib] = b_iwork[quartetOffset];
                                                    b_x[ib] = b_xwork[quartetOffset];
                                                    if (quartetOffset + 1 < bLen2) {
                                                        quartetOffset++;
                                                    }
                                                    else {
                                                        ib -= i3;
                                                        for (quartetOffset = i3 + 1; quartetOffset <= bLen;
                                                            quartetOffset++) {
                                                            idx_tmp = ib + quartetOffset;
                                                            b_idx[idx_tmp] = b_iwork[quartetOffset - 1];
                                                            b_x[idx_tmp] = b_xwork[quartetOffset - 1];
                                                        }
                                                        exitg1 = 1;
                                                    }
                                                }
                                            } while (exitg1 == 0);
                                        }
                                    }
                                }
                                ib = nBlocks << 8;
                                quartetOffset = nNonNaN - ib;
                                if (quartetOffset > 0) {
                                    merge_block(b_idx, b_x, ib, quartetOffset, 2, iwork, xwork);
                                }
                                ib = 8;
                            }
                        }
                        merge_block(b_idx, b_x, 0, nNonNaN, ib, iwork, xwork);
                    }
                    if ((nNaNs > 0) && (nNonNaN > 0)) {
                        for (k = 0; k < nNaNs; k++) {
                            ib = nNonNaN + k;
                            xwork[k] = b_x[ib];
                            iwork[k] = b_idx[ib];
                        }
                        for (k = nNonNaN; k >= 1; k--) {
                            ib = (nNaNs + k) - 1;
                            b_x[ib] = b_x[k - 1];
                            b_idx[ib] = b_idx[k - 1];
                        }
                        for (k = 0; k < nNaNs; k++) {
                            b_x[k] = xwork[k];
                            b_idx[k] = iwork[k];
                        }
                    }
                    ib = b_idx.size(0);
                    for (bLen = 0; bLen < ib; bLen++) {
                        idx[bLen] = b_idx[bLen];
                    }
                    ib = b_x.size(0);
                    for (bLen = 0; bLen < ib; bLen++) {
                        x[bLen] = b_x[bLen];
                    }
                }
            }
        }

    } // namespace internal
} // namespace coder

// End of code generation (sortIdx.cpp)
