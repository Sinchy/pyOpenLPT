//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// imhmax.cpp
//
// Code generation for function 'imhmax'
//

// Include files
#include "imhmax.h"
#include "BubbleCenterAndSizeByCircle_rtwutil.h"
#include "NeighborhoodProcessor.h"
#include "minOrMax.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include <cmath>
#include <cstring>

// Function Definitions
namespace coder {
    void imhmax(const ::coder::array<double, 2U>& b_I, double H,
        ::coder::array<double, 2U>& J)
    {
        images::internal::coder::NeighborhoodProcessor np;
        array<int, 2U> locationStack;
        array<int, 1U> r;
        double J_data[9];
        int imnhSubs[18];
        int iv[18];
        int imnhInds_[9];
        int imnhInds_data[9];
        int iv1[9];
        int J_size[2];
        int k[2];
        int pixelSub[2];
        int i;
        int ind;
        int loop_ub;
        int numPixels;
        int pInd;
        int partialTrueCount;
        int stackTop;
        int trueCount;
        bool isInside[9];
        bool exitg1;
        J.set_size(b_I.size(0), b_I.size(1));
        loop_ub = b_I.size(0) * b_I.size(1);
        for (i = 0; i < loop_ub; i++) {
            J[i] = b_I[i] - H;
        }
        loop_ub = J.size(0) * J.size(1) - 1;
        trueCount = 0;
        for (ind = 0; ind <= loop_ub; ind++) {
            if (J[ind] > b_I[ind]) {
                trueCount++;
            }
        }
        r.set_size(trueCount);
        partialTrueCount = 0;
        for (ind = 0; ind <= loop_ub; ind++) {
            if (J[ind] > b_I[ind]) {
                r[partialTrueCount] = ind + 1;
                partialTrueCount++;
            }
        }
        loop_ub = r.size(0);
        for (i = 0; i < loop_ub; i++) {
            J[r[i] - 1] = b_I[r[i] - 1];
        }
        np.ImageSize[0] = J.size(0);
        np.ImageSize[1] = J.size(1);
        for (ind = 0; ind < 9; ind++) {
            iv1[ind] = 0;
            imnhInds_[ind] = 0;
        }
        std::memset(&iv[0], 0, 18U * sizeof(int));
        for (i = 0; i < 9; i++) {
            isInside[i] = true;
        }
        images::internal::coder::NeighborhoodProcessor::computeParameters(
            np.ImageSize, isInside, iv1, imnhInds_, iv, pixelSub, k);
        numPixels = J.size(0) * J.size(1);
        if (J.size(0) == 1) {
            if (0 <= numPixels - 1) {
                J_size[0] = 1;
            }
            for (pInd = 0; pInd < numPixels; pInd++) {
                for (ind = 0; ind < 9; ind++) {
                    imnhInds_[ind] = (iv1[ind] + pInd) + 1;
                }
                if (np.ImageSize[0] == 0) {
                    loop_ub = 0;
                }
                else {
                    loop_ub = pInd - np.ImageSize[0] * div_s32(pInd, np.ImageSize[0]);
                }
                ind = pInd - loop_ub;
                if (np.ImageSize[0] == 0) {
                    if (ind == 0) {
                        partialTrueCount = 0;
                    }
                    else if (ind < 0) {
                        partialTrueCount = MIN_int32_T;
                    }
                    else {
                        partialTrueCount = MAX_int32_T;
                    }
                }
                else if (np.ImageSize[0] == 1) {
                    partialTrueCount = ind;
                }
                else {
                    if (ind >= 0) {
                        trueCount = ind;
                    }
                    else if (ind == MIN_int32_T) {
                        trueCount = MAX_int32_T;
                    }
                    else {
                        trueCount = -ind;
                    }
                    partialTrueCount = div_s32(trueCount, np.ImageSize[0]);
                    trueCount -= partialTrueCount * np.ImageSize[0];
                    if ((trueCount > 0) &&
                        (trueCount >= (np.ImageSize[0] >> 1) + (np.ImageSize[0] & 1))) {
                        partialTrueCount++;
                    }
                    if (ind < 0) {
                        partialTrueCount = -partialTrueCount;
                    }
                }
                pixelSub[1] = partialTrueCount + 1;
                pixelSub[0] = loop_ub + 1;
                for (ind = 0; ind < 2; ind++) {
                    for (trueCount = 0; trueCount < 9; trueCount++) {
                        loop_ub = trueCount + 9 * ind;
                        imnhSubs[loop_ub] = iv[loop_ub] + pixelSub[ind];
                    }
                }
                trueCount = 0;
                partialTrueCount = 0;
                for (ind = 0; ind < 9; ind++) {
                    isInside[ind] = true;
                    loop_ub = 0;
                    exitg1 = false;
                    while ((!exitg1) && (loop_ub < 2)) {
                        i = imnhSubs[ind + 9 * loop_ub];
                        if ((i < 1) || (i > np.ImageSize[loop_ub])) {
                            isInside[ind] = false;
                            exitg1 = true;
                        }
                        else {
                            loop_ub++;
                        }
                    }
                    if (isInside[ind]) {
                        trueCount++;
                        imnhInds_data[partialTrueCount] = imnhInds_[ind];
                        partialTrueCount++;
                    }
                }
                J_size[1] = trueCount;
                for (i = 0; i < trueCount; i++) {
                    J_data[i] = J[imnhInds_data[i] - 1];
                }
                J[pInd] = std::fmin(internal::maximum(J_data, J_size), b_I[pInd]);
            }
            locationStack.set_size(1, numPixels << 1);
            stackTop = -1;
            if (numPixels >= 1) {
                J_size[0] = 1;
            }
            for (pInd = numPixels; pInd >= 1; pInd--) {
                for (ind = 0; ind < 9; ind++) {
                    imnhInds_[ind] = iv1[ind] + pInd;
                }
                if (np.ImageSize[0] == 0) {
                    loop_ub = 1;
                }
                else {
                    loop_ub = pInd - np.ImageSize[0] * div_s32(pInd - 1, np.ImageSize[0]);
                }
                ind = pInd - loop_ub;
                if (np.ImageSize[0] == 0) {
                    if (ind == 0) {
                        partialTrueCount = 0;
                    }
                    else if (ind < 0) {
                        partialTrueCount = MIN_int32_T;
                    }
                    else {
                        partialTrueCount = MAX_int32_T;
                    }
                }
                else if (np.ImageSize[0] == 1) {
                    partialTrueCount = ind;
                }
                else {
                    if (ind >= 0) {
                        trueCount = ind;
                    }
                    else if (ind == MIN_int32_T) {
                        trueCount = MAX_int32_T;
                    }
                    else {
                        trueCount = -ind;
                    }
                    partialTrueCount = div_s32(trueCount, np.ImageSize[0]);
                    trueCount -= partialTrueCount * np.ImageSize[0];
                    if ((trueCount > 0) &&
                        (trueCount >= (np.ImageSize[0] >> 1) + (np.ImageSize[0] & 1))) {
                        partialTrueCount++;
                    }
                    if (ind < 0) {
                        partialTrueCount = -partialTrueCount;
                    }
                }
                pixelSub[1] = partialTrueCount + 1;
                pixelSub[0] = loop_ub;
                for (ind = 0; ind < 2; ind++) {
                    for (trueCount = 0; trueCount < 9; trueCount++) {
                        loop_ub = trueCount + 9 * ind;
                        imnhSubs[loop_ub] = iv[loop_ub] + pixelSub[ind];
                    }
                }
                trueCount = 0;
                partialTrueCount = 0;
                for (ind = 0; ind < 9; ind++) {
                    isInside[ind] = true;
                    loop_ub = 0;
                    exitg1 = false;
                    while ((!exitg1) && (loop_ub < 2)) {
                        i = imnhSubs[ind + 9 * loop_ub];
                        if ((i < 1) || (i > np.ImageSize[loop_ub])) {
                            isInside[ind] = false;
                            exitg1 = true;
                        }
                        else {
                            loop_ub++;
                        }
                    }
                    if (isInside[ind]) {
                        trueCount++;
                        imnhInds_data[partialTrueCount] = imnhInds_[ind];
                        partialTrueCount++;
                    }
                }
                J_size[1] = trueCount;
                for (i = 0; i < trueCount; i++) {
                    J_data[i] = J[imnhInds_data[i] - 1];
                }
                J[pInd - 1] = std::fmin(internal::maximum(J_data, J_size), b_I[pInd - 1]);
                ind = 0;
                exitg1 = false;
                while ((!exitg1) && (ind <= trueCount - 1)) {
                    if ((J[imnhInds_data[ind] - 1] < J[pInd - 1]) &&
                        (J[imnhInds_data[ind] - 1] < b_I[imnhInds_data[ind] - 1])) {
                        stackTop++;
                        locationStack[stackTop] = pInd;
                        exitg1 = true;
                    }
                    else {
                        ind++;
                    }
                }
            }
        }
        else {
            for (pInd = 0; pInd < numPixels; pInd++) {
                for (ind = 0; ind < 9; ind++) {
                    imnhInds_[ind] = (iv1[ind] + pInd) + 1;
                }
                if (np.ImageSize[0] == 0) {
                    loop_ub = 0;
                }
                else {
                    loop_ub = pInd - np.ImageSize[0] * div_s32(pInd, np.ImageSize[0]);
                }
                ind = pInd - loop_ub;
                if (np.ImageSize[0] == 0) {
                    if (ind == 0) {
                        partialTrueCount = 0;
                    }
                    else if (ind < 0) {
                        partialTrueCount = MIN_int32_T;
                    }
                    else {
                        partialTrueCount = MAX_int32_T;
                    }
                }
                else if (np.ImageSize[0] == 1) {
                    partialTrueCount = ind;
                }
                else {
                    if (ind >= 0) {
                        trueCount = ind;
                    }
                    else if (ind == MIN_int32_T) {
                        trueCount = MAX_int32_T;
                    }
                    else {
                        trueCount = -ind;
                    }
                    partialTrueCount = div_s32(trueCount, np.ImageSize[0]);
                    trueCount -= partialTrueCount * np.ImageSize[0];
                    if ((trueCount > 0) &&
                        (trueCount >= (np.ImageSize[0] >> 1) + (np.ImageSize[0] & 1))) {
                        partialTrueCount++;
                    }
                    if (ind < 0) {
                        partialTrueCount = -partialTrueCount;
                    }
                }
                pixelSub[1] = partialTrueCount + 1;
                pixelSub[0] = loop_ub + 1;
                for (ind = 0; ind < 2; ind++) {
                    for (trueCount = 0; trueCount < 9; trueCount++) {
                        loop_ub = trueCount + 9 * ind;
                        imnhSubs[loop_ub] = iv[loop_ub] + pixelSub[ind];
                    }
                }
                trueCount = 0;
                partialTrueCount = 0;
                for (ind = 0; ind < 9; ind++) {
                    isInside[ind] = true;
                    loop_ub = 0;
                    exitg1 = false;
                    while ((!exitg1) && (loop_ub < 2)) {
                        i = imnhSubs[ind + 9 * loop_ub];
                        if ((i < 1) || (i > np.ImageSize[loop_ub])) {
                            isInside[ind] = false;
                            exitg1 = true;
                        }
                        else {
                            loop_ub++;
                        }
                    }
                    if (isInside[ind]) {
                        trueCount++;
                        imnhInds_data[partialTrueCount] = imnhInds_[ind];
                        partialTrueCount++;
                    }
                }
                for (i = 0; i < trueCount; i++) {
                    J_data[i] = J[imnhInds_data[i] - 1];
                }
                J[pInd] = std::fmin(internal::maximum(J_data, trueCount), b_I[pInd]);
            }
            locationStack.set_size(1, numPixels << 1);
            stackTop = -1;
            for (pInd = numPixels; pInd >= 1; pInd--) {
                for (ind = 0; ind < 9; ind++) {
                    imnhInds_[ind] = iv1[ind] + pInd;
                }
                if (np.ImageSize[0] == 0) {
                    loop_ub = 1;
                }
                else {
                    loop_ub = pInd - np.ImageSize[0] * div_s32(pInd - 1, np.ImageSize[0]);
                }
                ind = pInd - loop_ub;
                if (np.ImageSize[0] == 0) {
                    if (ind == 0) {
                        partialTrueCount = 0;
                    }
                    else if (ind < 0) {
                        partialTrueCount = MIN_int32_T;
                    }
                    else {
                        partialTrueCount = MAX_int32_T;
                    }
                }
                else if (np.ImageSize[0] == 1) {
                    partialTrueCount = ind;
                }
                else {
                    if (ind >= 0) {
                        trueCount = ind;
                    }
                    else if (ind == MIN_int32_T) {
                        trueCount = MAX_int32_T;
                    }
                    else {
                        trueCount = -ind;
                    }
                    partialTrueCount = div_s32(trueCount, np.ImageSize[0]);
                    trueCount -= partialTrueCount * np.ImageSize[0];
                    if ((trueCount > 0) &&
                        (trueCount >= (np.ImageSize[0] >> 1) + (np.ImageSize[0] & 1))) {
                        partialTrueCount++;
                    }
                    if (ind < 0) {
                        partialTrueCount = -partialTrueCount;
                    }
                }
                pixelSub[1] = partialTrueCount + 1;
                pixelSub[0] = loop_ub;
                for (ind = 0; ind < 2; ind++) {
                    for (trueCount = 0; trueCount < 9; trueCount++) {
                        loop_ub = trueCount + 9 * ind;
                        imnhSubs[loop_ub] = iv[loop_ub] + pixelSub[ind];
                    }
                }
                trueCount = 0;
                partialTrueCount = 0;
                for (ind = 0; ind < 9; ind++) {
                    isInside[ind] = true;
                    loop_ub = 0;
                    exitg1 = false;
                    while ((!exitg1) && (loop_ub < 2)) {
                        i = imnhSubs[ind + 9 * loop_ub];
                        if ((i < 1) || (i > np.ImageSize[loop_ub])) {
                            isInside[ind] = false;
                            exitg1 = true;
                        }
                        else {
                            loop_ub++;
                        }
                    }
                    if (isInside[ind]) {
                        trueCount++;
                        imnhInds_data[partialTrueCount] = imnhInds_[ind];
                        partialTrueCount++;
                    }
                }
                for (i = 0; i < trueCount; i++) {
                    J_data[i] = J[imnhInds_data[i] - 1];
                }
                J[pInd - 1] =
                    std::fmin(internal::maximum(J_data, trueCount), b_I[pInd - 1]);
                ind = 0;
                exitg1 = false;
                while ((!exitg1) && (ind <= trueCount - 1)) {
                    if ((J[imnhInds_data[ind] - 1] < J[pInd - 1]) &&
                        (J[imnhInds_data[ind] - 1] < b_I[imnhInds_data[ind] - 1])) {
                        stackTop++;
                        locationStack[stackTop] = pInd;
                        exitg1 = true;
                    }
                    else {
                        ind++;
                    }
                }
            }
        }
        while (stackTop + 1 > 0) {
            pInd = locationStack[stackTop] - 1;
            stackTop--;
            for (ind = 0; ind < 9; ind++) {
                imnhInds_[ind] = (iv1[ind] + pInd) + 1;
            }
            if (np.ImageSize[0] == 0) {
                loop_ub = 0;
            }
            else {
                loop_ub = pInd - np.ImageSize[0] * div_s32(pInd, np.ImageSize[0]);
            }
            ind = pInd - loop_ub;
            if (np.ImageSize[0] == 0) {
                if (ind == 0) {
                    partialTrueCount = 0;
                }
                else if (ind < 0) {
                    partialTrueCount = MIN_int32_T;
                }
                else {
                    partialTrueCount = MAX_int32_T;
                }
            }
            else if (np.ImageSize[0] == 1) {
                partialTrueCount = ind;
            }
            else {
                if (ind >= 0) {
                    trueCount = ind;
                }
                else if (ind == MIN_int32_T) {
                    trueCount = MAX_int32_T;
                }
                else {
                    trueCount = -ind;
                }
                partialTrueCount = div_s32(trueCount, np.ImageSize[0]);
                trueCount -= partialTrueCount * np.ImageSize[0];
                if ((trueCount > 0) &&
                    (trueCount >= (np.ImageSize[0] >> 1) + (np.ImageSize[0] & 1))) {
                    partialTrueCount++;
                }
                if (ind < 0) {
                    partialTrueCount = -partialTrueCount;
                }
            }
            pixelSub[1] = partialTrueCount + 1;
            pixelSub[0] = loop_ub + 1;
            for (ind = 0; ind < 2; ind++) {
                for (trueCount = 0; trueCount < 9; trueCount++) {
                    loop_ub = trueCount + 9 * ind;
                    imnhSubs[loop_ub] = iv[loop_ub] + pixelSub[ind];
                }
            }
            trueCount = 0;
            partialTrueCount = 0;
            for (ind = 0; ind < 9; ind++) {
                isInside[ind] = true;
                loop_ub = 0;
                exitg1 = false;
                while ((!exitg1) && (loop_ub < 2)) {
                    i = imnhSubs[ind + 9 * loop_ub];
                    if ((i < 1) || (i > np.ImageSize[loop_ub])) {
                        isInside[ind] = false;
                        exitg1 = true;
                    }
                    else {
                        loop_ub++;
                    }
                }
                if (isInside[ind]) {
                    trueCount++;
                    imnhInds_data[partialTrueCount] = imnhInds_[ind];
                    partialTrueCount++;
                }
            }
            for (ind = 0; ind < trueCount; ind++) {
                i = imnhInds_data[ind];
                if ((J[i - 1] < J[pInd]) && (J[i - 1] != b_I[i - 1])) {
                    J[i - 1] = std::fmin(J[pInd], b_I[i - 1]);
                    stackTop++;
                    locationStack[stackTop] = i;
                }
            }
        }
    }

} // namespace coder

// End of code generation (imhmax.cpp)
