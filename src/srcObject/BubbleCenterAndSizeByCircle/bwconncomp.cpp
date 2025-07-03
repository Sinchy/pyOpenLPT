//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// bwconncomp.cpp
//
// Code generation for function 'bwconncomp'
//

// Include files
#include "bwconncomp.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Definitions
namespace coder {
    void bwconncomp(const ::coder::array<bool, 2U>& varargin_1,
        double* CC_Connectivity, double CC_ImageSize[2],
        double* CC_NumObjects,
        ::coder::array<double, 1U>& CC_RegionIndices,
        ::coder::array<int, 1U>& CC_RegionLengths)
    {
        array<int, 1U> endRow;
        array<int, 1U> idxCount;
        array<int, 1U> labelsRenumbered;
        array<int, 1U> pixelIdxList;
        array<int, 1U> startCol;
        array<int, 1U> startRow;
        array<int, 1U> x;
        double numComponents;
        int firstRunOnThisColumn;
        int k;
        int lastRunOnPreviousColumn;
        int numRuns;
        int row;
        numRuns = 0;
        if ((varargin_1.size(0) != 0) && (varargin_1.size(1) != 0)) {
            firstRunOnThisColumn = varargin_1.size(1);
            for (lastRunOnPreviousColumn = 0;
                lastRunOnPreviousColumn < firstRunOnThisColumn;
                lastRunOnPreviousColumn++) {
                if (varargin_1[varargin_1.size(0) * lastRunOnPreviousColumn]) {
                    numRuns++;
                }
                row = varargin_1.size(0);
                for (k = 0; k <= row - 2; k++) {
                    if (varargin_1[(k + varargin_1.size(0) * lastRunOnPreviousColumn) +
                        1] &&
                        (!varargin_1[k + varargin_1.size(0) * lastRunOnPreviousColumn])) {
                        numRuns++;
                    }
                }
            }
        }
        if (numRuns == 0) {
            CC_ImageSize[0] = varargin_1.size(0);
            CC_ImageSize[1] = varargin_1.size(1);
            numComponents = 0.0;
            CC_RegionIndices.set_size(0);
            CC_RegionLengths.set_size(1);
            CC_RegionLengths[0] = 0;
        }
        else {
            double y;
            int currentColumn;
            int runCounter;
            startRow.set_size(numRuns);
            endRow.set_size(numRuns);
            startCol.set_size(numRuns);
            currentColumn = varargin_1.size(0);
            runCounter = 0;
            firstRunOnThisColumn = varargin_1.size(1);
            for (lastRunOnPreviousColumn = 0;
                lastRunOnPreviousColumn < firstRunOnThisColumn;
                lastRunOnPreviousColumn++) {
                row = 1;
                while (row <= currentColumn) {
                    while (
                        (row <= currentColumn) &&
                        (!varargin_1[(row + varargin_1.size(0) * lastRunOnPreviousColumn) -
                            1])) {
                        row++;
                    }
                    if ((row <= currentColumn) &&
                        varargin_1[(row + varargin_1.size(0) * lastRunOnPreviousColumn) -
                        1]) {
                        startCol[runCounter] = lastRunOnPreviousColumn + 1;
                        startRow[runCounter] = row;
                        while (
                            (row <= currentColumn) &&
                            varargin_1[(row + varargin_1.size(0) * lastRunOnPreviousColumn) -
                            1]) {
                            row++;
                        }
                        endRow[runCounter] = row - 1;
                        runCounter++;
                    }
                }
            }
            CC_RegionLengths.set_size(numRuns);
            for (firstRunOnThisColumn = 0; firstRunOnThisColumn < numRuns;
                firstRunOnThisColumn++) {
                CC_RegionLengths[firstRunOnThisColumn] = 0;
            }
            k = 0;
            currentColumn = 1;
            runCounter = 1;
            row = -1;
            lastRunOnPreviousColumn = -1;
            firstRunOnThisColumn = 0;
            while (k + 1 <= numRuns) {
                if (startCol[k] == currentColumn + 1) {
                    row = firstRunOnThisColumn + 1;
                    firstRunOnThisColumn = k;
                    lastRunOnPreviousColumn = k;
                    currentColumn = startCol[k];
                }
                else if (startCol[k] > currentColumn + 1) {
                    row = -1;
                    lastRunOnPreviousColumn = -1;
                    firstRunOnThisColumn = k;
                    currentColumn = startCol[k];
                }
                if (row >= 0) {
                    for (int p{ row - 1 }; p < lastRunOnPreviousColumn; p++) {
                        if ((endRow[k] >= startRow[p] - 1) &&
                            (startRow[k] <= endRow[p] + 1)) {
                            if (CC_RegionLengths[k] == 0) {
                                CC_RegionLengths[k] = CC_RegionLengths[p];
                                runCounter++;
                            }
                            else if (CC_RegionLengths[k] != CC_RegionLengths[p]) {
                                int root_k;
                                int root_p;
                                for (root_k = k; root_k + 1 != CC_RegionLengths[root_k];
                                    root_k = CC_RegionLengths[root_k] - 1) {
                                    CC_RegionLengths[root_k] =
                                        CC_RegionLengths[CC_RegionLengths[root_k] - 1];
                                }
                                for (root_p = p; root_p + 1 != CC_RegionLengths[root_p];
                                    root_p = CC_RegionLengths[root_p] - 1) {
                                    CC_RegionLengths[root_p] =
                                        CC_RegionLengths[CC_RegionLengths[root_p] - 1];
                                }
                                if (root_k + 1 != root_p + 1) {
                                    if (root_p + 1 < root_k + 1) {
                                        CC_RegionLengths[root_k] = root_p + 1;
                                        CC_RegionLengths[k] = root_p + 1;
                                    }
                                    else {
                                        CC_RegionLengths[root_p] = root_k + 1;
                                        CC_RegionLengths[p] = root_k + 1;
                                    }
                                }
                            }
                        }
                    }
                }
                if (CC_RegionLengths[k] == 0) {
                    CC_RegionLengths[k] = runCounter;
                    runCounter++;
                }
                k++;
            }
            labelsRenumbered.set_size(CC_RegionLengths.size(0));
            numComponents = 0.0;
            for (k = 0; k < numRuns; k++) {
                if (CC_RegionLengths[k] == k + 1) {
                    numComponents++;
                    labelsRenumbered[k] = static_cast<int>(numComponents);
                }
                labelsRenumbered[k] = labelsRenumbered[CC_RegionLengths[k] - 1];
            }
            currentColumn = static_cast<int>(numComponents);
            CC_RegionLengths.set_size(currentColumn);
            for (firstRunOnThisColumn = 0; firstRunOnThisColumn < currentColumn;
                firstRunOnThisColumn++) {
                CC_RegionLengths[firstRunOnThisColumn] = 0;
            }
            for (k = 0; k < numRuns; k++) {
                CC_RegionLengths[labelsRenumbered[k] - 1] =
                    ((CC_RegionLengths[labelsRenumbered[k] - 1] + endRow[k]) -
                        startRow[k]) +
                    1;
            }
            if (CC_RegionLengths.size(0) == 0) {
                y = 0.0;
            }
            else {
                if (CC_RegionLengths.size(0) <= 1024) {
                    currentColumn = CC_RegionLengths.size(0);
                    lastRunOnPreviousColumn = 0;
                    row = 1;
                }
                else {
                    currentColumn = 1024;
                    row = CC_RegionLengths.size(0) / 1024;
                    lastRunOnPreviousColumn = CC_RegionLengths.size(0) - (row << 10);
                    if (lastRunOnPreviousColumn > 0) {
                        row++;
                    }
                    else {
                        lastRunOnPreviousColumn = 1024;
                    }
                }
                y = CC_RegionLengths[0];
                for (k = 2; k <= currentColumn; k++) {
                    y += static_cast<double>(CC_RegionLengths[k - 1]);
                }
                for (firstRunOnThisColumn = 2; firstRunOnThisColumn <= row;
                    firstRunOnThisColumn++) {
                    double bsum;
                    currentColumn = (firstRunOnThisColumn - 1) << 10;
                    bsum = CC_RegionLengths[currentColumn];
                    if (firstRunOnThisColumn == row) {
                        runCounter = lastRunOnPreviousColumn;
                    }
                    else {
                        runCounter = 1024;
                    }
                    for (k = 2; k <= runCounter; k++) {
                        bsum +=
                            static_cast<double>(CC_RegionLengths[(currentColumn + k) - 1]);
                    }
                    y += bsum;
                }
            }
            pixelIdxList.set_size(static_cast<int>(y));
            x.set_size(CC_RegionLengths.size(0));
            currentColumn = CC_RegionLengths.size(0);
            for (firstRunOnThisColumn = 0; firstRunOnThisColumn < currentColumn;
                firstRunOnThisColumn++) {
                x[firstRunOnThisColumn] = CC_RegionLengths[firstRunOnThisColumn];
            }
            if ((CC_RegionLengths.size(0) != 1) && (CC_RegionLengths.size(0) != 0) &&
                (CC_RegionLengths.size(0) != 1)) {
                firstRunOnThisColumn = CC_RegionLengths.size(0);
                for (k = 0; k <= firstRunOnThisColumn - 2; k++) {
                    x[k + 1] = x[k] + x[k + 1];
                }
            }
            idxCount.set_size(x.size(0) + 1);
            idxCount[0] = 0;
            currentColumn = x.size(0);
            for (firstRunOnThisColumn = 0; firstRunOnThisColumn < currentColumn;
                firstRunOnThisColumn++) {
                idxCount[firstRunOnThisColumn + 1] = x[firstRunOnThisColumn];
            }
            for (k = 0; k < numRuns; k++) {
                currentColumn = (startCol[k] - 1) * varargin_1.size(0);
                runCounter = labelsRenumbered[k] - 1;
                firstRunOnThisColumn = startRow[k];
                row = endRow[k];
                for (lastRunOnPreviousColumn = firstRunOnThisColumn;
                    lastRunOnPreviousColumn <= row; lastRunOnPreviousColumn++) {
                    idxCount[runCounter] = idxCount[runCounter] + 1;
                    pixelIdxList[idxCount[runCounter] - 1] =
                        lastRunOnPreviousColumn + currentColumn;
                }
            }
            CC_ImageSize[0] = varargin_1.size(0);
            CC_ImageSize[1] = varargin_1.size(1);
            CC_RegionIndices.set_size(pixelIdxList.size(0));
            currentColumn = pixelIdxList.size(0);
            for (firstRunOnThisColumn = 0; firstRunOnThisColumn < currentColumn;
                firstRunOnThisColumn++) {
                CC_RegionIndices[firstRunOnThisColumn] =
                    pixelIdxList[firstRunOnThisColumn];
            }
        }
        *CC_Connectivity = 8.0;
        *CC_NumObjects = numComponents;
    }

} // namespace coder

// End of code generation (bwconncomp.cpp)
