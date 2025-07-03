//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// regionprops.cpp
//
// Code generation for function 'regionprops'
//

// Include files
#include "regionprops.h"
#include "BubbleCenterAndSizeByCircle_internal_types.h"
#include "BubbleCenterAndSizeByCircle_rtwutil.h"
#include "bwconncomp.h"
#include "rt_nonfinite.h"
#include "sum.h"
#include "coder_array.h"
#include "coder_bounded_array.h"
#include <cstring>

// Type Definitions
struct d_struct_T {
    double Area;
    double Centroid[2];
    double BoundingBox[4];
    double MajorAxisLength;
    double MinorAxisLength;
    double Eccentricity;
    double Orientation;
    coder::empty_bounded_array<bool, 2U> Image;
    coder::empty_bounded_array<bool, 2U> FilledImage;
    double FilledArea;
    double EulerNumber;
    double Extrema[16];
    double EquivDiameter;
    double Extent;
    coder::array<double, 1U> PixelIdxList;
    coder::array<double, 2U> PixelList;
    double Perimeter;
    double Circularity;
    coder::array<double, 1U> PixelValues;
    double WeightedCentroid[2];
    double MeanIntensity;
    double MinIntensity;
    double MaxIntensity;
    coder::empty_bounded_array<double, 2U> SubarrayIdx;
    double SubarrayIdxLengths[2];
};

// Function Definitions
namespace coder {
    void regionprops(const ::coder::array<bool, 2U>& varargin_1,
        const ::coder::array<double, 2U>& varargin_2,
        ::coder::array<struct_T, 1U>& outstats)
    {
        array<d_struct_T, 1U> stats;
        array<double, 1U> CC_RegionIndices;
        array<int, 1U> idxCount;
        array<int, 1U> regionLengths;
        d_struct_T statsOneObj;
        struct_T s;
        double wc[2];
        double CC_NumObjects;
        double sumIntensity;
        int i;
        int i1;
        int k;
        int loop_ub;
        int loop_ub_tmp;
        bwconncomp(varargin_1, &sumIntensity, wc, &CC_NumObjects, CC_RegionIndices,
            regionLengths);
        s.WeightedCentroid[0] = 0.0;
        s.WeightedCentroid[1] = 0.0;
        loop_ub_tmp = static_cast<int>(CC_NumObjects);
        outstats.set_size(loop_ub_tmp);
        for (i = 0; i < loop_ub_tmp; i++) {
            outstats[i] = s;
        }
        statsOneObj.Area = 0.0;
        statsOneObj.Centroid[0] = 0.0;
        statsOneObj.Centroid[1] = 0.0;
        statsOneObj.BoundingBox[0] = 0.0;
        statsOneObj.BoundingBox[1] = 0.0;
        statsOneObj.BoundingBox[2] = 0.0;
        statsOneObj.BoundingBox[3] = 0.0;
        statsOneObj.MajorAxisLength = 0.0;
        statsOneObj.MinorAxisLength = 0.0;
        statsOneObj.Eccentricity = 0.0;
        statsOneObj.Orientation = 0.0;
        statsOneObj.Image.size[0] = 0;
        statsOneObj.Image.size[1] = 0;
        statsOneObj.FilledImage.size[0] = 0;
        statsOneObj.FilledImage.size[1] = 0;
        statsOneObj.FilledArea = 0.0;
        statsOneObj.EulerNumber = 0.0;
        std::memset(&statsOneObj.Extrema[0], 0, 16U * sizeof(double));
        statsOneObj.EquivDiameter = 0.0;
        statsOneObj.Extent = 0.0;
        statsOneObj.PixelIdxList.set_size(0);
        statsOneObj.PixelList.set_size(0, 2);
        statsOneObj.Perimeter = 0.0;
        statsOneObj.Circularity = 0.0;
        statsOneObj.PixelValues.set_size(0);
        statsOneObj.MeanIntensity = 0.0;
        statsOneObj.MinIntensity = 0.0;
        statsOneObj.MaxIntensity = 0.0;
        statsOneObj.SubarrayIdx.size[0] = 1;
        statsOneObj.SubarrayIdx.size[1] = 0;
        statsOneObj.WeightedCentroid[0] = 0.0;
        statsOneObj.SubarrayIdxLengths[0] = 0.0;
        statsOneObj.WeightedCentroid[1] = 0.0;
        statsOneObj.SubarrayIdxLengths[1] = 0.0;
        stats.set_size(loop_ub_tmp);
        for (i = 0; i < loop_ub_tmp; i++) {
            stats[i] = statsOneObj;
        }
        if (CC_NumObjects != 0.0) {
            if ((regionLengths.size(0) != 1) && (regionLengths.size(0) != 0) &&
                (regionLengths.size(0) != 1)) {
                i = regionLengths.size(0);
                for (k = 0; k <= i - 2; k++) {
                    regionLengths[k + 1] = regionLengths[k] + regionLengths[k + 1];
                }
            }
            idxCount.set_size(regionLengths.size(0) + 1);
            idxCount[0] = 0;
            loop_ub = regionLengths.size(0);
            for (i = 0; i < loop_ub; i++) {
                idxCount[i + 1] = regionLengths[i];
            }
            for (k = 0; k < loop_ub_tmp; k++) {
                i = idxCount[k + 1];
                if (idxCount[k] + 1 > i) {
                    i1 = 0;
                    i = 0;
                }
                else {
                    i1 = idxCount[k];
                }
                loop_ub = i - i1;
                stats[k].PixelIdxList.set_size(loop_ub);
                for (i = 0; i < loop_ub; i++) {
                    stats[k].PixelIdxList[i] = CC_RegionIndices[i1 + i];
                }
            }
        }
        i = stats.size(0);
        for (k = 0; k < i; k++) {
            loop_ub = stats[k].PixelIdxList.size(0);
            if (stats[k].PixelIdxList.size(0) != 0) {
                regionLengths.set_size(stats[k].PixelIdxList.size(0));
                for (i1 = 0; i1 < loop_ub; i1++) {
                    regionLengths[i1] = static_cast<int>(stats[k].PixelIdxList[i1]) - 1;
                }
                idxCount.set_size(regionLengths.size(0));
                loop_ub = regionLengths.size(0);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    idxCount[i1] = div_s32(regionLengths[i1], static_cast<int>(wc[0]));
                }
                loop_ub = regionLengths.size(0);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    regionLengths[i1] =
                        regionLengths[i1] - idxCount[i1] * static_cast<int>(wc[0]);
                }
                stats[k].PixelList.set_size(idxCount.size(0), 2);
                loop_ub = idxCount.size(0);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    stats[k].PixelList[i1] = idxCount[i1] + 1;
                }
                loop_ub = regionLengths.size(0);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    stats[k].PixelList[i1 + stats[k].PixelList.size(0)] =
                        regionLengths[i1] + 1;
                }
            }
            else {
                stats[k].PixelList.set_size(0, 2);
            }
        }
        i = stats.size(0);
        if (0 <= stats.size(0) - 1) {
            wc[0] = varargin_2.size(0);
        }
        for (k = 0; k < i; k++) {
            loop_ub = stats[k].PixelIdxList.size(0);
            stats[k].PixelValues.set_size(stats[k].PixelIdxList.size(0));
            regionLengths.set_size(stats[k].PixelIdxList.size(0));
            for (i1 = 0; i1 < loop_ub; i1++) {
                regionLengths[i1] = static_cast<int>(stats[k].PixelIdxList[i1]) - 1;
            }
            idxCount.set_size(regionLengths.size(0));
            loop_ub = regionLengths.size(0);
            for (i1 = 0; i1 < loop_ub; i1++) {
                idxCount[i1] =
                    div_s32(regionLengths[i1],
                        static_cast<int>(static_cast<unsigned int>(wc[0])));
            }
            loop_ub = regionLengths.size(0);
            for (i1 = 0; i1 < loop_ub; i1++) {
                regionLengths[i1] =
                    regionLengths[i1] -
                    idxCount[i1] * static_cast<int>(static_cast<unsigned int>(wc[0]));
            }
            loop_ub = regionLengths.size(0);
            for (i1 = 0; i1 < loop_ub; i1++) {
                regionLengths[i1] = regionLengths[i1] + 1;
            }
            loop_ub = idxCount.size(0);
            for (i1 = 0; i1 < loop_ub; i1++) {
                idxCount[i1] = idxCount[i1] + 1;
            }
            i1 = stats[k].PixelValues.size(0);
            for (loop_ub_tmp = 0; loop_ub_tmp < i1; loop_ub_tmp++) {
                stats[k].PixelValues[loop_ub_tmp] =
                    varargin_2[(regionLengths[loop_ub_tmp] +
                        varargin_2.size(0) * (idxCount[loop_ub_tmp] - 1)) -
                    1];
            }
        }
        i = stats.size(0);
        for (k = 0; k < i; k++) {
            sumIntensity = sum(stats[k].PixelValues);
            loop_ub = stats[k].PixelList.size(0);
            for (loop_ub_tmp = 0; loop_ub_tmp < 2; loop_ub_tmp++) {
                CC_RegionIndices.set_size(loop_ub);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    CC_RegionIndices[i1] =
                        stats[k].PixelList[i1 + stats[k].PixelList.size(0) * loop_ub_tmp] *
                        stats[k].PixelValues[i1];
                }
                wc[loop_ub_tmp] = sum(CC_RegionIndices) / sumIntensity;
            }
            stats[k].WeightedCentroid[0] = wc[0];
            stats[k].WeightedCentroid[1] = wc[1];
        }
        i = stats.size(0);
        for (k = 0; k < i; k++) {
            outstats[k].WeightedCentroid[0] = stats[k].WeightedCentroid[0];
            outstats[k].WeightedCentroid[1] = stats[k].WeightedCentroid[1];
        }
    }

} // namespace coder

// End of code generation (regionprops.cpp)
