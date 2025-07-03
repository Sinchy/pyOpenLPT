//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// medfilt2.cpp
//
// Code generation for function 'medfilt2'
//

// Include files
#include "medfilt2.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Declarations
namespace coder {
    namespace images {
        namespace internal {
            namespace coder {
                namespace optimized {
                    static double median25(double vec[25]);

                }
            } // namespace coder
        } // namespace internal
    } // namespace images
} // namespace coder

// Function Definitions
namespace coder {
    namespace images {
        namespace internal {
            namespace coder {
                namespace optimized {
                    static double median25(double vec[25])
                    {
                        double temp;
                        if (vec[0] > vec[1]) {
                            temp = vec[0];
                            vec[0] = vec[1];
                            vec[1] = temp;
                        }
                        if (vec[3] > vec[4]) {
                            temp = vec[3];
                            vec[3] = vec[4];
                            vec[4] = temp;
                        }
                        if (vec[2] > vec[4]) {
                            temp = vec[2];
                            vec[2] = vec[4];
                            vec[4] = temp;
                        }
                        if (vec[2] > vec[3]) {
                            temp = vec[2];
                            vec[2] = vec[3];
                            vec[3] = temp;
                        }
                        if (vec[6] > vec[7]) {
                            temp = vec[6];
                            vec[6] = vec[7];
                            vec[7] = temp;
                        }
                        if (vec[5] > vec[7]) {
                            temp = vec[5];
                            vec[5] = vec[7];
                            vec[7] = temp;
                        }
                        if (vec[5] > vec[6]) {
                            temp = vec[5];
                            vec[5] = vec[6];
                            vec[6] = temp;
                        }
                        if (vec[9] > vec[10]) {
                            temp = vec[9];
                            vec[9] = vec[10];
                            vec[10] = temp;
                        }
                        if (vec[8] > vec[10]) {
                            temp = vec[8];
                            vec[8] = vec[10];
                            vec[10] = temp;
                        }
                        if (vec[8] > vec[9]) {
                            temp = vec[8];
                            vec[8] = vec[9];
                            vec[9] = temp;
                        }
                        if (vec[12] > vec[13]) {
                            temp = vec[12];
                            vec[12] = vec[13];
                            vec[13] = temp;
                        }
                        if (vec[11] > vec[13]) {
                            temp = vec[11];
                            vec[11] = vec[13];
                            vec[13] = temp;
                        }
                        if (vec[11] > vec[12]) {
                            temp = vec[11];
                            vec[11] = vec[12];
                            vec[12] = temp;
                        }
                        if (vec[15] > vec[16]) {
                            temp = vec[15];
                            vec[15] = vec[16];
                            vec[16] = temp;
                        }
                        if (vec[14] > vec[16]) {
                            temp = vec[14];
                            vec[14] = vec[16];
                            vec[16] = temp;
                        }
                        if (vec[14] > vec[15]) {
                            temp = vec[14];
                            vec[14] = vec[15];
                            vec[15] = temp;
                        }
                        if (vec[18] > vec[19]) {
                            temp = vec[18];
                            vec[18] = vec[19];
                            vec[19] = temp;
                        }
                        if (vec[17] > vec[19]) {
                            temp = vec[17];
                            vec[17] = vec[19];
                            vec[19] = temp;
                        }
                        if (vec[17] > vec[18]) {
                            temp = vec[17];
                            vec[17] = vec[18];
                            vec[18] = temp;
                        }
                        if (vec[21] > vec[22]) {
                            temp = vec[21];
                            vec[21] = vec[22];
                            vec[22] = temp;
                        }
                        if (vec[20] > vec[22]) {
                            temp = vec[20];
                            vec[20] = vec[22];
                            vec[22] = temp;
                        }
                        if (vec[20] > vec[21]) {
                            temp = vec[20];
                            vec[20] = vec[21];
                            vec[21] = temp;
                        }
                        if (vec[23] > vec[24]) {
                            temp = vec[23];
                            vec[23] = vec[24];
                            vec[24] = temp;
                        }
                        if (vec[2] > vec[5]) {
                            temp = vec[2];
                            vec[2] = vec[5];
                            vec[5] = temp;
                        }
                        if (vec[3] > vec[6]) {
                            temp = vec[3];
                            vec[3] = vec[6];
                            vec[6] = temp;
                        }
                        if (vec[0] > vec[6]) {
                            temp = vec[0];
                            vec[0] = vec[6];
                            vec[6] = temp;
                        }
                        if (vec[0] > vec[3]) {
                            temp = vec[0];
                            vec[0] = vec[3];
                            vec[3] = temp;
                        }
                        if (vec[4] > vec[7]) {
                            temp = vec[4];
                            vec[4] = vec[7];
                            vec[7] = temp;
                        }
                        if (vec[1] > vec[7]) {
                            temp = vec[1];
                            vec[1] = vec[7];
                            vec[7] = temp;
                        }
                        if (vec[1] > vec[4]) {
                            temp = vec[1];
                            vec[1] = vec[4];
                            vec[4] = temp;
                        }
                        if (vec[11] > vec[14]) {
                            temp = vec[11];
                            vec[11] = vec[14];
                            vec[14] = temp;
                        }
                        if (vec[8] > vec[14]) {
                            temp = vec[8];
                            vec[8] = vec[14];
                            vec[14] = temp;
                        }
                        if (vec[8] > vec[11]) {
                            temp = vec[8];
                            vec[8] = vec[11];
                            vec[11] = temp;
                        }
                        if (vec[12] > vec[15]) {
                            temp = vec[12];
                            vec[12] = vec[15];
                            vec[15] = temp;
                        }
                        if (vec[9] > vec[15]) {
                            temp = vec[9];
                            vec[9] = vec[15];
                            vec[15] = temp;
                        }
                        if (vec[9] > vec[12]) {
                            temp = vec[9];
                            vec[9] = vec[12];
                            vec[12] = temp;
                        }
                        if (vec[13] > vec[16]) {
                            temp = vec[13];
                            vec[13] = vec[16];
                            vec[16] = temp;
                        }
                        if (vec[10] > vec[16]) {
                            temp = vec[10];
                            vec[10] = vec[16];
                            vec[16] = temp;
                        }
                        if (vec[10] > vec[13]) {
                            temp = vec[10];
                            vec[10] = vec[13];
                            vec[13] = temp;
                        }
                        if (vec[20] > vec[23]) {
                            temp = vec[20];
                            vec[20] = vec[23];
                            vec[23] = temp;
                        }
                        if (vec[17] > vec[23]) {
                            temp = vec[17];
                            vec[17] = vec[23];
                            vec[23] = temp;
                        }
                        if (vec[17] > vec[20]) {
                            temp = vec[17];
                            vec[17] = vec[20];
                            vec[20] = temp;
                        }
                        if (vec[21] > vec[24]) {
                            temp = vec[21];
                            vec[21] = vec[24];
                            vec[24] = temp;
                        }
                        if (vec[18] > vec[24]) {
                            temp = vec[18];
                            vec[18] = vec[24];
                            vec[24] = temp;
                        }
                        if (vec[18] > vec[21]) {
                            temp = vec[18];
                            vec[18] = vec[21];
                            vec[21] = temp;
                        }
                        if (vec[19] > vec[22]) {
                            temp = vec[19];
                            vec[19] = vec[22];
                            vec[22] = temp;
                        }
                        if (vec[8] > vec[17]) {
                            vec[17] = vec[8];
                        }
                        if (vec[9] > vec[18]) {
                            temp = vec[9];
                            vec[9] = vec[18];
                            vec[18] = temp;
                        }
                        if (vec[0] > vec[18]) {
                            temp = vec[0];
                            vec[0] = vec[18];
                            vec[18] = temp;
                        }
                        if (vec[0] > vec[9]) {
                            vec[9] = vec[0];
                        }
                        if (vec[10] > vec[19]) {
                            temp = vec[10];
                            vec[10] = vec[19];
                            vec[19] = temp;
                        }
                        if (vec[1] > vec[19]) {
                            temp = vec[1];
                            vec[1] = vec[19];
                            vec[19] = temp;
                        }
                        if (vec[1] > vec[10]) {
                            temp = vec[1];
                            vec[1] = vec[10];
                            vec[10] = temp;
                        }
                        if (vec[11] > vec[20]) {
                            temp = vec[11];
                            vec[11] = vec[20];
                            vec[20] = temp;
                        }
                        if (vec[2] > vec[20]) {
                            temp = vec[2];
                            vec[2] = vec[20];
                            vec[20] = temp;
                        }
                        if (vec[2] > vec[11]) {
                            vec[11] = vec[2];
                        }
                        if (vec[12] > vec[21]) {
                            temp = vec[12];
                            vec[12] = vec[21];
                            vec[21] = temp;
                        }
                        if (vec[3] > vec[21]) {
                            temp = vec[3];
                            vec[3] = vec[21];
                            vec[21] = temp;
                        }
                        if (vec[3] > vec[12]) {
                            temp = vec[3];
                            vec[3] = vec[12];
                            vec[12] = temp;
                        }
                        if (vec[13] > vec[22]) {
                            temp = vec[13];
                            vec[13] = vec[22];
                            vec[22] = temp;
                        }
                        if (vec[4] > vec[22]) {
                            vec[4] = vec[22];
                        }
                        if (vec[4] > vec[13]) {
                            temp = vec[4];
                            vec[4] = vec[13];
                            vec[13] = temp;
                        }
                        if (vec[14] > vec[23]) {
                            temp = vec[14];
                            vec[14] = vec[23];
                            vec[23] = temp;
                        }
                        if (vec[5] > vec[23]) {
                            temp = vec[5];
                            vec[5] = vec[23];
                            vec[23] = temp;
                        }
                        if (vec[5] > vec[14]) {
                            temp = vec[5];
                            vec[5] = vec[14];
                            vec[14] = temp;
                        }
                        if (vec[15] > vec[24]) {
                            temp = vec[15];
                            vec[15] = vec[24];
                            vec[24] = temp;
                        }
                        if (vec[6] > vec[24]) {
                            vec[6] = vec[24];
                        }
                        if (vec[6] > vec[15]) {
                            temp = vec[6];
                            vec[6] = vec[15];
                            vec[15] = temp;
                        }
                        if (vec[7] > vec[16]) {
                            vec[7] = vec[16];
                        }
                        if (vec[7] > vec[19]) {
                            vec[7] = vec[19];
                        }
                        if (vec[13] > vec[21]) {
                            vec[13] = vec[21];
                        }
                        if (vec[15] > vec[23]) {
                            vec[15] = vec[23];
                        }
                        if (vec[7] > vec[13]) {
                            vec[7] = vec[13];
                        }
                        if (vec[7] > vec[15]) {
                            vec[7] = vec[15];
                        }
                        if (vec[1] > vec[9]) {
                            vec[9] = vec[1];
                        }
                        if (vec[3] > vec[11]) {
                            vec[11] = vec[3];
                        }
                        if (vec[5] > vec[17]) {
                            vec[17] = vec[5];
                        }
                        if (vec[11] > vec[17]) {
                            vec[17] = vec[11];
                        }
                        if (vec[9] > vec[17]) {
                            vec[17] = vec[9];
                        }
                        if (vec[4] > vec[10]) {
                            temp = vec[4];
                            vec[4] = vec[10];
                            vec[10] = temp;
                        }
                        if (vec[6] > vec[12]) {
                            temp = vec[6];
                            vec[6] = vec[12];
                            vec[12] = temp;
                        }
                        if (vec[7] > vec[14]) {
                            temp = vec[7];
                            vec[7] = vec[14];
                            vec[14] = temp;
                        }
                        if (vec[4] > vec[6]) {
                            temp = vec[4];
                            vec[4] = vec[6];
                            vec[6] = temp;
                        }
                        if (vec[4] > vec[7]) {
                            vec[7] = vec[4];
                        }
                        if (vec[12] > vec[14]) {
                            temp = vec[12];
                            vec[12] = vec[14];
                            vec[14] = temp;
                        }
                        if (vec[10] > vec[14]) {
                            vec[10] = vec[14];
                        }
                        if (vec[6] > vec[7]) {
                            temp = vec[6];
                            vec[6] = vec[7];
                            vec[7] = temp;
                        }
                        if (vec[10] > vec[12]) {
                            temp = vec[10];
                            vec[10] = vec[12];
                            vec[12] = temp;
                        }
                        if (vec[6] > vec[10]) {
                            temp = vec[6];
                            vec[6] = vec[10];
                            vec[10] = temp;
                        }
                        if (vec[6] > vec[17]) {
                            vec[17] = vec[6];
                        }
                        if (vec[12] > vec[17]) {
                            temp = vec[12];
                            vec[12] = vec[17];
                            vec[17] = temp;
                        }
                        if (vec[7] > vec[17]) {
                            vec[7] = vec[17];
                        }
                        if (vec[7] > vec[10]) {
                            temp = vec[7];
                            vec[7] = vec[10];
                            vec[10] = temp;
                        }
                        if (vec[12] > vec[18]) {
                            temp = vec[12];
                            vec[12] = vec[18];
                            vec[18] = temp;
                        }
                        if (vec[7] > vec[12]) {
                            vec[12] = vec[7];
                        }
                        if (vec[10] > vec[18]) {
                            vec[10] = vec[18];
                        }
                        if (vec[12] > vec[20]) {
                            temp = vec[12];
                            vec[12] = vec[20];
                            vec[20] = temp;
                        }
                        if (vec[10] > vec[20]) {
                            vec[10] = vec[20];
                        }
                        if (vec[10] > vec[12]) {
                            vec[12] = vec[10];
                        }
                        return vec[12];
                    }

                    void medfilt2(::coder::array<double, 2U>& inImg,
                        ::coder::array<double, 2U>& outImg)
                    {
                        array<double, 2U> a;
                        double region[25];
                        int c_i;
                        int i2;
                        int i3;
                        int m;
                        int n;
                        m = inImg.size(1) - 1;
                        n = inImg.size(0);
                        outImg.set_size(inImg.size(0), inImg.size(1));
                        if ((inImg.size(0) == 0) || (inImg.size(1) == 0)) {
                            int loop_ub;
                            unsigned int sizeB_idx_0;
                            unsigned int sizeB_idx_1;
                            sizeB_idx_0 = inImg.size(0) + 4U;
                            sizeB_idx_1 = inImg.size(1) + 4U;
                            inImg.set_size(static_cast<int>(sizeB_idx_0),
                                static_cast<int>(sizeB_idx_1));
                            loop_ub = static_cast<int>(sizeB_idx_0) * static_cast<int>(sizeB_idx_1);
                            for (int i{ 0 }; i < loop_ub; i++) {
                                inImg[i] = 0.0;
                            }
                        }
                        else {
                            int b_i;
                            int i;
                            int i1;
                            int j;
                            int loop_ub;
                            a.set_size(inImg.size(0), inImg.size(1));
                            loop_ub = inImg.size(0) * inImg.size(1);
                            for (i = 0; i < loop_ub; i++) {
                                a[i] = inImg[i];
                            }
                            unsigned int sizeB_idx_0;
                            unsigned int sizeB_idx_1;
                            sizeB_idx_0 = inImg.size(0) + 4U;
                            sizeB_idx_1 = inImg.size(1) + 4U;
                            inImg.set_size(static_cast<int>(sizeB_idx_0),
                                static_cast<int>(sizeB_idx_1));
                            for (j = 0; j < 2; j++) {
                                i = inImg.size(0);
                                for (b_i = 0; b_i < i; b_i++) {
                                    inImg[b_i + inImg.size(0) * j] = 0.0;
                                }
                            }
                            i = a.size(1) + 3;
                            loop_ub = inImg.size(1);
                            for (j = i; j <= loop_ub; j++) {
                                i1 = inImg.size(0);
                                for (b_i = 0; b_i < i1; b_i++) {
                                    inImg[b_i + inImg.size(0) * (j - 1)] = 0.0;
                                }
                            }
                            i = a.size(1);
                            for (j = 0; j < i; j++) {
                                inImg[inImg.size(0) * (j + 2)] = 0.0;
                                inImg[inImg.size(0) * (j + 2) + 1] = 0.0;
                            }
                            i = a.size(1);
                            for (j = 0; j < i; j++) {
                                loop_ub = a.size(0) + 3;
                                i1 = inImg.size(0);
                                for (b_i = loop_ub; b_i <= i1; b_i++) {
                                    inImg[(b_i + inImg.size(0) * (j + 2)) - 1] = 0.0;
                                }
                            }
                            i = a.size(1);
                            for (j = 0; j < i; j++) {
                                loop_ub = a.size(0);
                                for (b_i = 0; b_i < loop_ub; b_i++) {
                                    inImg[(b_i + inImg.size(0) * (j + 2)) + 2] = a[b_i + a.size(0) * j];
                                }
                            }
                        }
#pragma omp parallel for num_threads(omp_get_max_threads()) private(           \
    region, c_i, i2, i3)

                        for (int b_j = 0; b_j <= m; b_j++) {
                            for (c_i = 0; c_i < n; c_i++) {
                                for (i2 = 0; i2 < 5; i2++) {
                                    for (i3 = 0; i3 < 5; i3++) {
                                        region[i3 + 5 * i2] =
                                            inImg[static_cast<int>(static_cast<unsigned int>(i3) + c_i) +
                                            inImg.size(0) *
                                            static_cast<int>(static_cast<unsigned int>(i2) + b_j)];
                                    }
                                }
                                outImg[c_i + outImg.size(0) * b_j] = median25(region);
                            }
                        }
                    }

                } // namespace optimized
            } // namespace coder
        } // namespace internal
    } // namespace images
} // namespace coder

// End of code generation (medfilt2.cpp)
