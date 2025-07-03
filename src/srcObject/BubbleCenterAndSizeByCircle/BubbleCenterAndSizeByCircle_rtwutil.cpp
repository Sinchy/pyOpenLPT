//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// BubbleCenterAndSizeByCircle_rtwutil.cpp
//
// Code generation for function 'BubbleCenterAndSizeByCircle_rtwutil'
//

// Include files
#include "BubbleCenterAndSizeByCircle_rtwutil.h"
#include "rt_nonfinite.h"

// Function Definitions
int div_s32(int numerator, int denominator)
{
    int quotient;
    if (denominator == 0) {
        if (numerator >= 0) {
            quotient = MAX_int32_T;
        }
        else {
            quotient = MIN_int32_T;
        }
    }
    else {
        unsigned int b_denominator;
        unsigned int b_numerator;
        if (numerator < 0) {
            b_numerator = ~static_cast<unsigned int>(numerator) + 1U;
        }
        else {
            b_numerator = static_cast<unsigned int>(numerator);
        }
        if (denominator < 0) {
            b_denominator = ~static_cast<unsigned int>(denominator) + 1U;
        }
        else {
            b_denominator = static_cast<unsigned int>(denominator);
        }
        b_numerator /= b_denominator;
        if ((numerator < 0) != (denominator < 0)) {
            quotient = -static_cast<int>(b_numerator);
        }
        else {
            quotient = static_cast<int>(b_numerator);
        }
    }
    return quotient;
}

// End of code generation (BubbleCenterAndSizeByCircle_rtwutil.cpp)
