//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// CircleIdentifier.h
//
// Code generation for function 'CircleIdentifier'
//

#ifndef CIRCLEIDENTIFIER_H
#define CIRCLEIDENTIFIER_H

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "omp.h"
#include <cstddef>
#include <cstdlib>
#include <deque>
#include "Position.h"
#include "Frame.h"

using namespace std;
// Type Definitions
class CircleIdentifier {
public:
    CircleIdentifier(int**& p, int rows, int cols);// , int threshold);
    ~CircleIdentifier();
    vector<double> BubbleCenterAndSizeByCircle(double rmin, double rmax, double sense);
    Frame CreateFrame();
    void SaveCenter(string file_path);
    void SaveRadius(string file_path);

private:
    int**& img_p;
    int rows;
    int cols;
    //int threshold;
    coder::array<unsigned char, 2U> img;
    deque<Position> center;
    deque<double> radius;
};

#endif
// End of code generation (CircleIdentifier.h)