//
//  Matrix.h
//
//  header file of class Matrix
//
//  Created  by Shijie Zhong on 07/02/2022
//

#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <iomanip>
#include <limits>
#include <math.h>
#include <stdlib.h>
#include <vector>


#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>


#include "STBCommons.h"

template <class T> class Matrix {
  int _dim_row = 0;
  int _dim_col = 0;
  int _n = 0;        // tot number of elem
  int _is_space = 0; // 0 for nothing installed; 1 for already claim space
  int mapID(int id_x, int id_y) const; // id_x*_dim_col + id_y
  T *_mtx;
  // std::vector<T> _mtx;

  // Create/Clear space
  void clear();
  // void ReCreateCol (int dim_col);
  void create(int dim_row, int dim_col);

public:
  // Constructor
  Matrix() {};
  Matrix(Matrix<T> const &mtx); // deep copy
  Matrix(int dim_row, int dim_col, T val);

  // dim_row, dim_col must be compatible with mtx
  Matrix(std::initializer_list<std::initializer_list<T>> mtx);

  // Load matrix from .csv file
  explicit Matrix(std::string file_name);

  // Load matrix from input stream
  explicit Matrix(int dim_row, int dim_col, std::istream &is);

  // Destructor
  ~Matrix();

  // Type transform
  Matrix<double> typeToDouble();
  Matrix<int> typeToInt();

  // Get/Assign value
  // Matrix.h
  T at(int r, int c) const;         // check index, for debug
  T &at(int r, int c);              // check index, for debug
  T operator()(int i, int j) const; // no check, fast
  T &operator()(int i, int j);      // no check, fast
  // Return _mtx[i], i = id_x*_dim_col + id_y
  // 0,1,2
  // 3,4,5 ...
  T operator[](int vec_i) const;
  T &operator[](int vec_i);

  // Get row
  std::vector<T> getRow(int row_i) const;
  // Get col
  std::vector<T> getCol(int col_j) const;

  // Get matrix info
  int getDimRow() const;
  int getDimCol() const;
  void print(int precision = 3) const;
  const T *data() const;
  void setData(const T *data, int size);

  // Matrix output
  void write(std::string file_name);
  void write(std::ostream &os);

  // Scalar operations
  double norm(); // sqrt(sum( xi^2 )) for all i

  // Matrix calculation
  Matrix<T> &operator=(Matrix<T> const &mtx);
  bool operator==(Matrix<T> const &mtx);
  bool operator!=(Matrix<T> const &mtx);
  Matrix<T> operator+(Matrix<T> const &mtx) const;
  Matrix<T> &operator+=(Matrix<T> const &mtx);
  Matrix<T> operator-(Matrix<T> const &mtx) const;
  Matrix<T> &operator-=(Matrix<T> const &mtx);
  Matrix<T> operator*(Matrix<T> const &mtx) const;
  Matrix<T> &operator*=(Matrix<T> const &mtx);
  Matrix<T> operator+(T delta) const;
  Matrix<T> &operator+=(T delta);
  Matrix<T> operator-(T delta) const;
  Matrix<T> &operator-=(T delta);
  Matrix<T> operator*(T ratio) const;
  Matrix<T> &operator*=(T ratio);
  Matrix<T> operator/(T ratio) const;
  Matrix<T> &operator/=(T ratio);

  // Matrix manipulation
  Matrix<T> transpose();
};

class Pt3D : public Matrix<double> {
public:
  Pt3D() : Matrix<double>(3, 1, 0) {};
  Pt3D(double x, double y, double z) : Matrix<double>({{x}, {y}, {z}}) {};
  Pt3D(const Pt3D &pt) : Matrix<double>(pt) {};
  Pt3D(const Matrix<double> &mtx)
      : Matrix<double>({{mtx[0]}, {mtx[1]}, {mtx[2]}}) {};
  Pt3D &operator=(const Pt3D &pt) = default;
  double operator*(Pt3D pt) const {
    return this->at(0, 0) * pt[0] + this->at(1, 0) * pt[1] +
           this->at(2, 0) * pt[2];
  }
  Pt3D operator*(double scalar) const {
    return Pt3D(this->at(0, 0) * scalar, this->at(1, 0) * scalar,
                this->at(2, 0) * scalar);
  }
  explicit Pt3D(std::string file_name) : Matrix<double>(file_name) {};
  explicit Pt3D(std::istream &is) : Matrix<double>(3, 1, is) {};
};

class Pt2D : public Matrix<double> {
public:
  Pt2D() : Matrix<double>(2, 1, 0) {};
  Pt2D(double x, double y) : Matrix<double>({{x}, {y}}) {};
  Pt2D(const Pt2D &pt) : Matrix<double>(pt) {};
  Pt2D(const Matrix<double> &mtx) : Matrix<double>({{mtx[0]}, {mtx[1]}}) {};
  Pt2D &operator=(const Pt2D &pt) = default;
  explicit Pt2D(std::string file_name) : Matrix<double>(file_name) {};
  explicit Pt2D(std::istream &is) : Matrix<double>(2, 1, is) {};
};

// Structure to store line
struct Line3D {
  Pt3D pt;
  Pt3D unit_vector;
};

struct Line2D {
  Pt2D pt;
  Pt2D unit_vector;
};

// Structure to store 3D plane
struct Plane3D {
  Pt3D pt;
  Pt3D norm_vector;
};

// Image: matrix with double type
// Image(row_id, col_id) = intensity
// row_id = img_y, col_id = img_x
class Image : public Matrix<double> {
public:
  Image() : Matrix<double>(1, 1, 0) {};
  Image(int dim_row, int dim_col, double val)
      : Matrix<double>(dim_row, dim_col, val) {};
  Image(std::initializer_list<std::initializer_list<double>> mtx)
      : Matrix<double>(mtx) {};
  Image(const Image &mtx) : Matrix<double>(mtx) {};
  Image(const Matrix<double> &mtx) : Matrix<double>(mtx) {};
  Image &operator=(const Image &mtx) = default;
  explicit Image(std::string file_name) : Matrix<double>(file_name) {};

  Image crop(int y0, int y1, int x0, int x1) const {
    const int H = y1 - y0, W = x1 - x0;
    Image sub(H, W, 0.0);
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        sub(r, c) = (*this)(y0 + r, x0 + c);
      }
    }
    return sub;
  }

  void save(const std::string &path, int bits_per_sample = 8, int n_channel = 1,
            std::uint16_t orientation = 1)
      const; // this function is in ImageIO.h because it uses ImageIO
};

#include "Matrix.hpp"

#endif
