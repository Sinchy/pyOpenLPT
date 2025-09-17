#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Matrix.h"

namespace py = pybind11;

// -------- Matrix<T> 通用绑定（精简且稳妥）--------
template <typename T>
void bind_MatrixT(py::module_ &m, const char* py_name) {
    using Mat = Matrix<T>;
    py::class_<Mat> cls(m, py_name);

    // 构造
    cls.def(py::init<>())
       .def(py::init<int,int,T>(), py::arg("rows"), py::arg("cols"), py::arg("val"))
       .def(py::init<std::initializer_list<std::initializer_list<T>>>(), py::arg("init"))
       .def(py::init<const Mat&>(), py::arg("other"));
       // 如果你确实有 Matrix(std::string) / Matrix(int,int,istream&) 的定义再打开：
       // .def(py::init<std::string>(), py::arg("file_name"))
       // .def(py::init<int,int,std::istream&>(), py::arg("rows"), py::arg("cols"), py::arg("istream"))

    // 维度/行列
    cls.def("getDimRow", &Mat::getDimRow)
       .def("getDimCol", &Mat::getDimCol)
       .def("getRow",    &Mat::getRow, py::arg("row_i"))
       .def("getCol",    &Mat::getCol, py::arg("col_j"));

    // 打印/范数/转置/类型转换
    cls.def("print",     &Mat::print, py::arg("precision")=3)
       .def("norm",      &Mat::norm)
       .def("transpose", &Mat::transpose);
    //    .def("typeToDouble", &Mat::typeToDouble)
    //    .def("typeToInt",    &Mat::typeToInt);

    // 索引（A[i,j] / A[i]）
    cls.def("__getitem__", [](const Mat& self, std::pair<int,int> ij){ return self(ij.first, ij.second); })
       .def("__setitem__", [](Mat& self, std::pair<int,int> ij, const T& v){ self(ij.first, ij.second) = v; })
       .def("__getitem__", [](const Mat& self, int k){ return self[k]; })
       .def("__setitem__", [](Mat& self, int k, const T& v){ self[k] = v; });

    // 写文件（只保留文件名版本；避免 ostream& 重载引发模板匹配混乱）
    cls.def("write", [](Mat& self, const std::string& path){ self.write(path); }, py::arg("file_name"));

    // 比较
    cls.def("__eq__", &Mat::operator==, py::is_operator())
       .def("__ne__", &Mat::operator!=, py::is_operator());

    // 下面所有运算都用 lambda 包裹，避免 MSVC 在重载选择、const 限定符上的纠结
    // 矩阵 ⊕ 矩阵
    cls.def("__add__", [](const Mat& a, const Mat& b){ Mat r=a; r+=b; return r; }, py::is_operator())
       .def("__sub__", [](const Mat& a, const Mat& b){ Mat r=a; r-=b; return r; }, py::is_operator())
       .def("__mul__", [](const Mat& a, const Mat& b){ Mat r=a; r*=b; return r; }, py::is_operator())
       // 原地
       .def("__iadd__", [](Mat& a, const Mat& b)->Mat&{ a+=b; return a; }, py::is_operator())
       .def("__isub__", [](Mat& a, const Mat& b)->Mat&{ a-=b; return a; }, py::is_operator())
       .def("__imul__", [](Mat& a, const Mat& b)->Mat&{ a*=b; return a; }, py::is_operator());

    // 矩阵 ⊕ 标量（右操作数）
    cls.def("__add__", [](const Mat& a, const T& s){ Mat r=a; r+=s; return r; }, py::is_operator())
       .def("__sub__", [](const Mat& a, const T& s){ Mat r=a; r-=s; return r; }, py::is_operator())
       .def("__mul__", [](const Mat& a, const T& s){ Mat r=a; r*=s; return r; }, py::is_operator())
       .def("__truediv__", [](const Mat& a, const T& s){ Mat r=a; r/=s; return r; }, py::is_operator())
       // 原地
       .def("__iadd__", [](Mat& a, const T& s)->Mat&{ a+=s; return a; }, py::is_operator())
       .def("__isub__", [](Mat& a, const T& s)->Mat&{ a-=s; return a; }, py::is_operator())
       .def("__imul__", [](Mat& a, const T& s)->Mat&{ a*=s; return a; }, py::is_operator())
       .def("__itruediv__", [](Mat& a, const T& s)->Mat&{ a/=s; return a; }, py::is_operator());

    // __repr__
    cls.def("__repr__", [py_name](const Mat& mtx){
        return std::string("<") + py_name + " " +
               std::to_string(mtx.getDimRow()) + "x" +
               std::to_string(mtx.getDimCol()) + ">";
    });

    cls.def("to_list", [](const Mat& self){
        std::vector<std::vector<T>> out(self.getDimRow(), std::vector<T>(self.getDimCol()));
        for (int i = 0; i < self.getDimRow(); ++i)
            for (int j = 0; j < self.getDimCol(); ++j)
                out[i][j] = self(i,j);
        return out;
    });

}

// -------- 非模板派生/结构 --------
void bind_Matrix(py::module_& m) {
    // 只实例化你要暴露的类型
    bind_MatrixT<double>(m, "MatrixDouble");
    bind_MatrixT<int>(m,    "MatrixInt");

    // Pt2D / Pt3D 继承 Matrix<double>
    py::class_<Pt2D, Matrix<double>>(m, "Pt2D")
        .def(py::init<>())
        .def(py::init<double,double>())
        .def(py::init<const Pt2D&>())
        .def(py::init<const Matrix<double>&>())
        .def(py::init<std::string>())
        .def("__repr__", [](const Pt2D& p){
            return "<Pt2D ("+std::to_string(p[0])+","+std::to_string(p[1])+")>";
        });

    py::class_<Pt3D, Matrix<double>>(m, "Pt3D")
        .def(py::init<>())
        .def(py::init<double,double,double>())
        .def(py::init<const Pt3D&>())
        .def(py::init<const Matrix<double>&>())
        .def(py::init<std::string>())
        .def("__repr__", [](const Pt3D& p){
            return "<Pt3D ("+std::to_string(p[0])+","+std::to_string(p[1])+","+std::to_string(p[2])+")>";
        });

    py::class_<Line2D>(m, "Line2D")
        .def(py::init<>())
        .def_readwrite("pt",          &Line2D::pt)
        .def_readwrite("unit_vector", &Line2D::unit_vector);

    py::class_<Line3D>(m, "Line3D")
        .def(py::init<>())
        .def_readwrite("pt",          &Line3D::pt)
        .def_readwrite("unit_vector", &Line3D::unit_vector);

    py::class_<Plane3D>(m, "Plane3D")
        .def(py::init<>())
        .def_readwrite("pt",          &Plane3D::pt)
        .def_readwrite("norm_vector", &Plane3D::norm_vector);

    py::class_<Image, Matrix<double>>(m, "Image")
        .def(py::init<>())
        .def(py::init<int,int,double>(), py::arg("rows"), py::arg("cols"), py::arg("val")=0.0)
        .def(py::init<const Image&>())
        .def(py::init<const Matrix<double>&>())
        .def(py::init<std::string>())
        .def("save",
            // 直接绑定成员函数；默认参数由 py::arg 指定
            &Image::save,
            py::arg("path"),
            py::arg("bits_per_sample") = 8,   // 8/16（常用），也支持 32/64 无符号
            py::arg("n_channel")       = 1,   // 灰度=1
            py::arg("orientation")     = 1,   // 1 == ORIENTATION_TOPLEFT（避免在头里引入 tiff.h）
            py::call_guard<py::gil_scoped_release>() // I/O 期间释放 GIL
            )
        .def("__repr__", [](const Image& img){
            return "<Image " + std::to_string(img.getDimRow()) + "x" + std::to_string(img.getDimCol()) + ">";
        });
}
