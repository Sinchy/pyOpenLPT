#ifndef STBCOMMONS_H
#define STBCOMMONS_H

#include <iostream>
#include <string>
#include "../error.hpp"

#define SAVEPRECISION 8

#define SMALLNUMBER 1e-8
#define SQRTSMALLNUMBER 1e-6
#define MAGSMALLNUMBER 1e-8
#define M_PI 3.14159265358979323846

// log small
#define LOGSMALLNUMBER 1e-4

// undistort
#define UNDISTORT_MAX_ITER 50
#define UNDISTORT_EPS 1e-5

// Image point init
// if 2d projection is not found, then the value is -10
#define IMGPTINIT -10

// Track prediction
#define WIENER_MAX_ITER 5000

// STB parameters
#define UNLINKED -1
#define MAX_ERR_LINEARFIT 5e-2
#define LEN_LONG_TRACK 7

// for multimorphic objects
// for cloning derived classes
#define ENABLE_CLONE(Derived, Base) \
    std::unique_ptr<Base> clone() const override { \
        return std::make_unique<Derived>(*this);   \
    }

// delete copy constructor and assignment operator
#define DISABLE_COPY(Class)                   \
    Class(const Class&) = delete;             \
    Class& operator=(const Class&) = delete;

// enable move constructor and assignment operator
#define ENABLE_MOVE_NOEXCEPT(Class)           \
    Class(Class&&) noexcept = default;        \
    Class& operator=(Class&&) noexcept = default;

// combine disable copy and enable move
#define NONCOPYABLE_MOVABLE(Class)            \
    DISABLE_COPY(Class)                       \
    ENABLE_MOVE_NOEXCEPT(Class)

// for debugging private members
#ifdef OPENLPT_EXPOSE_PRIVATE
  #define FRIEND_DEBUG(classname) friend struct DebugAccess_##classname;
#else
  #define FRIEND_DEBUG(classname)
#endif


struct PixelRange 
{
    // left is closed, right is open 
    // [min, max)
    int row_min = 0;
    int row_max = 0;
    int col_min = 0;
    int col_max = 0;

    // Note: before using SetRowRange or SetColRange
    //       make sure it has been initialized!!!
    void setRowRange (int row)
    {
        if (row > row_max)
        {
            row_max = row;
        }
        else if (row < row_min)
        {
            row_min = row;
        }
    }; 

    // Note: before using SetRowRange or SetColRange
    //       make sure it has been initialized!!!
    void setColRange (int col)
    {
        if (col > col_max)
        {
            col_max = col;
        }
        else if (col < col_min)
        {
            col_min = col;
        }
    }; 

    // Note: before using SetRowRange or SetColRange
    //       make sure it has been initialized!!!
    void setRange (int row, int col)
    {
        setRowRange(row);
        setColRange(col);
    };
    int getNumOfRow () const
    {
        return row_max - row_min;
    };
    int getNumOfCol() const
    {
        return col_max - col_min;
    };
};

struct AxisLimit 
{
    double x_min = 0;
    double x_max = 0;
    double y_min = 0;
    double y_max = 0;
    double z_min = 0;
    double z_max = 0;

    AxisLimit () {};

    AxisLimit (double x1, double x2, double y1, double y2, double z1, double z2) 
        : x_min(x1), x_max(x2), y_min(y1), y_max(y2), z_min(z1), z_max(z2) {};

    void operator= (AxisLimit const& limit)
    {
        x_min = limit.x_min;
        x_max = limit.x_max;
        y_min = limit.y_min;
        y_max = limit.y_max;
        z_min = limit.z_min;
        z_max = limit.z_max;
    };

    bool check (double x, double y, double z)
    {
        if (x > x_max || x < x_min || 
            y > y_max || y < y_min || 
            z > z_max || z < z_min)
        {
            return false;
        }

        return true;
    }
};

// Minimal CSV helper: read a comma-delimited field (no quotes handling).
// Strips trailing '\r' if present (Windows line endings).
static inline bool read_csv_field(std::istream& in, std::string& out, char delim = ',') {
    if (!std::getline(in, out, delim)) return false;
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
};

enum ErrorTypeID
{
    error_size = 1,
    error_type,
    error_range,
    error_space,
    error_io,
    error_div0,
    error_parallel
};

enum ObjectTypeID
{
    type_tracer,
    type_bubble,
    type_filament
};

enum FrameTypeID
{
    PREV_FRAME,
    CURR_FRAME
};

enum TrackStatusID
{
    LONG_ACTIVE,
    SHORT_ACTIVE,
    LONG_INACTIVE,
    EXIT
};

enum TrackPredID
{
    WIENER = 0,
    KALMAN
};

#endif // !STBCOMMONS
