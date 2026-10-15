// CU automatic image stages. Independent reconstruction adapted from the audited
// cu_reverse source ports. No vendor DLL, legacy ABI, or OpenCV C++ linkage.
#include "cu_automatic.hpp"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
namespace py = pybind11;
namespace {
struct CuBudgetError : std::runtime_error { using std::runtime_error::runtime_error; };
struct CuRuntime {
    py::object cv2, checkpoint;
    size_t budget, held=0, peak=0, ticks=0, max_candidates;
    CuRuntime(py::object check,size_t limit,size_t maximum):cv2(py::module_::import("cv2")),checkpoint(std::move(check)),budget(limit),max_candidates(maximum){}
    void check(){py::gil_scoped_acquire gil;if(!checkpoint.is_none())checkpoint();}
    void reserve(size_t bytes){if(bytes>budget-held)throw CuBudgetError("automatic_memory_budget");held+=bytes;peak=std::max(peak,held);}
    void release(size_t bytes){held-=bytes;}
};
thread_local CuRuntime* cu_runtime=nullptr;
void cu_tick(){if(cu_runtime && ((++cu_runtime->ticks&16383)==0))cu_runtime->check();}
void cu_check(){if(cu_runtime)cu_runtime->check();}
struct CuReservation {size_t bytes;explicit CuReservation(size_t b):bytes(b){cu_runtime->reserve(b);}~CuReservation(){cu_runtime->release(bytes);}};
struct CuBytes {CuReservation reservation;std::vector<unsigned char> value;explicit CuBytes(size_t n):reservation(n),value(n){}};
}
template<class T> struct CuAllocator {
    using value_type=T;
    using propagate_on_container_move_assignment=std::true_type;
    CuRuntime* runtime=cu_runtime;
    CuAllocator()=default;
    template<class U> CuAllocator(const CuAllocator<U>& other):runtime(other.runtime){}
    T* allocate(size_t n){
        if(n>SIZE_MAX/sizeof(T))throw CuBudgetError("automatic_memory_budget");
        size_t bytes=n*sizeof(T);runtime->reserve(bytes);
        try{return std::allocator<T>{}.allocate(n);}catch(...){runtime->release(bytes);throw;}
    }
    void deallocate(T* p,size_t n){std::allocator<T>{}.deallocate(p,n);runtime->release(n*sizeof(T));}
    template<class U> bool operator==(const CuAllocator<U>& other)const{return runtime==other.runtime;}
    template<class U> bool operator!=(const CuAllocator<U>& other)const{return !(*this==other);}
};
template<class T> using CuVector=std::vector<T,CuAllocator<T>>;
// Deliberately small local image adapter: numeric loops remain C++; only batch
// cv2 primitives cross the Python boundary. Images have explicit owning storage.
namespace cucv {
constexpr int CV_8UC1=1,CV_8UC3=3,THRESH_BINARY=0,BORDER_CONSTANT=0,MORPH_RECT=0,MORPH_ELLIPSE=2,COLOR_BGR2GRAY=6,RETR_TREE=3,CHAIN_APPROX_NONE=1,INTER_LINEAR=1,DIST_L1=1;
struct Point{int x=0,y=0;Point()=default;constexpr Point(int xx,int yy):x(xx),y(yy){}Point& operator-=(Point q){x-=q.x;y-=q.y;return *this;}Point& operator+=(Point q){x+=q.x;y+=q.y;return *this;}};
Point operator+(Point a,Point b){return {a.x+b.x,a.y+b.y};}Point operator-(Point a,Point b){return {a.x-b.x,a.y-b.y};}
bool operator==(Point a,Point b){return a.x==b.x&&a.y==b.y;}bool operator!=(Point a,Point b){return !(a==b);}
struct Point2f{float x=0,y=0;Point2f()=default;Point2f(float xx,float yy):x(xx),y(yy){}explicit Point2f(Point p):x(float(p.x)),y(float(p.y)){}};
struct Size{int width=0,height=0;Size()=default;Size(int w,int h):width(w),height(h){}};
struct Rect{int x,y,width,height;Rect(int xx,int yy,int w,int h):x(xx),y(yy),width(w),height(h){}};
struct Scalar{double values[4]{};Scalar()=default;Scalar(double a,double b=0,double c=0,double d=0):values{a,b,c,d}{}double operator[](int i)const{return values[i];}static Scalar all(double v){return {v,v,v,v};}};
using Vec4i=std::array<int,4>;using Vec4f=std::array<float,4>;
struct RotatedRect{struct{float width=0,height=0;}size;};
struct Mat {
    int rows=0,cols=0,channels=1;size_t step=0,offset=0;std::shared_ptr<CuBytes> storage;
    Mat()=default;
    Mat(int h,int w,int c,unsigned char* source=nullptr):rows(h),cols(w),channels(c),step(size_t(w)*c){
        if(h<0||w<0||c<1||size_t(h)>SIZE_MAX/std::max(size_t(1),step))throw CuBudgetError("automatic_image_size");
        storage=std::make_shared<CuBytes>(size_t(h)*step);if(source)std::memcpy(storage->value.data(),source,size_t(h)*step);
    }
    bool empty()const{return rows==0||cols==0;}int type()const{return channels;}Size size()const{return {cols,rows};}
    template<class T> T* ptr(int y){cu_tick();return reinterpret_cast<T*>(storage->value.data()+offset+size_t(y)*step);}
    template<class T> const T* ptr(int y)const{cu_tick();return reinterpret_cast<const T*>(storage->value.data()+offset+size_t(y)*step);}
    template<class T>T& at(Point p){if(p.x<0||p.y<0||p.x>=cols||p.y>=rows)throw std::out_of_range("automatic_pixel_outside_image");return ptr<T>(p.y)[p.x];}
    template<class T>const T& at(Point p)const{if(p.x<0||p.y<0||p.x>=cols||p.y>=rows)throw std::out_of_range("automatic_pixel_outside_image");return ptr<T>(p.y)[p.x];}
    Mat clone()const{Mat out(rows,cols,channels);for(int y=0;y<rows;++y)std::memcpy(out.ptr<unsigned char>(y),ptr<unsigned char>(y),size_t(cols)*channels);return out;}
    Mat operator()(Rect r)const{if(r.x<0||r.y<0||r.width<0||r.height<0||r.x+r.width>cols||r.y+r.height>rows)throw std::out_of_range("automatic_roi_outside_image");Mat out=*this;out.offset+=size_t(r.y)*step+size_t(r.x)*channels;out.rows=r.height;out.cols=r.width;return out;}
    static Mat zeros(Size s,int c){return Mat(s.height,s.width,c);}static Mat zeros(int h,int w,int c){return Mat(h,w,c);}
    void setTo(Scalar s,const Mat& mask){for(int y=0;y<rows;++y){auto p=ptr<unsigned char>(y);auto m=mask.ptr<unsigned char>(y);for(int x=0;x<cols;++x)if(m[x])for(int c=0;c<channels;++c)p[x*channels+c]=static_cast<unsigned char>(s[c]);}}
};
py::array array(const Mat& m){
    // Callers hold the GIL and keep Mat alive until the synchronous cv2 call ends.
    auto* data=const_cast<unsigned char*>(m.ptr<unsigned char>(0));
    if(m.channels==1)return py::array(py::dtype::of<unsigned char>(),{m.rows,m.cols},{py::ssize_t(m.step),py::ssize_t(1)},data,py::none());
    return py::array(py::dtype::of<unsigned char>(),{m.rows,m.cols,m.channels},{py::ssize_t(m.step),py::ssize_t(m.channels),py::ssize_t(1)},data,py::none());
}
Mat from_array(py::handle value){auto a=py::array_t<unsigned char,py::array::c_style|py::array::forcecast>::ensure(value);if(!a||a.ndim()<2||a.ndim()>3)throw std::invalid_argument("cv2 image shape");return Mat(int(a.shape(0)),int(a.shape(1)),a.ndim()==3?int(a.shape(2)):1,const_cast<unsigned char*>(a.data()));}
py::tuple color(Scalar s){return py::make_tuple(s[0],s[1],s[2],s[3]);}
py::array points(const CuVector<Point>& c){py::array_t<int> a({py::ssize_t(c.size()),py::ssize_t(1),py::ssize_t(2)});auto p=a.mutable_data();for(auto q:c){*p++=q.x;*p++=q.y;}return a;}
void threshold(const Mat& src,Mat& dst,double threshold_value,double maximum,int){Mat out(src.rows,src.cols,src.channels);for(int y=0;y<src.rows;++y){auto s=src.ptr<unsigned char>(y);auto d=out.ptr<unsigned char>(y);for(int x=0;x<src.cols*src.channels;++x)d[x]=s[x]>threshold_value?static_cast<unsigned char>(maximum):0;}dst=std::move(out);}
Mat operator-(Scalar s,const Mat& src){Mat out=src.clone();for(int y=0;y<src.rows;++y)for(int x=0;x<src.cols*src.channels;++x)out.ptr<unsigned char>(y)[x]=static_cast<unsigned char>(s[x%src.channels]-src.ptr<unsigned char>(y)[x]);return out;}
Scalar mean(const Mat& m){double sums[4]{};for(int y=0;y<m.rows;++y){auto p=m.ptr<unsigned char>(y);for(int x=0;x<m.cols;++x)for(int c=0;c<m.channels;++c)sums[c]+=p[x*m.channels+c];}double n=double(m.rows)*m.cols;return {sums[0]/n,sums[1]/n,sums[2]/n,sums[3]/n};}
int countNonZero(const Mat&m){int n=0;for(int y=0;y<m.rows;++y){auto p=m.ptr<unsigned char>(y);for(int x=0;x<m.cols;++x)n+=p[x]!=0;}return n;}
double norm(Point p){return std::sqrt(double(p.x)*p.x+double(p.y)*p.y);}
int floodFill(Mat&m,Point seed,Scalar s,void*,Scalar,Scalar,int){
 unsigned char before=m.at<unsigned char>(seed),after=static_cast<unsigned char>(s[0]);if(before==after)return 0;
 CuVector<Point> pending;pending.push_back(seed);int count=0;
 while(!pending.empty()){
  Point p=pending.back();pending.pop_back();auto row=m.ptr<unsigned char>(p.y);if(row[p.x]!=before)continue;
  int left=p.x,right=p.x;while(left>0&&row[left-1]==before)--left;while(right+1<m.cols&&row[right+1]==before)++right;
  std::memset(row+left,after,size_t(right-left+1));count+=right-left+1;
  for(int y:{p.y-1,p.y+1})if(y>=0&&y<m.rows){auto adjacent=m.ptr<unsigned char>(y);bool run=false;
   for(int x=left;x<=right;++x){if(adjacent[x]==before){if(!run)pending.push_back({x,y});run=true;}else run=false;}}
 }
 return count;
}
void cvtColor(const Mat&src,Mat&dst,int code){cu_check();py::gil_scoped_acquire gil;dst=from_array(cu_runtime->cv2.attr("cvtColor")(array(src),code));}
void Canny(const Mat&src,Mat&dst,double low,double high,int aperture,bool l2){cu_check();py::gil_scoped_acquire gil;dst=from_array(cu_runtime->cv2.attr("Canny")(array(src),low,high,py::arg("apertureSize")=aperture,py::arg("L2gradient")=l2));}
void resize(const Mat&src,Mat&dst,Size size,double fx,double fy,int interpolation){cu_check();py::gil_scoped_acquire gil;dst=from_array(cu_runtime->cv2.attr("resize")(array(src),py::make_tuple(size.width,size.height),py::arg("fx")=fx,py::arg("fy")=fy,py::arg("interpolation")=interpolation));}
void copyMakeBorder(const Mat&src,Mat&dst,int top,int bottom,int left,int right,int,Scalar value){Mat out(src.rows+top+bottom,src.cols+left+right,src.channels);for(int y=0;y<out.rows;++y){auto p=out.ptr<unsigned char>(y);for(int x=0;x<out.cols*src.channels;++x)p[x]=static_cast<unsigned char>(value[x%src.channels]);}for(int y=0;y<src.rows;++y)std::memcpy(out.ptr<unsigned char>(y+top)+left*src.channels,src.ptr<unsigned char>(y),size_t(src.cols)*src.channels);dst=std::move(out);}
void bitwise_or(const Mat&a,const Mat&b,Mat&dst){if(a.rows!=b.rows||a.cols!=b.cols||a.channels!=b.channels)throw std::invalid_argument("automatic_image_size_mismatch");Mat out(a.rows,a.cols,a.channels);for(int y=0;y<a.rows;++y){auto pa=a.ptr<unsigned char>(y),pb=b.ptr<unsigned char>(y);auto p=out.ptr<unsigned char>(y);for(int x=0;x<a.cols*a.channels;++x)p[x]=pa[x]|pb[x];}dst=std::move(out);}
Mat getStructuringElement(int type,Size s){py::gil_scoped_acquire gil;return from_array(cu_runtime->cv2.attr("getStructuringElement")(type,py::make_tuple(s.width,s.height)));}
Scalar morphologyDefaultBorderValue(){return Scalar(0);}
void morph(const char* name,const Mat&src,Mat&dst,const Mat&kernel,Point anchor,int iterations){cu_check();py::gil_scoped_acquire gil;dst=from_array(cu_runtime->cv2.attr(name)(array(src),array(kernel),py::arg("anchor")=py::make_tuple(anchor.x,anchor.y),py::arg("iterations")=iterations,py::arg("borderType")=0));}
void dilate(const Mat&src,Mat&dst,const Mat&kernel,Point anchor,int iterations,int,Scalar){morph("dilate",src,dst,kernel,anchor,iterations);}void erode(const Mat&src,Mat&dst,const Mat&kernel,Point anchor,int iterations,int,Scalar){morph("erode",src,dst,kernel,anchor,iterations);}
void findContours(const Mat&src,CuVector<CuVector<Point>>&contours,CuVector<Vec4i>&hierarchy,int mode,int method){
 // Python's contour arrays carry per-component object overhead. Horizontal
 // foreground runs bound the number of components/holes; exposed pixel sides
 // bound CHAIN_APPROX_NONE point storage. Reserve before cv2 allocates them.
 cu_check();size_t runs=0,sides=0;
 for(int y=0;y<src.rows;++y){const auto* row=src.ptr<unsigned char>(y);
  const auto* prior=y?src.ptr<unsigned char>(y-1):nullptr;
  const auto* next=y+1<src.rows?src.ptr<unsigned char>(y+1):nullptr;
  for(int x=0;x<src.cols;++x)if(row[x]){runs+=x==0||!row[x-1];
   sides+=(x==0||!row[x-1])+(x+1==src.cols||!row[x+1])+(!prior||!prior[x])+(!next||!next[x]);}}
 CuReservation contour_arrays(runs*192+sides*8);
py::gil_scoped_acquire gil;auto result=cu_runtime->cv2.attr("findContours")(array(src),mode,method).cast<py::tuple>();contours.clear();hierarchy.clear();for(py::handle item:result[0]){auto a=py::array_t<int,py::array::c_style|py::array::forcecast>::ensure(item);CuVector<Point> c;const int*p=a.data();for(py::ssize_t i=0;i<a.size()/2;++i)c.emplace_back(p[2*i],p[2*i+1]);contours.push_back(std::move(c));}if(!result[1].is_none()){auto a=py::array_t<int,py::array::c_style|py::array::forcecast>::ensure(result[1]);const int*p=a.data();for(size_t i=0;i<contours.size();++i)hierarchy.push_back({p[4*i],p[4*i+1],p[4*i+2],p[4*i+3]});}}
// OpenCV 1.0 grayscale LINE_AA compatibility. Only the one-channel,
// thickness-one path is retained. See fdm/_native/licenses/opencv-legacy.txt.
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/* Correction table depent on the slope */
static const unsigned char icvSlopeCorrTable[] = {
    181, 181, 181, 182, 182, 183, 184, 185, 187, 188, 190, 192, 194, 196, 198, 201,
    203, 206, 209, 211, 214, 218, 221, 224, 227, 231, 235, 238, 242, 246, 250, 254
};

/* Gaussian for antialiasing filter */
static const int icvFilterTable[] = {
    168, 177, 185, 194, 202, 210, 218, 224, 231, 236, 241, 246, 249, 252, 254, 254,
    254, 254, 252, 249, 246, 241, 236, 231, 224, 218, 210, 202, 194, 185, 177, 168,
    158, 149, 140, 131, 122, 114, 105, 97, 89, 82, 75, 68, 62, 56, 50, 45,
    40, 36, 32, 28, 25, 22, 19, 16, 14, 12, 11, 9, 8, 7, 5, 5
};

static bool legacy_clip_line(Size img_size,Point* pt1,Point* pt2) {
 int x1,y1,x2,y2,c1,c2;
 int right=img_size.width-1,bottom=img_size.height-1;
 if(right<0||bottom<0)return false;
    x1 = pt1->x; y1 = pt1->y; x2 = pt2->x; y2 = pt2->y;
    c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
    c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;

    if( (c1 & c2) == 0 && (c1 | c2) != 0 )
    {
        int a;

        if( c1 & 12 )
        {
            a = c1 < 8 ? 0 : bottom;
            x1 += (int) (((int64_t) (a - y1)) * (x2 - x1) / (y2 - y1));
            y1 = a;
            c1 = (x1 < 0) + (x1 > right) * 2;
        }
        if( c2 & 12 )
        {
            a = c2 < 8 ? 0 : bottom;
            x2 += (int) (((int64_t) (a - y2)) * (x2 - x1) / (y2 - y1));
            y2 = a;
            c2 = (x2 < 0) + (x2 > right) * 2;
        }
        if( (c1 & c2) == 0 && (c1 | c2) != 0 )
        {
            if( c1 )
            {
                a = c1 == 1 ? 0 : right;
                y1 += (int) (((int64_t) (a - x1)) * (y2 - y1) / (x2 - x1));
                x1 = a;
                c1 = 0;
            }
            if( c2 )
            {
                a = c2 == 1 ? 0 : right;
                y2 += (int) (((int64_t) (a - x2)) * (y2 - y1) / (x2 - x1));
                x2 = a;
                c2 = 0;
            }
        }



        pt1->x = x1;
        pt1->y = y1;
        pt2->x = x2;
        pt2->y = y2;
    }

 return (c1|c2)==0;
}
static void legacy_line_aa(Mat& img, Point pt1,Point pt2,int cb) {
 if(img.rows<5||img.cols<5)return;
 constexpr int XY_SHIFT=16,XY_ONE=1<<16;
 pt1.x*=XY_ONE;pt1.y*=XY_ONE;pt2.x*=XY_ONE;pt2.y*=XY_ONE;
 int dx,dy,ecount,scount=0,slope,ax,ay,x_step,y_step,i,j,ep_table[9],_cb;
 unsigned char* ptr=img.ptr<unsigned char>(0);int step=int(img.step);Size size=img.size();
    pt1.x -= XY_ONE*2;
    pt1.y -= XY_ONE*2;
    pt2.x -= XY_ONE*2;
    pt2.y -= XY_ONE*2;
    ptr += step*2 + 2;

    size.width = ((size.width - 5) << XY_SHIFT) + 1;
    size.height = ((size.height - 5) << XY_SHIFT) + 1;

    if( !legacy_clip_line( size, &pt1, &pt2 ))
        return;

    dx = pt2.x - pt1.x;
    dy = pt2.y - pt1.y;

    j = dx < 0 ? -1 : 0;
    ax = (dx ^ j) - j;
    i = dy < 0 ? -1 : 0;
    ay = (dy ^ i) - i;

    if( ax > ay )
    {
        dx = ax;
        dy = (dy ^ j) - j;
        pt1.x ^= pt2.x & j;
        pt2.x ^= pt1.x & j;
        pt1.x ^= pt2.x & j;
        pt1.y ^= pt2.y & j;
        pt2.y ^= pt1.y & j;
        pt1.y ^= pt2.y & j;

        x_step = XY_ONE;
        y_step = (int) (((int64_t) dy * XY_ONE) / (ax | 1));
        pt2.x += XY_ONE;
        ecount = (pt2.x >> XY_SHIFT) - (pt1.x >> XY_SHIFT);
        j = -(pt1.x & (XY_ONE - 1));
        pt1.y += (int) ((((int64_t) y_step) * j) >> XY_SHIFT) + (XY_ONE >> 1);
        slope = (y_step >> (XY_SHIFT - 5)) & 0x3f;
        slope ^= (y_step < 0 ? 0x3f : 0);

        /* Get 4-bit fractions for end-point adjustments */
        i = (pt1.x >> (XY_SHIFT - 7)) & 0x78;
        j = (pt2.x >> (XY_SHIFT - 7)) & 0x78;
    }
    else
    {
        dy = ay;
        dx = (dx ^ i) - i;
        pt1.x ^= pt2.x & i;
        pt2.x ^= pt1.x & i;
        pt1.x ^= pt2.x & i;
        pt1.y ^= pt2.y & i;
        pt2.y ^= pt1.y & i;
        pt1.y ^= pt2.y & i;

        x_step = (int) (((int64_t) dx * XY_ONE) / (ay | 1));
        y_step = XY_ONE;
        pt2.y += XY_ONE;
        ecount = (pt2.y >> XY_SHIFT) - (pt1.y >> XY_SHIFT);
        j = -(pt1.y & (XY_ONE - 1));
        pt1.x += (int) ((((int64_t) x_step) * j) >> XY_SHIFT) + (XY_ONE >> 1);
        slope = (x_step >> (XY_SHIFT - 5)) & 0x3f;
        slope ^= (x_step < 0 ? 0x3f : 0);

        /* Get 4-bit fractions for end-point adjustments */
        i = (pt1.y >> (XY_SHIFT - 7)) & 0x78;
        j = (pt2.y >> (XY_SHIFT - 7)) & 0x78;
    }

    slope = (slope & 0x20) ? 0x100 : icvSlopeCorrTable[slope];

    /* Calc end point correction table */
    {
        int t0 = slope << 7;
        int t1 = ((0x78 - i) | 4) * slope;
        int t2 = (j | 4) * slope;

        ep_table[0] = 0;
        ep_table[8] = slope;
        ep_table[1] = ep_table[3] = ((((j - i) & 0x78) | 4) * slope >> 8) & 0x1ff;
        ep_table[2] = (t1 >> 8) & 0x1ff;
        ep_table[4] = ((((j - i) + 0x80) | 4) * slope >> 8) & 0x1ff;
        ep_table[5] = ((t1 + t0) >> 8) & 0x1ff;
        ep_table[6] = (t2 >> 8) & 0x1ff;
        ep_table[7] = ((t2 + t0) >> 8) & 0x1ff;
    }

        #define  ICV_PUT_POINT()            \
        {                                   \
            _cb = tptr[0];                  \
            _cb += ((cb - _cb)*a + 127)>> 8;\
            tptr[0] = (unsigned char)_cb;           \
        }

        if( ax > ay )
        {
            ptr += (pt1.x >> XY_SHIFT);

            while( ecount >= 0 )
            {
                unsigned char *tptr = ptr + ((pt1.y >> XY_SHIFT) - 1) * step;

                int ep_corr = ep_table[(((scount >= 2) + 1) & (scount | 2)) * 3 +
                                       (((ecount >= 2) + 1) & (ecount | 2))];
                int a, dist = (pt1.y >> (XY_SHIFT - 5)) & 31;

                a = (ep_corr * icvFilterTable[dist + 32] >> 8) & 0xff;
                ICV_PUT_POINT();
                ICV_PUT_POINT();

                tptr += step;
                a = (ep_corr * icvFilterTable[dist] >> 8) & 0xff;
                ICV_PUT_POINT();
                ICV_PUT_POINT();

                tptr += step;
                a = (ep_corr * icvFilterTable[63 - dist] >> 8) & 0xff;
                ICV_PUT_POINT();
                ICV_PUT_POINT();

                pt1.y += y_step;
                ptr++;
                scount++;
                ecount--;
            }
        }
        else
        {
            ptr += (pt1.y >> XY_SHIFT) * step;

            while( ecount >= 0 )
            {
                unsigned char *tptr = ptr + ((pt1.x >> XY_SHIFT) - 1);

                int ep_corr = ep_table[(((scount >= 2) + 1) & (scount | 2)) * 3 +
                                       (((ecount >= 2) + 1) & (ecount | 2))];
                int a, dist = (pt1.x >> (XY_SHIFT - 5)) & 31;

                a = (ep_corr * icvFilterTable[dist + 32] >> 8) & 0xff;
                ICV_PUT_POINT();
                ICV_PUT_POINT();

                tptr++;
                a = (ep_corr * icvFilterTable[dist] >> 8) & 0xff;
                ICV_PUT_POINT();
                ICV_PUT_POINT();

                tptr++;
                a = (ep_corr * icvFilterTable[63 - dist] >> 8) & 0xff;
                ICV_PUT_POINT();
                ICV_PUT_POINT();

                pt1.x += x_step;
                ptr += step;
                scount++;
                ecount--;
            }
        }
        #undef ICV_PUT_POINT

}
void drawContours(Mat&dst,const CuVector<CuVector<Point>>&cs,int i,Scalar s,int thickness,int line_type){cu_check();if(thickness==1&&line_type==16&&dst.channels==1){const auto& c=cs.at(size_t(i));for(size_t k=0;k<c.size();++k)legacy_line_aa(dst,c[k],c[(k+1)%c.size()],int(s[0]));return;}py::gil_scoped_acquire gil;py::list selected;selected.append(points(cs.at(size_t(i))));cu_runtime->cv2.attr("drawContours")(array(dst),selected,0,color(s),thickness,line_type);}
double contourArea(const CuVector<Point>&c){py::gil_scoped_acquire gil;return cu_runtime->cv2.attr("contourArea")(points(c)).cast<double>();}
double arcLength(const CuVector<Point>&c,bool closed){py::gil_scoped_acquire gil;return cu_runtime->cv2.attr("arcLength")(points(c),closed).cast<double>();}
// OpenCV 3.3 rotating-calipers dimensions. CU's strict aspect-ratio cutoffs
// depend on the original float operations and clockwise convex-hull order.
// Only width/height are needed; no center/angle or legacy OpenCV ABI is retained.
RotatedRect minAreaRect(const CuVector<Point>& contour){
 CuVector<Point2f> hull;
 {py::gil_scoped_acquire gil;auto result=py::array_t<int,py::array::c_style|py::array::forcecast>::ensure(cu_runtime->cv2.attr("convexHull")(points(contour),py::arg("clockwise")=true));
  const int* values=result.data();for(py::ssize_t i=0;i<result.size()/2;++i)hull.emplace_back(float(values[i*2]),float(values[i*2+1]));}
 RotatedRect box;const int n=int(hull.size());if(n<2)return box;
 if(n==2){double dx=hull[1].x-hull[0].x,dy=hull[1].y-hull[0].y;box.size.width=float(std::sqrt(dx*dx+dy*dy));return box;}
 CuVector<float> inverse(size_t(n),0);CuVector<Point2f> vector(size_t(n),Point2f{});
 int left=0,bottom=0,right=0,top=0;Point2f previous=hull[0];
 float left_x=previous.x,right_x=previous.x,top_y=previous.y,bottom_y=previous.y;
 for(int i=0;i<n;++i){
  if(previous.x<left_x){left_x=previous.x;left=i;}if(previous.x>right_x){right_x=previous.x;right=i;}
  if(previous.y>top_y){top_y=previous.y;top=i;}if(previous.y<bottom_y){bottom_y=previous.y;bottom=i;}
  Point2f next=hull[(i+1)%n];double dx=next.x-previous.x,dy=next.y-previous.y;
  vector[i]={float(dx),float(dy)};inverse[i]=float(1./std::sqrt(dx*dx+dy*dy));previous=next;
 }
 float orientation=0;double ax=vector[n-1].x,ay=vector[n-1].y;
 for(int i=0;i<n;++i){double bx=vector[i].x,by=vector[i].y,cross=ax*by-ay*bx;if(cross!=0){orientation=cross>0?1.f:-1.f;break;}ax=bx;ay=by;}
 if(orientation==0)throw std::runtime_error("automatic_degenerate_convex_hull");
 float base_a=orientation,base_b=0,minarea=std::numeric_limits<float>::max();
 float saved_a=0,saved_b=0,saved_width=0,saved_height=0;int sequence[4]={bottom,right,top,left};
 for(int k=0;k<n;++k){cu_tick();
  float dot[4]={base_a*vector[sequence[0]].x+base_b*vector[sequence[0]].y,
   -base_b*vector[sequence[1]].x+base_a*vector[sequence[1]].y,
   -base_a*vector[sequence[2]].x-base_b*vector[sequence[2]].y,
   base_b*vector[sequence[3]].x-base_a*vector[sequence[3]].y};
  float maxcos=dot[0]*inverse[sequence[0]];int main=0;
  for(int i=1;i<4;++i){float cosine=dot[i]*inverse[sequence[i]];if(cosine>maxcos){main=i;maxcos=cosine;}}
  int index=sequence[main];float lead_x=vector[index].x*inverse[index],lead_y=vector[index].y*inverse[index];
  switch(main){case 0:base_a=lead_x;base_b=lead_y;break;case 1:base_a=lead_y;base_b=-lead_x;break;
   case 2:base_a=-lead_x;base_b=-lead_y;break;case 3:base_a=-lead_y;base_b=lead_x;break;}
  sequence[main]=(sequence[main]+1)%n;
  float dx=hull[sequence[1]].x-hull[sequence[3]].x,dy=hull[sequence[1]].y-hull[sequence[3]].y;
  float width=dx*base_a+dy*base_b;
  dx=hull[sequence[2]].x-hull[sequence[0]].x;dy=hull[sequence[2]].y-hull[sequence[0]].y;
  float height=-dx*base_b+dy*base_a,area=width*height;
  if(area<=minarea){minarea=area;saved_a=base_a;saved_b=base_b;saved_width=width;saved_height=height;}
 }
 float x1=saved_a*saved_width,y1=saved_b*saved_width,x2=-saved_b*saved_height,y2=saved_a*saved_height;
 box.size.width=float(std::sqrt(double(x1)*x1+double(y1)*y1));
 box.size.height=float(std::sqrt(double(x2)*x2+double(y2)*y2));return box;
}
// Compatibility for the OpenCV 3.3 L1 fitLine stopping rule. OpenCV 4.13
// updates its best fit inside each iteration; CU used the older outer update.
// Adapted from OpenCV 3.3.0 modules/imgproc/src/linefit.cpp (Intel BSD license).
void legacy_fit_line_l1(const CuVector<Point2f>& points,Vec4f& output,float reps,float aeps){
 const int count=int(points.size());
 const double epsilon=count*std::numeric_limits<float>::epsilon();
 CuVector<float> weights(size_t(count),0),distances(size_t(count),0);
 auto fit=[&](Vec4f& line){
  double x=0,y=0,x2=0,y2=0,xy=0,w=0;
  for(int i=0;i<count;++i){float weight=weights[i];x+=weight*points[i].x;y+=weight*points[i].y;
   x2+=weight*points[i].x*points[i].x;y2+=weight*points[i].y*points[i].y;xy+=weight*points[i].x*points[i].y;w+=weight;}
  x/=w;y/=w;x2/=w;y2/=w;xy/=w;
  float angle=float(std::atan2(2*(xy-x*y),(x2-x*x)-(y2-y*y)))/2;
  line={float(std::cos(double(angle))),float(std::sin(double(angle))),float(x),float(y)};
 };
 uint64_t rng=UINT64_MAX;auto random=[&](){rng=uint64_t(uint32_t(rng))*4164903690U+uint32_t(rng>>32);return uint32_t(rng);};
 const float radius=reps!=0?reps:1.f,angle_limit=aeps!=0?aeps:.01f;
 double minimum=std::numeric_limits<double>::max(),error=0;output={0,0,0,0};
 for(int trial=0;trial<20;++trial){
  cu_check();std::fill(weights.begin(),weights.end(),0.f);
  for(int selected=0;selected<std::min(count,10);){int index=int(random()%uint32_t(count));if(weights[index]<std::numeric_limits<float>::epsilon()){weights[index]=1.f;++selected;}}
  Vec4f current{},previous{};fit(current);
  for(int iteration=0;iteration<30;++iteration){
   if(iteration){double dot=current[0]*previous[0]+current[1]*previous[1];dot=std::min(1.,std::max(-1.,dot));
    if(std::abs(std::acos(dot))<angle_limit){float dx=std::abs(current[2]-previous[2]),dy=std::abs(current[3]-previous[3]);if(std::max(dx,dy)<radius)break;}}
   error=0;float nx=current[1],ny=-current[0];
   for(int i=0;i<count;++i){float dx=points[i].x-current[2],dy=points[i].y-current[3];distances[i]=std::abs(nx*dx+ny*dy);error+=distances[i];}
   if(error<epsilon)break;
   double sum=0;for(int i=0;i<count;++i){weights[i]=float(1./std::max(std::abs(double(distances[i])),1.e-6));sum+=weights[i];}
   if(std::abs(sum)>std::numeric_limits<float>::epsilon()){sum=1./sum;for(float& weight:weights)weight=float(weight*sum);}
   else std::fill(weights.begin(),weights.end(),1.f);
   previous=current;fit(current);
  }
  if(error<minimum){minimum=error;output=current;if(error<epsilon)break;}
 }
}
void fitLine(const CuVector<Point2f>& c,Vec4f& line,int dist,double,double reps,double aeps){if(dist!=DIST_L1||c.empty())throw std::invalid_argument("legacy_fitLine requires nonempty L1 points");legacy_fit_line_l1(c,line,float(reps),float(aeps));}
double pointPolygonTest(const CuVector<Point>&c,Point2f p,bool distance){py::gil_scoped_acquire gil;return cu_runtime->cv2.attr("pointPolygonTest")(points(c),py::make_tuple(p.x,p.y),distance).cast<double>();}
void line(Mat&dst,Point a,Point b,Scalar s,int thick,int type,int shift=0){cu_tick();py::gil_scoped_acquire gil;cu_runtime->cv2.attr("line")(array(dst),py::make_tuple(a.x,a.y),py::make_tuple(b.x,b.y),color(s),thick,type,shift);}
void circle(Mat&dst,Point p,int radius,Scalar s,int thick,int type){cu_tick();py::gil_scoped_acquire gil;cu_runtime->cv2.attr("circle")(array(dst),py::make_tuple(p.x,p.y),radius,color(s),thick,type);}
}
using cucv::CV_8UC1;using cucv::CV_8UC3;
#define CV_Assert(x) do{if(!(x))throw std::invalid_argument("automatic_internal_invariant: " #x);}while(0)

namespace cu_legacy {
using Point=cucv::Point;
struct Image {cucv::Mat matrix;int width=0,height=0,channels=1,depth=8,image_size=0,step=0;unsigned char* data=nullptr;void sync(){width=matrix.cols;height=matrix.rows;channels=matrix.channels;step=int(matrix.step);image_size=height*step;data=matrix.ptr<unsigned char>(0);}};
struct Circle{int owner,used,endpoint,reserved,cross;Point center,p,q;int pad;double radius;int extra[2];};
struct Fiber{int first=0,last=0,count=0;int indices[301]{};double diameter=0,length=0;};
struct Order{int index,neighbors;};
constexpr Point neighbors[8]={{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1},{1,1}};
constexpr double pi=3.1415926;
constexpr double deg=0.017453292222222222;
int manhattan(Point a,Point b) { return std::abs(a.x-b.x)+std::abs(a.y-b.y); }
double length(Point a,Point b) {
    double dx=double(b.x-a.x),dy=double(b.y-a.y);
    return std::sqrt(dx*dx+dy*dy);
}
double dot(Point a,Point b,Point c,Point d) {
    return double(b.x-a.x)*double(d.x-c.x)+double(b.y-a.y)*double(d.y-c.y);
}
double cosine(Point a,Point b,Point c,Point d) { return dot(a,b,c,d)/(length(a,b)*length(c,d)); }
bool inside(Image* im,Point p) { return p.x>=0 && p.y>=0 && p.x<im->width && p.y<im->height; }
bool interior(Image* im,Point p) { return p.x>0 && p.y>0 && p.x<im->width && p.y<im->height; }
unsigned char& pixel(Image* im,Point p) { cu_tick(); return im->data[p.y*im->step+p.x]; }
unsigned char read(Image* im,Point p) { return inside(im,p)?pixel(im,p):255; }

struct Engine {
    // Stable prefix is compatible with the original result reader. Private C++
    // ownership follows it, so a runner can read +8/+14/+94/+98 unchanged.
    std::map<size_t,int> ints; std::map<size_t,double> doubles;
    Image* gray_image=nullptr; Image* mask_image=nullptr; Image* contour_image=nullptr;
    CuVector<Circle> circles;
    CuVector<Fiber> fibers;
    CuVector<Order> order;
    CuVector<CuVector<signed char>> relation;
    CuVector<CuVector<int>> radii, map0, map1;
    int capacity;
    double angles[7]{};

    template<class T> T& field(size_t off) { if constexpr(std::is_same_v<T,int>)return ints[off];else return doubles[off]; }
    Image*& gray() { return gray_image; }
    Image*& mask() { return mask_image; }
    Image*& contour() { return contour_image; }

    int& count() { return field<int>(0x98); }
    int& split() { return field<int>(0xa4); }
    explicit Engine(size_t maximum):circles(maximum),capacity(int(maximum)) {

        field<int>(0x40)=7; field<int>(0x44)=50; field<int>(0x48)=250;
        field<double>(0x50)=field<double>(0x78)=30;
        field<double>(0x58)=field<double>(0x80)=.3;
        field<int>(0x60)=field<int>(0x70)=5;
        field<int>(0x64)=field<int>(0x88)=8;
        field<int>(0x68)=field<int>(0x8c)=60;
        field<int>(0x6c)=field<int>(0x90)=1;
        field<double>(0xa8)=std::cos(0.34906584444444444);
        split()=-1;
        angles[0]=1;
        for(int i=1;i<6;++i) angles[i]=std::cos(double(i)*pi*(1.0/12.0));
        angles[6]=0;

    }
    ~Engine(){delete gray_image;delete mask_image;delete contour_image;}
    // 0x10005b10: global maximum between-class-variance threshold, first maximum.
    static int threshold(Image* im) {
        int hist[256]{};
        for(int y=0;y<im->height;++y) for(int x=0;x<im->width;++x) ++hist[pixel(im,{x,y})];
        double total=0,weighted=0;
        for(int i=0;i<256;++i) { total+=hist[i]; weighted+=double(hist[i])*(i+1); }
        double scores[256]{};
        for(int t=0;t<256;++t) {
            double n0=0,s0=0;
            for(int i=0;i<=t;++i) { n0+=hist[i]; s0+=double(hist[i])*(i+1); }
            double n1=total-n0,m0=n0?s0/n0:0,m1=n1?(weighted-s0)/n1:0;
            scores[t]=(m0-m1)*(m0-m1)*n1*n0;
        }
        int selected=0;
        for(int t=1;t<256;++t) if(scores[t]>scores[selected]) selected=t;
        for(int y=0;y<im->height;++y) for(int x=0;x<im->width;++x)
            pixel(im,{x,y})=pixel(im,{x,y})<=selected?0:255;
        return selected;
    }
    // 0x10005d40: labels 125/254/1 prevent revisiting recolored components.
    static void cleanup(Image* im) {
        CuVector<Point> q; q.reserve(size_t(im->width)*im->height);
        for(int y=0;y<im->height;++y) for(int x=0;x<im->width;++x) {
            unsigned char value=pixel(im,{x,y});
            if(value!=0 && value!=255) continue;
            q.clear(); q.push_back({x,y}); pixel(im,{x,y})=125;
            for(size_t i=0;i<q.size();++i) {
                Point p=q[i];
                for(Point d:neighbors) {
                    Point n={p.x+d.x,p.y+d.y};
                    if(inside(im,n) && pixel(im,n)==value) { pixel(im,n)=125; q.push_back(n); }
                }
            }
            unsigned char label=value==255?(q.size()>250?254:1):(q.size()>50?1:254);
            for(Point p:q) pixel(im,p)=label;
        }
        for(int y=0;y<im->height;++y) for(int x=0;x<im->width;++x)
            pixel(im,{x,y})=pixel(im,{x,y})==1?0:255;
    }
    // 0x100060f0 and 0x10006310: old OpenCV contour drawing is antialiased.
    int preprocess(Image* g,Image* edge,Image* binary) {
        {
            py::gil_scoped_acquire gil;
            // Legacy cvSmooth uses replicate borders, unlike GaussianBlur's default.
            binary->matrix=cucv::from_array(cu_runtime->cv2.attr("GaussianBlur")(cucv::array(g->matrix),py::make_tuple(7,7),0,py::arg("borderType")=1));
            binary->sync();
        }
        threshold(binary);cleanup(binary);
        cucv::Mat inverted=cucv::Scalar::all(255)-binary->matrix;
        // OpenCV 1.0 contour scanning clears the image border before tracing.
        for(int x=0;x<inverted.cols;++x){inverted.ptr<unsigned char>(0)[x]=0;inverted.ptr<unsigned char>(inverted.rows-1)[x]=0;}
        for(int y=0;y<inverted.rows;++y){inverted.ptr<unsigned char>(y)[0]=0;inverted.ptr<unsigned char>(y)[inverted.cols-1]=0;}
        CuVector<CuVector<cucv::Point>> contours;CuVector<cucv::Vec4i> hierarchy;
        cucv::findContours(inverted,contours,hierarchy,3,1);
        edge->matrix=cucv::Mat(g->height,g->width,1);
        std::fill(edge->matrix.storage->value.begin(),edge->matrix.storage->value.end(),255);
        // cvDrawContours(first, maxLevel=2) traverses all roots and their children.
        for(size_t i=0;i<contours.size();++i){int depth=0,parent=hierarchy[i][3];while(parent>=0){++depth;parent=hierarchy[parent][3];}if(depth<2)cucv::drawContours(edge->matrix,contours,int(i),cucv::Scalar(0),1,16);}
        edge->sync();threshold(edge);
        return 0;
    }

    // 0x100053b0. Regression swaps axes when y has the larger extent.
    static bool fitting(CuVector<Point>& pts,double& k,double& b) {
        int n=int(pts.size()); if(n<5) return false;
        int xmin=999999,ymin=999999,xmax=0,ymax=0;
        for(Point p:pts) { xmin=std::min(xmin,p.x);ymin=std::min(ymin,p.y);xmax=std::max(xmax,p.x);ymax=std::max(ymax,p.y); }
        bool swap=xmax-xmin<ymax-ymin;
        double sxy=0,sxx=0,sx=0,sy=0;
        for(Point p:pts) { if(swap) std::swap(p.x,p.y); sxy+=double(p.x*p.y);sxx+=double(p.x*p.x);sx+=p.x;sy+=p.y; }
        double denom=double(n)*sxx-sx*sx;
        if(std::abs(denom)<=1e-5) {
            if(!swap) return false;
            k=0; b=sx/double(n); return true;
        }
        k=(double(n)*sxy-sy*sx)/denom; b=(sy-k*sx)/double(n);
        if(swap) { if(std::abs(k)<.0001) return false; k=1/k; b=-(k*b); }
        return true;
    }
    // Shared 8-neighbor collection; order is significant because the first ten
    // contour pixels determine the fitted line and endpoints saved by the DLL.
    bool gather(Point seed,CuVector<Point>& pts,bool stop_at_border,bool require_seed) {
        Image* im=contour(); if(!interior(im,seed)) return false;
        Point start=seed;
        if(pixel(im,start)!=0) {
            bool found=false;
            for(Point d:neighbors) { Point p={seed.x+d.x,seed.y+d.y}; if(inside(im,p)&&pixel(im,p)==0){start=p;found=true;break;} }
            if(!found) {
                if(require_seed)return false;
                // IsContourSlope's first edge, unlike its second edge, falls
                // through with the last neighbor even when no zero was found.
                start={seed.x+neighbors[7].x,seed.y+neighbors[7].y};
                if(!inside(im,start))return false; // bound the original invalid-pointer corner case
            }
        }
        pts.clear(); pts.push_back(start); pixel(im,start)=1;
        bool proceed=true;
        for(size_t i=0;i<pts.size() && pts.size()<10 && proceed;++i) {
            Point p=pts[i];
            for(Point d:neighbors) {
                Point q={p.x+d.x,p.y+d.y};
                if(!inside(im,q)) { if(stop_at_border){proceed=false;break;} continue; }
                if(pixel(im,q)==0) { if(pts.size()>=10) break; pts.push_back(q); pixel(im,q)=1; }
            }
        }
        for(Point p:pts) if(!stop_at_border||interior(im,p)) pixel(im,p)=0;
        return true;
    }
    // 0x10001a80. Horizontal/vertical special cases deliberately scan toward
    // increasing x/y only. Oblique cases choose the more distant directional hit.
    bool opposite(Point seed,Point& result) {
        CuVector<Point> pts;
        if(!gather(seed,pts,false,true)) return false;
        double k=0,b=0;
        if(!fitting(pts,k,b)) {
            int x=seed.x; while(x<mask()->width && read(mask(),{x,seed.y})!=255) ++x;
            result={x-1,seed.y}; return true;
        }
        if(std::abs(k)<=1e-5) {
            int y=seed.y; while(y<mask()->height && read(mask(),{seed.x,y})!=255) ++y;
            result={seed.x,y-1}; return true;
        }
        double normal=-1/k,intercept=double(seed.y)-double(seed.x)*normal;
        Point hit[2]={{-1,-1},{-1,-1}};
        for(int pass=0;pass<2;++pass) {
            double t=std::abs(normal)>1?seed.y:seed.x;
            int direction=pass?1:-1;
            for(;;) {
                t+=direction;
                double x=std::abs(normal)>1?(t-intercept)/normal:t;
                double y=std::abs(normal)>1?t:t*normal+intercept;
                if(x<=0||y<=0||x>=mask()->width||y>=mask()->height) break;
                Point p={int(x+.5),int(y+.5)}; hit[pass]=p;
                if(read(mask(),p)==255) break;
            }
        }
        result=manhattan(hit[0],seed)<manhattan(hit[1],seed)?hit[1]:hit[0];
        return true;
    }
    // 0x100024c0: retain n2*k1-1 denominator, rather than textbook k1*k2+1.
    bool contour_slope(Point& p,Point& q,Point& center,double& radius) {
        if(!interior(contour(),p)||!interior(contour(),q)) return false;
        CuVector<Point> a,b;
        if(!gather(p,a,true,false)||!gather(q,b,true,true)) return false;
        double k1=0,b1=0,k2=0,b2=0;
        bool fa=fitting(a,k1,b1),fb=fitting(b,k2,b2);
        if(fa!=fb) return false;
        if(fa) { double t=(k2-k1)/(double(b.size())*k1+1-2);
            if(!(std::cos(field<double>(0x78)*deg)<std::sqrt(1/(t*t+1)))) return false; }
        radius=std::sqrt(double((a[0].x-b[0].x)*(a[0].x-b[0].x)+(a[0].y-b[0].y)*(a[0].y-b[0].y)))*.5;
        center={(p.x+q.x)/2,(p.y+q.y)/2};
        p={a[0].y,a[0].x}; q={b[0].y,b[0].x};
        return true;
    }
    // 0x10002160: preserve the two original, distinct raster recurrences.
    bool draw_circle(Point c,int radius,unsigned char mark,int next,int& replacement) {
        if(radius==0||c.x-radius<0||c.y-radius<0||c.x+radius>=mask()->width||c.y+radius>=mask()->height) return false;
        int previous=radii[c.x][c.y]; bool replace=previous>0;
        if(replace && radius<=previous) return false;
        int error=0,x=c.x,y=c.y+radius;
        while(x!=c.x+radius && y!=c.y) {
            if(error<0) {++x;error+=2*(x-c.x)+1;}
            else {--y;error+=1+2*(c.y-y);}
            for(int a=2*c.x-x;a<=x;++a) {
                unsigned char p=read(mask(),{a,y}),q=read(mask(),{a,2*c.y-y});
                if(p==255||q==255||p==mark||q==mark) return false;
            }
        }
        if(replace) for(int i=0;i<next;++i) if(circles[i].center.x==c.x&&circles[i].center.y==c.y){replacement=i;break;}
        radii[c.x][c.y]=radius;
        int dx=0,dy=radius,right=c.x,left=c.x;
        auto paint=[&](int y0,int x0,int x1){for(int z=x0;z<x1;++z) if(inside(mask(),{z,y0})) {
            pixel(mask(),{z,y0})=mark; (split()<0?map0:map1)[y0][z]=next;
        }};
        int guard=0;
        do {
            if(error<0){++dx;++right;--left;error+=1+2*dx;}
            else {--dy;error+=1-2*dy;}
            paint(c.y+dy,left,right); paint(c.y-c.x+right,c.x-dy,c.x+dy);
            paint(c.y-c.x+left,c.x-dy,c.x+dy); paint(c.y-dy,left,right);
            if(++guard>4*radius+16) throw std::runtime_error("original circle recurrence exceeded bounds");
        } while(dx!=dy);
        return true;
    }

    // 0x100035a0, Bresenham branches with original endpoint acceptance rule.
    bool clear_line(Point a,Point b,int forbidden=255) {
        double slope=double(b.y-a.y)/double(b.x-a.x);
        if(b.x<a.x) std::swap(a,b);
        int dx=b.x-a.x,dy=b.y-a.y,x=a.x,y=a.y; bool clear=true;
        auto test=[&](){if(read(mask(),{x,y})==forbidden) clear=false;};
        if(slope>0 && slope<=1) {int e=dx-2*dy;while(x!=b.x||y!=b.y){if(e<1){++y;e+=2*(dx-dy);}else e-=2*dy;++x;test();if(!clear)break;}}
        else if(slope<0 && slope>=-1) {int e=-(dx+2*dy);while(x!=b.x||y!=b.y){if(e<1)e-=2*dy;else{--y;e-=2*(dx+dy);}++x;test();if(!clear)break;}}
        else if(slope>1) {int e=2*dx-dy;while(x!=b.x||y!=b.y){if(e<1)e+=2*dx;else{++x;e+=2*(dx-dy);}++y;test();if(!clear)break;}}
        else if(slope<-1) {int e=-(2*dx+dy);while(x!=b.x||y!=b.y){if(e<1){++x;e-=2*(dx+dy);}else e-=2*dx;--y;test();if(!clear)break;}}
        else if(slope==0) {while(x!=b.x||y!=b.y){if(dx>0)++x;else if(dx<0)--x;test();if(!clear)break;}}
        return (x==b.x&&y==b.y)||clear;
    }
    // 0x10003b10 uses the same historical check recurrence as draw_circle.
    bool scan_circle(Point c,int radius) {
        if(c.x-radius<0||c.y-radius<0||c.x+radius>=mask()->width||c.y+radius>=mask()->height) return false;
        int x=c.x,y=c.y+radius,e=0;
        while(x!=c.x+radius && y!=c.y) {
            if(e<0){++x;e+=2*(x-c.x)+1;}else{--y;e+=1+2*(c.y-y);}
            if(read(mask(),{x,y})==255||read(mask(),{x,2*c.y-y})==255||read(mask(),{2*c.x-x,y})==255||read(mask(),{2*c.x-x,2*c.y-y})==255)return false;
        }
        return true;
    }
    int connect(int a,int b) {
        cu_tick();signed char cached=relation[a][b]; if(cached!=-1) return cached;
        const Circle& p=circles[a];const Circle& q=circles[b];
        if(p.center.x==q.center.x&&p.center.y==q.center.y)return 0;
        if(!clear_line(p.center,q.center))return 0;
        int code=scan_circle({(p.center.x+q.center.x)/2,(p.center.y+q.center.y)/2},int((p.radius+q.radius)*.5*.8+.5))?2:1;
        double ratio=(std::max(p.radius,q.radius)-std::min(p.radius,q.radius))/std::min(p.radius,q.radius);
        if(ratio<field<double>(0x80))code+=2;
        return code;
    }
    bool same_scan(int a,int b) {return (a<split())==(b<split());}
    int is_cross(int at) {
        Point origin=circles[at].center,first{},second{};int found=0;
        for(int i=0;i<count();++i) if(i!=at) {
            int code=connect(i,at);if(code!=2&&code!=4)continue;
            Point p=circles[i].center;double r=circles[i].radius+circles[at].radius;
            int dx=origin.x-p.x,dy=origin.y-p.y;
            if(r*r>double(dx*dx+dy*dy))continue;
            if(!found){first=p;found=1;}
            else if(found==1){if(cosine(origin,first,origin,p)<field<double>(0xa8)){second=p;found=2;}}
            else if(cosine(origin,first,origin,p)<=field<double>(0xa8)&&cosine(origin,second,origin,p)<=field<double>(0xa8))return 1;
        }
        return 0;
    }
    bool is_end(int at) {
        int found=0;bool same=false,ok=true;Point first{},origin=circles[at].center;
        for(int i=0;i<count();++i) if(i!=at && connect(i,at)==4) {
            if(!found){found=1;first=circles[i].center;same=same_scan(i,at);}
            else {++found;Point p=circles[i].center;
                if(dot(origin,first,origin,p)<=0||(same&&same_scan(i,at)&&cosine(origin,first,origin,p)<field<double>(0xa8))){ok=false;break;}}
        }
        return found==1?(same&&ok):(found>0&&ok);
    }
    // 0x100045b0. Equal-neighbor ordering must remain the original quicksort.
    void sort_order(int left,int right) {
        do {int pivot=order[(left+right)/2].neighbors,i=left,j=right;
            do {while(order[i].neighbors<pivot&&i<right)++i;while(pivot<order[j].neighbors&&left<j)--j;
                if(j<i)break;std::swap(order[i],order[j]);++i;--j;
            }while(i<=j);
            if(left<j)sort_order(left,j);left=i;if(right<=i)return;
        }while(true);
    }
    // 0x10003c60. Nearest first point is selected by Manhattan distance.
    void grow(int start,Fiber& f,int owner) {
        if(circles[start].cross)return;
        Point origin=circles[start].center,second{};int next=0,best=0;bool found=false;
        for(int i=0;i<count();++i) if(i!=start&&connect(start,i)==4&&!circles[i].cross&&circles[i].owner==-1) {
            int d=manhattan(origin,circles[i].center);
            if(!found||d<best){found=true;best=d;next=i;second=circles[i].center;}
        }
        if(!(length(origin,second)<circles[start].radius*6))return;
        circles[start].used=circles[next].used=1;circles[start].owner=circles[next].owner=owner;
        f.first=start;f.last=next;f.count=2;f.indices[0]=start;f.indices[1]=next;f.length=length(origin,second);
        Point prev=origin,current=second;int current_index=next;
        for(;;) {
            int chosen[6]{},distance[6]{};bool filled[6]{};
            for(int i=0;i<count();++i) if(i!=current_index&&connect(current_index,i)==4&&!circles[i].cross&&circles[i].owner==-1) {
                double angle=cosine(prev,current,current,circles[i].center);
                for(int band=0;band<6;++band) {
                    if((angle<angles[band]||std::abs(angle-angles[band])<.0001)&&angles[band+1]<angle) {
                        int d=manhattan(current,circles[i].center);
                        if(!filled[band]||d<distance[band]){filled[band]=true;distance[band]=d;chosen[band]=i;break;}
                    }
                }
            }
            int band=0;while(band<6&&!filled[band])++band;if(band==6)return;
            next=chosen[band];circles[next].used=1;circles[next].owner=owner;
            prev=current;current=circles[next].center;current_index=next;
            if(f.count<100){f.indices[f.count++]=next;f.last=next;f.length+=length(prev,current);}
        }
    }
    // 0x10005190 is a measurement-side effect, despite its drawing-like name:
    // accepted centerline pixels assign unowned overlapping circles to the fiber.
    void mark_covered(const Fiber& f,int owner) {
        for(int j=1;j<f.count;++j) {
            Point a=circles[f.indices[j-1]].center,b=circles[f.indices[j]].center;
            int dx=b.x-a.x,dy=b.y-a.y,ax=std::abs(dx),ay=std::abs(dy);
            int extent=std::max(ax,ay);
            for(int t=1;t<extent;++t) {
                Point p;
                if(ay<ax)p={a.x+(dx>0?t:-t),a.y+int(double(t)*double(dy)/double(ax)+.5)};
                else p={a.x+int(double(t)*double(dx)/double(ay)+.5),a.y+(dy>0?t:-t)};
                if(!inside(mask(),p))continue;
                for(int index:{map0[p.y][p.x],map1[p.y][p.x]})
                    if(index>=0&&index<count()&&circles[index].owner==-1)circles[index].owner=owner;
            }
        }
    }
    void group() {
        cu_check();if(size_t(count())*size_t(count())>cu_runtime->budget-cu_runtime->held)throw CuBudgetError("automatic_memory_budget");
        relation.assign(count(),CuVector<signed char>(count(),-1));
        for(int i=0;i<count();++i)relation[i][i]=0;
        for(int i=0;i<count();++i)for(int j=i+1;j<count();++j)relation[i][j]=relation[j][i]=static_cast<signed char>(connect(i,j));
        order.resize(count());
        for(int i=0;i<count();++i){order[i]={i,0};for(int j=0;j<count();++j)if(relation[i][j]>0)++order[i].neighbors;}
        if(count())sort_order(0,count()-1);
        for(int i=0;i<count();++i)circles[i].cross=is_cross(i);
        fibers.clear();fibers.emplace_back();int accepted=0;
        for(Order o:order){Circle& c=circles[o.index];if(c.used||c.owner!=-1||!is_end(o.index))continue;
            Fiber& f=fibers[accepted];std::memset(&f,0,sizeof(f));c.endpoint=1;grow(o.index,f,accepted);
            if(f.count<=field<int>(0x88)||f.length<=double(field<int>(0x8c))) {
                for(int j=0;j<f.count;++j){circles[f.indices[j]].used=0;circles[f.indices[j]].owner=-1;}
            }else {mark_covered(f,accepted);++accepted;if(accepted>=capacity)throw CuBudgetError("automatic_candidate_limit");fibers.emplace_back();}
        }
        for(int i=0;i<accepted;++i){double sum=0;Fiber& f=fibers[i];for(int j=0;j<f.count;++j)if(f.indices[j]<count())sum+=length(circles[f.indices[j]].p,circles[f.indices[j]].q);f.diameter=sum/double(f.count);}
        field<int>(0x94)=accepted;
    }
    void candidates() {
        count()=0;field<double>(0xe8)=0;split()=-1;
        std::memset(circles.data(),0,circles.size()*sizeof(Circle));
        radii.assign(mask()->width,CuVector<int>(mask()->height,-1));
        map0.assign(mask()->height,CuVector<int>(mask()->width,-1));map1=map0;
        auto attempt=[&](Point p,unsigned char mark){Point q{},center{};double r=0;
            if(!opposite(p,q)||!contour_slope(p,q,center,r))return;
            int replace=-1;
            if(!draw_circle(center,int(r+.5),mark,count(),replace))return;
            if(replace<0&&size_t(count())>=circles.size())throw CuBudgetError("automatic_candidate_limit");
            if(replace<0){Circle& c=circles[count()++];c.center=center;c.p=p;c.q=q;c.radius=r;c.owner=-1;c.used=0;field<double>(0xe8)+=r;}
            else {Circle& c=circles[replace];field<double>(0xe8)=field<double>(0xe8)-c.radius+r;c.p=p;c.q=q;c.radius=r;}
        };
        for(int x=5;x<mask()->width-5;++x)for(int y=5;y<mask()->height-5;++y)
            if(pixel(mask(),{x,y-1})==255&&pixel(mask(),{x,y})==0)attempt({x,y},150);
        split()=count();
        for(int y=5;y<mask()->height-5;++y)for(int x=5;x<mask()->width-5;++x)
            if(pixel(mask(),{x-1,y})==255&&pixel(mask(),{x,y})==0)attempt({x,y},200);
        field<int>(0x28)=count();field<double>(0xe8)/=double(count());group();
    }
    int load(const cucv::Mat& input) {
        gray_image=new Image;mask_image=new Image;contour_image=new Image;
        if(input.channels==1)gray_image->matrix=input.clone();else cucv::cvtColor(input,gray_image->matrix,6);
        gray_image->sync();mask_image->matrix=cucv::Mat(input.rows,input.cols,1);mask_image->sync();contour_image->matrix=cucv::Mat(input.rows,input.cols,1);contour_image->sync();
        preprocess(gray(),contour(),mask());candidates();return 0;
    }
};

}

// Adapted from algtdl_length_port.hpp
// Independent source restoration of the length/centerline stages in the shipped
// AlgtDL.dll (image base 0x10000000). No AlgtDL function is imported or called.
// The companion stage probe, unlike this implementation, is a comparison oracle.

namespace algt_length {

struct Note {
    cucv::Point point{};
    int degree = 1;
    int kind = 0; // uvNote +0x10: 0 endpoint, 1 junction
    int remaining = 0;
    CuVector<int> linked_segments;
    CuVector<cucv::Point> linked_points;
};
struct Segment {
    int number = -1;
    Note start, end;
    CuVector<cucv::Point> points; // exact trace order, including duplicate ends
};
struct Fiber {
    CuVector<Segment> segments;
    bool along_x = true; // uvGlassFibre +0x18
    bool valid = true;   // uvGlassFibre +0x28
    cucv::Point start{}, end{};
    int start_segment = -1, end_segment = -1;
};
struct Parameters {
    int edge_width = 12;               // +0x3c
    int gap_length = 20;               // +0x40
    int component_area = 150;          // +0x44
    double compactness = 1.2;          // +0x48
    double aspect_ratio = 1.6;         // +0x50
    double contour_area = 280.0;       // +0x58
    int hole_area_1 = 1500;            // +0x60
    int hole_width_1 = 9;              // +0x64
    int hole_area_2 = 5000;            // +0x68
    int hole_width_2 = 9;              // +0x6c
    double link_angle_1 = 15;          // +0x70
    double link_distance_1 = 9;        // +0x78
    double link_angle_2 = 11;          // +0x80
    double link_distance_2 = 10;       // +0x88
    double link_distance_3 = 9;        // +0x90
    int min_fiber_points = 100;        // +0x98
    int min_segment_points = 15;       // +0x9c
    double relative_link_distance = .5; // +0xa0
    double length_ratio = .1;          // +0xa8
};
struct State {
    Parameters parameters;
    cucv::Mat original_gray; // +4
    cucv::Mat binary;        // +0xb0; later skeleton
    cucv::Mat filtered;      // +0xe8
    cucv::Mat hole_mask;     // +0x120
    cucv::Mat scratch;       // +0x158
    CuVector<CuVector<cucv::Point>> contours;
    CuVector<cucv::Vec4i> hierarchy;
    CuVector<Note> notes;
    CuVector<Segment> segments;
    CuVector<Fiber> fibers;
};

// Integer Otsu helper is supplied by the outer port. These stages retain the
// original explicit threshold arguments and exact loop bounds/order.
void enhance_gray(const cucv::Mat& source, cucv::Mat& destination, int otsu_threshold);
void remove_components(cucv::Mat& image, int maximum_area, unsigned char source,
                       unsigned char replacement, unsigned char temporary = 8);
void thin_binary(cucv::Mat& image);
void prune_short_segments(cucv::Mat& image, int maximum_points, int mode);
void fill_edge_mizi(cucv::Mat& image, int border);
void find_notes_segments(State& state);
void set_original_gray(State& state, const cucv::Mat& gray);
void preprocess_1(State& state, int otsu_threshold);
void preprocess_2(State& state);
void link_gap(State& state);
void get_holes(State& state);
void sift_holes(State& state);
void thin_remove_thin(State& state);
void sift_short(State& state);
// Preserve the original caller's filtering precondition. In diameter mode,
// _JudgeSegment removes segments below 10 points before min_segment_points=5
// is applied. Invalid sampling indices are rejected instead of emulating the
// original helper's out-of-bounds heap reads when that precondition is bypassed.
void link_segments(State& state, bool diameter_mode, bool option = false);
void minus_edge_pixels(State& state, int border);

} // namespace algt_length

// Adapted from algtdl_geometry_port.hpp

// Source reconstruction from AlgtDL b4379a4c...; no AlgtDL calls.
namespace algt_geometry {
struct WidthSample {
    cucv::Point center, first, second;
    double raw_width = -1, corrected_width = -1;
    bool retained = false;
};
struct LineFit { double slope=0, intercept=0; bool vertical=false, valid=false; };
int otsu_threshold(const cucv::Mat& gray);
cucv::Mat canny(const cucv::Mat& gray);
CuVector<cucv::Point> interior_line(cucv::Point start,cucv::Point end);
bool reach_edge(const cucv::Mat& edge,cucv::Point hit,cucv::Point outside,CuVector<cucv::Point>& points,int count=8);
LineFit line_fit(const CuVector<cucv::Point>& points);
bool measure_part(const cucv::Mat& edge,const cucv::Mat& mask,cucv::Point first,cucv::Point second,WidthSample& sample);
bool measure_point(const cucv::Mat& edge,const cucv::Mat& mask,int radius,cucv::Point center,WidthSample& sample);
double measure_samples(const cucv::Mat& edge,const cucv::Mat& mask,const CuVector<cucv::Point>& points,int radius,double bias,CuVector<WidthSample>& samples);
void fill_holes(const cucv::Mat& input,cucv::Mat& output);
void remove_bad_shadows(const cucv::Mat& input,cucv::Mat& output);
}

// Adapted from algtdl_judgement_port.hpp

// Independent source reconstruction of AlgtDL's geometric acceptance tests.
// No AlgtDL function or embedded machine code is called by this implementation.
namespace algt_judgement {
bool points_have_angle(const CuVector<cucv::Point>& points, int window, int stride, int angle);
bool segment_has_angle(const algt_length::Segment& segment, int window, int stride, int angle);
bool fiber_has_angle(const algt_length::Fiber& fiber, int window, int stride, int angle);
void filter_segments(CuVector<algt_length::Segment>& segments);
bool reject_fiber(const algt_length::Fiber& fiber, const cucv::Mat& mask, double fraction);
}

// Adapted from algtdl_port.hpp

namespace algt_port {
struct Fiber {
    algt_length::Fiber geometry;
    int pass=0;
    bool valid=false;
    double raw_diameter_px=-1;
    CuVector<algt_geometry::WidthSample> samples;
};
struct Result {
    std::array<CuVector<Fiber>,3> passes;
    cucv::Mat original,mask,edges,selection,overlay;
};
Result measure(const unsigned char* bgr,int width,int height);
}

// Adapted from algtdl_length_port.cpp

namespace algt_length {
namespace {
void require_gray(const cucv::Mat& image) {
    if (image.empty() || image.type() != CV_8UC1)
        throw std::invalid_argument("AlgtDL length stage requires CV_8UC1");
}
unsigned char pixel(const cucv::Mat& image, int x, int y) {
    // The original only reaches these helpers with a cleared outer border.
    // Explicit bounds retain zero-border semantics without its out-of-range read.
    return (x >= 0 && y >= 0 && x < image.cols && y < image.rows)
        ? image.ptr<unsigned char>(y)[x] : 0;
}
constexpr int dx[8] = {-1,0,1,1,1,0,-1,-1};
constexpr int dy_forward[8] = {-1,-1,-1,0,1,1,1,0};
constexpr int dy_backward[8] = {1,1,1,0,-1,-1,-1,0};
constexpr int next_direction[8] = {5,0,0,2,2,4,4,6};

// 0x10013870 and 0x1001f880. The first neighbor is counted separately;
// the closing NW/W group is corrected only after endpoint classification.
CuVector<Note> detect_notes(const cucv::Mat& source, cucv::Mat& marked,
                               unsigned char endpoint, unsigned char junction) {
    marked = source.clone();
    CuVector<Note> notes;
    for (int x=0; x<source.cols; ++x) for (int y=0; y<source.rows; ++y) {
        if (!pixel(source,x,y)) continue;
        bool previous=false; int degree=0;
        for (int n=0; n<8; ++n) {
            bool present=pixel(source,x+dx[n],y+dy_forward[n])==255;
            if (present && !previous) ++degree;
            previous=present;
        }
        if (degree==1) {
            Note note;note.point={x,y};note.degree=degree;note.kind=0;note.remaining=degree;
            notes.push_back(note);marked.ptr<unsigned char>(y)[x]=endpoint;
        }
        if (pixel(source,x-1,y-1)!=0 && pixel(source,x-1,y)!=0) --degree;
        if (degree>2) {
            Note note;note.point={x,y};note.degree=degree;note.kind=1;note.remaining=degree;
            notes.push_back(note);marked.ptr<unsigned char>(y)[x]=junction;
        }
    }
    return notes;
}

// Trace order shared by 0x10013d80 and 0x1001fca0. A duplicate terminal point
// is appended when the search exhausts its eight neighbors; this is intentional.
CuVector<Segment> trace_segments(cucv::Mat& marked, bool preserve_endpoint_kind) {
    CuVector<Segment> segments;
    int number=0;
    for(int y=0;y<marked.rows;++y) for(int x=0;x<marked.cols;++x) {
        if(pixel(marked,x,y)!=255) continue;
        Segment segment;segment.number=number++;
        segment.points.emplace_back(x,y);marked.ptr<unsigned char>(y)[x]=150;
        for(int pass=0;pass<2;++pass) {
            cucv::Point current(x,y); int direction=pass==0?3:0;
            for(;;) {
                bool found=false, terminal=false; int kind=0;
                cucv::Point selected=current;int selected_direction=0;
                for(int n=0;n<8;++n) {
                    int d=(direction+n+8)%8;
                    cucv::Point candidate(current.x+dx[d],current.y+(pass==0?dy_forward[d]:dy_backward[d]));
                    unsigned char value=pixel(marked,candidate.x,candidate.y);
                    if(value==200 || value==100 || value==255) {
                        found=true;selected=candidate;selected_direction=d;
                        terminal=value!=255;kind=value==200?1:0;break;
                    }
                }
                if(!found) terminal=true;
                if(pass==0) segment.points.push_back(selected);
                else segment.points.insert(segment.points.begin(),selected);
                if(terminal) {
                    Note& endpoint=pass==0?segment.end:segment.start;
                    endpoint.point=selected;endpoint.kind=preserve_endpoint_kind?kind:0;
                    break;
                }
                marked.ptr<unsigned char>(selected.y)[selected.x]=150;
                current=selected;direction=next_direction[selected_direction];
            }
        }
        if(segment.points.size()>1) segments.push_back(std::move(segment));
    }
    return segments;
}

bool symmetric_pixel(const cucv::Mat& image,int x,int y,int radius,int maximum_gap) {
    constexpr int sx[4]={1,0,1,-1},sy[4]={0,1,1,1};
    for(int axis=0;axis<4;++axis) {
        int positive=0,negative=0;
        for(int k=1;k<=radius;++k) {
            if(!positive && pixel(image,x+k*sx[axis],y+k*sy[axis]))positive=k;
            if(!negative && pixel(image,x-k*sx[axis],y-k*sy[axis]))negative=k;
            if(positive && negative)break;
        }
        if(positive && negative && (positive+negative-1)*(axis<2?1.0:1.414)<maximum_gap)return true;
    }
    return false;
}

// Separate float-histogram Otsu used by 0x1001dee0, helper 0x1001e290.
int threshold_otsu_float(const cucv::Mat& image) {
    float histogram[256]{};
    for(int y=0;y<image.rows;++y)for(int x=0;x<image.cols;++x) histogram[image.ptr<unsigned char>(y)[x]]+=1.0f;
    float count=static_cast<float>(image.rows*image.cols);
    for(float& h:histogram)h/=count;
    float mean=0;
    for(int i=0;i<256;++i)mean=mean+static_cast<float>(i)*histogram[i];
    float weight=0,weighted=0,best=0;int result=image.rows;
    for(int i=0;i<256;++i) {
        weight=weight+histogram[i];weighted=weighted+static_cast<float>(i)*histogram[i];
        float difference=weight*mean-weighted;
        float score=(difference*difference)/((1.0f-weight)*weight);
        // COMISS/JBE also leaves the previous threshold for an unordered score.
        if(score>best){best=score;result=i;}
    }
    return result;
}
}

void enhance_gray(const cucv::Mat& source, cucv::Mat& destination, int otsu_threshold) {
 require_gray(source);cucv::Mat input=source;destination=source.clone();
 const int bright_threshold=static_cast<int>(otsu_threshold*1.15);
 if(input.rows<7||input.cols<7)return;
 CuVector<int> columns(size_t(input.cols),0);
 for(int y=0;y<7;++y){const auto* row=input.ptr<unsigned char>(y);for(int x=0;x<input.cols;++x)columns[x]+=row[x];}
 for(int y=3;y<=input.rows-4;++y){
  cu_check();const auto* row=input.ptr<unsigned char>(y);auto* out=destination.ptr<unsigned char>(y);
  int sum=0;for(int x=0;x<7;++x)sum+=columns[x];
  for(int x=3;x<=input.cols-4;++x){
   int value=row[x];double local_mean=double(sum)/49.;double changed=value;
   if(.97<=value/local_mean){if(value<otsu_threshold)changed=value*.95;}
   else changed=value*(value<bright_threshold?.9:.93);
   out[x]=static_cast<unsigned char>(static_cast<int>(changed));
   if(x<input.cols-4)sum+=columns[x+4]-columns[x-3];
  }
  if(y<input.rows-4){const auto* old=input.ptr<unsigned char>(y-3);const auto* next=input.ptr<unsigned char>(y+4);
   for(int x=0;x<input.cols;++x)columns[x]+=int(next[x])-int(old[x]);}
 }
}

void remove_components(cucv::Mat& image,int maximum_area,unsigned char source,
                       unsigned char replacement,unsigned char temporary) {
    require_gray(image);
    for(int x=0;x<image.cols;++x)for(int y=0;y<image.rows;++y)if(image.ptr<unsigned char>(y)[x]==source) {
        int area=cucv::floodFill(image,{x,y},cucv::Scalar(temporary),nullptr,cucv::Scalar(),cucv::Scalar(),4);
        if(area<=maximum_area)cucv::floodFill(image,{x,y},cucv::Scalar(replacement),nullptr,cucv::Scalar(),cucv::Scalar(),4);
    }
    for(int x=0;x<image.cols;++x)for(int y=0;y<image.rows;++y)if(image.ptr<unsigned char>(y)[x]==temporary)
        cucv::floodFill(image,{x,y},cucv::Scalar(source),nullptr,cucv::Scalar(),cucv::Scalar(),4);
}

void thin_binary(cucv::Mat& image) {
    require_gray(image);
    cucv::Mat remove=cucv::Mat::zeros(image.size(),CV_8UC1);
    bool changed;
    // VA 0x1001f050, first phase: simultaneous deletion with a persistent mask.
    do {
        changed=false;
        for(int y=3;y<=image.rows-4;++y)for(int x=3;x<=image.cols-4;++x) {
            if(pixel(image,x,y)!=255)continue;
            const bool a=pixel(image,x-1,y-1)==255,b=pixel(image,x,y-1)==255,c=pixel(image,x+1,y-1)==255;
            const bool d=pixel(image,x-1,y)==255,f=pixel(image,x+1,y)==255,f2=pixel(image,x+2,y)==255;
            const bool g=pixel(image,x-1,y+1)==255,h=pixel(image,x,y+1)==255,i=pixel(image,x+1,y+1)==255,h2=pixel(image,x,y+2)==255;
            bool erase=(!g&&h&&i&&!d&&f&&f2&&!a&&c)
                ||(h&&i&&!d&&f&&f2&&!a&&!b)
                ||(!g&&!h&&!d&&f&&f2&&b&&c)
                ||(!g&&!h&&!i&&a&&b&&c)
                ||(g&&!i&&d&&!f&&a&&!c)
                ||(h&&d&&!f&&!b&&!c)
                ||(!h&&!i&&d&&!f&&b)
                ||(h2&&g&&h&&i&&!a&&!b&&!c);
            if(erase){remove.ptr<unsigned char>(y)[x]=255;changed=true;}
        }
        image.setTo(cucv::Scalar(0),remove);
    }while(changed);
    // Second phase updates the image immediately in row-major order.
    do {
        changed=false;
        for(int y=3;y<=image.rows-4;++y)for(int x=3;x<=image.cols-4;++x) {
            if(pixel(image,x,y)!=255)continue;
            const bool a=pixel(image,x-1,y-1)==255,b=pixel(image,x,y-1)==255,c=pixel(image,x+1,y-1)==255;
            const bool d=pixel(image,x-1,y)==255,f=pixel(image,x+1,y)==255;
            const bool g=pixel(image,x-1,y+1)==255,h=pixel(image,x,y+1)==255,i=pixel(image,x+1,y+1)==255;
            bool erase=(!g&&i&&!d&&f&&!a&&c)
                ||(!g&&!h&&!d&&f&&b)
                ||(g&&h&&i&&!a&&!b&&!c)
                ||(h&&!d&&f&&!a&&!b)
                ||(h&&d&&f&&!a&&!b&&!c)
                ||(!g&&!h&&!i&&d&&f&&b)
                ||(h&&!i&&d&&!f&&b&&!c)
                ||(!g&&h&&!d&&f&&!a&&b);
            if(erase){image.ptr<unsigned char>(y)[x]=0;changed=true;}
        }
    }while(changed);
}

void prune_short_segments(cucv::Mat& image,int maximum_points,int mode) {
    require_gray(image);if(maximum_points<=1)return;
    cucv::Mat marked;auto notes=detect_notes(image,marked,100,200);image=marked;
    auto segments=trace_segments(image,true);
    for(const Segment& segment:segments) {
        if(segment.points.size()>static_cast<size_t>(maximum_points))continue;
        int start=segment.start.kind,end=segment.end.kind;
        bool erase=(mode==0 || mode==2) && start==0&&end==0;
        bool branch=(mode==1 || mode==2) && start!=end;
        if(erase||branch)for(const cucv::Point& p:segment.points)image.ptr<unsigned char>(p.y)[p.x]=0;
        if(branch){cucv::Point p=start==1?segment.start.point:segment.end.point;image.ptr<unsigned char>(p.y)[p.x]=200;}
    }
    for(const Note& note:notes) {
        cucv::Point p=note.point;if(!pixel(image,p.x,p.y))continue;
        bool adjacent=false;
        for(int n=0;n<8;++n)if(pixel(image,p.x+dx[n],p.y+dy_forward[n])==150){adjacent=true;break;}
        if(!adjacent)image.ptr<unsigned char>(p.y)[p.x]=0;
    }
    cucv::threshold(image,image,0,255,cucv::THRESH_BINARY);
}

void fill_edge_mizi(cucv::Mat& image,int border) {
    require_gray(image);
    const int right=image.cols-1+border,bottom=image.rows-1+border;
    cucv::copyMakeBorder(image,image,border,border,border,border,cucv::BORDER_CONSTANT,cucv::Scalar(0));
    cucv::Mat output=image.clone();
    auto fill=[&](int x,int y){if(!pixel(image,x,y)&&symmetric_pixel(image,x,y,border,8))output.ptr<unsigned char>(y)[x]=255;};
    for(int x=border;x<right;++x)for(int y=border;y<border*2;++y)fill(x,y);
    for(int x=border;x<right;++x)for(int y=bottom-border;y<bottom;++y)fill(x,y);
    for(int x=border;x<border*2;++x)for(int y=border;y<bottom;++y)fill(x,y);
    for(int x=right-border;x<right;++x)for(int y=border;y<bottom;++y)fill(x,y);
    image=output;
}

void find_notes_segments(State& state) {
    cucv::Mat marked;state.notes=detect_notes(state.binary,marked,180,180);
    for(const Note& note:state.notes)marked.ptr<unsigned char>(note.point.y)[note.point.x]=note.kind==0?100:200;
    state.segments=trace_segments(marked,false);
}

void set_original_gray(State& state,const cucv::Mat& gray) {
    require_gray(gray);state.original_gray=gray.clone();state.binary=gray.clone();
}

void preprocess_1(State& state,int otsu_threshold) {
    cucv::Mat enhanced;enhance_gray(state.binary,enhanced,otsu_threshold);
    cucv::Canny(enhanced,state.binary,std::max(1.0,otsu_threshold*.4),std::min(254.0,otsu_threshold*1.1),3,false);
    state.filtered=state.binary.clone();
    const int threshold=threshold_otsu_float(enhanced);
    cucv::threshold(enhanced,enhanced,threshold,255,cucv::THRESH_BINARY);
    enhanced=cucv::Scalar::all(255)-enhanced;
    cucv::bitwise_or(enhanced,state.binary,state.binary);
    remove_components(state.binary,state.parameters.component_area,255,0,8);
}

void preprocess_2(State& state) {
    auto& p=state.parameters;
    cucv::dilate(state.filtered,state.filtered,cucv::getStructuringElement(cucv::MORPH_RECT,{3,3}),{-1,-1},1,cucv::BORDER_CONSTANT,cucv::morphologyDefaultBorderValue());
    fill_edge_mizi(state.filtered,p.edge_width/2);
    remove_components(state.filtered,p.hole_area_1,0,255,8);
    remove_components(state.filtered,static_cast<int>(p.contour_area),255,0,8);
    CuVector<CuVector<cucv::Point>> contours;CuVector<cucv::Vec4i> hierarchy;
    cucv::findContours(state.filtered,contours,hierarchy,cucv::RETR_TREE,cucv::CHAIN_APPROX_NONE);
    for(size_t i=0;i<contours.size();++i)if(hierarchy[i][3]==-1) {
        double area=cucv::contourArea(contours[i]);
        double circularity=cucv::arcLength(contours[i],true)/(std::sqrt(area/3.141592653589793)*6.283185307179586);
        auto box=cucv::minAreaRect(contours[i]);
        float aspect=box.size.width<=box.size.height?box.size.height/box.size.width:box.size.width/box.size.height;
        if(area<=p.contour_area || (aspect<=p.aspect_ratio&&circularity<=p.compactness))
            cucv::drawContours(state.filtered,contours,static_cast<int>(i),cucv::Scalar(0),-1,8);
    }
}

void link_gap(State& state) {
    cucv::Mat temp=state.binary.clone();
    remove_components(temp,state.parameters.hole_width_1,0,255,8);thin_binary(temp);prune_short_segments(temp,state.parameters.edge_width,2);
    remove_components(temp,state.parameters.hole_width_1,0,255,8);thin_binary(temp);prune_short_segments(temp,3,2);
    remove_components(temp,state.parameters.hole_width_1,0,255,8);thin_binary(temp);
    int border=state.parameters.edge_width/2;CuVector<cucv::Point> ends;
    for(int x=border;x<temp.cols-border;++x)for(int y=border;y<temp.rows-border;++y)
        if(pixel(temp,x,y)==255 && cucv::countNonZero(temp(cucv::Rect(x-1,y-1,3,3)))<3)ends.emplace_back(x,y);
    for(size_t i=0;i+1<ends.size();++i)for(size_t j=i+1;j<ends.size();++j)
        if(cucv::norm(ends[i]-ends[j])<state.parameters.gap_length)
            cucv::line(state.binary,ends[i],ends[j],cucv::Scalar(255),2,8);
}

void get_holes(State& state) {
    state.hole_mask=cucv::Mat::zeros(state.binary.size(),CV_8UC1);
    cucv::findContours(state.binary,state.contours,state.hierarchy,cucv::RETR_TREE,cucv::CHAIN_APPROX_NONE);
    for(size_t i=0;i<state.contours.size();++i)if(state.hierarchy[i][3]==-1)
        cucv::drawContours(state.hole_mask,state.contours,static_cast<int>(i),cucv::Scalar(255),-1,8);
}

void sift_holes(State& state) {
    cucv::Mat accepted=cucv::Mat::zeros(state.binary.size(),CV_8UC1);
    auto draw=[&](cucv::Mat& image,size_t i,int value){cucv::drawContours(image,state.contours,static_cast<int>(i),cucv::Scalar(value),-1,8);};
    const auto& p=state.parameters;
    for(size_t i=0;i<state.contours.size();++i) {
        auto hierarchy=state.hierarchy[i];
        if(hierarchy[3]==-1)continue;
        double area=cucv::contourArea(state.contours[i]);
        if(hierarchy[2]==-1) {
            draw(accepted,i,255);bool keep=false;
            if(area<p.hole_area_2 && area>=p.hole_width_2) {
                auto box=cucv::minAreaRect(state.contours[i]);
                float minimum=std::min(box.size.width,box.size.height);
                float aspect=std::max(box.size.width,box.size.height)/minimum;
                keep=minimum<p.edge_width*1.5 && ((area<60)||(area<80&&aspect>1.2)||(area<100&&aspect>2.3)
                    ||(area<150&&aspect>2.5)||(area<300&&aspect>3)||(area<400&&aspect>5)||(area<500&&aspect>6)||aspect>10);
                double circularity=cucv::arcLength(state.contours[i],true)/(std::sqrt(area/3.141592653589793)*6.283185307179586);
                if((area<56&&circularity>2.3)||(area>800&&circularity<2.2)||(area>1800&&circularity<3))keep=false;
            }
            if(!keep){draw(accepted,i,0);if(area>150)draw(state.hole_mask,i,0);}
        } else if(area>p.hole_area_2)draw(state.hole_mask,i,0);
    }
    for(int x=0;x<accepted.cols;++x)for(int y=0;y<accepted.rows;++y)
        if(pixel(accepted,x,y)==255 && pixel(state.binary,x,y)==255)accepted.ptr<unsigned char>(y)[x]=0;
    state.binary=accepted;
}

void thin_remove_thin(State& state) {
    thin_binary(state.binary);prune_short_segments(state.binary,3,2);
    remove_components(state.binary,9,0,255,8);thin_binary(state.binary);
}
void sift_short(State& state) {
    for(int x=0;x<state.binary.cols;++x)for(int y=0;y<state.binary.rows;++y)
        if(pixel(state.binary,x,y)==255 && pixel(state.filtered,x,y)==0)state.binary.ptr<unsigned char>(y)[x]=0;
}
void minus_edge_pixels(State& state,int border) {
    for(Fiber& fiber:state.fibers){fiber.start-=cucv::Point(border,border);fiber.end-=cucv::Point(border,border);}
    for(Fiber& fiber:state.fibers)for(Segment& segment:fiber.segments) {
        segment.start.point-=cucv::Point(border,border);segment.end.point-=cucv::Point(border,border);
        for(cucv::Point& point:segment.points)point-=cucv::Point(border,border);
    }
}

namespace {
struct LinkInfo {
    int a=0,b=0,c=0,d=0;
    bool used=false,assigned=false,long_segment=false;
    double angle=0;
};
struct OrderedSegment {int id;bool start_is_min;bool processed=false;};
int coordinate(cucv::Point p,bool along_x){return along_x?p.x:p.y;}
double segment_angle(const CuVector<cucv::Point2f>& points) {
    if(points.size()<2)return 0;
    bool same_x=true,same_y=true;
    for(size_t i=1;i<points.size();++i)if(std::abs(double(points[i].x)-double(points[0].x))>.5){same_x=false;break;}
    if(same_x)return 90;
    for(size_t i=1;i<points.size();++i)if(std::abs(double(points[i].y)-double(points[0].y))>.5){same_y=false;break;}
    if(same_y)return 0;
    cucv::Vec4f line;cucv::fitLine(points,line,cucv::DIST_L1,0,.01,.01);
    float radians=static_cast<float>(std::acos(double(line[0])));
    double angle=double(radians)*180.0/3.141592653589793;
    if((double(points.front().y)-points.back().y)*(double(points.front().x)-points.back().x)<0)angle=180-angle;
    return angle;
}
int angle_at(cucv::Point a,cucv::Point vertex,cucv::Point b) {
    double ax=double(a.x)-vertex.x,ay=double(a.y)-vertex.y;
    double bx=double(b.x)-vertex.x,by=double(b.y)-vertex.y;
    double norm=std::sqrt(ax*ax+ay*ay)*std::sqrt(bx*bx+by*by);
    if(norm<=0)return -1;
    double cosine=(bx*ax+by*ay)/norm;cosine=std::min(1.0,std::max(-1.0,cosine));
    return static_cast<int>((std::acos(cosine)*180.0)/3.141592653589793);
}
CuVector<cucv::Point> bridge_points(cucv::Point a,cucv::Point b) {
    // VA 0x10020ce0, excludes both endpoints; truncation after +0.5 is retained.
    CuVector<cucv::Point> points;
    double dxv=double(b.x-a.x),dyv=double(b.y-a.y),ax=std::abs(dxv),ay=std::abs(dyv);
    if(ax<1e-6){for(int i=1;i<ay;++i)points.emplace_back(a.x,int((dyv/ay)*i+a.y+.5));return points;}
    if(ay<1e-6){for(int i=1;i<ax;++i)points.emplace_back(int((dxv/ax)*i+a.x+.5),a.y);return points;}
    double sx=dxv/ax,sy=dyv/ay;
    if(ax<ay){double ratio=ax/ay;for(int i=1;i<ay;++i)points.emplace_back(int(i*sx*ratio+.5+a.x),int(i*sy+a.y+.5));}
    else {double ratio=ay/ax;for(int i=1;i<ax;++i)points.emplace_back(int(i*sx+a.x+.5),int(i*sy*ratio+.5+a.y));}
    return points;
}
bool marked_segment(const Segment& segment,const cucv::Mat& marked,double threshold) {
    double counts[3]{};
    for(cucv::Point p:segment.points){
        if(pixel(marked,p.x,p.y)==120)++counts[0];
        if(pixel(marked,std::min(p.x+1,marked.cols-1),std::min(p.y+1,marked.rows-1))==120)++counts[1];
        if(pixel(marked,std::max(p.x-1,0),std::max(p.y-1,0))==120)++counts[2];
    }
    double n=double(segment.points.size());
    return !(counts[0]/n<=threshold&&counts[1]/n<=threshold&&counts[2]/n<=threshold);
}
bool clear_bridge(const cucv::Mat& mask,cucv::Point a,cucv::Point b,int maximum_index=-1) {
    auto points=bridge_points(a,b);
    for(size_t i=0;i<points.size();++i)
        if(pixel(mask,points[i].x,points[i].y)==0 || (maximum_index>=0&&int(i)>maximum_index))return false;
    return true;
}
}

void link_segments(State& state,bool diameter_mode,bool option) {
    if(option)throw std::invalid_argument("interactive LinkSegment debug windows are disabled in the static demo");
    const auto& p=state.parameters;state.fibers.clear();
    CuVector<LinkInfo> info(state.segments.size());
    for(size_t i=0;i<state.segments.size();++i) {
        auto& q=info[i];const auto& points=state.segments[i].points;int n=int(points.size());
        q.long_segment=n>=p.min_segment_points;
        if(!q.long_segment){q.a=q.d=n-1;q.b=q.c=0;}
        else if(diameter_mode){
            if(n<41){q.a=q.d=n-10;q.b=q.c=3;}
            else if(n<81){q.a=25;q.b=16;q.c=n-26;q.d=n-17;}
            else {q.a=40;q.b=18;q.c=n-41;q.d=n-19;}
        }
        else if(n<41){q.a=q.d=n-3;q.b=q.c=2;}
        else if(n<81){q.a=33;q.b=8;q.c=n-34;q.d=n-9;}
        else {q.a=45;q.b=12;q.c=n-46;q.d=n-13;}
        // The actual Thin caller first runs _JudgeSegment, which excludes
        // fewer than 10 points. Calling LinkSegment_dia directly with 5..9
        // points and min_segment_points=5 makes the original read points[n-10]
        // before its vector allocation. Reject that invalid internal state;
        // uninitialized heap bytes have no reproducible algorithmic meaning.
        if(q.a<0||q.b<0||q.c<0||q.d<0||q.a>=n||q.b>=n||q.c>=n||q.d>=n)
            throw std::invalid_argument("AlgtDL LinkSegment sample index is outside segment; apply the original caller's segment filter before linking");
        CuVector<cucv::Point2f> fit;
        if(!q.long_segment)for(cucv::Point point:points)fit.emplace_back(float(point.x),float(point.y));
        else for(int j=q.b;j<q.d;++j)fit.emplace_back(float(points[j].x),float(points[j].y));
        q.angle=segment_angle(fit);
    }
    cucv::Mat marked=state.hole_mask.clone();
    auto mark_judge=[&](int id,double threshold){
        if(marked_segment(state.segments[id],marked,threshold)){info[id].used=true;return true;}return false;
    };
    for(size_t contour=0;contour<state.contours.size();++contour) {
        if(state.hierarchy[contour][3]!=-1 || (!diameter_mode&&state.hierarchy[contour][2]==-1))continue;
        CuVector<int> long_ids,short_ids;
        for(size_t i=0;i<state.segments.size();++i) {
            if(info[i].assigned)continue;
            if(cucv::pointPolygonTest(state.contours[contour],cucv::Point2f(state.segments[i].start.point),false)<0)continue;
            (info[i].long_segment?long_ids:short_ids).push_back(int(i));info[i].assigned=true;
        }
        std::sort(long_ids.begin(),long_ids.end(),[&](int a,int b){return state.segments[a].points.size()>state.segments[b].points.size();});
        for(int seed:long_ids) {
            if(info[seed].used || mark_judge(seed,p.relative_link_distance))continue;
            bool along_x=info[seed].angle<45||info[seed].angle>135;
            CuVector<OrderedSegment> ordered;
            auto add_ordered=[&](int id){const auto& s=state.segments[id];ordered.push_back({id,coordinate(s.start.point,along_x)<=coordinate(s.end.point,along_x),false});};
            add_ordered(seed);
            for(int id:long_ids)if(id!=seed&&!info[id].used&&!mark_judge(id,p.relative_link_distance)){
                double angle=std::abs(info[seed].angle-info[id].angle);
                if(std::min(angle,180-angle)<p.link_angle_1)add_ordered(id);
            }
            auto minimum=[&](const OrderedSegment& entry){const auto& s=state.segments[entry.id];return entry.start_is_min?s.start.point:s.end.point;};
            auto maximum=[&](const OrderedSegment& entry){const auto& s=state.segments[entry.id];return entry.start_is_min?s.end.point:s.start.point;};
            std::sort(ordered.begin(),ordered.end(),[&](const OrderedSegment&a,const OrderedSegment&b){return coordinate(minimum(a),along_x)<coordinate(minimum(b),along_x);});
            int seed_position=0;for(size_t i=0;i<ordered.size();++i)if(ordered[i].id==seed){seed_position=int(i);break;}
            ordered[seed_position].processed=true;info[seed].used=true;
            CuVector<int> accepted{seed};int left=seed_position,right=seed_position;
            auto sample=[&](const OrderedSegment& entry,bool right_side){
                const auto& s=state.segments[entry.id];const auto& q=info[entry.id];
                int near_index,far_index;
                if(right_side==entry.start_is_min){near_index=q.d;far_index=q.c;}
                else {near_index=q.b;far_index=q.a;}
                return std::make_pair(s.points[near_index],s.points[far_index]);
            };
            for(int direction=-1;direction<=1;direction+=2) {
                for(int k=seed_position+direction;k>=0&&k<int(ordered.size());k+=direction) {
                    if(ordered[k].processed||k==seed_position)continue;
                    if(mark_judge(ordered[k].id,p.relative_link_distance))continue;
                    CuVector<int> candidates{k};
                    int extent=coordinate(direction<0?minimum(ordered[k]):maximum(ordered[k]),along_x);
                    for(int j=k+direction;j>=0&&j<int(ordered.size());j+=direction) {
                        if(ordered[j].processed)continue;
                        if(mark_judge(ordered[j].id,p.relative_link_distance))continue;
                        if(coordinate(minimum(ordered[j]),along_x)<extent && extent<coordinate(maximum(ordered[j]),along_x)){
                            candidates.push_back(j);ordered[j].processed=true;
                        }
                    }
                    double best=99999;int selected=-1;cucv::Point best_a,best_b;
                    for(int candidate:candidates) {
                        int lower=direction<0?candidate:right,upper=direction<0?left:candidate;
                        auto lo=sample(ordered[lower],true),hi=sample(ordered[upper],false);
                        double angle1=std::abs(180-angle_at(lo.second,lo.first,hi.first));
                        double angle2=std::abs(180-angle_at(lo.first,hi.first,hi.second));
                        size_t n1=state.segments[ordered[lower].id].points.size(),n2=state.segments[ordered[upper].id].points.size();
                        double tolerance=p.link_angle_2;
                        if(n1>40&&n2>40)tolerance=(n1>80&&n2>80)?p.link_distance_1:p.link_distance_2;
                        if(angle1<tolerance&&angle2<tolerance&&clear_bridge(state.hole_mask,lo.first,hi.first)&&angle1+angle2<best){
                            best=angle1+angle2;selected=candidate;best_a=lo.first;best_b=hi.first;
                        }
                    }
                    if(selected>=0){
                        int id=ordered[selected].id;info[id].used=true;accepted.push_back(id);
                        cucv::line(marked,best_a,best_b,cucv::Scalar(120),7,8);
                        if(direction<0)left=selected;else right=selected;
                    }
                }
            }
            for(int id:accepted)for(cucv::Point point:state.segments[id].points)cucv::circle(marked,point,1,cucv::Scalar(120),7,8);
            std::sort(accepted.begin(),accepted.end(),[&](int a,int b){
                const auto& x=state.segments[a];const auto& y=state.segments[b];
                return std::min(coordinate(x.start.point,along_x),coordinate(x.end.point,along_x))<std::min(coordinate(y.start.point,along_x),coordinate(y.end.point,along_x));
            });
            Fiber fiber;fiber.along_x=along_x;fiber.start_segment=ordered[left].id;fiber.end_segment=ordered[right].id;
            for(int id:accepted)fiber.segments.push_back(state.segments[id]);
            auto end_by_axis=[&](const Segment& s,bool high){
                bool start_lower=coordinate(s.start.point,along_x)<=coordinate(s.end.point,along_x);
                return high?(start_lower?s.end.point:s.start.point):(start_lower?s.start.point:s.end.point);
            };
            fiber.start=end_by_axis(fiber.segments.front(),false);fiber.end=end_by_axis(fiber.segments.back(),true);
            bool extended_left=false,extended_right=false;
            for(int id:short_ids) {
                if(info[id].used||mark_judge(id,p.length_ratio))continue;
                auto& segment=state.segments[id];
                for(cucv::Point point:segment.points)state.hole_mask.ptr<unsigned char>(point.y)[point.x]=50;
                for(cucv::Point point:segment.points)state.hole_mask.ptr<unsigned char>(point.y)[point.x]=255;
                cucv::Point middle=segment.points[segment.points.size()/2];
                auto left_sample=sample(ordered[left],false),right_sample=sample(ordered[right],true);
                if(!extended_left && std::abs(180-angle_at(middle,left_sample.first,left_sample.second))<p.link_distance_3
                    &&clear_bridge(state.hole_mask,middle,left_sample.first,p.min_fiber_points)) {
                    extended_left=true;fiber.segments.insert(fiber.segments.begin(),segment);fiber.start=middle;
                }
                // Both original branches rasterize the same midpoint->left near
                // bridge (0x1000e6ef / 0x1000e906); preserve that unusual behavior.
                if(!extended_right && std::abs(180-angle_at(right_sample.second,right_sample.first,middle))<p.link_distance_3
                    &&clear_bridge(state.hole_mask,middle,left_sample.first,p.min_fiber_points)) {
                    extended_right=true;fiber.segments.push_back(segment);fiber.end=middle;
                }
            }
            state.fibers.push_back(std::move(fiber));
        }
    }
}
} // namespace algt_length

// Adapted from algtdl_geometry_port.cpp

namespace algt_geometry {
namespace {
constexpr double pi=3.141592653589793;
bool inside(const cucv::Mat& m,cucv::Point p){return p.x>=0&&p.y>=0&&p.x<m.cols&&p.y<m.rows;}
double angle(const LineFit& a,const LineFit& b){
    double aa=a.vertical?90.:std::atan(a.slope)*180./pi;
    double ab=b.vertical?90.:std::atan(b.slope)*180./pi;
    double d=std::abs(aa-ab);return d>90.?180.-d:d;
}
double aspect(const CuVector<cucv::Point>& c){
    auto r=cucv::minAreaRect(c);float w=r.size.width,h=r.size.height;
    return w<=h?h/w:w/h;
}
}

// FUN_1001e580: float histogram, float probabilities and between-class variance.
int otsu_threshold(const cucv::Mat& gray){
    CV_Assert(gray.type()==CV_8UC1);
    float histogram[256]={};
    for(int y=0;y<gray.rows;++y)for(int x=0;x<gray.cols;++x)histogram[gray.ptr<unsigned char>(y)[x]]+=1.f;
    float total=float(gray.rows*gray.cols),mean=0.f;
    for(float& h:histogram)h/=total;
    for(int i=0;i<256;++i)mean+=float(i)*histogram[i];
    float weight=0.f,partial=0.f,best=0.f;int threshold=0;
    for(int i=0;i<256;++i){
        weight+=histogram[i];partial+=float(i)*histogram[i];
        float delta=weight*mean-partial;
        float score=delta*delta/((1.f-weight)*weight);
        if(score>best){best=score;threshold=i;}
    }
    return threshold;
}

// CMeasureGlassFiberDia::_GetCanny @ 10005440.
cucv::Mat canny(const cucv::Mat& gray){
    double high=double(otsu_threshold(gray))*1.1;
    double low=high*.3;
    high=std::min(254.,std::max(1.,high));low=std::min(254.,std::max(1.,low));
    cucv::Mat result;cucv::Canny(gray,result,low,high,3,false);return result;
}

// FUN_10020ce0 uses floating interpolation, excludes both endpoints, truncates (v+.5).
CuVector<cucv::Point> interior_line(cucv::Point start,cucv::Point end){
    CuVector<cucv::Point> result;
    double dx=end.x-start.x,dy=end.y-start.y,ax=std::abs(dx),ay=std::abs(dy);
    double count=std::max(ax,ay);if(count<1.e-6)return result;
    for(int i=1;double(i)<count;++i){
        int x=ax<1.e-6?start.x:int(double(i)*(dx/ax)*(ax/count)+double(start.x)+.5);
        int y=ay<1.e-6?start.y:int(double(i)*(dy/ay)*(ay/count)+double(start.y)+.5);
        result.emplace_back(x,y);
    }
    return result;
}

// GetPointReachEdge @ 10006e60. The second half is appended, not prepended.
bool reach_edge(const cucv::Mat& edge,cucv::Point hit,cucv::Point outside,CuVector<cucv::Point>& points,int count){
    static const int dx[2][8]={{1,1,0,-1,-1,-1,0,1},{-1,-1,0,1,1,1,0,-1}};
    static const int dy[8]={0,-1,-1,-1,0,1,1,1};
    points.clear();points.push_back(hit);
    for(int direction=0;direction<2;++direction){
        int start=0;
        while(start<8&&(dx[direction][start]!=outside.x-hit.x||dy[start]!=outside.y-hit.y))++start;
        if(start==8)return false;
        cucv::Point current=hit;int advance=0,scan=0;
        while(advance<count){
            if(scan>7)return false;
            int index=(scan+start)%8;
            cucv::Point next=current+cucv::Point(dx[direction][index],dy[index]);
            if(!inside(edge,next))return false;
            if(edge.at<unsigned char>(next)==255){
                current=next;points.push_back(next);++advance;start=(index+5)%8;scan=0;
            }
            ++scan;
        }
    }
    return true;
}

// CMeasureGlassFiber_ListPoint::GetLineParam @ 10019530.
LineFit line_fit(const CuVector<cucv::Point>& points){
    LineFit r;if(points.size()<2)return r;
    double sx=0,sy=0,sxy=0,sxx=0,syy=0,n=double(points.size());
    for(auto p:points){double x=p.x,y=p.y;sx+=x;sy+=y;sxy+=x*y;sxx+=x*x;syy+=y*y;}
    auto delta=points.front()-points.back();
    if(std::abs(delta.x)<std::abs(delta.y)){
        double denominator=sx*sy-n*sxy;
        if(denominator==0.){r.valid=true;r.vertical=true;r.slope=1.;return r;}
        r.slope=(sy*sy-n*syy)/denominator;
    }else{
        double denominator=sx*sx-n*sxx;
        if(denominator!=0.)r.slope=(sx*sy-n*sxy)/denominator;
    }
    r.intercept=(sy-sx*r.slope)/n;r.valid=true;return r;
}

// _MeasureOnePart @ 10006440: endpoints are required to lie outside BOTH maps.
bool measure_part(const cucv::Mat& edge,const cucv::Mat& mask,cucv::Point first,cucv::Point second,WidthSample& sample){
    if(!inside(edge,first)||!inside(edge,second)||!inside(mask,first)||!inside(mask,second))return false;
    if(edge.at<unsigned char>(first)||edge.at<unsigned char>(second)||mask.at<unsigned char>(first)||mask.at<unsigned char>(second))return false;
    auto points=interior_line(first,second);int n=int(points.size()),a=0,b=n-1;
    while(a<n&&!edge.at<unsigned char>(points[a]))++a;
    while(b>0&&!edge.at<unsigned char>(points[b]))--b;
    if(b-a<=1)return false;
    cucv::Point pa=points[a],pb=points[b],oa=a==0?first:points[a-1],ob=b==n-1?second:points[b+1];
    CuVector<cucv::Point> sideA,sideB;
    if(!reach_edge(edge,pa,oa,sideA)||!reach_edge(edge,pb,ob,sideB))return false;
    auto spanA=sideA.front()-sideA.back(),spanB=sideB.front()-sideB.back();
    if(std::abs(spanA.x)+std::abs(spanA.y)<4||std::abs(spanB.x)+std::abs(spanB.y)<4)return false;
    LineFit la=line_fit(sideA),lb=line_fit(sideB),cut=line_fit({first,second});
    if(!la.valid||!lb.valid||!cut.valid||angle(la,lb)>=10.||angle(la,cut)<=60.)return false;
    double da=la.vertical?std::abs(pb.x-pa.x):std::abs(double(pb.x)*la.slope-double(pb.y)+la.intercept)/std::sqrt(la.slope*la.slope+1.);
    double db=lb.vertical?std::abs(pa.x-pb.x):std::abs(double(pa.x)*lb.slope-double(pa.y)+lb.intercept)/std::sqrt(lb.slope*lb.slope+1.);
    sample.first=pa;sample.second=pb;sample.raw_width=(da+db)*.5;return true;
}

// _MeasureOnePnt @ 10006210 preserves first-success order: X, Y, +diag, -diag.
bool measure_point(const cucv::Mat& edge,const cucv::Mat& mask,int radius,cucv::Point center,WidthSample& sample){
    int diag=int(double(radius)/1.4142135623731+.5);
    cucv::Point a[]={center+cucv::Point(-radius,0),center+cucv::Point(0,-radius),center+cucv::Point(diag,diag),center+cucv::Point(diag,-diag)};
    cucv::Point b[]={center+cucv::Point(radius,0),center+cucv::Point(0,radius),center+cucv::Point(-diag,-diag),center+cucv::Point(-diag,diag)};
    sample.center=center;sample.raw_width=-1.;
    for(int i=0;i<4;++i)if(measure_part(edge,mask,a[i],b[i],sample))return true;
    return false;
}

// MeasureGlassFibre @ 10005a40, valid-point mean and strict 30% trimming.
double measure_samples(const cucv::Mat& edge,const cucv::Mat& mask,const CuVector<cucv::Point>& points,int radius,double bias,CuVector<WidthSample>& samples){
    int pad=radius+1;cucv::Mat padded_edge,padded_mask;
    cucv::copyMakeBorder(edge,padded_edge,pad,pad,pad,pad,cucv::BORDER_CONSTANT,cucv::Scalar(0));
    cucv::copyMakeBorder(mask,padded_mask,pad,pad,pad,pad,cucv::BORDER_CONSTANT,cucv::Scalar(0));
    double sum=0;samples.clear();
    for(cucv::Point p:points){WidthSample sample;
        if(measure_point(padded_edge,padded_mask,radius,p+cucv::Point(pad,pad),sample)){
            sample.corrected_width=sample.raw_width-bias;
            if(sample.corrected_width>0){
                // Original accumulation performs addition before subtracting bias.
                sum=(sample.raw_width+sum)-bias;
                sample.center-=cucv::Point(pad,pad);sample.first-=cucv::Point(pad,pad);sample.second-=cucv::Point(pad,pad);
                samples.push_back(sample);
            }
        }
    }
    if(samples.empty())return -1.;
    double initial=sum/double(samples.size()),retained_sum=0;int count=0;
    for(auto& sample:samples)if(std::abs(sample.corrected_width-initial)/initial<.3){sample.retained=true;retained_sum+=sample.corrected_width;++count;}
    return count?retained_sum/count:-1.;
}

// _FillHoles @ 10004c20. OpenCV contour retrieval TREE, CHAIN_APPROX_NONE.
void fill_holes(const cucv::Mat& input,cucv::Mat& output){
    output=input.clone();cucv::Mat work=input.clone();
    CuVector<CuVector<cucv::Point>> contours;CuVector<cucv::Vec4i> hierarchy;
    cucv::findContours(work,contours,hierarchy,cucv::RETR_TREE,cucv::CHAIN_APPROX_NONE);
    for(size_t i=0;i<contours.size();++i)if(hierarchy[i][3]!=-1){
        double area=cucv::contourArea(contours[i]),ratio=aspect(contours[i]);
        bool fill=area<500.||(area<700.&&ratio>3.)||(area<1500.&&ratio>3.4)||ratio>4.;
        if(fill)cucv::drawContours(output,contours,int(i),cucv::Scalar(255),-1,8);
    }
}

// _RemoveBadShadows @ 10005000, area/circularity/aspect thresholds.
void remove_bad_shadows(const cucv::Mat& input,cucv::Mat& output){
    output=input.clone();cucv::Mat work=input.clone();
    CuVector<CuVector<cucv::Point>> contours;CuVector<cucv::Vec4i> hierarchy;
    cucv::findContours(work,contours,hierarchy,cucv::RETR_TREE,cucv::CHAIN_APPROX_NONE);
    for(size_t i=0;i<contours.size();++i)if(hierarchy[i][3]==-1){
        double area=cucv::contourArea(contours[i]),circ=cucv::arcLength(contours[i],true)/(std::sqrt(area/pi)*(2*pi));
        double ratio=aspect(contours[i]);
        bool remove=area<100.||(area<300.&&circ<1.1)||(area<500.&&ratio<2.2&&circ<1.11);
        if(remove)cucv::drawContours(output,contours,int(i),cucv::Scalar(0),-1,8);
    }
}
}

// Adapted from algtdl_judgement_port.cpp

namespace algt_judgement {
namespace {
constexpr double pi = 3.141592653589793;

// 0x10007770 / 0x10007930. The original truncates the angle to an integer
// before its inclusive comparison. A duplicate arm yields -1 and is rejected.
int bend_angle(cucv::Point first, cucv::Point middle, cucv::Point last) {
    const double ax = double(first.x) - double(middle.x);
    const double ay = double(first.y) - double(middle.y);
    const double bx = double(last.x) - double(middle.x);
    const double by = double(last.y) - double(middle.y);
    const double first_length = std::sqrt(ax * ax + ay * ay);
    const double second_length = std::sqrt(bx * bx + by * by);
    const double denominator = first_length * second_length;
    if (!(denominator > 0.0)) return -1;
    double cosine = (ax * bx + ay * by) / denominator;
    if (cosine < -1.0) cosine = -1.0;
    if (cosine > 1.0) cosine = 1.0;
    return int((std::acos(cosine) * 180.0) / pi);
}

// 0x10004964..0x10004ac5. OpenCV returns float dimensions; the division
// and area multiplication remain float before the original widens to double.
bool keep_shape(const algt_length::Segment& segment) {
    const cucv::RotatedRect box = cucv::minAreaRect(segment.points);
    const float width = box.size.width, height = box.size.height;
    const float ratio32 = width <= height ? height / width : width / height;
    const float area32 = width * height;
    const double ratio = double(ratio32);
    const double points = double(segment.points.size());
    const double occupancy = (points / double(area32)) * 100.0;
    if (points < 10.0) return false;
    if (points < 20.0)
        return (ratio > 10.0 && occupancy > 45.0) || occupancy > 80.0;
    if (points < 60.0)
        return (ratio > 10.0 && occupancy > 30.0) ||
               (ratio > 8.0 && occupancy > 40.0);
    if (points < 200.0)
        return (ratio > 7.0 && occupancy > 7.0) || ratio > 20.0 || occupancy > 15.0;
    return ratio > 15.0 || occupancy > 15.0 ||
           (ratio > 10.0 && occupancy > 5.0);
}
}

bool points_have_angle(const CuVector<cucv::Point>& points, int window, int stride, int angle) {
    if (window < 0 || stride <= 0)
        throw std::invalid_argument("AlgtDL angle window must be nonnegative and stride positive");
    const size_t span = size_t(window), half = size_t(window / 2);
    for (size_t start = 0; start + span < points.size(); start += size_t(stride))
        if (bend_angle(points[start], points[start + half], points[start + span]) <= angle)
            return true;
    return false;
}

bool segment_has_angle(const algt_length::Segment& segment, int window, int stride, int angle) {
    return points_have_angle(segment.points, window, stride, angle);
}

// 0x10007550 concatenates every stored segment point, then inserts interior
// bridge points from the current end note to the next start note. It neither
// reverses segments nor removes duplicate endpoints at this stage.
bool fiber_has_angle(const algt_length::Fiber& fiber, int window, int stride, int angle) {
    CuVector<cucv::Point> points;
    for (size_t i = 0; i < fiber.segments.size(); ++i) {
        const auto& segment = fiber.segments[i];
        points.insert(points.end(), segment.points.begin(), segment.points.end());
        if (i + 1 < fiber.segments.size()) {
            const auto bridge = algt_geometry::interior_line(segment.end.point, fiber.segments[i + 1].start.point);
            points.insert(points.end(), bridge.begin(), bridge.end());
        }
    }
    return points_have_angle(points, window, stride, angle);
}

// 0x10004770. Original Mat argument is unused; the accepted segment order and
// all point/note fields are retained exactly when the filtered vector is built.
void filter_segments(CuVector<algt_length::Segment>& segments) {
    CuVector<algt_length::Segment> kept;
    kept.reserve(segments.size());
    for (const auto& segment : segments)
        if (!segment_has_angle(segment, 12, 2, 80) &&
            !segment_has_angle(segment, 20, 3, 90) && keep_shape(segment))
            kept.push_back(segment);
    segments.swap(kept);
}

// 0x100073c0. The caller applies this result by clearing its validity flag;
// passing does not reset an already-invalid fiber. Only stored points count
// toward the mask fraction. Bridge points participate solely in angle testing.
bool reject_fiber(const algt_length::Fiber& fiber, const cucv::Mat& mask, double fraction) {
    if (fiber_has_angle(fiber, 26, 3, 90)) return true;
    CV_Assert(mask.type() == CV_8UC1);
    double nonzero = 0.0, count = 0.0;
    for (const auto& segment : fiber.segments) {
        count += double(segment.points.size());
        for (cucv::Point point : segment.points) {
            // Bound the original raw-pointer read for invalid caller geometry.
            if (point.x < 0 || point.y < 0 || point.x >= mask.cols || point.y >= mask.rows)
                throw std::out_of_range("AlgtDL fiber point lies outside judgement mask");
            if (mask.ptr<unsigned char>(point.y)[point.x] != 0) nonzero += 1.0;
        }
    }
    return fraction < nonzero / count; // Strict comparison, including original 0/0 behavior.
}
}

// Adapted from algtdl_port.cpp
// Independent reconstruction of CMeasureGlassFiberDia::ConBineCalc and pass logic.
// Original target sha256 b4379a4cf2936e8d26e115c6832c464b74c63cc582a890e27d480002b2a0986d.

namespace algt_port {
namespace {
using algt_length::State;

CuVector<cucv::Point> all_points(const algt_length::Fiber& fiber){
    CuVector<cucv::Point> points;
    for(const auto& segment:fiber.segments)points.insert(points.end(),segment.points.begin(),segment.points.end());
    return points;
}

void scale_points(State& state,double scale,int width,int height){
    // ResizeGlassPnt @ 10005500 clamps to width/height, not width-1/height-1.
    auto transform=[&](cucv::Point& p){p.x=std::min(width,int(double(p.x)*scale+.5));p.y=std::min(height,int(double(p.y)*scale+.5));};
    for(auto& fiber:state.fibers)for(auto& segment:fiber.segments){transform(segment.start.point);transform(segment.end.point);for(auto& p:segment.points)transform(p);}
}

void draw_gray(cucv::Mat& image,const algt_length::Fiber& fiber,int thickness){
    // DrawGlassFibreGray @ 10005770, including extrema-to-extrema bridge order.
    if(!fiber.valid)return;
    for(size_t i=0;i<fiber.segments.size();++i){
        const auto& segment=fiber.segments[i];if(segment.points.empty())continue;
        cucv::Point previous=segment.points.front();
        for(auto point:segment.points){cucv::line(image,previous,point,cucv::Scalar(255),thickness,8,0);previous=point;}
        if(i){
            const auto& prior=fiber.segments[i-1];
            cucv::Point first=prior.end.point,second=segment.start.point;
            if(fiber.along_x){if(prior.start.point.x>=prior.end.point.x)first=prior.start.point;if(segment.start.point.x>=segment.end.point.x)second=segment.end.point;}
            else {if(prior.start.point.y>=prior.end.point.y)first=prior.start.point;if(segment.start.point.y>=segment.end.point.y)second=segment.end.point;}
            cucv::line(image,first,second,cucv::Scalar(255),thickness,8,0);
        }
    }
}

void draw_gray(cucv::Mat& image,const CuVector<Fiber>& fibers,int thickness){
    for(const auto& fiber:fibers)if(fiber.valid)draw_gray(image,fiber.geometry,thickness);
}

CuVector<Fiber> take_fibers(State& state,int pass){
    CuVector<Fiber> result;result.reserve(state.fibers.size());
    for(auto& geometry:state.fibers){Fiber fiber;fiber.valid=geometry.valid;fiber.geometry=std::move(geometry);fiber.pass=pass;result.push_back(std::move(fiber));}
    state.fibers.clear();return result;
}

void measure_fibers(CuVector<Fiber>& fibers,const cucv::Mat& edge,const cucv::Mat& mask,int radius,double minimum,double maximum,double bias){
    for(auto& fiber:fibers){
        if(!fiber.valid)continue;
        fiber.raw_diameter_px=algt_geometry::measure_samples(edge,mask,all_points(fiber.geometry),radius,bias,fiber.samples);
        fiber.valid=fiber.raw_diameter_px>minimum&&fiber.raw_diameter_px<=maximum;
        fiber.geometry.valid=fiber.valid;
    }
}

void reject_mask_overlap(CuVector<Fiber>& fibers,const cucv::Mat& mask,double fraction){
    for(auto& fiber:fibers)if(algt_judgement::reject_fiber(fiber.geometry,mask,fraction)){fiber.valid=false;fiber.geometry.valid=false;}
}

void length_pipeline(State& state,const cucv::Mat& gray){
    algt_length::set_original_gray(state,gray);
    algt_length::preprocess_1(state,algt_geometry::otsu_threshold(gray));
    algt_length::preprocess_2(state);
    algt_length::fill_edge_mizi(state.binary,state.parameters.edge_width/2);
    algt_length::link_gap(state);algt_length::get_holes(state);algt_length::sift_holes(state);
    algt_length::thin_remove_thin(state);algt_length::sift_short(state);
    algt_length::find_notes_segments(state);algt_length::link_segments(state,false,false);
    algt_length::minus_edge_pixels(state,state.parameters.edge_width/2);
}

cucv::Mat enhanced_eroded_gray(const cucv::Mat& original,int erode_size){
    cucv::Mat eroded,gray;
    cucv::erode(original,eroded,cucv::getStructuringElement(cucv::MORPH_ELLIPSE,{erode_size,erode_size}),{-1,-1},1,cucv::BORDER_CONSTANT,cucv::morphologyDefaultBorderValue());
    cucv::cvtColor(eroded,gray,cucv::COLOR_BGR2GRAY);
    algt_length::enhance_gray(gray,gray,algt_geometry::otsu_threshold(gray));return gray;
}

void mid(Result& result){
    cucv::Mat gray=enhanced_eroded_gray(result.original,3);
    result.edges=algt_geometry::canny(gray);
    cucv::dilate(result.edges,result.edges,cucv::getStructuringElement(0,{2,2}),{-1,-1},1,cucv::BORDER_CONSTANT,cucv::morphologyDefaultBorderValue());
    cucv::resize(gray,gray,{},.66,.66,cucv::INTER_LINEAR);
    State state;length_pipeline(state,gray);scale_points(state,1./.66,result.original.cols,result.original.rows);
    // 10004175 zeroes DL (source=0); stack 0xff is replacement, so fill holes.
    algt_length::remove_components(result.edges,1000,0,255,8);
    result.selection=cucv::Mat::zeros(result.edges.size(),CV_8UC1);
    result.passes[0]=take_fibers(state,1);draw_gray(result.selection,result.passes[0],22);
    measure_fibers(result.passes[0],result.edges,result.selection,14,6.,26.,4.);
}

void coarse(Result& result){
    cucv::Mat gray=enhanced_eroded_gray(result.original,7);
    cucv::resize(gray,gray,{},.33,.33,cucv::INTER_LINEAR);
    State state;length_pipeline(state,gray);scale_points(state,1./.33,result.original.cols,result.original.rows);
    result.passes[1]=take_fibers(state,2);
    reject_mask_overlap(result.passes[1],result.mask,.3);
    draw_gray(result.selection,result.passes[1],44);
    measure_fibers(result.passes[1],result.edges,result.selection,25,8.,100.,5.);
}

void thin(Result& result){
    cucv::Mat gray,work,filled,edge;
    cucv::cvtColor(result.original,gray,cucv::COLOR_BGR2GRAY);
    algt_length::enhance_gray(gray,gray,algt_geometry::otsu_threshold(gray));
    work=algt_geometry::canny(gray);
    cucv::dilate(work,work,cucv::getStructuringElement(0,{3,3}),{-1,-1},1,cucv::BORDER_CONSTANT,cucv::morphologyDefaultBorderValue());
    cucv::copyMakeBorder(work,work,1,1,1,1,cucv::BORDER_CONSTANT,cucv::Scalar(255));
    algt_geometry::fill_holes(work,filled);work=filled(cucv::Rect(1,1,filled.cols-2,filled.rows-2)).clone();
    cucv::copyMakeBorder(work,work,8,8,8,8,cucv::BORDER_CONSTANT,cucv::Scalar(0));
    algt_geometry::remove_bad_shadows(work,filled);work=filled;edge=work.clone();
    cucv::copyMakeBorder(result.mask,result.mask,8,8,8,8,cucv::BORDER_CONSTANT,cucv::Scalar(0));
    State state;cucv::bitwise_or(result.mask,work,state.filtered);state.binary=state.filtered;
    algt_length::get_holes(state);
    algt_length::thin_binary(work);algt_length::prune_short_segments(work,10,0);algt_length::prune_short_segments(work,16,1);
    state.binary=work;algt_length::find_notes_segments(state);
    algt_judgement::filter_segments(state.segments);
    auto& p=state.parameters;p.link_angle_1=15.;p.link_distance_1=14.;p.link_angle_2=11.;p.link_distance_2=12.;p.link_distance_3=12.;p.min_fiber_points=100;p.min_segment_points=5;p.relative_link_distance=.5;
    algt_length::link_segments(state,true,false);
    result.passes[2]=take_fibers(state,3);
    measure_fibers(result.passes[2],edge,result.mask,10,0.,20.,3.);
    reject_mask_overlap(result.passes[2],result.mask,.25);
    for(auto& fiber:result.passes[2]){
        for(auto& segment:fiber.geometry.segments){segment.start.point-=cucv::Point(8,8);segment.end.point-=cucv::Point(8,8);for(auto& point:segment.points)point-=cucv::Point(8,8);}
        for(auto& sample:fiber.samples){sample.center-=cucv::Point(8,8);sample.first-=cucv::Point(8,8);sample.second-=cucv::Point(8,8);}
    }
}
}

Result measure(const unsigned char* bgr,int width,int height){
    Result result;
    result.original=cucv::Mat(height,width,CV_8UC3,const_cast<unsigned char*>(bgr));
    mid(result);result.mask=cucv::Mat::zeros(height,width,CV_8UC1);draw_gray(result.mask,result.passes[0],22);
    coarse(result);draw_gray(result.mask,result.passes[1],44);thin(result);
    return result;
}
}

namespace {
py::list cu_point(cucv::Point p){py::list r;r.append(p.x);r.append(p.y);return r;}
py::dict automatic_entry(py::array_t<unsigned char,py::array::c_style> input,const std::string& algorithm,py::object checkpoint,size_t maximum,size_t budget,const std::string& preset,bool debug){
    if(input.ndim()!=2 && !(input.ndim()==3&&input.shape(2)==3))throw std::invalid_argument("automatic input must be uint8 gray or BGR");
    if(input.shape(0)>32766||input.shape(1)>32766)throw std::invalid_argument("automatic input dimensions exceed fixed-point drawing range");
    if(input.shape(0)<16||input.shape(1)<16)throw std::invalid_argument("automatic input dimensions must be at least 16");
    if(maximum<1||maximum>1000000)throw std::invalid_argument("max_candidates must be in 1..1000000");
    if(algorithm!="traditional"&&algorithm!="multiscale")throw std::invalid_argument("unknown automatic algorithm");
    if(preset!="fiber0"&&preset!="fiber1"&&preset!="constructor")throw std::invalid_argument("unknown traditional preset");
    if(cu_runtime)throw std::runtime_error("automatic kernel cannot be reentered");
    CuRuntime runtime(std::move(checkpoint),budget,maximum);
    struct Scope{Scope(CuRuntime* r){cu_runtime=r;}~Scope(){cu_runtime=nullptr;}} scope(&runtime);
    cu_check();
    // Reserve transient cv2 buffers, including the Canny edge stack.
    // Owning images and actual graph/vector capacities are metered separately;
    // reserving a second full-image graph allowance would reject valid 6K frames.
    size_t pixels=size_t(input.shape(0))*size_t(input.shape(1));
    CuReservation workspace(pixels*16);
    cucv::Mat image(int(input.shape(0)),int(input.shape(1)),input.ndim()==3?3:1,const_cast<unsigned char*>(input.data()));
    py::list candidates;py::dict diagnostics;diagnostics["algorithm"]=algorithm;diagnostics["preset"]=preset;diagnostics["cv2_version"]=runtime.cv2.attr("__version__");
    if(algorithm=="traditional"){
        cu_legacy::Engine engine(maximum);
        if(preset=="fiber1"){engine.field<int>(0x88)=4;engine.field<int>(0x8c)=70;}
        {py::gil_scoped_release release;engine.load(image);}
        size_t output_bytes=0;for(int i=0;i<engine.field<int>(0x94);++i)output_bytes+=1024+size_t(engine.fibers[i].count)*768;CuReservation output_budget(output_bytes);
        diagnostics["circle_count"]=engine.count();diagnostics["relation_entries"]=size_t(engine.count())*engine.count();
        for(int i=0;i<engine.field<int>(0x94);++i){
            auto& f=engine.fibers[i];py::dict candidate;py::list path,samples,indices;
            for(int j=0;j<f.count;++j){int index=f.indices[j];auto& c=engine.circles[index];path.append(cu_point(c.center));py::dict s;s["start"]=cu_point({c.p.y,c.p.x});s["end"]=cu_point({c.q.y,c.q.x});samples.append(s);indices.append(index);}
            auto& representative=engine.circles[f.indices[f.count/2]];
            candidate["width_px"]=f.diameter;candidate["start"]=cu_point({representative.p.y,representative.p.x});candidate["end"]=cu_point({representative.q.y,representative.q.x});candidate["path"]=path;candidate["samples"]=samples;candidate["circle_indices"]=indices;candidate["centerline_length_px"]=f.length;candidate["pass"]=0;candidates.append(candidate);
        }
        if(debug){CuReservation debug_arrays(pixels*3);diagnostics["gray"]=cucv::array(engine.gray()->matrix).attr("copy")();diagnostics["contour"]=cucv::array(engine.contour()->matrix).attr("copy")();diagnostics["mask_labeled"]=cucv::array(engine.mask()->matrix).attr("copy")();}
    }else{
        algt_port::Result result;
        {py::gil_scoped_release release;if(image.channels==1){cucv::Mat color(image.rows,image.cols,3);for(int y=0;y<image.rows;++y)for(int x=0;x<image.cols;++x)for(int c=0;c<3;++c)color.ptr<unsigned char>(y)[x*3+c]=image.ptr<unsigned char>(y)[x];image=std::move(color);}result=algt_port::measure(image.ptr<unsigned char>(0),image.cols,image.rows);}
        size_t output_bytes=0;for(auto& pass:result.passes)for(auto& f:pass)if(f.valid){output_bytes+=1024+f.samples.size()*1024;for(auto& segment:f.geometry.segments)output_bytes+=256+segment.points.size()*160;}CuReservation output_budget(output_bytes);
        py::list passes,pass_axes;
        for(auto& pass:result.passes){if(debug){py::list axes;for(const auto& fiber:pass)axes.append(fiber.geometry.along_x);pass_axes.append(axes);}int valid=0;for(auto& f:pass)if(f.valid){if(size_t(candidates.size())>=maximum)throw CuBudgetError("automatic_candidate_limit");++valid;
            py::dict candidate;py::list path,segments,samples;const algt_geometry::WidthSample* representative=nullptr;
            for(auto& segment:f.geometry.segments){py::list points;for(auto p:segment.points){auto q=cu_point(p);points.append(q);path.append(q);}py::dict s;s["number"]=segment.number;s["points"]=points;segments.append(s);}
            for(auto& sample:f.samples){py::dict s;s["start"]=cu_point(sample.first);s["end"]=cu_point(sample.second);s["center"]=cu_point(sample.center);s["raw_width"]=sample.raw_width;s["corrected_width"]=sample.corrected_width;s["retained"]=sample.retained;samples.append(s);if(sample.retained&&!representative)representative=&sample;}
            if(!representative)throw std::runtime_error("automatic_valid_fiber_has_no_width_sample");
            candidate["width_px"]=f.raw_diameter_px;candidate["start"]=cu_point(representative->first);candidate["end"]=cu_point(representative->second);candidate["path"]=path;candidate["segments"]=segments;candidate["samples"]=samples;candidate["pass"]=f.pass;candidate["along_x"]=f.geometry.along_x;candidate["fiber_start"]=cu_point(f.geometry.start);candidate["fiber_end"]=cu_point(f.geometry.end);candidates.append(candidate);
        }py::dict p;p["count"]=pass.size();p["valid"]=valid;passes.append(p);}diagnostics["passes"]=passes;if(debug)diagnostics["pass_axes"]=pass_axes;
        if(debug){CuReservation debug_arrays(size_t(result.mask.rows)*result.mask.cols+size_t(result.edges.rows)*result.edges.cols+size_t(result.selection.rows)*result.selection.cols);diagnostics["mask"]=cucv::array(result.mask).attr("copy")();diagnostics["edges"]=cucv::array(result.edges).attr("copy")();diagnostics["selection"]=cucv::array(result.selection).attr("copy")();}
    }
    cu_check();diagnostics["native_peak_reserved_bytes"]=runtime.peak;diagnostics["native_loop_checkpoints"]=runtime.ticks/16384;
    py::dict out;out["status"]=candidates.empty()?"empty":"ok";out["reason"]=candidates.empty()?"no_valid_fibers":"";out["candidates"]=candidates;out["diagnostics"]=diagnostics;return out;
}
}
void register_cu_automatic(py::module_& module){
    py::register_exception<CuBudgetError>(module,"CuAutomaticResourceError",PyExc_MemoryError);
    module.def("_cu_automatic",&automatic_entry,py::arg("image_bgr").noconvert(),py::arg("algorithm")="traditional",py::arg("checkpoint")=py::none(),py::arg("max_candidates")=10000,py::arg("memory_budget")=size_t(2)*1024*1024*1024,py::arg("preset")="fiber1",py::arg("debug")=false);
}
