// CU-assisted line measurement, reconstructed from the documented calculation
// rules. No CU binaries, OpenCV ABI, Win32/GDI calls, or image files are used.
#include "cu_manual.hpp"
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace {
struct Point { int x, y; };
bool operator==(Point a, Point b) { return a.x == b.x && a.y == b.y; }
int manhattan(Point a, Point b) { return std::abs(a.x-b.x)+std::abs(a.y-b.y); }

class Checkpoint {
public:
    explicit Checkpoint(py::object callback): callback_(std::move(callback)) {
        if (!callback_.is_none() && !PyCallable_Check(callback_.ptr()))
            throw py::type_error("checkpoint must be callable or None");
    }
    void check() const {
        if (!callback_.is_none()) { py::gil_scoped_acquire lock; callback_(); }
    }
private:
    py::object callback_;
};

struct Region {
    int x, y, width, height, frame_width, frame_height;
    std::vector<std::uint8_t> gray, mask;
    bool inside(Point p) const { return p.x >= x && p.y >= y && p.x < x+width && p.y < y+height; }
    std::size_t index(Point p) const { return std::size_t(p.y-y)*width+(p.x-x); }
    std::uint8_t value(Point p) const { return mask[index(p)]; }
};

bool clamp_endpoint(Point& p, int width, int height) {
    bool changed=false;
    if (p.x<=10) { p.x=13; changed=true; }
    if (p.x>=width-10) { p.x=width-13; changed=true; }
    if (p.y<=10) { p.y=13; changed=true; }
    if (p.y>=height-10) { p.y=height-13; changed=true; }
    return changed;
}
Point parse_point(const py::object& value, int width, int height) {
    if (!py::isinstance<py::sequence>(value) || py::len(value)!=2)
        throw py::value_error("测量端点必须包含两个原图坐标。");
    const auto sequence=py::reinterpret_borrow<py::sequence>(value);
    const double x=py::cast<double>(sequence[0]), y=py::cast<double>(sequence[1]);
    if (!std::isfinite(x) || !std::isfinite(y) || x<0 || y<0 || x>=width || y>=height)
        throw py::value_error("测量端点必须位于原图内，且坐标必须为有限数值。");
    // CU divides out display zoom, then truncates positive source coordinates.
    return {int(x),int(y)};
}

// Win32 LineDDA's major-axis stepping, including the first point, excluding the
// last. Strict tie handling is retained for reversed and diagonal gestures.
std::vector<Point> line_points(Point a, Point b) {
    const int dx=std::abs(b.x-a.x), dy=std::abs(b.y-a.y);
    const int sx=b.x>a.x?1:-1, sy=b.y>a.y?1:-1;
    std::vector<Point> result;
    result.reserve(std::size_t(std::max(dx,dy)));
    if (dx>=dy) {
        int error=dx/2;
        for (int i=0;i<dx;++i) {
            result.push_back(a); a.x+=sx; error-=dy;
            if (error<0) { a.y+=sy; error+=dx; }
        }
    } else {
        int error=dy/2;
        for (int i=0;i<dy;++i) {
            result.push_back(a); a.y+=sy; error-=dx;
            if (error<0) { a.x+=sx; error+=dy; }
        }
    }
    return result;
}

// This is the recovered successive-score threshold rule, NOT global-max Otsu.
// Binary64 operation ordering follows the default x87 precision-control=53-bit
// reference runs. No fused multiply-add or fast-math may be enabled.
int histogram_threshold(const std::array<int,256>& h, int pixels) {
    double total_weight=0;
    for (int i=0;i<256;++i) total_weight+=double(std::int64_t(i+1)*h[i]);
    double lower=h[0], upper=double(pixels)-lower;
    double mean_lower=lower==0?0:1;
    double mean_upper=upper==0?total_weight:(total_weight-lower*mean_lower)/upper;
    double difference=mean_lower-mean_upper;
    double previous=((lower*upper)*difference)*difference/(lower+upper);
    int threshold=0;
    for (int intensity=2;intensity<=255;++intensity) {
        const double next_lower=lower+h[intensity-1];
        const double next_upper=double(pixels)-next_lower;
        if (next_upper<=0) break;
        const double next_mean_lower=next_lower==0?0:
            (mean_lower*lower+double(std::int64_t(intensity)*h[intensity-1]))/next_lower;
        const double next_mean_upper=(total_weight-next_mean_lower*next_lower)/next_upper;
        difference=next_mean_upper-next_mean_lower;
        const double score=((next_lower*next_upper)*difference)*difference/(next_lower+next_upper);
        if (score>previous) threshold=intensity;
        lower=next_lower; upper=next_upper; mean_lower=next_mean_lower;
        mean_upper=next_mean_upper; previous=score;
    }
    return threshold;
}

int endpoint_mean(const Region& r, Point p) {
    int sum=0;
    for (int x=p.x-2;x<p.x+3;++x) for (int y=p.y-2;y<p.y+3;++y)
        sum+=r.gray[r.index({x,y})];
    return sum/25;
}

// The reference labels and restores an entire component on EVERY line pixel.
// At 37 connected pixels it is already proven not to be removed. Stopping there
// is exact, bounds work per seed, and never clips a small component at the ROI.
std::size_t clean_small_components(Region& r, const std::vector<Point>& line,
                                 const Checkpoint& checkpoint) {
    const Point delta[4]={{-1,0},{1,0},{0,-1},{0,1}};
    std::vector<std::uint32_t> visited(r.mask.size(),0);
    std::vector<std::uint8_t> known_large(r.mask.size(),0);
    std::uint32_t stamp=0;
    std::array<Point,37> queue{};
    std::size_t removed=0;
    for (std::size_t n=0;n<line.size();++n) {
        if ((n&63)==0) checkpoint.check();
        const Point seed=line[n];
        if (!r.value(seed) || known_large[r.index(seed)]) continue;
        ++stamp;
        std::size_t count=1,head=0;
        queue[0]=seed; visited[r.index(seed)]=stamp;
        bool large=false;
        while (head<count && !large) {
            const Point p=queue[head++];
            for (Point d:delta) {
                const Point q{p.x+d.x,p.y+d.y};
                if (q.x<0 || q.y<0 || q.x>=r.frame_width || q.y>=r.frame_height) continue;
                // A component with <=36 pixels cannot reach beyond the 38-pixel
                // halo. Reaching its edge is sufficient proof of a large one.
                if (!r.inside(q)) { large=true; break; }
                const auto index=r.index(q);
                if (r.mask[index]==255 && visited[index]!=stamp) {
                    visited[index]=stamp; queue[count++]=q;
                    if (count==queue.size()) { large=true; break; }
                }
            }
        }
        for (std::size_t i=0;i<count;++i) {
            const auto index=r.index(queue[i]);
            if (large) known_large[index]=1;
            else { r.mask[index]=0; ++removed; }
        }
    }
    return removed;
}

bool track_edge(const Region& r, Point hit, Point outside, std::vector<Point>& points) {
    points.push_back(hit);
    const int dy[8]={0,-1,-1,-1,0,1,1,1};
    const int dx[2][8]={{1,1,0,-1,-1,-1,0,1},{-1,-1,0,1,1,1,0,-1}};
    for (int side=0;side<2;++side) {
        int start=-1;
        for(int i=0;i<8;++i) if (dx[side][i]==outside.x-hit.x && dy[i]==outside.y-hit.y) { start=i; break; }
        if (start<0) return false;
        Point at=hit; int count=0,scan=0;
        while (scan<8) {
            if (count>9) break;
            const int index=(scan+start)%8;
            const Point next{at.x+dx[side][index],at.y+dy[index]};
            if (!r.inside(next)) return false;
            if (r.value(next)==255) {
                points.push_back(next); ++count; start=(index+5)%8; scan=0; at=next;
            }
            ++scan;
        }
        if (count<10) return false;
    }
    return true;
}
struct Fit { double slope=0, intercept=0; bool vertical=false, valid=false; };
Fit fit_points(const std::vector<Point>& points) {
    Fit fit;
    if (points.size()<2) return fit;
    double sx=0,sy=0,sxy=0,sxx=0,syy=0;
    for (Point p:points) {
        const double x=p.x,y=p.y;
        sx+=x; sy+=y; sxy+=x*y; sxx+=x*x; syy+=y*y;
    }
    const double n=double(points.size());
    if (std::abs(points.front().x-points.back().x)<std::abs(points.front().y-points.back().y)) {
        const double denominator=sx*sy-n*sxy;
        if (denominator==0) { fit.slope=1; fit.vertical=true; fit.valid=true; return fit; }
        fit.slope=(sy*sy-n*syy)/denominator;
    } else {
        const double denominator=sx*sx-n*sxx;
        if (denominator!=0) fit.slope=(sx*sy-n*sxy)/denominator;
    }
    fit.intercept=(sy-fit.slope*sx)/n; fit.valid=true;
    return fit;
}
double angle(const Fit& a,const Fit& b) {
    if (!a.valid || !b.valid) return -1;
    constexpr double pi=3.141592653589793;
    const double aa=a.vertical?90:(std::atan(a.slope)*180)/pi;
    const double bb=b.vertical?90:(std::atan(b.slope)*180)/pi;
    double result=std::abs(bb-aa);
    return result>90?180-result:result;
}
double width(Point a,Point b,const Fit& first,const Fit& second) {
    const double da=first.vertical?std::abs(a.x-b.x):
        std::abs((double(b.x)*first.slope-double(b.y))+first.intercept)/std::sqrt(first.slope*first.slope+1);
    const double db=second.vertical?std::abs(a.x-b.x):
        std::abs((double(a.x)*second.slope-double(a.y))+second.intercept)/std::sqrt(second.slope*second.slope+1);
    return (da+db)*.5;
}

py::list points_to_python(const std::vector<Point>& points) {
    py::list result;
    for(Point p:points) result.append(py::make_tuple(p.x,p.y));
    return result;
}

py::dict measure(const py::array& image, py::object start_xy, py::object end_xy,
                 py::object callback) {
    if (!image.dtype().equal(py::dtype::of<std::uint8_t>())) throw py::type_error("测量图像必须为 uint8。");
    if (image.ndim()!=2 && !(image.ndim()==3 && image.shape(2)==3))
        throw py::value_error("测量图像必须为二维灰度图或三通道 BGR 图。");
    if (image.shape(0)<24 || image.shape(1)<24 || image.shape(0)>20000 || image.shape(1)>20000)
        throw py::value_error("测量图像宽高必须在 24 至 20000 像素之间。");
    const int height=int(image.shape(0)),frame_width=int(image.shape(1));
    const Point requested_first=parse_point(start_xy,frame_width,height), requested_second=parse_point(end_xy,frame_width,height);
    Point first=requested_first,second=requested_second;
    const bool adjusted_first=clamp_endpoint(first,frame_width,height);
    const bool adjusted_second=clamp_endpoint(second,frame_width,height);
    if (adjusted_second) std::swap(first,second);
    Checkpoint checkpoint(std::move(callback));
    checkpoint.check();
    int left=std::min(first.x,second.x),right=std::max(first.x,second.x);
    int top=std::min(first.y,second.y),bottom=std::max(first.y,second.y);
    if (bottom-top<6) { top-=3; bottom+=3; }
    if (right-left<6) { left-=3; right+=3; }
    const int rx=std::max(0,std::min(first.x,second.x)-38), ry=std::max(0,std::min(first.y,second.y)-38);
    Region region{rx,ry,std::min(frame_width,std::max(first.x,second.x)+39)-rx,
        std::min(height,std::max(first.y,second.y)+39)-ry,frame_width,height,{},{}};
    const std::size_t region_pixels=std::size_t(region.width)*region.height;
    // gray + mask + visited(uint32) + known_large consume seven bytes/pixel.
    // A gesture across an enormous frame must not allocate gigabytes of scratch.
    if(region_pixels>8u*1024u*1024u)
        throw py::value_error("人工测径局部区域超过 8 Mi 像素预算，请缩短划线后重试。");
    region.gray.resize(region_pixels);
    region.mask.resize(region.gray.size());
    const auto* base=static_cast<const std::uint8_t*>(image.data());
    const auto row_stride=image.strides(0),column_stride=image.strides(1);
    const bool color=image.ndim()==3;
    const auto channel_stride=color?image.strides(2):0;
    // Immutable task frames permit the bounded copy to release the GIL too.
    // Writable caller arrays remain protected from ordinary Python mutation;
    // neither kind may be mutated through another native alias concurrently.
    const auto copy_snapshot=[&] {
        for (int y=0;y<region.height;++y) {
            if ((y&63)==0) checkpoint.check();
            for (int x=0;x<region.width;++x) {
                const auto* pixel=base+py::ssize_t(y+ry)*row_stride+py::ssize_t(x+rx)*column_stride;
                int value=*pixel;
                if (color) {
                    const int b=pixel[0],g=pixel[channel_stride],r=pixel[2*channel_stride];
                    value=(b*1868+g*9617+r*4899+8192)>>14;
                }
                region.gray[std::size_t(y)*region.width+x]=std::uint8_t(int(std::clamp(double(value)*1.3-90.,0.,180.)+.5));
            }
        }
    };
    if(!image.writeable()) { py::gil_scoped_release release; copy_snapshot(); }
    else copy_snapshot();
    std::vector<Point> edge_a,edge_b;
    std::string reason="ok";
    double measured=-1,between=-1,across=-1;
    int threshold=-1,background_mean=-1;
    std::size_t removed=0;
    Point hit_first=first,hit_last=second;
    {
        py::gil_scoped_release release;
        checkpoint.check();
        std::array<int,256> histogram{};
        for (int y=top;y<bottom;++y) {
            if ((y&63)==0) checkpoint.check();
            for(int x=left;x<right;++x) ++histogram[region.gray[region.index({x,y})]];
        }
        threshold=histogram_threshold(histogram,(right-left)*(bottom-top));
        background_mean=(endpoint_mean(region,first)+endpoint_mean(region,second))/2;
        int lower=0,upper=0;
        for (int i=0;i<threshold;++i) lower+=histogram[i];
        for (int i=threshold;i<256;++i) upper+=histogram[i];
        if (lower>upper) threshold=background_mean<=100?80:background_mean-20;
        for (std::size_t i=0;i<region.gray.size();++i) {
            if ((i&16383)==0) checkpoint.check();
            region.mask[i]=region.gray[i]<=threshold?255:0;
        }
        const auto line=line_points(first,second);
        removed=clean_small_components(region,line,checkpoint);
        if (manhattan(first,second)<5) reason="line_too_short";
        else if (region.value(first)!=0 || region.value(second)!=0) reason="endpoints_not_background";
        else {
            Point before{-1,-1},after{-1,-1};
            hit_first=hit_last={-1,-1}; int phase=0;
            for (Point p:line) {
                const auto v=region.value(p);
                if (phase==0) { if(v==0) { before=p; continue; } hit_first=p; phase=1; }
                if (phase==1) { if(v!=0) { hit_last=p; continue; } after=p; phase=2; }
                if (phase==2 && v==255) { after={-1,-1}; phase=1; hit_last=p; }
            }
            if (hit_first.x<0 || hit_last.x<0 || before.x<0 || after.x<0) reason="no_complete_crossing";
            else if (!track_edge(region,hit_first,before,edge_a) || !track_edge(region,hit_last,after,edge_b)) reason="edge_tracking_failed";
            else if (manhattan(edge_a.front(),edge_a.back())<4 || manhattan(edge_b.front(),edge_b.back())<4) reason="edge_span_too_short";
            else {
                const auto a=fit_points(edge_a),b=fit_points(edge_b);
                between=angle(a,b); across=angle(a,fit_points({first,second}));
                if (!(between>=0 && between<30)) reason="edges_not_parallel";
                else if (!(across>60)) reason="crossing_too_oblique";
                else {
                    measured=width(hit_first,hit_last,a,b);
                    if (!std::isfinite(measured) || measured<=0) { measured=-1; reason="invalid_width"; }
                    else { first=hit_first; second=hit_last; }
                }
            }
        }
        checkpoint.check();
    }
    py::dict diagnostic;
    diagnostic["algorithm_revision"]="cu-manual-portable-v1";
    diagnostic["threshold"]=threshold;
    diagnostic["endpoint_mean"]=background_mean;
    diagnostic["edge_angle_degrees"]=between;
    diagnostic["crossing_angle_degrees"]=across;
    diagnostic["boundary_adjusted"]=adjusted_first||adjusted_second;
    diagnostic["roi"]=py::make_tuple(region.x,region.y,region.width,region.height);
    diagnostic["processed_pixels"]=region.gray.size();
    diagnostic["removed_small_component_pixels"]=removed;
    diagnostic["input_start"]=py::make_tuple(requested_first.x,requested_first.y);
    diagnostic["input_end"]=py::make_tuple(requested_second.x,requested_second.y);
    py::dict result;
    result["status"]=reason=="ok"?"ok":"rejected";
    result["reason"]=reason;
    result["width_px"]=measured>0?py::cast(measured):py::none();
    result["start"]=py::make_tuple(first.x,first.y);
    result["end"]=py::make_tuple(second.x,second.y);
    result["edge_a"]=points_to_python(edge_a);
    result["edge_b"]=points_to_python(edge_b);
    result["diagnostics"]=diagnostic;
    return result;
}
} // namespace

void register_cu_manual(py::module_& module) {
    module.def("cu_manual_measure", &measure, py::arg("image"), py::arg("start_xy"),
               py::arg("end_xy"), py::arg("checkpoint")=py::none(),
               "Measure a CU-style assisted fiber crossing on an owned local image snapshot.");
}
