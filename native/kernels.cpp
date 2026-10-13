// Exact numerical kernels. Python owns policy, task scheduling and diagnostics.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

#ifndef FDM_NATIVE_API_VERSION
#define FDM_NATIVE_API_VERSION 1
#endif
#ifndef FDM_NATIVE_SOURCE_SHA256
#define FDM_NATIVE_SOURCE_SHA256 "unavailable"
#endif
#ifndef FDM_NATIVE_PYTHON_ABI
#define FDM_NATIVE_PYTHON_ABI "unavailable"
#endif
#ifndef FDM_NATIVE_BUILD_TYPE
#define FDM_NATIVE_BUILD_TYPE "unspecified"
#endif
#define FDM_STRINGIFY_DETAIL(value) #value
#define FDM_STRINGIFY(value) FDM_STRINGIFY_DETAIL(value)

namespace {

constexpr std::size_t kMaxEdges = 2'000'000;
constexpr std::size_t kScanCheckpointMask = 65'535;
constexpr std::size_t kWorkCheckpointMask = 16'383;

// This object is constructed and destroyed while holding the GIL. Its owning
// reference also keeps a bound callback's captured input/task state alive.
class Checkpoint {
public:
    explicit Checkpoint(py::object callback) : callback_(std::move(callback)) {
        if (!callback_.is_none() && !PyCallable_Check(callback_.ptr())) {
            throw py::type_error("checkpoint must be callable or None");
        }
    }

    bool enabled() const { return !callback_.is_none(); }

    void check() const {
        if (enabled()) {
            py::gil_scoped_acquire acquire;
            callback_();
        }
    }

private:
    py::object callback_;
};

void require_c_array(const py::array& array, int ndim, const char* name) {
    if (array.ndim() != ndim) {
        throw py::value_error(std::string(name) + " has an invalid number of dimensions");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C-contiguous");
    }
}

template <typename T>
bool has_dtype(const py::array& array) {
    return array.dtype().equal(py::dtype::of<T>());
}

template <typename T>
void require_dtype(const py::array& array, const char* name) {
    if (!has_dtype<T>(array)) {
        throw py::type_error(std::string(name) + " has an unsupported dtype");
    }
    if (reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T)) {
        throw py::value_error(std::string(name) + " must have aligned storage");
    }
}

std::size_t checked_pixels(const py::array& array) {
    const auto height = static_cast<std::size_t>(array.shape(0));
    const auto width = static_cast<std::size_t>(array.shape(1));
    const auto limit = static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max());
    if (width > limit - 1 || (width && height > limit / width)) {
        throw py::value_error("array dimensions exceed supported address space");
    }
    return height * width;
}

py::array_t<double> boundary_segments(const py::array& mask, py::object callback) {
    require_c_array(mask, 2, "mask");
    require_dtype<bool>(mask, "mask");
    checked_pixels(mask);
    const auto height = static_cast<std::size_t>(mask.shape(0));
    const auto width = static_cast<std::size_t>(mask.shape(1));
    // NumPy bool buffers can contain any nonzero byte (for example a uint8
    // view). Reading those bytes as C++ bool would have undefined behavior.
    const auto* pixels = static_cast<const std::uint8_t*>(mask.data());
    Checkpoint checkpoint(std::move(callback));
    std::array<std::size_t, 4> counts{};
    std::size_t total = 0;
    {
        py::gil_scoped_release release;
        checkpoint.check();
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const auto index = y * width + x;
                if (!(index & kScanCheckpointMask)) checkpoint.check();
                if (!pixels[index]) continue;
                const std::array<bool, 4> exposed{
                    y == 0 || !pixels[index - width],
                    y + 1 == height || !pixels[index + width],
                    x == 0 || !pixels[index - 1],
                    x + 1 == width || !pixels[index + 1],
                };
                for (std::size_t side = 0; side < exposed.size(); ++side) {
                    if (exposed[side]) {
                        ++counts[side];
                        if (++total > kMaxEdges) {
                            throw py::value_error("轮廓碎片过多，请先剔除噪声或框选目标。");
                        }
                    }
                }
            }
        }
        checkpoint.check();
    }

    // Allocate only after counting and enforcing the cap: no temporary padded
    // masks, edge arrays or final concatenate copy are necessary.
    py::array_t<double> output(std::vector<py::ssize_t>{static_cast<py::ssize_t>(total), 2, 2});
    auto* result = output.mutable_data();
    std::array<std::size_t, 4> offsets{0, counts[0], counts[0] + counts[1],
                                      counts[0] + counts[1] + counts[2]};
    {
        py::gil_scoped_release release;
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const auto index = y * width + x;
                if (!(index & kScanCheckpointMask)) checkpoint.check();
                if (!pixels[index]) continue;
                const double xd = static_cast<double>(x), yd = static_cast<double>(y);
                auto write = [&](std::size_t side, double x0, double y0, double x1, double y1) {
                    auto* edge = result + 4 * offsets[side]++;
                    edge[0] = x0; edge[1] = y0; edge[2] = x1; edge[3] = y1;
                };
                // Each group retains NumPy nonzero's row-major order. Both
                // horizontal groups point right; both vertical groups point down.
                if (y == 0 || !pixels[index - width]) write(0, xd, yd, xd + 1, yd);
                if (y + 1 == height || !pixels[index + width]) write(1, xd, yd + 1, xd + 1, yd + 1);
                if (x == 0 || !pixels[index - 1]) write(2, xd, yd, xd, yd + 1);
                if (x + 1 == width || !pixels[index + 1]) write(3, xd + 1, yd, xd + 1, yd + 1);
            }
        }
        checkpoint.check();
    }
    return output;
}

py::tuple contour_sections(const py::array& segments, const py::array& heights,
                           double step, py::object callback) {
    require_c_array(segments, 3, "segments");
    require_dtype<double>(segments, "segments");
    if (segments.shape(1) != 2 || segments.shape(2) != 2) {
        throw py::value_error("segments must have shape (E, 2, 2)");
    }
    require_c_array(heights, 1, "heights");
    require_dtype<double>(heights, "heights");
    if (heights.shape(0) == 0) throw py::value_error("heights must not be empty");
    if (!std::isfinite(step) || step <= 0) throw py::value_error("step must be finite and positive");
    const auto edges = static_cast<std::size_t>(segments.shape(0));
    const auto rows = static_cast<std::size_t>(heights.shape(0));
    if (rows >= static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max()) ||
        edges > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max()) / 4) {
        throw py::value_error("array dimensions exceed supported address space");
    }
    const auto* points = static_cast<const double*>(segments.data());
    const auto* samples = static_cast<const double*>(heights.data());
    Checkpoint checkpoint(std::move(callback));
    std::vector<std::int64_t> offsets;
    std::vector<double> intervals;
    {
        py::gil_scoped_release release;
        checkpoint.check();
        for (std::size_t row = 0; row < rows; ++row) {
            if (!(row & kWorkCheckpointMask)) checkpoint.check();
            if (!std::isfinite(samples[row])) throw py::value_error("heights must be finite");
        }
        std::vector<std::vector<double>> hits(rows);
        std::size_t active = 0, intersections = 0;
        const auto grid_index = [&](double value) -> std::size_t {
            const double index = std::ceil((value - samples[0]) / step - 1e-10);
            // Clamp before integer conversion, including +/- infinity from an
            // otherwise finite subtraction/division at extreme input scales.
            if (index <= 0) return 0;
            if (index >= static_cast<double>(rows)) return rows;
            return static_cast<std::size_t>(index);
        };
        for (std::size_t edge = 0; edge < edges; ++edge) {
            if (!(edge & kWorkCheckpointMask)) checkpoint.check();
            const auto* p = points + 4 * edge;
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) ||
                !std::isfinite(p[2]) || !std::isfinite(p[3])) {
                throw py::value_error("segments must be finite");
            }
            const double dy = p[3] - p[1], dx = p[2] - p[0];
            if (!std::isfinite(dy) || !std::isfinite(dx)) {
                throw py::value_error("segment differences exceed floating-point range");
            }
            const auto start = grid_index(std::min(p[1], p[3]));
            const auto stop = grid_index(std::max(p[1], p[3]));
            if (stop <= start || std::abs(dy) <= 1e-12) continue;
            if (!(active++ % 2048)) checkpoint.check();
            for (auto row = start; row < stop; ++row) {
                if (!(intersections++ & kWorkCheckpointMask)) checkpoint.check();
                const double fraction = (samples[row] - p[1]) / dy;
                const double x = p[0] + fraction * dx;
                if (!std::isfinite(x)) throw py::value_error("intersection exceeds floating-point range");
                hits[row].push_back(x);
            }
        }

        offsets.reserve(rows + 1);
        offsets.push_back(0);
        for (auto& values : hits) {
            checkpoint.check();
            if (checkpoint.enabled()) {
                std::size_t comparisons = 0;
                std::stable_sort(values.begin(), values.end(), [&](double first, double second) {
                    if (!(comparisons++ & kWorkCheckpointMask)) checkpoint.check();
                    return first < second;
                });
            } else {
                std::stable_sort(values.begin(), values.end());
            }
            if (values.size() % 2) throw py::value_error("轮廓截线存在未闭合交点，请检查掩膜。");
            const auto row_begin = intervals.size();
            for (std::size_t index = 0; index < values.size(); index += 2) {
                if (!(index & kWorkCheckpointMask)) checkpoint.check();
                const double left = values[index], right = values[index + 1];
                if (right - left <= 1e-8) continue;
                if (intervals.size() > row_begin && std::abs(left - intervals.back()) < 1e-8) {
                    intervals.back() = right;
                } else {
                    intervals.push_back(left);
                    intervals.push_back(right);
                }
            }
            if (intervals.size() / 2 > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
                throw py::value_error("too many contour intervals");
            }
            offsets.push_back(static_cast<std::int64_t>(intervals.size() / 2));
        }
        checkpoint.check();
    }
    if (intervals.size() / 2 > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw py::value_error("too many contour intervals");
    }
    py::array_t<std::int64_t> output_offsets(static_cast<py::ssize_t>(offsets.size()));
    py::array_t<double> output_intervals(std::vector<py::ssize_t>{static_cast<py::ssize_t>(intervals.size() / 2), 2});
    auto* offsets_data = output_offsets.mutable_data();
    auto* intervals_data = output_intervals.mutable_data();
    {
        py::gil_scoped_release release;
        for (std::size_t index = 0; index < offsets.size(); ++index) {
            if (!(index & kWorkCheckpointMask)) checkpoint.check();
            offsets_data[index] = offsets[index];
        }
        for (std::size_t index = 0; index < intervals.size(); ++index) {
            if (!(index & kWorkCheckpointMask)) checkpoint.check();
            intervals_data[index] = intervals[index];
        }
        checkpoint.check();
    }
    return py::make_tuple(std::move(output_offsets), std::move(output_intervals));
}

// Under MSVC /fp:strict, std::isfinite(double) can call the CRT for every
// pixel. Classify the IEEE NumPy storage without float conversion or arithmetic;
// memcpy preserves aliasing rules and the NaN/Inf rejection contract.
bool finite_distance(float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559);
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000U) != 0x7f800000U;
}

bool finite_distance(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

template <typename Distance>
void prune_impl(bool* usable, const std::uint8_t* source, const Distance* distance_map,
                std::size_t height, std::size_t width, const Checkpoint& checkpoint) {
    const auto pixels = height * width;
    for (std::size_t index = 0; index < pixels; ++index) {
        if (!(index & kScanCheckpointMask)) checkpoint.check();
        if (!finite_distance(distance_map[index])) {
            throw py::value_error("distance_map must be finite");
        }
        usable[index] = source[index] != 0;
    }
    std::vector<std::uint8_t> neighbors(pixels), degrees(pixels), removed(pixels, 0);
    std::vector<std::size_t> endpoints, removed_pixels, path;
    const std::array<std::ptrdiff_t, 8> offsets{
        -static_cast<std::ptrdiff_t>(width), -static_cast<std::ptrdiff_t>(width) + 1,
        1, static_cast<std::ptrdiff_t>(width) + 1, static_cast<std::ptrdiff_t>(width),
        static_cast<std::ptrdiff_t>(width) - 1, -1, -static_cast<std::ptrdiff_t>(width) - 1,
    };
    const double diagonal_length = std::hypot(1.0, 1.0);
    for (int iteration = 0; iteration < 8; ++iteration) {
        checkpoint.check();
        endpoints.clear();
        removed_pixels.clear();
        std::size_t usable_count = 0;
        bool has_junction = false;
        double max_radius = -std::numeric_limits<double>::infinity();
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const auto index = y * width + x;
                if (!(index & kScanCheckpointMask)) checkpoint.check();
                if (!usable[index]) continue;
                ++usable_count;
                const bool north = y > 0 && usable[index - width];
                const bool east = x + 1 < width && usable[index + 1];
                const bool south = y + 1 < height && usable[index + width];
                const bool west = x > 0 && usable[index - 1];
                const std::array<bool, 8> adjacent{
                    north,
                    y > 0 && x + 1 < width && usable[index - width + 1] && !(north || east),
                    east,
                    y + 1 < height && x + 1 < width && usable[index + width + 1] && !(east || south),
                    south,
                    y + 1 < height && x > 0 && usable[index + width - 1] && !(south || west),
                    west,
                    y > 0 && x > 0 && usable[index - width - 1] && !(west || north),
                };
                std::uint8_t mask = 0, degree = 0;
                for (std::size_t direction = 0; direction < adjacent.size(); ++direction) {
                    if (adjacent[direction]) {
                        mask |= static_cast<std::uint8_t>(1U << direction);
                        ++degree;
                    }
                }
                neighbors[index] = mask;
                degrees[index] = degree;
                if (degree == 1) endpoints.push_back(index);
                if (degree > 2) {
                    has_junction = true;
                    max_radius = std::max(max_radius, static_cast<double>(distance_map[index]));
                }
            }
        }
        if (!has_junction) break;
        const double max_length = 2.0 * max_radius;
        if (!std::isfinite(max_length)) throw py::value_error("junction diameter exceeds floating-point range");
        for (const auto endpoint : endpoints) {
            checkpoint.check();
            auto current = endpoint;
            auto previous = pixels;  // Sentinel outside the valid index range.
            path.clear();
            double length = 0;
            while (length <= max_length) {
                if (!(path.size() % 128)) checkpoint.check();
                if (degrees[current] > 2) {
                    if (length <= 2.0 * static_cast<double>(distance_map[current])) {
                        for (const auto point : path) {
                            if (!removed[point]) {
                                removed[point] = 1;
                                removed_pixels.push_back(point);
                            }
                        }
                    }
                    break;
                }
                path.push_back(current);
                std::size_t next = pixels;
                std::size_t next_direction = 0;
                int next_count = 0;
                for (std::size_t direction = 0; direction < offsets.size(); ++direction) {
                    if (!(neighbors[current] & (1U << direction))) continue;
                    const auto point = static_cast<std::size_t>(static_cast<std::ptrdiff_t>(current) + offsets[direction]);
                    if (point == previous) continue;
                    next = point;
                    next_direction = direction;
                    ++next_count;
                }
                if (next_count != 1) break;
                length += (next_direction % 2) ? diagonal_length : 1.0;
                previous = current;
                current = next;
            }
        }
        if (removed_pixels.empty()) break;
        // Freeze all degrees/edges for this pass. Applying each arm's removal
        // immediately would change the algorithm and can destroy short crosses.
        if (static_cast<double>(usable_count - removed_pixels.size()) < std::max(6.0, std::ceil(max_length))) break;
        for (std::size_t index = 0; index < removed_pixels.size(); ++index) {
            if (!(index & kWorkCheckpointMask)) checkpoint.check();
            const auto point = removed_pixels[index];
            usable[point] = false;
            removed[point] = 0;
        }
    }
    checkpoint.check();
}

py::array_t<bool> prune_short_spurs(const py::array& skeleton, const py::array& distance_map,
                                  py::object callback) {
    require_c_array(skeleton, 2, "skeleton");
    require_dtype<bool>(skeleton, "skeleton");
    require_c_array(distance_map, 2, "distance_map");
    if (skeleton.shape(0) != distance_map.shape(0) || skeleton.shape(1) != distance_map.shape(1)) {
        throw py::value_error("skeleton and distance_map shapes must match");
    }
    const bool float32 = has_dtype<float>(distance_map), float64 = has_dtype<double>(distance_map);
    if (!float32 && !float64) throw py::type_error("distance_map must have dtype float32 or float64");
    if (float32) require_dtype<float>(distance_map, "distance_map");
    else require_dtype<double>(distance_map, "distance_map");
    checked_pixels(skeleton);
    const auto height = static_cast<std::size_t>(skeleton.shape(0));
    const auto width = static_cast<std::size_t>(skeleton.shape(1));
    const auto* source = static_cast<const std::uint8_t*>(skeleton.data());
    Checkpoint checkpoint(std::move(callback));
    py::array_t<bool> output(std::vector<py::ssize_t>{skeleton.shape(0), skeleton.shape(1)});
    auto* usable = output.mutable_data();
    // Resolve the dtype/pointer with the GIL held. Neither input can be
    // destroyed while these array arguments remain in the bound call frame.
    const auto* distance_data = distance_map.data();
    {
        py::gil_scoped_release release;
        checkpoint.check();
        if (float32) prune_impl(usable, source, static_cast<const float*>(distance_data), height, width, checkpoint);
        else prune_impl(usable, source, static_cast<const double*>(distance_data), height, width, checkpoint);
    }
    return output;
}

py::dict build_info() {
    py::dict info;
    info["api_version"] = FDM_NATIVE_API_VERSION;
    info["source_sha256"] = FDM_NATIVE_SOURCE_SHA256;
    info["python_abi"] = FDM_NATIVE_PYTHON_ABI;
    info["python_version"] = PY_VERSION;
    info["pybind11_version"] = FDM_STRINGIFY(PYBIND11_VERSION_MAJOR) "." FDM_STRINGIFY(PYBIND11_VERSION_MINOR) "." FDM_STRINGIFY(PYBIND11_VERSION_PATCH);
    info["build_type"] = FDM_NATIVE_BUILD_TYPE;
#ifdef _MSVC_LANG
    info["cpp_standard"] = _MSVC_LANG;
#else
    info["cpp_standard"] = __cplusplus;
#endif
#if defined(__clang__)
    info["compiler"] = "Clang " __clang_version__;
#elif defined(_MSC_VER)
    info["compiler"] = "MSVC " FDM_STRINGIFY(_MSC_VER);
#elif defined(__GNUC__)
    info["compiler"] = "GCC " __VERSION__;
#else
    info["compiler"] = "unknown";
#endif
    return info;
}

}  // namespace

PYBIND11_MODULE(_kernels, module) {
    module.doc() = "Exact C++ kernels for FDM contour comparison and fiber geometry.";
    module.def("boundary_segments", &boundary_segments, py::arg("mask").noconvert(),
               py::arg("checkpoint") = py::none());
    module.def("contour_sections", &contour_sections, py::arg("segments").noconvert(),
               py::arg("heights").noconvert(), py::arg("step"), py::arg("checkpoint") = py::none());
    module.def("prune_short_spurs", &prune_short_spurs, py::arg("skeleton").noconvert(),
               py::arg("distance_map").noconvert(), py::arg("checkpoint") = py::none());
    module.def("build_info", &build_info);
}
