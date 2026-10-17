// Microview VER5 SDK ABI. Kept separate from the portable measurement kernels.
// SDK callbacks never enter Python, acquire the GIL, allocate, or wait for GUI.
#define NOMINMAX
#include <windows.h>
#include <pybind11/pybind11.h>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <thread>
#include <string>
#include <vector>

namespace py = pybind11;
struct ImageInfo { uint32_t length, color, height, width, skip; };
static_assert(sizeof(ImageInfo) == 20, "Microview MV_IMAGEINFO ABI");
using Callback = BOOL (WINAPI *)(void*, ImageInfo*, void*, ULONG);
using SetCallback = BOOL (WINAPI *)(HANDLE, Callback, void*, int);

static volatile LONG64* marker_address(const py::buffer_info& buffer) {
    if (buffer.itemsize != 1 || buffer.ndim != 1 || buffer.strides[0] != 1 || buffer.size != 8 ||
        reinterpret_cast<uintptr_t>(buffer.ptr) % alignof(LONG64))
        throw std::invalid_argument("Microview latest marker must be an aligned writable 8-byte buffer");
    return static_cast<volatile LONG64*>(buffer.ptr);
}

static uint64_t read_latest_marker(const py::buffer& marker) {
    const auto buffer = marker.request(true);
    return static_cast<uint64_t>(InterlockedCompareExchange64(marker_address(buffer), 0, 0));
}

static double timestamp() noexcept {
    LARGE_INTEGER value, frequency;
    QueryPerformanceCounter(&value);
    QueryPerformanceFrequency(&frequency);
    return double(value.QuadPart) / double(frequency.QuadPart);
}

class FrameEvent {
    HANDLE handle_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
public:
    FrameEvent() {
        if (!handle_) throw std::runtime_error("Microview frame event creation failed");
    }
    ~FrameEvent() { CloseHandle(handle_); }
    void signal() noexcept { SetEvent(handle_); }
    bool wait(unsigned timeout_ms) {
        const DWORD status = WaitForSingleObject(handle_, timeout_ms);
        if (status == WAIT_FAILED) throw std::runtime_error("Microview frame event wait failed");
        return status == WAIT_OBJECT_0;
    }
};

class FrameTap {
    struct Slot {
        std::vector<uint8_t> bytes;
        ImageInfo info{};
        uint64_t sequence = 0;
        double received = 0;
        ULONG index = 0;
    };
    std::array<Slot, 2> slots_;
    std::mutex mutex_;
    FrameEvent wake_event_;
    SetCallback set_callback_;
    HANDLE device_;
    std::atomic<size_t> capacity_;
    std::atomic<size_t> required_capacity_{0};
    std::atomic<bool> accepting_{false};
    // Requested SDK drawing state. Some shipped board DLLs ignore AFTER's
    // return value: the GUI must also hide the live, dedicated native host
    // whenever the trial canvas owns presentation. Never disable processing.
    std::atomic<bool> native_preview_visible_{true};
    std::atomic<uint32_t> in_flight_{0};
    std::atomic<uint64_t> callbacks_{0}, dropped_{0}, invalid_{0}, fields_{0};
    std::atomic<uint64_t> latest_sequence_{0};
    // The helper owns this shared marker. The callback only touches its native
    // pointer; replacing or releasing the Python owner requires the slot lock.
    py::object latest_marker_owner_;
    std::unique_ptr<py::buffer_info> latest_marker_buffer_;
    volatile LONG64* latest_marker_ = nullptr;
    uint64_t delivered_ = 0;
    uint32_t expected_height_;
    bool registered_ = false;

    static BOOL WINAPI callback(void* data, ImageInfo* info, void* context, ULONG index) noexcept {
        auto* self = static_cast<FrameTap*>(context);
        if (!self) return FALSE;
        self->in_flight_.fetch_add(1);
        struct Exit { FrameTap* self; ~Exit() { self->in_flight_.fetch_sub(1); } } guard{self};
        if (!self->accepting_.load()) return FALSE;
        const auto display = [self]() noexcept {
            return self->native_preview_visible_.load() ? TRUE : FALSE;
        };
        const uint64_t sequence = self->callbacks_.fetch_add(1) + 1;
        uint32_t bpp = 0;
        if (info) {
            switch (info->color) {
            case 0: case 8: bpp = 1; break;
            case 1: case 5: case 6: case 15: case 16: bpp = 2; break;
            case 2: case 24: bpp = 3; break;
            case 3: case 32: bpp = 4; break;
            }
        }
        if (!data || !info || !bpp || !info->width || !info->height ||
            info->skip >= info->width || info->length > 128u * 1024 * 1024 ||
            uint64_t(info->width) * info->height * bpp > info->length) {
            self->invalid_.fetch_add(1);
            return display();
        }
        if (info->length > self->capacity_.load()) {
            self->required_capacity_.store(info->length);
            self->dropped_.fetch_add(1);
            self->wake_event_.signal();
            return display();
        }
        // A half-height field is not a complete frame. Never silently stretch it.
        if (self->expected_height_ && info->height * 2 == self->expected_height_) {
            self->fields_.fetch_add(1);
            return display();
        }
        std::unique_lock<std::mutex> lock(self->mutex_, std::try_to_lock);
        if (!lock.owns_lock()) { self->dropped_.fetch_add(1); return display(); }
        auto& slot = self->slots_[self->slots_[0].sequence <= self->slots_[1].sequence ? 0 : 1];
        std::memcpy(slot.bytes.data(), data, info->length);
        slot.info = *info;
        slot.sequence = sequence;
        slot.index = index;
        slot.received = timestamp();
        self->latest_sequence_.store(sequence);
        if (self->latest_marker_)
            InterlockedExchange64(self->latest_marker_, static_cast<LONG64>(sequence));
        self->wake_event_.signal();
        // AFTER_PROCESS supplies SDK display-processed pixels. The manual says
        // before drawing, but the shipped SAA7130 draws first and ignores BOOL.
        // Do not disable native preview just because the measurement tap drops
        // a frame; its display must remain independent of Python/Qt consumers.
        return display();
    }
public:
    // The helper waits for a frame or a control command, not a short polling
    // timer. Auto-reset retains a wake that arrives immediately before wait.
    void wake() noexcept { wake_event_.signal(); }
    bool wait(unsigned timeout_ms) { return wake_event_.wait(timeout_ms); }
    void bind_latest_marker(py::object marker) {
        auto buffer = marker.is_none() ? nullptr :
            std::make_unique<py::buffer_info>(py::buffer(marker).request(true));
        volatile LONG64* pointer = buffer ? marker_address(*buffer) : nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        latest_marker_ = pointer;
        // Retain the actual buffer export as well as its Python owner. An owner
        // alone does not prevent bytearray resize or memoryview.release().
        latest_marker_buffer_ = std::move(buffer);
        latest_marker_owner_ = std::move(marker);
        if (latest_marker_)
            InterlockedExchange64(latest_marker_, static_cast<LONG64>(latest_sequence_.load()));
    }
    void reserve(size_t capacity) {
        if (capacity > 128u * 1024 * 1024 || capacity < capacity_.load())
            throw std::invalid_argument("invalid Microview resize capacity");
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& slot : slots_) slot.bytes.resize(capacity);
        capacity_.store(capacity);
        required_capacity_.store(0);
    }
    struct Probe {};
    explicit FrameTap(Probe) : set_callback_(nullptr), device_(nullptr), capacity_(64), expected_height_(2) {
        for (auto& slot : slots_) slot.bytes.resize(capacity_);
        accepting_.store(true);
    }
    static py::dict self_check() {
        FrameTap tap(Probe{});
        std::array<uint8_t, 18> pixels{90,90,90,1,2,3,4,5,6,91,91,91,7,8,9,10,11,12};
        ImageInfo info{18,24,2,3,1};
        bool native_display = callback(pixels.data(), &info, &tap, 0) == TRUE;
        py::bytearray output(std::string(64, '\0'));
        auto result = tap.poll_into(output);
        auto view = py::buffer(output).request();
        bool exact = std::memcmp(view.ptr, pixels.data(), pixels.size()) == 0;
        bool once = tap.poll_into(output).is_none();
        tap.set_native_preview_visible(false);
        bool qt_display = callback(pixels.data(), &info, &tap, 0) == FALSE;
        bool processed_pixels_continue = !tap.poll_into(output).is_none();
        py::bytearray marker(std::string(8, '\0'));
        tap.bind_latest_marker(marker);
        const auto before = read_latest_marker(py::buffer(marker));
        callback(pixels.data(), &info, &tap, 0);
        bool marker_updated = before == 2 && read_latest_marker(py::buffer(marker)) == 3;
        py::bytearray replacement(std::string(8, '\0'));
        tap.bind_latest_marker(replacement);
        callback(pixels.data(), &info, &tap, 0);
        bool marker_rebound = read_latest_marker(py::buffer(marker)) == 3 &&
            read_latest_marker(py::buffer(replacement)) == 4;
        tap.bind_latest_marker(py::none());
        info.height = 1;
        callback(pixels.data(), &info, &tap, 1);
        py::dict checks;
        checks["callback_copy"] = !result.is_none() && exact;
        checks["no_replay"] = once;
        checks["field_rejected"] = tap.fields_.load() == 1;
        checks["metadata_size"] = sizeof(ImageInfo) == 20;
        checks["display_owner_switch"] = native_display && qt_display && processed_pixels_continue;
        checks["latest_sequence_marker"] = marker_updated && marker_rebound;
        bool callback_wake = tap.wait(0) && !tap.wait(0);
        tap.wake();
        checks["frame_control_wakeup"] = callback_wake && tap.wait(0) && !tap.wait(0);
        return checks;
    }
    FrameTap(uintptr_t module, uintptr_t device, size_t capacity, uint32_t expected_height,
             bool native_preview_visible = true)
        : device_(reinterpret_cast<HANDLE>(device)), capacity_(capacity),
          native_preview_visible_(native_preview_visible), expected_height_(expected_height) {
        if (!module || !device || capacity < 1 || capacity > 128 * 1024 * 1024)
            throw std::invalid_argument("invalid Microview callback capacity/handle");
        set_callback_ = reinterpret_cast<SetCallback>(GetProcAddress(reinterpret_cast<HMODULE>(module), "MV_SetCallBack"));
        if (!set_callback_) throw std::runtime_error("MV_SetCallBack export is missing");
        for (auto& slot : slots_) slot.bytes.resize(capacity);
    }
    void start() {
        if (registered_) return;
        accepting_.store(true);
        if (!set_callback_(device_, callback, this, 1)) {
            accepting_.store(false);
            throw std::runtime_error("MV_SetCallBack(AFTER_PROCESS) failed");
        }
        registered_ = true;
    }
    void set_native_preview_visible(bool visible) noexcept {
        native_preview_visible_.store(visible);
    }
    void stop() {
        accepting_.store(false);
        wake_event_.signal();
        if (registered_) {
            if (!set_callback_(device_, nullptr, nullptr, 1))
                throw std::runtime_error("MV_SetCallBack unregister failed; retain callback until device stops");
            registered_ = false;
        }
    }
    void wait_idle(unsigned timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (in_flight_.load()) {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Microview callback did not retire");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    py::object poll_into(py::buffer output) {
        auto buffer = output.request(true);
        if (buffer.itemsize != 1 || buffer.ndim != 1 || buffer.strides[0] != 1 || size_t(buffer.size) < capacity_)
            throw std::invalid_argument("Microview destination must be a contiguous writable byte buffer");
        ImageInfo info{}; uint64_t sequence = 0; ULONG index = 0; double received = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            auto& slot = slots_[slots_[0].sequence >= slots_[1].sequence ? 0 : 1];
            if (slot.sequence > delivered_) {
                info = slot.info; sequence = slot.sequence; received = slot.received; index = slot.index;
                std::memcpy(buffer.ptr, slot.bytes.data(), info.length);
                delivered_ = sequence;
            }
        }
        if (!sequence) return py::none();
        py::dict result;
        result["length"] = info.length; result["width"] = info.width; result["height"] = info.height;
        result["color"] = info.color; result["skip"] = info.skip; result["sequence"] = sequence;
        result["index"] = index; result["received_at"] = received;
        result["callback_stage"] = "after_process";
        return result;
    }
    py::dict diagnostics() const {
        py::dict result;
        result["callbacks"] = callbacks_.load(); result["dropped_busy"] = dropped_.load();
        result["invalid"] = invalid_.load(); result["fields_rejected"] = fields_.load();
        result["required_capacity"] = required_capacity_.load();
        result["native_slot_bytes"] = capacity_.load() * 2; result["in_flight"] = in_flight_.load();
        result["context_address"] = reinterpret_cast<uintptr_t>(this);
        result["callback_stage"] = "after_process";
        result["native_preview_visible"] = native_preview_visible_.load();
        result["latest_valid_sequence"] = latest_sequence_.load();
        return result;
    }
    // Ownership is explicit: backend unregisters, stops/closes the SDK and
    // waits for in-flight callbacks before releasing this object or the DLL.
    // The helper is terminated if that ordering cannot be confirmed.
};

PYBIND11_MODULE(_microview, module) {
    module.def("self_check", &FrameTap::self_check);
    py::class_<FrameTap>(module, "FrameTap")
        .def(py::init<uintptr_t, uintptr_t, size_t, uint32_t, bool>(),
             py::arg("module"), py::arg("device"), py::arg("capacity"),
             py::arg("expected_height"), py::arg("native_preview_visible") = true)
        .def("reserve", &FrameTap::reserve, py::call_guard<py::gil_scoped_release>())
        .def("start", &FrameTap::start, py::call_guard<py::gil_scoped_release>())
        .def("set_native_preview_visible", &FrameTap::set_native_preview_visible)
        .def("bind_latest_marker", &FrameTap::bind_latest_marker)
        .def("wake", &FrameTap::wake)
        .def("wait", &FrameTap::wait, py::arg("timeout_ms"), py::call_guard<py::gil_scoped_release>())
        .def("stop", &FrameTap::stop, py::call_guard<py::gil_scoped_release>())
        .def("wait_idle", &FrameTap::wait_idle, py::arg("timeout_ms") = 2000, py::call_guard<py::gil_scoped_release>())
        .def("poll_into", &FrameTap::poll_into)
        .def("diagnostics", &FrameTap::diagnostics);
    module.def("read_latest_marker", &read_latest_marker);
    module.def("build_info", [] {
        py::dict result;
        result["api_version"] = 3; result["image_info_size"] = sizeof(ImageInfo);
        result["pointer_size"] = sizeof(void*); result["source_sha256"] = FDM_NATIVE_SOURCE_SHA256;
        result["python_abi"] = FDM_NATIVE_PYTHON_ABI;
        return result;
    });
}
