// A real Windows DLL with the vendor's ABI, used only for software validation.
// Never shipped with the application and never installed as a device driver.
#define NOMINMAX
#include <windows.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>
#include <cstdint>
#include <array>
struct ImageInfo { ULONG Length, nColor, Heigth, Width, SkipPixel; };
static_assert(sizeof(ImageInfo) == 20);
using Callback = BOOL (WINAPI *)(PVOID, ImageInfo*, PVOID, ULONG);
struct Device {
    std::mutex lock;
    std::array<Callback, 2> callbacks{};
    std::array<void*, 2> contexts{};
    // Model the documented order, not a substitute for vendor device testing:
    // raw callback -> display processing -> processed callback -> native draw.
    std::array<std::atomic<ULONG>, 4> counts{};
    std::atomic<ULONG> displayed_first_pixel{0};
    std::atomic<bool> ignore_after_return{false};
    std::atomic<ULONG> visible_draws{0}, target_window{0};
    std::atomic<bool> display_presence{true};
    std::atomic<bool> run{false};
    std::thread worker;
    ULONG width = 768, height = 576, skip = 0, fps = 25;
    bool fields = false;
    bool fixed_period = false, reuse_processed = false, no_frames = false;
    const void* reused_input = nullptr;
    std::vector<unsigned char> reused_output;
    std::atomic<double> work_ms{0}, schedule_late_ms{0};
    std::atomic<ULONG> completed_frames{0}, missed_periods{0};
};
#define API extern "C" __declspec(dllexport)
API DWORD WINAPI MV_GetDeviceNumber() { return 1; }
API DWORD WINAPI MV_GetLastError(BOOL) { return 0; }
API HANDLE WINAPI MV_OpenDevice(DWORD, BOOL) {
    auto* d = new Device;
    char value[4]{};
    d->fixed_period = GetEnvironmentVariableA("FDM_TEST_MICROVIEW_FIXED_PERIOD", value, 4) && value[0]=='1';
    d->reuse_processed = GetEnvironmentVariableA("FDM_TEST_MICROVIEW_REUSE_PROCESSED", value, 4) && value[0]=='1';
    d->no_frames = GetEnvironmentVariableA("FDM_TEST_MICROVIEW_NO_FRAMES", value, 4) && value[0]=='1';
    return d;
}
API LONG WINAPI MV_GetDeviceParameter(HANDLE h, int parameter) {
    auto* d = static_cast<Device*>(h);
    if (parameter == 62) return d->height;
    if (parameter == 63) return d->width;
    if (parameter == 7) return d->target_window.load();
    if (parameter == 6) return d->display_presence.load();
    return 0;
}
API BOOL WINAPI MV_SetDeviceParameter(HANDLE h, int parameter, DWORD value) {
    // The unsafe pointer-through-DWORD operation must never be requested.
    auto* d = static_cast<Device*>(h);
    if (parameter == 7) d->target_window.store(value);
    if (parameter == 6) d->display_presence.store(value != 0);
    return parameter != 2;
}
API BOOL WINAPI MV_SetCallBack(HANDLE h, Callback cb, PVOID context, int kind) {
    if (kind != 0 && kind != 1) return FALSE;
    auto* d = static_cast<Device*>(h);
    std::lock_guard<std::mutex> guard(d->lock);
    d->callbacks[kind] = cb; d->contexts[kind] = context;
    return TRUE;
}
API BOOL WINAPI MV_TestEmit(HANDLE h, PVOID pixels, ImageInfo* info, ULONG index) {
    auto* d = static_cast<Device*>(h);
    std::array<Callback, 2> callbacks; std::array<void*, 2> contexts;
    { std::lock_guard<std::mutex> guard(d->lock); callbacks=d->callbacks; contexts=d->contexts; }
    BOOL display = TRUE;
    if (callbacks[0]) {
        d->counts[0].fetch_add(1);
        display = callbacks[0](pixels, info, contexts[0], index);
    }
    // The shipped SAA7130 display thread gates processing and AFTER on
    // DISP_PRESENCE. Keep it enabled even when its dedicated HWND is hidden.
    if (!d->display_presence.load()) return FALSE;
    std::vector<unsigned char> processed;
    PVOID output = pixels;
    if (d->reuse_processed && pixels == d->reused_input && !d->reused_output.empty()) {
        output = d->reused_output.data();
    } else if (pixels && info && info->Length && info->Length <= 128u*1024*1024) {
        auto* input = static_cast<unsigned char*>(pixels);
        processed.assign(input, input+info->Length);
        // Deliberately distinguish processed pixels without inventing a device
        // deinterlacing algorithm. Production must use what the SDK provides.
        for (auto& pixel : processed) pixel ^= 0x40;
        output = processed.data();
    }
    d->counts[1].fetch_add(1);
    auto draw = [&] {
        d->counts[3].fetch_add(1);
        if (output && info && info->Length)
            d->displayed_first_pixel.store(static_cast<unsigned char*>(output)[0]);
        const auto target = d->target_window.load();
        if (!target || IsWindowVisible(reinterpret_cast<HWND>(uintptr_t(target))))
            d->visible_draws.fetch_add(1);
    };
    // SAA7130's shipped binary draws before AFTER and ignores its return.
    // Keep that variant explicit so a permissive mock cannot certify hiding.
    const bool ignores_after = d->ignore_after_return.load();
    if (ignores_after && display) draw();
    if (callbacks[1]) {
        d->counts[2].fetch_add(1);
        BOOL after = callbacks[1](output, info, contexts[1], index);
        if (!ignores_after) display = after && display;
    }
    if (!ignores_after && display) draw();
    return display;
}
API ULONG WINAPI MV_TestGetCounter(HANDLE h, ULONG index) {
    auto* d = static_cast<Device*>(h);
    if (index < 4) return d->counts[index].load();
    if (index == 5) return d->visible_draws.load();
    if (index == 6) return d->target_window.load();
    if (index == 7) return d->display_presence.load();
    return d->displayed_first_pixel.load();
}
API void WINAPI MV_TestIgnoreAfterReturn(HANDLE h, BOOL ignore) {
    static_cast<Device*>(h)->ignore_after_return.store(ignore != FALSE);
}
API void WINAPI MV_TestConfigure(HANDLE h, ULONG width, ULONG height, ULONG skip, ULONG fps, BOOL fields) {
    auto* d = static_cast<Device*>(h);
    d->width=width; d->height=height; d->skip=skip; d->fps=fps; d->fields=fields != FALSE;
}
API double WINAPI MV_TestMetric(HANDLE h, ULONG index) {
    auto* d = static_cast<Device*>(h);
    if (index == 0) return d->work_ms.load();
    if (index == 1) return d->schedule_late_ms.load();
    if (index == 2) return d->completed_frames.load();
    if (index == 3) return d->missed_periods.load();
    if (index == 4) return d->fixed_period;
    if (index == 5) return d->reuse_processed;
    return -1;
}
API int WINAPI MV_OperateDevice(HANDLE h, int operation) {
    auto* d = static_cast<Device*>(h);
    if (!operation) {
        d->run.store(false);
        if (d->worker.joinable()) d->worker.join();
        return 0;
    }
    if (d->run.exchange(true)) return 1;
    d->worker=std::thread([d] {
        if (d->no_frames) {
            while (d->run.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
            return;
        }
        const ULONG height=d->fields ? d->height/2 : d->height;
        std::vector<unsigned char> data(d->width*height*4, 240);
        for (ULONG y=0; y<height; ++y) for (ULONG x=0; x<d->width; ++x) {
            const auto offset=(y*d->width+x)*4;
            data[offset+3]=255;
            if (y>=height/2-6 && y<height/2+6) data[offset]=data[offset+1]=data[offset+2]=35;
        }
        ImageInfo info{ULONG(data.size()),32,height,d->width,d->skip};
        if (d->reuse_processed) {
            d->reused_input = data.data();
            d->reused_output = data;
            for (auto& pixel : d->reused_output) pixel ^= 0x40;
        }
        ULONG sequence=0;
        const auto period = std::chrono::nanoseconds(1000000000 / d->fps);
        auto scheduled = std::chrono::steady_clock::now();
        while (d->run.load()) {
            if (d->fixed_period) std::this_thread::sleep_until(scheduled);
            const auto start = std::chrono::steady_clock::now();
            d->schedule_late_ms.store(d->fixed_period ?
                std::chrono::duration<double, std::milli>(start - scheduled).count() : 0);
            MV_TestEmit(d, data.data(), &info, sequence++);
            const auto end = std::chrono::steady_clock::now();
            d->work_ms.store(std::chrono::duration<double, std::milli>(end - start).count());
            d->completed_frames.fetch_add(1);
            if (d->fixed_period) {
                scheduled += period;
                // Skip missed source ticks instead of manufacturing a burst
                // of old frames that a real latest-frame source would drop.
                if (end >= scheduled) {
                    const auto missed = (end - scheduled) / period + 1;
                    d->missed_periods.fetch_add(static_cast<ULONG>(missed));
                    scheduled += period * missed;
                }
            } else {
                // Legacy probe mode: work time + 40 ms is not fixed 25 FPS.
                std::this_thread::sleep_for(period);
            }
        }
        d->reused_input = nullptr;
    });
    return 1;
}
API void WINAPI MV_CloseDevice(HANDLE h) { if (h) { MV_OperateDevice(h,0); delete static_cast<Device*>(h); } }
API PVOID WINAPI MV_CaptureSingle(HANDLE, BOOL, PVOID, ULONG, ImageInfo*) { return nullptr; }
API BOOL WINAPI MV_TestSignal(HANDLE, DWORD, DWORD) { return TRUE; }
API DWORD WINAPI MV_GetSignalParam(HANDLE, int, float* f, DWORD* i) { *f=0; *i=0; return 0; }
API BOOL WINAPI MV_SaveSignalParamToIni(HANDLE) { return TRUE; }
