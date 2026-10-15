// HWND capture via Windows Graphics Capture. Kept in its own Windows extension;
// it does not link CU, camera SDK, Qt, or the numerical _kernels module.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <roapi.h>
#include <dwmapi.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <thread>
#include <vector>

#ifndef FDM_NATIVE_SOURCE_SHA256
#define FDM_NATIVE_SOURCE_SHA256 "unavailable"
#endif
#ifndef FDM_NATIVE_PYTHON_ABI
#define FDM_NATIVE_PYTHON_ABI "unavailable"
#endif
namespace py = pybind11;
using namespace winrt;
using namespace winrt::Windows::Graphics;
using namespace winrt::Windows::Graphics::Capture;
using namespace winrt::Windows::Graphics::DirectX;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;

namespace {
struct Identity { DWORD pid; std::uint64_t created; };
Identity identity(HWND hwnd) {
    if (!IsWindow(hwnd)) throw std::runtime_error("capture_window_closed");
    DWORD pid=0; GetWindowThreadProcessId(hwnd,&pid);
    HANDLE process=OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,FALSE,pid);
    if (!process) throw std::runtime_error("capture_process_identity_unavailable");
    FILETIME creation{},exit{},kernel{},user{};
    const BOOL ok=GetProcessTimes(process,&creation,&exit,&kernel,&user);
    CloseHandle(process);
    if(!ok) throw std::runtime_error("capture_process_identity_unavailable");
    return {pid,(std::uint64_t(creation.dwHighDateTime)<<32)|creation.dwLowDateTime};
}
struct DpiScope {
    DPI_AWARENESS_CONTEXT previous;
    DpiScope():previous(SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {}
    ~DpiScope(){if(previous)SetThreadDpiAwarenessContext(previous);}
};
struct Geometry { int x,y,width,height,root_width,root_height; UINT dpi; };
Geometry geometry(HWND root, HWND child) {
    DpiScope dpi;
    if (!IsWindow(root)||!IsWindow(child)||GetAncestor(child,GA_ROOT)!=root)
        throw std::runtime_error("capture_window_identity_changed");
    DWORD cloaked=0;
    DwmGetWindowAttribute(root,DWMWA_CLOAKED,&cloaked,sizeof(cloaked));
    if (!IsWindowVisible(root)||!IsWindowVisible(child)||IsIconic(root)||cloaked)
        throw std::runtime_error("capture_window_unavailable");
    RECT outer{},client{};
    if (FAILED(DwmGetWindowAttribute(root,DWMWA_EXTENDED_FRAME_BOUNDS,&outer,sizeof(outer))))
        if(!GetWindowRect(root,&outer))throw std::runtime_error("capture_window_geometry_unavailable");
    if(!GetClientRect(child,&client))throw std::runtime_error("capture_child_geometry_unavailable");
    POINT first{client.left,client.top},last{client.right,client.bottom};
    if(!ClientToScreen(child,&first)||!ClientToScreen(child,&last))
        throw std::runtime_error("capture_child_geometry_unavailable");
    const int x=first.x-outer.left,y=first.y-outer.top,w=last.x-first.x,h=last.y-first.y;
    if(w<=0||h<=0||x<0||y<0||x+w>outer.right-outer.left||y+h>outer.bottom-outer.top)
        throw std::runtime_error("capture_child_is_clipped");
    if(std::uint64_t(w)*h>64ull*1024*1024)throw std::runtime_error("capture_frame_exceeds_budget");
    return {x,y,w,h,outer.right-outer.left,outer.bottom-outer.top,GetDpiForWindow(root)};
}
bool operator==(const Geometry& a,const Geometry& b) {
    return a.x==b.x&&a.y==b.y&&a.width==b.width&&a.height==b.height&&a.root_width==b.root_width&&a.root_height==b.root_height&&a.dpi==b.dpi;
}

class WindowCapture {
public:
    WindowCapture(std::uintptr_t root, std::uintptr_t child):
        root_(reinterpret_cast<HWND>(root)),child_(reinterpret_cast<HWND>(child)),owner_(GetCurrentThreadId()) {
        const HRESULT apartment=RoInitialize(RO_INIT_MULTITHREADED);
        if (FAILED(apartment)&&apartment!=RPC_E_CHANGED_MODE)check_hresult(apartment);
        apartment_owned_=SUCCEEDED(apartment);
        try {
            if(!GraphicsCaptureSession::IsSupported())throw std::runtime_error("windows_graphics_capture_not_supported");
            identity_=identity(root_);
            const auto child_identity=identity(child_);
            if(child_identity.pid!=identity_.pid||child_identity.created!=identity_.created)
                throw std::runtime_error("capture_child_process_mismatch");
            geometry_=geometry(root_,child_);
            UINT flags=D3D11_CREATE_DEVICE_BGRA_SUPPORT;
            HRESULT hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,flags,nullptr,0,D3D11_SDK_VERSION,
                                        device_.put(),nullptr,context_.put());
            if(FAILED(hr))hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,flags,nullptr,0,D3D11_SDK_VERSION,
                                        device_.put(),nullptr,context_.put());
            check_hresult(hr);
            auto dxgi=device_.as<IDXGIDevice>();
            com_ptr<IInspectable> inspectable;
            check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(),inspectable.put()));
            runtime_device_=inspectable.as<IDirect3DDevice>();
            auto factory=get_activation_factory<GraphicsCaptureItem,IGraphicsCaptureItemInterop>();
            check_hresult(factory->CreateForWindow(root_,guid_of<GraphicsCaptureItem>(),put_abi(item_)));
            size_=item_.Size();
            pool_=Direct3D11CaptureFramePool::CreateFreeThreaded(runtime_device_,DirectXPixelFormat::B8G8R8A8UIntNormalized,1,size_);
            session_=pool_.CreateCaptureSession(item_);
            // Cursor control was added after 1903. Older supported systems keep
            // the system default; no optional API is needed to start capture.
            try { session_.IsCursorCaptureEnabled(false); } catch(const hresult_error&) {}
            session_.StartCapture();
        } catch(...) { close(); throw; }
    }
    ~WindowCapture(){close();}
    void close() noexcept {
        if(closed_)return;
        closed_=true;
        try {if(session_)session_.Close();}catch(...){}
        try {if(pool_)pool_.Close();}catch(...){}
        session_=nullptr;pool_=nullptr;item_=nullptr;runtime_device_=nullptr;
        staging_=nullptr;context_=nullptr;device_=nullptr;
        if(apartment_owned_&&GetCurrentThreadId()==owner_){RoUninitialize();apartment_owned_=false;}
    }
    py::dict get_identity() const {
        py::dict result;
        result["pid"]=identity_.pid;result["process_created"]=identity_.created;
        result["root_hwnd"]=reinterpret_cast<std::uintptr_t>(root_);
        result["child_hwnd"]=reinterpret_cast<std::uintptr_t>(child_);
        return result;
    }
    py::object poll(int timeout_ms) {
        if(GetCurrentThreadId()!=owner_)throw std::runtime_error("capture_session_thread_mismatch");
        if(closed_)throw std::runtime_error("capture_session_closed");
        timeout_ms=std::clamp(timeout_ms,0,1000);
        py::array_t<std::uint8_t> pixels;
        std::uint8_t* destination=nullptr;
        int array_width=0,array_height=0;
        bool has_pixels=false;
        Geometry captured=geometry_;
        std::int64_t timestamp=0;
        std::uint64_t gpu_pool_bytes=0,cpu_staging_bytes=0,cpu_staging_row_pitch=0;
        {
            py::gil_scoped_release release;
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
            do {
                const auto current_identity=identity(root_);
                if(current_identity.pid!=identity_.pid||current_identity.created!=identity_.created)
                    throw std::runtime_error("capture_process_identity_changed");
                auto current=geometry(root_,child_);
                auto frame=pool_.TryGetNextFrame();
                if(frame) {
                    const auto content=frame.ContentSize();
                    if(content.Width!=size_.Width||content.Height!=size_.Height) {
                        frame.Close();size_=content;
                        pool_.Recreate(runtime_device_,DirectXPixelFormat::B8G8R8A8UIntNormalized,1,size_);
                        continue;
                    }
                    timestamp=frame.SystemRelativeTime().count();
                    if(timestamp<=last_timestamp_){frame.Close();continue;}
                    // Do not stretch stale frame pixels to a newly resized HWND.
                    if(content.Width!=current.root_width||content.Height!=current.root_height) {
                        frame.Close();continue;
                    }
                    auto access=frame.Surface().as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
                    com_ptr<ID3D11Texture2D> texture;
                    check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D),texture.put_void()));
                    D3D11_TEXTURE2D_DESC description{};texture->GetDesc(&description);
                    if(description.Format!=DXGI_FORMAT_B8G8R8A8_UNORM)
                        throw std::runtime_error("capture_pixel_format_changed");
                    gpu_pool_bytes=std::uint64_t(description.Width)*description.Height*4;
                    if(!staging_||staging_width_!=current.width||staging_height_!=current.height) {
                        D3D11_TEXTURE2D_DESC target{};
                        target.Width=current.width;target.Height=current.height;target.MipLevels=target.ArraySize=1;
                        target.Format=DXGI_FORMAT_B8G8R8A8_UNORM;target.SampleDesc.Count=1;
                        target.Usage=D3D11_USAGE_STAGING;target.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
                        staging_=nullptr;check_hresult(device_->CreateTexture2D(&target,nullptr,staging_.put()));
                        staging_width_=current.width;staging_height_=current.height;
                    }
                    D3D11_BOX box{UINT(current.x),UINT(current.y),0,UINT(current.x+current.width),UINT(current.y+current.height),1};
                    context_->CopySubresourceRegion(staging_.get(),0,0,0,0,texture.get(),0,&box);
                    D3D11_MAPPED_SUBRESOURCE mapped{};
                    check_hresult(context_->Map(staging_.get(),0,D3D11_MAP_READ,0,&mapped));
                    cpu_staging_row_pitch=mapped.RowPitch;
                    cpu_staging_bytes=std::uint64_t(mapped.RowPitch)*current.height;
                    try {
                        if(array_width!=current.width||array_height!=current.height) {
                            py::gil_scoped_acquire acquire;
                            // Drop a stale-size allocation before allocating a
                            // replacement; never keep a third full-frame buffer.
                            pixels=py::array_t<std::uint8_t>();
                            pixels=py::array_t<std::uint8_t>(std::vector<py::ssize_t>{current.height,current.width,3});
                            destination=pixels.mutable_data();
                            array_width=current.width;array_height=current.height;
                        }
                        // Preserve every B/G/R byte at the original resolution;
                        // WGC's unused alpha never becomes a fourth CPU channel.
                        // No intermediate BGRA NumPy/vector allocation exists.
                        for(int y=0;y<current.height;++y) {
                            const auto* row=static_cast<const std::uint8_t*>(mapped.pData)+std::size_t(y)*mapped.RowPitch;
                            auto* output=destination+std::size_t(y)*current.width*3;
                            for(int x=0;x<current.width;++x) {
                                output[3*x]=row[4*x];output[3*x+1]=row[4*x+1];output[3*x+2]=row[4*x+2];
                            }
                        }
                    }catch(...){context_->Unmap(staging_.get(),0);throw;}
                    context_->Unmap(staging_.get(),0);frame.Close();
                    if(!(current==geometry(root_,child_)))continue;
                    if(!(current==geometry_)){++generation_;geometry_=current;}
                    captured=current;last_timestamp_=timestamp;++sequence_;
                    has_pixels=true;
                    break;
                }
                if(timeout_ms==0)break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }while(std::chrono::steady_clock::now()<deadline);
        }
        if(!has_pixels)return py::none();
        py::dict result;
        result["pixels"]=pixels;result["sequence"]=sequence_;result["generation"]=generation_;
        result["source_timestamp_100ns"]=timestamp;result["dpi"]=captured.dpi;
        result["crop"]=py::make_tuple(captured.x,captured.y,captured.width,captured.height);
        result["backend"]="windows_graphics_capture";
        const auto next_delivery_bytes=std::uint64_t(captured.width)*captured.height*3;
        result["gpu_pool_buffers"]=1;result["gpu_pool_bytes"]=gpu_pool_bytes;
        result["cpu_staging_row_pitch"]=cpu_staging_row_pitch;
        result["cpu_staging_bytes"]=cpu_staging_bytes;
        result["next_delivery_bytes"]=next_delivery_bytes;
        // This excludes the returned frame once handed to the parent. The
        // parent accounts its current/presented/mailbox/draft owners separately.
        // GPU bytes describe frame payloads, not opaque driver heap overhead.
        result["acquisition_reserved_bytes"]=gpu_pool_bytes+cpu_staging_bytes+next_delivery_bytes;
        return result;
    }
private:
    HWND root_,child_;DWORD owner_;bool apartment_owned_=false,closed_=false;
    Identity identity_{};Geometry geometry_{};
    std::uint64_t sequence_=0,generation_=0;std::int64_t last_timestamp_=0;
    int staging_width_=0,staging_height_=0;
    com_ptr<ID3D11Device> device_;com_ptr<ID3D11DeviceContext> context_;com_ptr<ID3D11Texture2D> staging_;
    IDirect3DDevice runtime_device_{nullptr};GraphicsCaptureItem item_{nullptr};SizeInt32 size_{};
    Direct3D11CaptureFramePool pool_{nullptr};GraphicsCaptureSession session_{nullptr};
};
}
PYBIND11_MODULE(_windows_capture,module) {
    py::register_exception_translator([](std::exception_ptr failure) {
        try { if(failure)std::rethrow_exception(failure); }
        catch(const winrt::hresult_error& error) {
            const auto message=winrt::to_string(error.message());
            PyErr_SetString(PyExc_RuntimeError,message.c_str());
        }
    });
    module.def("build_info",[]{py::dict r;r["api_version"]=1;r["backend"]="WGC/D3D11";
        r["minimum_windows_build"]=18362;r["source_sha256"]=FDM_NATIVE_SOURCE_SHA256;
        r["python_abi"]=FDM_NATIVE_PYTHON_ABI;return r;});
    py::class_<WindowCapture>(module,"WindowCapture")
        .def(py::init([](std::uintptr_t root,std::uintptr_t child){
            py::gil_scoped_release release;
            return std::make_unique<WindowCapture>(root,child);
        }),py::arg("root_hwnd"),py::arg("child_hwnd"))
        .def("poll",&WindowCapture::poll,py::arg("timeout_ms")=100)
        .def("identity",&WindowCapture::get_identity)
        .def("close",&WindowCapture::close,py::call_guard<py::gil_scoped_release>());
}
