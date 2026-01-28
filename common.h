#pragma once
// ============== COMMON DEFINITIONS ==============
// Shared types, globals, and declarations for all renderers

#include <Windows.h>
#include <CommCtrl.h>
#include <dxgi1_6.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <cstdarg>

// ============== RENDERER TYPES ==============
enum RendererType {
    RENDERER_D3D11,
    RENDERER_D3D12,
    RENDERER_D3D12_DXR10,   // DXR 1.0 - TraceRay with raygen/hit/miss shaders
    RENDERER_D3D12_RT,      // DXR 1.1 - Inline RayQuery in pixel shader
    RENDERER_D3D12_PT,
    RENDERER_D3D12_PT_DLSS,
    RENDERER_OPENGL,
    RENDERER_VULKAN,
    RENDERER_VULKAN_RT,     // Vulkan RT - VK_KHR_ray_tracing_pipeline
    RENDERER_VULKAN_RQ      // Vulkan RayQuery - VK_KHR_ray_query (inline ray tracing)
};

// ============== VULKAN RT FEATURE FLAGS ==============
// For Vulkan RT renderer (VK_KHR_ray_tracing_pipeline)
// Each feature compiles into shader via #ifdef
struct VulkanRTFeatures {
    bool spotlight;         // Spotlight cone lighting
    bool softShadows;       // Soft shadows (4 samples)
    bool ambientOcclusion;  // Ambient occlusion (3 samples)
    bool globalIllum;       // Global illumination (1 bounce)
    bool reflections;       // Mirror reflections
    bool glassRefraction;   // Glass transparency with fresnel

    // Parameters
    int shadowSamples;      // 1, 4, 8
    int aoSamples;          // 1, 3, 5
    float aoRadius;         // 0.1 - 1.0
    float lightRadius;      // For soft shadows, 0.05 - 0.3

    // Initialize with defaults - most features ON
    void SetDefaults() {
        spotlight = true;
        softShadows = true;
        ambientOcclusion = true;
        globalIllum = true;
        reflections = true;
        glassRefraction = true;
        shadowSamples = 4;
        aoSamples = 3;
        aoRadius = 0.3f;
        lightRadius = 0.15f;
    }

    // Comparison for detecting changes
    bool operator==(const VulkanRTFeatures& other) const {
        return spotlight == other.spotlight &&
               softShadows == other.softShadows &&
               ambientOcclusion == other.ambientOcclusion &&
               globalIllum == other.globalIllum &&
               reflections == other.reflections &&
               glassRefraction == other.glassRefraction;
    }
    bool operator!=(const VulkanRTFeatures& other) const { return !(*this == other); }
};

extern VulkanRTFeatures g_vulkanRTFeatures;

// ============== GPU INFO ==============
struct GPUInfo {
    std::wstring name;
    IDXGIAdapter1* adapter;  // DXGI adapter pointer for D3D11/D3D12
    SIZE_T vram;             // Dedicated video memory
};

// ============== SETTINGS ==============
struct Settings {
    int selectedGPU = 0;
    RendererType renderer = RENDERER_D3D11;
};

// ============== WINDOW SIZE ==============
extern const UINT W;
extern const UINT H;

// ============== GLOBALS ==============
extern HWND g_hMainWnd;
extern LARGE_INTEGER g_startTime;
extern LARGE_INTEGER g_perfFreq;
extern std::vector<GPUInfo> g_gpuList;
extern Settings g_settings;
extern HINSTANCE g_hInstance;
extern bool g_tearingSupported;
extern std::wstring gpuName;
extern int fps;

// ============== LOGGING ==============
void InitLog();
void Log(const char* fmt, ...);
void LogHR(const char* operation, HRESULT hr);
void CloseLog();

// ============== GPU ENUMERATION ==============
void EnumerateGPUs();
void FreeGPUList();

// ============== TEXT VERTEX (shared for all renderers) ==============
struct TextVert {
    float x, y;
    float u, v;
    float r, g, b, a;
};

// ============== SIMPLE 8x8 BITMAP FONT (ASCII 32-127) ==============
extern const unsigned char g_font8x8[96][8];
