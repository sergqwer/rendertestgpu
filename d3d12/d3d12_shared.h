#pragma once
// ============== D3D12 SHARED DECLARATIONS ==============
// All D3D12 globals are defined in d3d12_globals.cpp

#include <d3d12.h>
#include <dxgi1_6.h>
#include "../common.h"

// ============== CONSTANTS ==============
#define FRAME_COUNT 3
#define MAX_TEXT_VERTS 6000

// ============== D3D12 BASE GLOBALS ==============
extern bool g_tearingSupported12;

extern ID3D12Device* dev12;
extern ID3D12CommandQueue* cmdQueue;
extern ID3D12CommandAllocator* cmdAlloc[3];
extern ID3D12GraphicsCommandList* cmdList;
extern IDXGISwapChain3* swap12;
extern ID3D12DescriptorHeap* rtvHeap12;
extern ID3D12DescriptorHeap* dsvHeap12;
extern ID3D12DescriptorHeap* srvHeap12;
extern ID3D12Resource* renderTargets12[3];
extern ID3D12Resource* depthStencil12;
extern ID3D12RootSignature* rootSig;
extern ID3D12PipelineState* pso;
extern ID3D12PipelineState* textPso;
extern ID3D12Resource* vb12;
extern ID3D12Resource* ib12;
extern ID3D12Resource* cbUpload12;
extern ID3D12Resource* fontTex12;
extern ID3D12Resource* textVB12;
extern ID3D12RootSignature* textRootSig12;

// Synchronization
extern ID3D12Fence* fence;
extern UINT64 fenceValues[3];
extern HANDLE fenceEvent;
extern UINT frameIndex;
extern UINT rtvDescSize;

// Buffer views
extern D3D12_VERTEX_BUFFER_VIEW vbView12;
extern D3D12_INDEX_BUFFER_VIEW ibView12;
extern D3D12_VERTEX_BUFFER_VIEW textVbView12;

// Persistent mapped pointers
extern void* cbMapped12;
extern void* textVbMapped12;

// Geometry counts
extern UINT totalIndices12;
extern UINT totalVertices12;

// ============== DXR FEATURE FLAGS ==============
// Each feature can be toggled independently
// When disabled, falls back to rasterization equivalent
struct DXRFeatures {
    bool useRayQuery;       // Use SM 6.5 RayQuery (requires real DXR GPU). If false, use SM 6.0 compatible mode
    bool rtLighting;        // Use RT lighting model (spotlight + GI) vs simple
    bool rtShadows;         // Ray-traced hard shadows (requires useRayQuery)
    bool rtSoftShadows;     // Soft shadows (multiple samples) - requires rtShadows
    bool rtReflections;     // Ray-traced reflections on cubes (requires useRayQuery)
    bool rtAO;              // Ray-traced ambient occlusion (requires useRayQuery)
    bool rtGI;              // Global illumination (color bleeding) (requires useRayQuery)

    // Soft shadow parameters
    int softShadowSamples;  // 4, 8, 16, 32
    float shadowSoftness;   // 0.0-1.0 (light source radius)

    // Reflection parameters
    float reflectionStrength; // 0.0-1.0
    float roughness;          // 0.0-1.0 (blur reflections)

    // AO parameters
    int aoSamples;          // 4, 8, 16
    float aoRadius;         // World space radius
    float aoStrength;       // 0.0-1.0

    // GI parameters
    int giBounces;          // 1, 2, 3
    float giStrength;       // 0.0-1.0

    // Debug visualization mode (press 0-6 to change)
    // 0 = Normal rendering
    // 1 = Object IDs (color-coded)
    // 2 = Normals
    // 3 = Reflection directions (for mirror)
    // 4 = Shadow rays (show light visibility)
    // 5 = UV coordinates
    // 6 = Depth
    int debugMode;

    // Temporal denoising settings
    bool enableTemporalDenoise;  // Temporal denoising for shadows/AO/GI
    float denoiseBlendFactor;    // 0.0-1.0, higher = more temporal smoothing (0.9 = 90% history)

    // Initialize with defaults - most features ON by default
    void SetDefaults() {
        useRayQuery = true;    // SM 6.5 RayQuery ON (set false for compatibility mode)
        rtLighting = true;     // Spotlight lighting ON
        rtShadows = true;      // Shadows ON
        rtSoftShadows = true;  // Soft shadows ON
        rtReflections = true;  // Reflections ON
        rtAO = true;           // Ambient Occlusion ON
        rtGI = true;           // Global Illumination ON
        softShadowSamples = 8;
        shadowSoftness = 0.2f;
        reflectionStrength = 0.85f;  // Strong reflections
        roughness = 0.2f;
        aoSamples = 8;
        aoRadius = 0.5f;
        aoStrength = 0.5f;
        giBounces = 1;
        giStrength = 0.3f;
        debugMode = 0;  // Normal rendering by default
        // Temporal denoising defaults
        // OFF by default - can cause crashes on some virtual/emulated GPU drivers
        enableTemporalDenoise = false;
        denoiseBlendFactor = 0.9f;     // 90% history, 10% new frame
    }
};

extern DXRFeatures g_dxrFeatures;

// ============== DXR 1.0 FEATURE FLAGS ==============
// For DXR 1.0 renderer (TraceRay with raygen/hit/miss shaders)
// Each feature compiles into shader via #ifdef
struct DXR10Features {
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
    bool operator==(const DXR10Features& other) const {
        return spotlight == other.spotlight &&
               softShadows == other.softShadows &&
               ambientOcclusion == other.ambientOcclusion &&
               globalIllum == other.globalIllum &&
               reflections == other.reflections &&
               glassRefraction == other.glassRefraction;
    }
    bool operator!=(const DXR10Features& other) const { return !(*this == other); }
};

extern DXR10Features g_dxr10Features;

// ============== DXR RAY TRACING GLOBALS ==============
extern ID3D12Device5* dev12RT;
extern ID3D12GraphicsCommandList4* cmdListRT;
extern ID3D12Resource* blasBuffer;
extern ID3D12Resource* tlasBuffer;
extern ID3D12Resource* instanceBuffer;
extern ID3D12Resource* scratchBuffer;
extern ID3D12DescriptorHeap* srvHeapRT;
extern ID3D12RootSignature* rootSigRT;
extern ID3D12PipelineState* psoRT;
extern bool g_dxrSupported;

// ============== PATH TRACING GLOBALS ==============
extern ID3D12Resource* pathTraceOutput;
extern ID3D12Resource* denoiseTemp;
extern ID3D12RootSignature* pathTraceRootSig;
extern ID3D12PipelineState* pathTracePSO;
extern ID3D12RootSignature* denoiseRootSig;
extern ID3D12PipelineState* denoisePSO;
extern ID3D12DescriptorHeap* pathTraceSrvUavHeap;
extern ID3D12Resource* pathTraceCB;
extern ID3D12Resource* denoiseCB;
extern void* pathTraceCBMapped;
extern void* denoiseCBMapped;
extern UINT g_frameCount;

// Denoise modes
enum DenoiseMode { DENOISE_OFF, DENOISE_ATROUS, DENOISE_TEMPORAL };
extern DenoiseMode g_denoiseMode;
extern UINT g_temporalFrameCount;

// ============== DLSS GLOBALS ==============
struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;
extern NVSDK_NGX_Handle* g_dlssRRHandle;
extern NVSDK_NGX_Parameter* g_ngxParams;
extern bool g_ngxInitialized;
extern bool g_dlssRRSupported;

extern ID3D12Resource* g_gbufferAlbedo;
extern ID3D12Resource* g_gbufferNormal;
extern ID3D12Resource* g_gbufferMotion;
extern ID3D12Resource* g_gbufferDepth;
extern ID3D12Resource* g_dlssOutput;
extern ID3D12DescriptorHeap* g_gbufferHeap;

// ============== TEXT RENDERING ==============
// MAX_TEXT_VERTS is defined as 6000 above
extern TextVert g_textVerts[MAX_TEXT_VERTS];
extern UINT g_textVertCount;
extern int g_cachedFps;
extern bool g_textNeedsRebuild;

// ============== DXC SHADER COMPILER ==============
typedef HRESULT(WINAPI* DxcCreateInstanceProc)(REFCLSID, REFIID, LPVOID*);
extern HMODULE g_dxcModule;
extern DxcCreateInstanceProc g_DxcCreateInstance;

// ============== SHARED HELPER FUNCTIONS ==============
void WaitForGpu();
void MoveToNextFrame();
void DrawText12(const char* text, float x, float y, float r, float g, float b, float a, float scale);
void DrawTextDirect(const char* text, float x, float y, float r, float g, float b, float a, float scale);
bool LoadDXC();
bool InitGPUText12();  // Text rendering init - shared by base, PT, and DLSS renderers

// DXR support check (defined in renderer_d3d12_rt.cpp)
bool CheckDXRSupport(struct IDXGIAdapter1* adapter);

// Path Tracing TLAS update functions (defined in renderer_d3d12_pt.cpp)
// Used by DLSS renderer to update cube transform and rebuild TLAS
void UpdateCubeTransformPT(float time);
void RebuildTLAS_PT(ID3D12GraphicsCommandList4* cmdListRT);
