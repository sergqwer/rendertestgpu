// ============== D3D12 SHARED GLOBALS ==============
// This file defines all D3D12 globals shared between renderers
// (Base, RT, PT, DLSS)

#include <d3d12.h>
#include <dxgi1_6.h>
#include "../common.h"
#include "d3d12_shared.h"

// DLSS SDK - set to 0 if SDK not available
#define ENABLE_DLSS 0

#if ENABLE_DLSS
// Include DLSS headers for DLSS globals
#define USE_DIRECTX
#include "nvsdk_ngx.h"
#endif

// FRAME_COUNT and MAX_TEXT_VERTS are #defined in d3d12_shared.h

// ============== D3D12 BASE GLOBALS ==============
bool g_tearingSupported12 = false;

ID3D12Device* dev12 = nullptr;
ID3D12CommandQueue* cmdQueue = nullptr;
ID3D12CommandAllocator* cmdAlloc[3] = {};
ID3D12GraphicsCommandList* cmdList = nullptr;
IDXGISwapChain3* swap12 = nullptr;
ID3D12DescriptorHeap* rtvHeap12 = nullptr;
ID3D12DescriptorHeap* dsvHeap12 = nullptr;
ID3D12DescriptorHeap* srvHeap12 = nullptr;
ID3D12Resource* renderTargets12[3] = {};
ID3D12Resource* depthStencil12 = nullptr;
ID3D12RootSignature* rootSig = nullptr;
ID3D12PipelineState* pso = nullptr;
ID3D12PipelineState* textPso = nullptr;
ID3D12Resource* vb12 = nullptr;
ID3D12Resource* ib12 = nullptr;
ID3D12Resource* cbUpload12 = nullptr;
ID3D12Resource* fontTex12 = nullptr;
ID3D12Resource* textVB12 = nullptr;
ID3D12RootSignature* textRootSig12 = nullptr;

// Synchronization
ID3D12Fence* fence = nullptr;
UINT64 fenceValues[3] = {};
HANDLE fenceEvent = nullptr;
UINT frameIndex = 0;
UINT rtvDescSize = 0;

// Buffer views
D3D12_VERTEX_BUFFER_VIEW vbView12 = {};
D3D12_INDEX_BUFFER_VIEW ibView12 = {};
D3D12_VERTEX_BUFFER_VIEW textVbView12 = {};

// Persistent mapped pointers
void* cbMapped12 = nullptr;
void* textVbMapped12 = nullptr;

// Geometry counts
UINT totalIndices12 = 0;
UINT totalVertices12 = 0;

// ============== DXR FEATURE FLAGS ==============
DXRFeatures g_dxrFeatures = {};

// ============== DXR 1.0 FEATURE FLAGS ==============
DXR10Features g_dxr10Features = {};

// ============== DXR RAY TRACING GLOBALS ==============
ID3D12Device5* dev12RT = nullptr;
ID3D12GraphicsCommandList4* cmdListRT = nullptr;
ID3D12Resource* blasBuffer = nullptr;
ID3D12Resource* tlasBuffer = nullptr;
ID3D12Resource* instanceBuffer = nullptr;
ID3D12Resource* scratchBuffer = nullptr;
ID3D12DescriptorHeap* srvHeapRT = nullptr;
ID3D12RootSignature* rootSigRT = nullptr;
ID3D12PipelineState* psoRT = nullptr;
bool g_dxrSupported = false;

// ============== PATH TRACING GLOBALS ==============
ID3D12Resource* pathTraceOutput = nullptr;
ID3D12Resource* denoiseTemp = nullptr;
ID3D12RootSignature* pathTraceRootSig = nullptr;
ID3D12PipelineState* pathTracePSO = nullptr;
ID3D12RootSignature* denoiseRootSig = nullptr;
ID3D12PipelineState* denoisePSO = nullptr;
ID3D12DescriptorHeap* pathTraceSrvUavHeap = nullptr;
ID3D12Resource* pathTraceCB = nullptr;
ID3D12Resource* denoiseCB = nullptr;
void* pathTraceCBMapped = nullptr;
void* denoiseCBMapped = nullptr;
UINT g_frameCount = 0;

// Denoise modes (enum defined in d3d12_shared.h)
DenoiseMode g_denoiseMode = DENOISE_ATROUS;
UINT g_temporalFrameCount = 0;

// ============== DLSS RAY RECONSTRUCTION GLOBALS ==============
NVSDK_NGX_Handle* g_dlssRRHandle = nullptr;
NVSDK_NGX_Parameter* g_ngxParams = nullptr;
bool g_ngxInitialized = false;
bool g_dlssRRSupported = false;

// G-Buffer resources for DLSS-RR
ID3D12Resource* g_gbufferAlbedo = nullptr;
ID3D12Resource* g_gbufferNormal = nullptr;
ID3D12Resource* g_gbufferMotion = nullptr;
ID3D12Resource* g_gbufferDepth = nullptr;
ID3D12Resource* g_dlssOutput = nullptr;
ID3D12DescriptorHeap* g_gbufferHeap = nullptr;

// ============== TEXT RENDERING ==============
// MAX_TEXT_VERTS is #defined in d3d12_shared.h as 6000
TextVert g_textVerts[MAX_TEXT_VERTS];
UINT g_textVertCount = 0;
int g_cachedFps = -1;
bool g_textNeedsRebuild = true;

// ============== DXC SHADER COMPILER ==============
typedef HRESULT(WINAPI* DxcCreateInstanceProc)(REFCLSID, REFIID, LPVOID*);
HMODULE g_dxcModule = nullptr;
DxcCreateInstanceProc g_DxcCreateInstance = nullptr;
