// ============== D3D12 PATH TRACING + DLSS RAY RECONSTRUCTION ==============
// This file implements D3D12 Path Tracing with NVIDIA DLSS Ray Reconstruction
// Outputs G-Buffer textures which DLSS-RR then denoises

// DLSS SDK - set to 0 if SDK not available
#define ENABLE_DLSS 0

#if !ENABLE_DLSS
// Stub implementations when DLSS SDK is not available
#include <windows.h>
#include "../common.h"
bool InitD3D12PT_DLSS(HWND hwnd) {
    Log("[ERROR] DLSS renderer not available - SDK not included in build\n");
    return false;
}
void RenderD3D12PT_DLSS() {}
void CleanupD3D12PT_DLSS() {}
#else

// Local includes
#include "../common.h"
#include "d3d12_shared.h"
#include "renderer_d3d12.h"
#include "../shaders/d3d12_dlss_shaders.h"

// NVIDIA NGX SDK for DLSS Ray Reconstruction
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_defs_dlssd.h"
#include "nvsdk_ngx_params_dlssd.h"
#include "nvsdk_ngx_helpers_dlssd.h"

// System includes
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <dxcapi.h>
#include <vector>

// Linker directives
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "nvsdk_ngx_d.lib")

using namespace DirectX;

// ============== DLSS GLOBALS ==============
// Core DLSS globals are defined in d3d12_globals.cpp and declared in d3d12_shared.h
// DLSS-specific local resources:
static ID3D12Resource* g_gbufferColor = nullptr;             // HDR noisy path traced color
static ID3D12Resource* g_gbufferDiffuseAlbedo = nullptr;     // RGB diffuse albedo
static ID3D12Resource* g_gbufferSpecularAlbedo = nullptr;    // RGB specular (F0)
static ID3D12Resource* g_gbufferNormals = nullptr;           // World-space normals
static ID3D12Resource* g_gbufferRoughness = nullptr;         // Roughness
static ID3D12Resource* g_gbufferMotionVectors = nullptr;     // Screen-space motion vectors

// DLSS-specific resources
ID3D12RootSignature* g_pathTraceGbufferRootSig = nullptr;  // Root sig for G-Buffer PT
ID3D12PipelineState* g_pathTraceGbufferPSO = nullptr;      // PSO for G-Buffer PT
ID3D12DescriptorHeap* g_dlssSrvUavHeap = nullptr;          // SRV/UAV heap for DLSS

// Tone mapping for HDR->LDR conversion
ID3D12RootSignature* g_tonemapRootSig = nullptr;
ID3D12PipelineState* g_tonemapPSO = nullptr;
ID3D12DescriptorHeap* g_tonemapSrvHeap = nullptr;

// DLSS constant buffer
static ID3D12Resource* g_dlssCB = nullptr;
static void* g_dlssCBMapped = nullptr;

// Previous frame's ViewProj matrix for motion vectors
static XMMATRIX g_prevViewProj = XMMatrixIdentity();

// ============== CONSTANT BUFFER STRUCTURES ==============

struct CB {
    float time;
    float _pad[3];
};

struct PathTraceCBData {
    XMFLOAT4X4 InvView;
    XMFLOAT4X4 InvProj;
    float Time;
    UINT FrameCount;
    UINT Width;
    UINT Height;
};

struct PathTraceDlssCBData {
    XMFLOAT4X4 InvView;
    XMFLOAT4X4 InvProj;
    XMFLOAT4X4 PrevViewProj;
    float Time;
    UINT FrameCount;
    UINT Width;
    UINT Height;
};

struct Vert {
    XMFLOAT3 p, n;
    UINT cubeID;
};

// ============== NGX INITIALIZATION ==============

static bool InitNGX()
{
    Log("[INFO] Initializing NVIDIA NGX SDK...\n");

    // Get executable directory for NGX data path
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    wchar_t* lastSlash = wcsrchr(exePath, L'\\');
    if (lastSlash) *lastSlash = L'\0';

    // Initialize NGX with project ID (must be GUID-like for CUSTOM engine type)
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_Init_with_ProjectID(
        "c1d3d12a-b001-4fea-90ae-c4035c19df01",   // Project ID (GUID format required for CUSTOM)
        NVSDK_NGX_ENGINE_TYPE_CUSTOM,            // Engine type
        "1.0",                                    // Engine version
        exePath,                                  // Data path
        dev12,                                    // D3D12 device
        nullptr,                                  // Feature info (optional)
        NVSDK_NGX_Version_API                     // SDK version
    );

    if (NVSDK_NGX_FAILED(result)) {
        Log("[ERROR] NGX initialization failed: 0x%08X\n", result);
        return false;
    }

    Log("[INFO] NGX initialized successfully\n");

    // Get capability parameters
    result = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_ngxParams);
    if (NVSDK_NGX_FAILED(result)) {
        Log("[ERROR] Failed to get NGX capability parameters: 0x%08X\n", result);
        return false;
    }

    // Check if DLSS Ray Reconstruction is supported
    int dlssRRSupported = 0;
    g_ngxParams->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &dlssRRSupported);

    // Also check for Ray Reconstruction specifically
    unsigned int needsUpdatedDriver = 0;
    result = g_ngxParams->Get(NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsUpdatedDriver);

    if (!dlssRRSupported) {
        Log("[WARNING] DLSS not available on this system\n");
        g_dlssRRSupported = false;
        return false;
    }

    g_dlssRRSupported = true;
    g_ngxInitialized = true;
    Log("[INFO] DLSS Ray Reconstruction is supported\n");
    return true;
}

// ============== CREATE DLSS-RR FEATURE ==============

static bool CreateDLSSRRFeature()
{
    if (!g_ngxInitialized || !g_dlssRRSupported) {
        Log("[ERROR] NGX not initialized or DLSS-RR not supported\n");
        return false;
    }

    Log("[INFO] Creating DLSS Ray Reconstruction feature...\n");

    // Create DLSS-RR feature parameters
    NVSDK_NGX_Parameter* ngxParams = nullptr;
    NVSDK_NGX_Result result = NVSDK_NGX_D3D12_AllocateParameters(&ngxParams);
    if (NVSDK_NGX_FAILED(result)) {
        Log("[ERROR] Failed to allocate DLSS-RR parameters: 0x%08X\n", result);
        return false;
    }

    // Fill the DLSSD Create Params structure (required by the helper function!)
    NVSDK_NGX_DLSSD_Create_Params dlssdCreateParams = {};
    dlssdCreateParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;  // DL-based denoiser
    dlssdCreateParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;  // Separate roughness buffer
    dlssdCreateParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;  // We output linear depth
    dlssdCreateParams.InWidth = W;
    dlssdCreateParams.InHeight = H;
    dlssdCreateParams.InTargetWidth = W;   // No upscaling, same resolution
    dlssdCreateParams.InTargetHeight = H;
    dlssdCreateParams.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value_Balanced;
    dlssdCreateParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
                                              NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    dlssdCreateParams.InEnableOutputSubrects = false;

    // Set Ray Reconstruction preset (D for transformer model - best quality)
    ngxParams->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced,
        (int)NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D);

    // Create command list for feature creation
    cmdList->Reset(cmdAlloc[0], nullptr);

    // Create DLSS-RR feature using the proper structure
    result = NGX_D3D12_CREATE_DLSSD_EXT(
        cmdList,
        1,  // Creation node mask
        1,  // Visibility node mask
        &g_dlssRRHandle,
        ngxParams,
        &dlssdCreateParams  // Now passing the proper structure!
    );

    // Execute command list
    cmdList->Close();
    ID3D12CommandList* ppCommandLists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, ppCommandLists);
    WaitForGpu();

    NVSDK_NGX_D3D12_DestroyParameters(ngxParams);

    if (NVSDK_NGX_FAILED(result)) {
        Log("[ERROR] Failed to create DLSS-RR feature: 0x%08X\n", result);
        return false;
    }

    Log("[INFO] DLSS Ray Reconstruction feature created successfully\n");
    return true;
}

// ============== CREATE G-BUFFER TEXTURES ==============

static bool CreateGBufferTextures()
{
    Log("[INFO] Creating G-Buffer textures for DLSS-RR...\n");

    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = W;
    texDesc.Height = H;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

    // HDR noisy color output (RGBA16F) - for DLSS-RR input
    texDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (FAILED(dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_gbufferColor)))) {
        Log("[ERROR] Failed to create HDR color texture\n");
        return false;
    }

    // Diffuse albedo (RGBA16F)
    if (FAILED(dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_gbufferDiffuseAlbedo)))) {
        Log("[ERROR] Failed to create diffuse albedo texture\n");
        return false;
    }

    // Specular albedo (RGBA16F)
    if (FAILED(dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_gbufferSpecularAlbedo)))) {
        Log("[ERROR] Failed to create specular albedo texture\n");
        return false;
    }

    // Normals (RGBA16F)
    if (FAILED(dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_gbufferNormals)))) {
        Log("[ERROR] Failed to create normals texture\n");
        return false;
    }

    // Roughness (R16F)
    texDesc.Format = DXGI_FORMAT_R16_FLOAT;
    if (FAILED(dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_gbufferRoughness)))) {
        Log("[ERROR] Failed to create roughness texture\n");
        return false;
    }

    // Depth (R32F)
    texDesc.Format = DXGI_FORMAT_R32_FLOAT;
    if (FAILED(dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_gbufferDepth)))) {
        Log("[ERROR] Failed to create depth texture\n");
        return false;
    }

    // Motion vectors (RG16F)
    texDesc.Format = DXGI_FORMAT_R16G16_FLOAT;
    if (FAILED(dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_gbufferMotionVectors)))) {
        Log("[ERROR] Failed to create motion vectors texture\n");
        return false;
    }

    // DLSS output (RGBA16F)
    texDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    if (FAILED(dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&g_dlssOutput)))) {
        Log("[ERROR] Failed to create DLSS output texture\n");
        return false;
    }

    Log("[INFO] G-Buffer textures created successfully\n");
    return true;
}

// ============== INIT D3D12 PATH TRACING + DLSS ==============

bool InitD3D12PT_DLSS(HWND hwnd)
{
    Log("[INFO] Initializing Direct3D 12 with Path Tracing + DLSS Ray Reconstruction...\n");

    // First initialize the base D3D12 PT
    if (!InitD3D12PT(hwnd)) {
        Log("[ERROR] Failed to initialize base D3D12 PT\n");
        return false;
    }

    // Initialize NGX
    if (!InitNGX()) {
        Log("[ERROR] NGX initialization failed - DLSS-RR not available\n");
        MessageBoxW(hwnd, L"DLSS Ray Reconstruction is not available on this system.\nFalling back to standard path tracing.",
            L"DLSS Not Available", MB_OK | MB_ICONWARNING);
        return true;  // Continue with basic PT
    }

    // Create G-Buffer textures
    if (!CreateGBufferTextures()) {
        Log("[ERROR] Failed to create G-Buffer textures\n");
        return false;
    }

    // Create DLSS-RR feature
    if (!CreateDLSSRRFeature()) {
        Log("[ERROR] Failed to create DLSS-RR feature\n");
        g_dlssRRSupported = false;
        return true;  // Continue with basic PT
    }

    // Check if device was removed during DLSS feature creation
    HRESULT hrRemoved = dev12->GetDeviceRemovedReason();
    if (FAILED(hrRemoved)) {
        Log("[ERROR] Device removed after DLSS feature creation: 0x%08X\n", hrRemoved);
        return false;
    }

    // Create G-Buffer root signature (CBV + SRVs + 7 UAVs)
    {
        D3D12_DESCRIPTOR_RANGE1 srvUavRanges[2] = {};
        // SRVs: t0 = TLAS, t1 = Vertices, t2 = Indices
        srvUavRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvUavRanges[0].NumDescriptors = 3;
        srvUavRanges[0].BaseShaderRegister = 0;
        srvUavRanges[0].RegisterSpace = 0;
        srvUavRanges[0].OffsetInDescriptorsFromTableStart = 0;
        // UAVs: u0-u6 = G-Buffer outputs (7 total)
        srvUavRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        srvUavRanges[1].NumDescriptors = 7;
        srvUavRanges[1].BaseShaderRegister = 0;
        srvUavRanges[1].RegisterSpace = 0;
        srvUavRanges[1].OffsetInDescriptorsFromTableStart = 3;

        D3D12_ROOT_PARAMETER1 rootParams[2] = {};
        rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rootParams[0].Descriptor.ShaderRegister = 0;
        rootParams[0].Descriptor.RegisterSpace = 0;
        rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParams[1].DescriptorTable.NumDescriptorRanges = 2;
        rootParams[1].DescriptorTable.pDescriptorRanges = srvUavRanges;
        rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC rootSigDesc = {};
        rootSigDesc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        rootSigDesc.Desc_1_1.NumParameters = 2;
        rootSigDesc.Desc_1_1.pParameters = rootParams;
        rootSigDesc.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ID3DBlob* sigBlob = nullptr;
        ID3DBlob* errBlob = nullptr;
        if (FAILED(D3D12SerializeVersionedRootSignature(&rootSigDesc, &sigBlob, &errBlob))) {
            if (errBlob) { Log("[ERROR] G-Buffer root sig: %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
            return false;
        }
        if (FAILED(dev12->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&g_pathTraceGbufferRootSig)))) {
            Log("[ERROR] Failed to create G-Buffer root signature\n");
            sigBlob->Release();
            return false;
        }
        sigBlob->Release();
        Log("[INFO] G-Buffer root signature created\n");
    }

    // Compile G-Buffer path tracing shader
    {
        Log("[INFO] Compiling G-Buffer path tracing shader...\n");
        HMODULE dxcModule = LoadLibraryW(L"dxcompiler.dll");
        if (!dxcModule) {
            Log("[ERROR] Failed to load dxcompiler.dll\n");
            return false;
        }
        DxcCreateInstanceProc DxcCreateInstance = (DxcCreateInstanceProc)GetProcAddress(dxcModule, "DxcCreateInstance");

        IDxcCompiler3* compiler = nullptr;
        IDxcUtils* utils = nullptr;
        DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));
        DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));

        IDxcBlobEncoding* sourceBlob = nullptr;
        utils->CreateBlob(g_ptDlssShaderCode, (UINT)strlen(g_ptDlssShaderCode), CP_UTF8, &sourceBlob);

        LPCWSTR args[] = { L"-T", L"cs_6_5", L"-E", L"PathTraceDlssCS" };
        DxcBuffer sourceBuffer = { sourceBlob->GetBufferPointer(), sourceBlob->GetBufferSize(), CP_UTF8 };

        IDxcResult* result = nullptr;
        compiler->Compile(&sourceBuffer, args, _countof(args), nullptr, IID_PPV_ARGS(&result));

        HRESULT hrStatus;
        result->GetStatus(&hrStatus);
        if (FAILED(hrStatus)) {
            IDxcBlobEncoding* errors = nullptr;
            result->GetErrorBuffer(&errors);
            Log("[ERROR] G-Buffer shader compile failed: %s\n", (char*)errors->GetBufferPointer());
            errors->Release();
            result->Release();
            sourceBlob->Release();
            compiler->Release();
            utils->Release();
            return false;
        }

        IDxcBlob* shaderBlob = nullptr;
        result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr);

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = g_pathTraceGbufferRootSig;
        psoDesc.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
        psoDesc.CS.BytecodeLength = shaderBlob->GetBufferSize();

        if (FAILED(dev12->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&g_pathTraceGbufferPSO)))) {
            Log("[ERROR] Failed to create G-Buffer PSO\n");
            shaderBlob->Release();
            result->Release();
            sourceBlob->Release();
            compiler->Release();
            utils->Release();
            return false;
        }

        Log("[INFO] G-Buffer PSO created (shader size: %zu)\n", shaderBlob->GetBufferSize());
        shaderBlob->Release();
        result->Release();
        sourceBlob->Release();
        compiler->Release();
        utils->Release();

        // Check if device was removed during PSO creation
        hrRemoved = dev12->GetDeviceRemovedReason();
        if (FAILED(hrRemoved)) {
            Log("[ERROR] Device removed after G-Buffer PSO creation: 0x%08X\n", hrRemoved);
            return false;
        }
    }

    // Create descriptor heap for G-Buffer (3 SRVs + 7 UAVs = 10 descriptors)
    {
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.NumDescriptors = 10;
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        if (FAILED(dev12->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&g_dlssSrvUavHeap)))) {
            Log("[ERROR] Failed to create G-Buffer descriptor heap\n");
            return false;
        }

        UINT descSize = dev12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = g_dlssSrvUavHeap->GetCPUDescriptorHandleForHeapStart();

        // SRV 0: TLAS
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.RaytracingAccelerationStructure.Location = tlasBuffer->GetGPUVirtualAddress();
        dev12->CreateShaderResourceView(nullptr, &srvDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // SRV 1: Vertices
        srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.NumElements = totalVertices12;
        srvDesc.Buffer.StructureByteStride = sizeof(Vert);
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        dev12->CreateShaderResourceView(vb12, &srvDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // SRV 2: Indices
        srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Buffer.NumElements = totalIndices12;
        srvDesc.Buffer.StructureByteStride = sizeof(UINT);
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        dev12->CreateShaderResourceView(ib12, &srvDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // UAV 0: HDR noisy color output (RGBA16F)
        D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        dev12->CreateUnorderedAccessView(g_gbufferColor, nullptr, &uavDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // UAV 1: Diffuse Albedo (RGBA16F)
        dev12->CreateUnorderedAccessView(g_gbufferDiffuseAlbedo, nullptr, &uavDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // UAV 2: Specular Albedo
        dev12->CreateUnorderedAccessView(g_gbufferSpecularAlbedo, nullptr, &uavDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // UAV 3: Normals
        dev12->CreateUnorderedAccessView(g_gbufferNormals, nullptr, &uavDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // UAV 4: Roughness
        uavDesc.Format = DXGI_FORMAT_R16_FLOAT;
        dev12->CreateUnorderedAccessView(g_gbufferRoughness, nullptr, &uavDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // UAV 5: Depth
        uavDesc.Format = DXGI_FORMAT_R32_FLOAT;
        dev12->CreateUnorderedAccessView(g_gbufferDepth, nullptr, &uavDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // UAV 6: Motion Vectors
        uavDesc.Format = DXGI_FORMAT_R16G16_FLOAT;
        dev12->CreateUnorderedAccessView(g_gbufferMotionVectors, nullptr, &uavDesc, cpuHandle);

        Log("[INFO] G-Buffer descriptor heap created (10 descriptors)\n");

        // Check if device was removed during descriptor heap creation
        hrRemoved = dev12->GetDeviceRemovedReason();
        if (FAILED(hrRemoved)) {
            Log("[ERROR] Device removed after descriptor heap creation: 0x%08X\n", hrRemoved);
            return false;
        }
    }

    // Wait for GPU to finish all previous operations before creating more resources
    WaitForGpu();

    // Check device status after WaitForGpu
    hrRemoved = dev12->GetDeviceRemovedReason();
    if (FAILED(hrRemoved)) {
        Log("[ERROR] Device removed after WaitForGpu: 0x%08X\n", hrRemoved);
        return false;
    }

    // Create DLSS constant buffer (larger, includes PrevViewProj)
    D3D12_HEAP_PROPERTIES heapProps = {};
    heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;

    D3D12_RESOURCE_DESC cbDesc = {};
    cbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    cbDesc.Width = (sizeof(PathTraceDlssCBData) + 255) & ~255;
    cbDesc.Height = 1;
    cbDesc.DepthOrArraySize = 1;
    cbDesc.MipLevels = 1;
    cbDesc.SampleDesc.Count = 1;
    cbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hrCB = dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &cbDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_dlssCB));
    if (FAILED(hrCB)) {
        Log("[ERROR] Failed to create DLSS constant buffer: 0x%08X (size=%zu)\n", hrCB, (size_t)cbDesc.Width);
        return false;
    }

    D3D12_RANGE readRange = { 0, 0 };
    g_dlssCB->Map(0, &readRange, &g_dlssCBMapped);

    // ===== CREATE TONE MAPPING PSO =====
    // This converts HDR (RGBA16F) to LDR (RGBA8) for display
    {
        // Root signature: 1 descriptor table with SRV
        D3D12_DESCRIPTOR_RANGE srvRange = {};
        srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        srvRange.NumDescriptors = 1;
        srvRange.BaseShaderRegister = 0;

        D3D12_ROOT_PARAMETER rootParam = {};
        rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        rootParam.DescriptorTable.NumDescriptorRanges = 1;
        rootParam.DescriptorTable.pDescriptorRanges = &srvRange;
        rootParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_STATIC_SAMPLER_DESC sampler = {};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.ShaderRegister = 0;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

        D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
        rsDesc.NumParameters = 1;
        rsDesc.pParameters = &rootParam;
        rsDesc.NumStaticSamplers = 1;
        rsDesc.pStaticSamplers = &sampler;
        rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

        ID3DBlob* sigBlob = nullptr;
        ID3DBlob* errorBlob = nullptr;
        D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errorBlob);
        if (errorBlob) errorBlob->Release();

        dev12->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&g_tonemapRootSig));
        sigBlob->Release();

        // Compile tone mapping shaders
        const char* tonemapShader = R"(
            Texture2D<float4> hdrInput : register(t0);
            SamplerState samp : register(s0);

            struct VSOut {
                float4 pos : SV_Position;
                float2 uv : TEXCOORD0;
            };

            // Fullscreen triangle vertex shader
            VSOut VSMain(uint vertexId : SV_VertexID) {
                VSOut o;
                // Generate fullscreen triangle
                o.uv = float2((vertexId << 1) & 2, vertexId & 2);
                o.pos = float4(o.uv * 2.0 - 1.0, 0.0, 1.0);
                o.uv.y = 1.0 - o.uv.y;  // Flip Y
                return o;
            }

            // Pixel shader with tone mapping
            float4 PSMain(VSOut input) : SV_Target {
                float3 hdr = hdrInput.Sample(samp, input.uv).rgb;

                // Reinhard tone mapping
                float3 ldr = hdr / (1.0 + hdr);

                // Gamma correction
                ldr = pow(ldr, 1.0 / 2.2);

                return float4(ldr, 1.0);
            }
        )";

        // Use DXC compiler for tone mapping shaders
        HMODULE dxcModule = LoadLibraryW(L"dxcompiler.dll");
        DxcCreateInstanceProc DxcCreateInstance = (DxcCreateInstanceProc)GetProcAddress(dxcModule, "DxcCreateInstance");

        IDxcUtils* utils = nullptr;
        IDxcCompiler3* compiler = nullptr;
        DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
        DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

        IDxcBlobEncoding* sourceBlob = nullptr;
        utils->CreateBlob(tonemapShader, (UINT)strlen(tonemapShader), CP_UTF8, &sourceBlob);

        DxcBuffer srcBuffer = { sourceBlob->GetBufferPointer(), sourceBlob->GetBufferSize(), CP_UTF8 };

        // Compile VS
        LPCWSTR vsArgs[] = { L"-E", L"VSMain", L"-T", L"vs_6_0" };
        IDxcResult* vsResult = nullptr;
        compiler->Compile(&srcBuffer, vsArgs, _countof(vsArgs), nullptr, IID_PPV_ARGS(&vsResult));
        IDxcBlob* vsBlob = nullptr;
        vsResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&vsBlob), nullptr);
        vsResult->Release();

        // Compile PS
        LPCWSTR psArgs[] = { L"-E", L"PSMain", L"-T", L"ps_6_0" };
        IDxcResult* psResult = nullptr;
        compiler->Compile(&srcBuffer, psArgs, _countof(psArgs), nullptr, IID_PPV_ARGS(&psResult));
        IDxcBlob* psBlob = nullptr;
        psResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&psBlob), nullptr);
        psResult->Release();

        sourceBlob->Release();
        compiler->Release();
        utils->Release();

        // Create PSO
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = g_tonemapRootSig;
        psoDesc.VS = { vsBlob->GetBufferPointer(), vsBlob->GetBufferSize() };
        psoDesc.PS = { psBlob->GetBufferPointer(), psBlob->GetBufferSize() };
        psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        psoDesc.SampleMask = UINT_MAX;
        psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleDesc.Count = 1;

        dev12->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&g_tonemapPSO));
        vsBlob->Release();
        psBlob->Release();

        Log("[INFO] Tone mapping PSO created\n");

        // Create SRV heap for tone mapping (1 descriptor for HDR input)
        D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
        srvHeapDesc.NumDescriptors = 2;  // One for g_gbufferColor, one for g_dlssOutput
        srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        dev12->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&g_tonemapSrvHeap));

        UINT descSize = dev12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE cpuHandle = g_tonemapSrvHeap->GetCPUDescriptorHandleForHeapStart();

        // SRV 0: g_gbufferColor (noisy HDR)
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 1;
        dev12->CreateShaderResourceView(g_gbufferColor, &srvDesc, cpuHandle);
        cpuHandle.ptr += descSize;

        // SRV 1: g_dlssOutput (denoised HDR)
        dev12->CreateShaderResourceView(g_dlssOutput, &srvDesc, cpuHandle);

        Log("[INFO] Tone mapping SRV heap created\n");
    }

    // Initialize text rendering (shared with base D3D12 renderer)
    if (!InitGPUText12()) {
        Log("[ERROR] Failed to initialize text rendering for DLSS!\n");
        return false;
    }

    Log("[INFO] D3D12 + Path Tracing + DLSS Ray Reconstruction initialization complete\n");
    return true;
}

// ============== RENDER D3D12 PT + DLSS ==============

void RenderD3D12PT_DLSS()
{
    cmdAlloc[frameIndex]->Reset();
    cmdList->Reset(cmdAlloc[frameIndex], nullptr);  // Start with no PSO

    // Update constant buffer
    LARGE_INTEGER nowTime;
    QueryPerformanceCounter(&nowTime);
    float t = (float)(nowTime.QuadPart - g_startTime.QuadPart) / g_perfFreq.QuadPart;

    // ===== UPDATE CUBE TRANSFORM AND REBUILD TLAS =====
    UpdateCubeTransformPT(t);
    RebuildTLAS_PT(cmdListRT);

    // Build camera matrices (looking at room from outside)
    XMMATRIX view = XMMatrixLookAtLH(
        XMVectorSet(0, 0, -3.5f, 1),  // Eye position (further back)
        XMVectorSet(0, 0, 1, 1),      // Look at back wall
        XMVectorSet(0, 1, 0, 0)       // Up
    );
    XMMATRIX proj = XMMatrixPerspectiveFovLH(XM_PI / 3.0f, (float)W / H, 0.1f, 100.0f);  // 60° FOV
    XMMATRIX invView = XMMatrixInverse(nullptr, view);
    XMMATRIX invProj = XMMatrixInverse(nullptr, proj);
    XMMATRIX viewProj = view * proj;

    // Use DLSS constant buffer with PrevViewProj for motion vectors
    if (g_dlssRRSupported && g_dlssCBMapped) {
        PathTraceDlssCBData dlssCbData;
        XMStoreFloat4x4(&dlssCbData.InvView, XMMatrixTranspose(invView));
        XMStoreFloat4x4(&dlssCbData.InvProj, XMMatrixTranspose(invProj));
        XMStoreFloat4x4(&dlssCbData.PrevViewProj, XMMatrixTranspose(g_prevViewProj));
        dlssCbData.Time = t;
        dlssCbData.FrameCount = g_frameCount++;
        dlssCbData.Width = W;
        dlssCbData.Height = H;
        memcpy(g_dlssCBMapped, &dlssCbData, sizeof(dlssCbData));
    } else {
        PathTraceCBData cbData;
        XMStoreFloat4x4(&cbData.InvView, XMMatrixTranspose(invView));
        XMStoreFloat4x4(&cbData.InvProj, XMMatrixTranspose(invProj));
        cbData.Time = t;
        cbData.FrameCount = g_frameCount++;
        cbData.Width = W;
        cbData.Height = H;
        memcpy(pathTraceCBMapped, &cbData, sizeof(cbData));
    }

    // Store current ViewProj for next frame's motion vectors
    g_prevViewProj = viewProj;

    // Also update regular CB for text rendering
    CB textCbData = { t };
    memcpy(cbMapped12, &textCbData, sizeof(CB));

    // ===== PATH TRACING WITH G-BUFFER OUTPUT =====
    if (g_dlssRRSupported && g_pathTraceGbufferRootSig && g_dlssSrvUavHeap && g_pathTraceGbufferPSO) {
        cmdList->SetPipelineState(g_pathTraceGbufferPSO);
        cmdList->SetComputeRootSignature(g_pathTraceGbufferRootSig);
        cmdList->SetComputeRootConstantBufferView(0, g_dlssCB->GetGPUVirtualAddress());

        ID3D12DescriptorHeap* heaps[] = { g_dlssSrvUavHeap };
        cmdList->SetDescriptorHeaps(1, heaps);
        cmdList->SetComputeRootDescriptorTable(1, g_dlssSrvUavHeap->GetGPUDescriptorHandleForHeapStart());
    } else if (pathTracePSO && pathTraceRootSig) {
        // Fallback to standard PT (no DLSS)
        cmdList->SetPipelineState(pathTracePSO);
        cmdList->SetComputeRootSignature(pathTraceRootSig);
        cmdList->SetComputeRootConstantBufferView(0, pathTraceCB->GetGPUVirtualAddress());

        ID3D12DescriptorHeap* heaps[] = { pathTraceSrvUavHeap };
        cmdList->SetDescriptorHeaps(1, heaps);
        cmdList->SetComputeRootDescriptorTable(1, pathTraceSrvUavHeap->GetGPUDescriptorHandleForHeapStart());
    }

    UINT groupsX = (W + 7) / 8;
    UINT groupsY = (H + 7) / 8;
    cmdList->Dispatch(groupsX, groupsY, 1);

    // ===== DLSS RAY RECONSTRUCTION EVALUATION =====
    D3D12_RESOURCE_BARRIER barriers[8] = {};
    ID3D12Resource* outputToCopy = g_gbufferColor;  // Default to HDR noisy output

    if (g_dlssRRSupported && g_dlssRRHandle) {
        // Transition G-Buffer textures to shader resource state for DLSS to read
        barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[0].Transition.pResource = g_gbufferColor;
        barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        barriers[1] = barriers[0];
        barriers[1].Transition.pResource = g_gbufferDiffuseAlbedo;

        barriers[2] = barriers[0];
        barriers[2].Transition.pResource = g_gbufferSpecularAlbedo;

        barriers[3] = barriers[0];
        barriers[3].Transition.pResource = g_gbufferNormals;

        barriers[4] = barriers[0];
        barriers[4].Transition.pResource = g_gbufferRoughness;

        barriers[5] = barriers[0];
        barriers[5].Transition.pResource = g_gbufferDepth;

        barriers[6] = barriers[0];
        barriers[6].Transition.pResource = g_gbufferMotionVectors;

        cmdList->ResourceBarrier(7, barriers);

        // Allocate parameters for DLSS evaluation
        NVSDK_NGX_Parameter* evalParams = nullptr;
        NVSDK_NGX_D3D12_AllocateParameters(&evalParams);

        // Fill DLSS-RR evaluation parameters
        NVSDK_NGX_D3D12_DLSSD_Eval_Params dlssEvalParams = {};
        dlssEvalParams.pInColor = g_gbufferColor;
        dlssEvalParams.pInOutput = g_dlssOutput;
        dlssEvalParams.pInDepth = g_gbufferDepth;
        dlssEvalParams.pInMotionVectors = g_gbufferMotionVectors;
        dlssEvalParams.pInDiffuseAlbedo = g_gbufferDiffuseAlbedo;
        dlssEvalParams.pInSpecularAlbedo = g_gbufferSpecularAlbedo;
        dlssEvalParams.pInNormals = g_gbufferNormals;
        dlssEvalParams.pInRoughness = g_gbufferRoughness;
        dlssEvalParams.InJitterOffsetX = 0.0f;  // We use full-res, no jitter
        dlssEvalParams.InJitterOffsetY = 0.0f;
        dlssEvalParams.InRenderSubrectDimensions.Width = W;
        dlssEvalParams.InRenderSubrectDimensions.Height = H;
        dlssEvalParams.InMVScaleX = 1.0f;
        dlssEvalParams.InMVScaleY = 1.0f;
        dlssEvalParams.InReset = 0;

        // Execute DLSS Ray Reconstruction
        NVSDK_NGX_Result result = NGX_D3D12_EVALUATE_DLSSD_EXT(
            cmdList,
            g_dlssRRHandle,
            evalParams,
            &dlssEvalParams
        );

        NVSDK_NGX_D3D12_DestroyParameters(evalParams);

        if (NVSDK_NGX_SUCCEED(result)) {
            outputToCopy = g_dlssOutput;  // Use denoised output
        }

        // Transition back to UAV for next frame
        for (int i = 0; i < 7; i++) {
            barriers[i].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            barriers[i].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        cmdList->ResourceBarrier(7, barriers);
    }

    // ===== TONE MAPPING: HDR -> LDR =====
    // Transition backbuffer to render target and HDR texture to SRV
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = renderTargets12[frameIndex];
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = outputToCopy;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(2, barriers);

    // Render fullscreen triangle with tone mapping
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap12->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += frameIndex * rtvDescSize;
    cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);

    D3D12_VIEWPORT vp = {0, 0, (float)W, (float)H, 0, 1};
    D3D12_RECT sr = {0, 0, (LONG)W, (LONG)H};
    cmdList->RSSetViewports(1, &vp);
    cmdList->RSSetScissorRects(1, &sr);

    cmdList->SetPipelineState(g_tonemapPSO);
    cmdList->SetGraphicsRootSignature(g_tonemapRootSig);

    ID3D12DescriptorHeap* heaps[] = { g_tonemapSrvHeap };
    cmdList->SetDescriptorHeaps(1, heaps);

    // Select correct SRV: index 0 = g_gbufferColor (noisy), index 1 = g_dlssOutput (denoised)
    UINT descSize = dev12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = g_tonemapSrvHeap->GetGPUDescriptorHandleForHeapStart();
    if (outputToCopy == g_dlssOutput) {
        gpuHandle.ptr += descSize;  // Use SRV 1 (denoised)
    }
    cmdList->SetGraphicsRootDescriptorTable(0, gpuHandle);

    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmdList->DrawInstanced(3, 1, 0, 0);  // Fullscreen triangle

    // Transition HDR texture back to UAV for next frame
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList->ResourceBarrier(1, &barriers[1]);

    // ===== TEXT OVERLAY =====
    // rtvHandle already computed above in tone mapping section

    if (fps != g_cachedFps || g_textNeedsRebuild) {
        g_cachedFps = fps;
        g_textNeedsRebuild = false;

        static char gpuNameA[128] = {0};
        if (gpuNameA[0] == 0) {
            size_t converted;
            wcstombs_s(&converted, gpuNameA, sizeof(gpuNameA), gpuName.c_str(), _TRUNCATE);
        }

        char infoText[512];
        const char* dlssStatus = g_dlssRRSupported ? "DLSS-RR Active" : "DLSS-RR N/A (fallback)";
        sprintf_s(infoText,
            "API: D3D12 + PT + DLSS RR\n"
            "GPU: %s\n"
            "FPS: %d\n"
            "Triangles: %u\n"
            "Resolution: %ux%u\n"
            "Rays: 1 SPP | Bounces: 3\n"
            "%s",
            gpuNameA, fps, totalIndices12 / 3, W, H, dlssStatus);

        g_textVertCount = 0;
        DrawTextDirect(infoText, 12.0f, 12.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.5f);
        DrawTextDirect(infoText, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.5f);
        memcpy(textVbMapped12, g_textVerts, g_textVertCount * sizeof(TextVert));
    }

    // Draw text
    if (g_textVertCount > 0) {
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        D3D12_VIEWPORT vp = {0, 0, (float)W, (float)H, 0, 1};
        D3D12_RECT sr = {0, 0, (LONG)W, (LONG)H};
        cmdList->RSSetViewports(1, &vp);
        cmdList->RSSetScissorRects(1, &sr);
        cmdList->SetPipelineState(textPso);
        cmdList->SetGraphicsRootSignature(textRootSig12);
        ID3D12DescriptorHeap* srvHeaps[] = { srvHeap12 };
        cmdList->SetDescriptorHeaps(1, srvHeaps);
        cmdList->SetGraphicsRootDescriptorTable(0, srvHeap12->GetGPUDescriptorHandleForHeapStart());
        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmdList->IASetVertexBuffers(0, 1, &textVbView12);
        cmdList->DrawInstanced(g_textVertCount, 1, 0, 0);
    }

    // Transition backbuffer to present
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = renderTargets12[frameIndex];
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    cmdList->ResourceBarrier(1, barriers);

    cmdList->Close();

    ID3D12CommandList* lists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, lists);

    UINT syncInterval = 0;  // Vsync off
    UINT presentFlags = g_tearingSupported12 ? DXGI_PRESENT_ALLOW_TEARING : 0;
    swap12->Present(syncInterval, presentFlags);

    MoveToNextFrame();
}

// ============== CLEANUP D3D12 PT + DLSS ==============

void CleanupD3D12PT_DLSS()
{
    WaitForGpu();

    // Release DLSS-RR resources
    if (g_dlssRRHandle) {
        NVSDK_NGX_D3D12_ReleaseFeature(g_dlssRRHandle);
        g_dlssRRHandle = nullptr;
    }

    // Release G-Buffer PSO and root signature
    if (g_pathTraceGbufferPSO) { g_pathTraceGbufferPSO->Release(); g_pathTraceGbufferPSO = nullptr; }
    if (g_pathTraceGbufferRootSig) { g_pathTraceGbufferRootSig->Release(); g_pathTraceGbufferRootSig = nullptr; }
    if (g_dlssSrvUavHeap) { g_dlssSrvUavHeap->Release(); g_dlssSrvUavHeap = nullptr; }

    // Release tone mapping resources
    if (g_tonemapPSO) { g_tonemapPSO->Release(); g_tonemapPSO = nullptr; }
    if (g_tonemapRootSig) { g_tonemapRootSig->Release(); g_tonemapRootSig = nullptr; }
    if (g_tonemapSrvHeap) { g_tonemapSrvHeap->Release(); g_tonemapSrvHeap = nullptr; }

    // Release G-Buffer textures
    if (g_gbufferColor) { g_gbufferColor->Release(); g_gbufferColor = nullptr; }
    if (g_gbufferDiffuseAlbedo) { g_gbufferDiffuseAlbedo->Release(); g_gbufferDiffuseAlbedo = nullptr; }
    if (g_gbufferSpecularAlbedo) { g_gbufferSpecularAlbedo->Release(); g_gbufferSpecularAlbedo = nullptr; }
    if (g_gbufferNormals) { g_gbufferNormals->Release(); g_gbufferNormals = nullptr; }
    if (g_gbufferRoughness) { g_gbufferRoughness->Release(); g_gbufferRoughness = nullptr; }
    if (g_gbufferDepth) { g_gbufferDepth->Release(); g_gbufferDepth = nullptr; }
    if (g_gbufferMotionVectors) { g_gbufferMotionVectors->Release(); g_gbufferMotionVectors = nullptr; }
    if (g_dlssOutput) { g_dlssOutput->Release(); g_dlssOutput = nullptr; }
    if (g_dlssCB) { g_dlssCB->Release(); g_dlssCB = nullptr; }

    // Shutdown NGX
    if (g_ngxInitialized && dev12) {
        NVSDK_NGX_D3D12_Shutdown1(dev12);
        g_ngxInitialized = false;
    }

    // Call base PT cleanup
    CleanupD3D12PT();
}

#endif // ENABLE_DLSS
