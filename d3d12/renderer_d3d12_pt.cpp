// ============== D3D12 PATH TRACING RENDERER ==============
// Compute shader-based path tracing with DXR inline ray tracing (RayQuery)

#include "../common.h"
#include "d3d12_shared.h"
#include "renderer_d3d12.h"
#include "../shaders/d3d12_pt_shaders.h"
#include "../shaders/d3d12_denoise_shaders.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <dxcapi.h>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;

// ============== LOCAL CONSTANT BUFFER STRUCTURES ==============
struct PathTraceCBData {
    XMFLOAT4X4 InvView;
    XMFLOAT4X4 InvProj;
    float Time;
    UINT FrameCount;
    UINT Width;
    UINT Height;
};

struct DenoiseCBData {
    UINT Width;
    UINT Height;
    UINT StepSize;
    float ColorSigma;
};

struct CB {
    float time;
    float _pad[3];
};

// ============== LOCAL VERTEX STRUCTURE ==============
// Unique name to avoid ODR violation with other renderers' Vert structs
// Force no padding - must be exactly 32 bytes
#pragma pack(push, 1)
struct PTVert {
    XMFLOAT3 p;         // Position (offset 0)
    XMFLOAT3 n;         // Normal (offset 12)
    UINT objectID;      // Object ID for color (offset 24)
    UINT materialType;  // Material type (offset 28)
};
#pragma pack(pop)
static_assert(sizeof(PTVert) == 32, "PTVert struct must be exactly 32 bytes!");

// ============== HELPER FUNCTIONS (extern declarations) ==============
extern void WaitForGpu();
extern void MoveToNextFrame();
extern bool CheckDXRSupport(IDXGIAdapter1* adapter);
extern void DrawTextDirect(const char* text, float x, float y, float r, float g, float b, float a, float scale);

// Material types for Cornell Box scene (same as RT renderer)
enum MaterialType { MAT_DIFFUSE = 0, MAT_MIRROR = 1, MAT_GLASS = 2, MAT_EMISSIVE = 3 };
enum ObjectID {
    OBJ_FLOOR = 0, OBJ_CEILING = 1, OBJ_BACK_WALL = 2, OBJ_LEFT_WALL = 3, OBJ_RIGHT_WALL = 4,
    OBJ_LIGHT = 5, OBJ_CUBE = 6, OBJ_MIRROR = 7, OBJ_GLASS = 8, OBJ_SMALL_CUBE = 9, OBJ_FRONT_WALL = 10
};

// Local static variables for PT renderer
static ID3D12Resource* s_vbStatic = nullptr;
static ID3D12Resource* s_ibStatic = nullptr;
static ID3D12Resource* s_vbCube = nullptr;
static ID3D12Resource* s_ibCube = nullptr;
static ID3D12Resource* s_blasStatic = nullptr;
static ID3D12Resource* s_blasCube = nullptr;
static ID3D12Resource* s_tlasBuffer = nullptr;
static ID3D12Resource* s_scratchBuffer = nullptr;
static ID3D12Resource* s_instanceBuffer = nullptr;
static void* s_instanceMapped = nullptr;
static UINT s_vertCountStatic = 0, s_indCountStatic = 0;
static UINT s_vertCountCube = 0, s_indCountCube = 0;

// Add a quad (two triangles)
static void AddQuad(std::vector<PTVert>& verts, std::vector<UINT>& inds,
    XMFLOAT3 p0, XMFLOAT3 p1, XMFLOAT3 p2, XMFLOAT3 p3,
    XMFLOAT3 normal, UINT objID, UINT matType)
{
    UINT base = (UINT)verts.size();
    PTVert v; v.n = normal; v.objectID = objID; v.materialType = matType;
    v.p = p0; verts.push_back(v);
    v.p = p1; verts.push_back(v);
    v.p = p2; verts.push_back(v);
    v.p = p3; verts.push_back(v);
    inds.push_back(base + 0); inds.push_back(base + 1); inds.push_back(base + 2);
    inds.push_back(base + 0); inds.push_back(base + 2); inds.push_back(base + 3);
}

// Add a box (6 quads)
static void AddBox(std::vector<PTVert>& verts, std::vector<UINT>& inds,
    XMFLOAT3 center, XMFLOAT3 halfSize, UINT objID, UINT matType)
{
    float hx = halfSize.x, hy = halfSize.y, hz = halfSize.z;
    float cx = center.x, cy = center.y, cz = center.z;
    AddQuad(verts, inds, {cx-hx, cy-hy, cz+hz}, {cx+hx, cy-hy, cz+hz}, {cx+hx, cy+hy, cz+hz}, {cx-hx, cy+hy, cz+hz}, {0, 0, 1}, objID, matType);
    AddQuad(verts, inds, {cx+hx, cy-hy, cz-hz}, {cx-hx, cy-hy, cz-hz}, {cx-hx, cy+hy, cz-hz}, {cx+hx, cy+hy, cz-hz}, {0, 0, -1}, objID, matType);
    AddQuad(verts, inds, {cx+hx, cy-hy, cz+hz}, {cx+hx, cy-hy, cz-hz}, {cx+hx, cy+hy, cz-hz}, {cx+hx, cy+hy, cz+hz}, {1, 0, 0}, objID, matType);
    AddQuad(verts, inds, {cx-hx, cy-hy, cz-hz}, {cx-hx, cy-hy, cz+hz}, {cx-hx, cy+hy, cz+hz}, {cx-hx, cy+hy, cz-hz}, {-1, 0, 0}, objID, matType);
    AddQuad(verts, inds, {cx-hx, cy+hy, cz+hz}, {cx+hx, cy+hy, cz+hz}, {cx+hx, cy+hy, cz-hz}, {cx-hx, cy+hy, cz-hz}, {0, 1, 0}, objID, matType);
    AddQuad(verts, inds, {cx-hx, cy-hy, cz-hz}, {cx+hx, cy-hy, cz-hz}, {cx+hx, cy-hy, cz+hz}, {cx-hx, cy-hy, cz+hz}, {0, -1, 0}, objID, matType);
}

// Build static geometry (room, mirror, light) - same as RT renderer
static void BuildStaticGeometry(std::vector<PTVert>& verts, std::vector<UINT>& inds)
{
    verts.clear(); inds.clear();
    const float s = 2.0f; // Room half-size (increased for full camera view)

    // Floor
    AddQuad(verts, inds, {-s, -s, -s}, {s, -s, -s}, {s, -s, s}, {-s, -s, s}, {0, 1, 0}, OBJ_FLOOR, MAT_DIFFUSE);
    // Ceiling
    AddQuad(verts, inds, {-s, s, s}, {s, s, s}, {s, s, -s}, {-s, s, -s}, {0, -1, 0}, OBJ_CEILING, MAT_DIFFUSE);
    // Back wall
    AddQuad(verts, inds, {-s, -s, s}, {-s, s, s}, {s, s, s}, {s, -s, s}, {0, 0, -1}, OBJ_BACK_WALL, MAT_DIFFUSE);
    // Left wall (RED)
    AddQuad(verts, inds, {-s, -s, -s}, {-s, s, -s}, {-s, s, s}, {-s, -s, s}, {1, 0, 0}, OBJ_LEFT_WALL, MAT_DIFFUSE);
    // Right wall (GREEN)
    AddQuad(verts, inds, {s, -s, s}, {s, s, s}, {s, s, -s}, {s, -s, -s}, {-1, 0, 0}, OBJ_RIGHT_WALL, MAT_DIFFUSE);

    // Light (slightly inset from ceiling, bigger for larger room)
    const float ls = 0.5f;
    AddQuad(verts, inds, {-ls, s - 0.01f, -ls}, {ls, s - 0.01f, -ls}, {ls, s - 0.01f, ls}, {-ls, s - 0.01f, ls}, {0, -1, 0}, OBJ_LIGHT, MAT_EMISSIVE);

    // Angled mirror (45 degrees, floor to ceiling, in back-left corner)
    const float mWidth = 0.8f;  // Mirror width
    float mx = -1.2f, mz = 1.2f;  // Position in back-left corner
    float nx = 0.707f, nz = -0.707f;  // Normal facing toward center
    float tx = -nz, tz = nx;  // Tangent along mirror surface
    XMFLOAT3 mNorm = {nx, 0, nz};
    AddQuad(verts, inds,
        {mx - tx * mWidth, -s, mz - tz * mWidth},
        {mx + tx * mWidth, -s, mz + tz * mWidth},
        {mx + tx * mWidth, s - 0.05f, mz + tz * mWidth},  // Full height (floor to ceiling)
        {mx - tx * mWidth, s - 0.05f, mz - tz * mWidth},
        mNorm, OBJ_MIRROR, MAT_MIRROR);

    // Small cube behind glass panel
    const float sc = 0.15f;
    AddBox(verts, inds, {1.5f, -s + sc, 0.5f}, {sc, sc, sc}, OBJ_SMALL_CUBE, MAT_DIFFUSE);

    // Glass panel (right side, towards back)
    const float gw = 0.4f, gh = 0.6f;
    float glassX = 1.2f;   // Near right wall
    float glassZ = 0.5f;   // In the back half
    AddQuad(verts, inds, {glassX, -s, glassZ - gw}, {glassX, -s, glassZ + gw}, {glassX, -s + gh * 2, glassZ + gw}, {glassX, -s + gh * 2, glassZ - gw}, {-1, 0, 0}, OBJ_GLASS, MAT_GLASS);
    AddQuad(verts, inds, {glassX, -s, glassZ + gw}, {glassX, -s, glassZ - gw}, {glassX, -s + gh * 2, glassZ - gw}, {glassX, -s + gh * 2, glassZ + gw}, {1, 0, 0}, OBJ_GLASS, MAT_GLASS);

    // No front wall - camera is inside the room looking at the scene
}

// Build dynamic cube geometry (8 small cubes in 2x2x2 formation, at origin) - same as RT
static void BuildDynamicCubes(std::vector<PTVert>& verts, std::vector<UINT>& inds)
{
    verts.clear(); inds.clear();
    const float smallSize = 0.11f;
    const float spacing = smallSize;  // Cubes touch each other

    int cubeIdx = 0;
    for (int x = 0; x < 2; x++) {
        for (int y = 0; y < 2; y++) {
            for (int z = 0; z < 2; z++) {
                float cx = (x - 0.5f) * spacing * 2;
                float cy = (y - 0.5f) * spacing * 2;
                float cz = (z - 0.5f) * spacing * 2;
                // Use materialType to store cube index for per-cube coloring
                AddBox(verts, inds, {cx, cy, cz}, {smallSize, smallSize, smallSize}, OBJ_CUBE, (UINT)cubeIdx);
                cubeIdx++;
            }
        }
    }
}

// Update cube instance transform (called each frame)
// Not static - accessible from DLSS renderer
void UpdateCubeTransformPT(float time)
{
    if (!s_instanceMapped) return;

    float angleY = time * 1.2f;
    float angleX = time * 0.7f;
    float cosY = cosf(angleY), sinY = sinf(angleY);
    float cosX = cosf(angleX), sinX = sinf(angleX);

    // Combined rotation: RotY * RotX (transposed for TLAS)
    float m00 = cosY, m01 = sinY * sinX, m02 = sinY * cosX;
    float m10 = 0, m11 = cosX, m12 = -sinX;
    float m20 = -sinY, m21 = cosY * sinX, m22 = cosY * cosX;

    // Cube position (centered in larger room)
    float tx = 0.0f, ty = 0.0f, tz = 0.5f;

    D3D12_RAYTRACING_INSTANCE_DESC* instances = (D3D12_RAYTRACING_INSTANCE_DESC*)s_instanceMapped;
    // Instance 1: Cube with rotation (transposed)
    instances[1].Transform[0][0] = m00; instances[1].Transform[0][1] = m10; instances[1].Transform[0][2] = m20; instances[1].Transform[0][3] = tx;
    instances[1].Transform[1][0] = m01; instances[1].Transform[1][1] = m11; instances[1].Transform[1][2] = m21; instances[1].Transform[1][3] = ty;
    instances[1].Transform[2][0] = m02; instances[1].Transform[2][1] = m12; instances[1].Transform[2][2] = m22; instances[1].Transform[2][3] = tz;
}

// Rebuild TLAS after transform update (called from DLSS renderer)
void RebuildTLAS_PT(ID3D12GraphicsCommandList4* cmdListRT)
{
    if (!s_tlasBuffer || !s_instanceBuffer || !s_scratchBuffer) return;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 2;
    tlasInputs.InstanceDescs = s_instanceBuffer->GetGPUVirtualAddress();
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
    tlasBuildDesc.Inputs = tlasInputs;
    tlasBuildDesc.DestAccelerationStructureData = s_tlasBuffer->GetGPUVirtualAddress();
    tlasBuildDesc.SourceAccelerationStructureData = s_tlasBuffer->GetGPUVirtualAddress();
    tlasBuildDesc.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    cmdListRT->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = s_tlasBuffer;
    cmdListRT->ResourceBarrier(1, &uavBarrier);
}

// ============== INITIALIZATION ==============
bool InitD3D12PT(HWND hwnd)
{
    Log("[INFO] Initializing Direct3D 12 with Path Tracing...\n");
    HRESULT hr;

    // Enable debug layer
    ID3D12Debug* debugController = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
        debugController->EnableDebugLayer();
        debugController->Release();
        Log("[INFO] D3D12 Debug layer enabled\n");
    }

    // Get selected adapter
    IDXGIFactory4* factory = nullptr;
    CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    IDXGIAdapter1* adapter = g_gpuList[g_settings.selectedGPU].adapter;
    gpuName = g_gpuList[g_settings.selectedGPU].name;

    // Create DXR-capable device (skip DXR support check)
    hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev12));
    if (FAILED(hr)) { LogHR("CreateDevice", hr); factory->Release(); return false; }

    hr = dev12->QueryInterface(IID_PPV_ARGS(&dev12RT));
    if (FAILED(hr)) { LogHR("QueryInterface Device5", hr); factory->Release(); return false; }
    Log("[INFO] D3D12 Device5 (DXR) created for Path Tracing\n");

    // Command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = dev12->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&cmdQueue));
    if (FAILED(hr)) { LogHR("CreateCommandQueue", hr); factory->Release(); return false; }

    // Swap chain
    IDXGIFactory5* factory5 = nullptr;
    factory->QueryInterface(IID_PPV_ARGS(&factory5));

    BOOL tearingSupport = FALSE;
    if (factory5 && SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearingSupport, sizeof(tearingSupport)))) {
        g_tearingSupported12 = (tearingSupport == TRUE);
    }

    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.Width = W; scd.Height = H;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;  // No UAV - we copy from separate texture
    scd.BufferCount = FRAME_COUNT;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.Flags = g_tearingSupported12 ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    Log("[INFO] Creating swap chain: %ux%u, BufferCount=%u, Tearing=%s\n",
        scd.Width, scd.Height, scd.BufferCount, g_tearingSupported12 ? "YES" : "NO");

    IDXGISwapChain1* swap1 = nullptr;
    hr = factory5->CreateSwapChainForHwnd(cmdQueue, hwnd, &scd, nullptr, nullptr, &swap1);
    if (factory5) factory5->Release();
    factory->Release();
    if (FAILED(hr)) {
        LogHR("CreateSwapChain", hr);
        Log("[ERROR] Swap chain creation failed. Params: Format=%u, Usage=0x%X, SwapEffect=%u\n",
            scd.Format, scd.BufferUsage, scd.SwapEffect);
        return false;
    }
    Log("[INFO] Swap chain created successfully\n");
    swap1->QueryInterface(IID_PPV_ARGS(&swap12));
    swap1->Release();
    frameIndex = swap12->GetCurrentBackBufferIndex();
    Log("[INFO] Initial frame index: %u\n", frameIndex);

    // RTV heap
    Log("[INFO] Creating RTV heap...\n");
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = FRAME_COUNT;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    hr = dev12->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap12));
    if (FAILED(hr)) { LogHR("CreateRTVHeap", hr); return false; }
    rtvDescSize = dev12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Create RTVs
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap12->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < FRAME_COUNT; i++) {
        hr = swap12->GetBuffer(i, IID_PPV_ARGS(&renderTargets12[i]));
        if (FAILED(hr)) { LogHR("GetBuffer", hr); return false; }
        dev12->CreateRenderTargetView(renderTargets12[i], nullptr, rtvHandle);
        rtvHandle.ptr += rtvDescSize;
    }
    Log("[INFO] RTVs created for %u buffers\n", FRAME_COUNT);

    // DSV heap & depth buffer (for text rendering compatibility)
    Log("[INFO] Creating DSV heap and depth buffer...\n");
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    hr = dev12->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap12));
    if (FAILED(hr)) { LogHR("CreateDSVHeap", hr); return false; }

    D3D12_HEAP_PROPERTIES defaultHeap = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC dsDesc = {};
    dsDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    dsDesc.Width = W; dsDesc.Height = H; dsDesc.DepthOrArraySize = 1;
    dsDesc.MipLevels = 1; dsDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dsDesc.SampleDesc.Count = 1;
    dsDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clearVal = {}; clearVal.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; clearVal.DepthStencil.Depth = 1.0f;
    hr = dev12->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &dsDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clearVal, IID_PPV_ARGS(&depthStencil12));
    if (FAILED(hr)) { LogHR("CreateDepthStencil", hr); return false; }
    dev12->CreateDepthStencilView(depthStencil12, nullptr, dsvHeap12->GetCPUDescriptorHandleForHeapStart());
    Log("[INFO] DSV created\n");

    // Command allocators
    Log("[INFO] Creating command allocators...\n");
    for (UINT i = 0; i < FRAME_COUNT; i++) {
        hr = dev12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAlloc[i]));
        if (FAILED(hr)) { LogHR("CreateCommandAllocator", hr); return false; }
    }
    Log("[INFO] %u command allocators created\n", FRAME_COUNT);

    // Fence
    Log("[INFO] Creating fence and event...\n");
    hr = dev12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) { LogHR("CreateFence", hr); return false; }
    fenceValues[0] = fenceValues[1] = fenceValues[2] = 1;
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!fenceEvent) { Log("[ERROR] CreateEvent failed!\n"); return false; }

    // ===== BUILD GEOMETRY (Static + Dynamic Cubes) =====
    Log("[INFO] Building geometry...\n");
    std::vector<PTVert> vertsStatic, vertsCube;
    std::vector<UINT> indsStatic, indsCube;
    BuildStaticGeometry(vertsStatic, indsStatic);
    BuildDynamicCubes(vertsCube, indsCube);
    s_vertCountStatic = (UINT)vertsStatic.size();
    s_indCountStatic = (UINT)indsStatic.size();
    s_vertCountCube = (UINT)vertsCube.size();
    s_indCountCube = (UINT)indsCube.size();
    Log("[INFO] Static: %u verts, %u inds | Cubes: %u verts, %u inds\n",
        s_vertCountStatic, s_indCountStatic, s_vertCountCube, s_indCountCube);

    // Upload buffers
    D3D12_HEAP_PROPERTIES uploadHeap = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_RESOURCE_DESC bufDesc = {}; bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Height = 1; bufDesc.DepthOrArraySize = 1; bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1; bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    void* mapped;

    // Static VB/IB
    UINT vbSizeStatic = s_vertCountStatic * sizeof(PTVert);
    bufDesc.Width = vbSizeStatic;
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_vbStatic));
    s_vbStatic->Map(0, nullptr, &mapped); memcpy(mapped, vertsStatic.data(), vbSizeStatic); s_vbStatic->Unmap(0, nullptr);

    UINT ibSizeStatic = s_indCountStatic * sizeof(UINT);
    bufDesc.Width = ibSizeStatic;
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_ibStatic));
    s_ibStatic->Map(0, nullptr, &mapped); memcpy(mapped, indsStatic.data(), ibSizeStatic); s_ibStatic->Unmap(0, nullptr);

    // Cube VB/IB
    UINT vbSizeCube = s_vertCountCube * sizeof(PTVert);
    bufDesc.Width = vbSizeCube;
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_vbCube));
    s_vbCube->Map(0, nullptr, &mapped); memcpy(mapped, vertsCube.data(), vbSizeCube); s_vbCube->Unmap(0, nullptr);

    UINT ibSizeCube = s_indCountCube * sizeof(UINT);
    bufDesc.Width = ibSizeCube;
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_ibCube));
    s_ibCube->Map(0, nullptr, &mapped); memcpy(mapped, indsCube.data(), ibSizeCube); s_ibCube->Unmap(0, nullptr);

    // Also keep vb12/ib12 pointing to static for shader StructuredBuffer access
    vb12 = s_vbStatic;  // Shader reads vertices from here
    ib12 = s_ibStatic;
    totalVertices12 = s_vertCountStatic;
    totalIndices12 = s_indCountStatic;

    // Path tracing constant buffer
    bufDesc.Width = 256;
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&pathTraceCB));
    pathTraceCB->Map(0, nullptr, &pathTraceCBMapped);
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cbUpload12));
    cbUpload12->Map(0, nullptr, &cbMapped12);

    // Create command list
    dev12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAlloc[0], nullptr, IID_PPV_ARGS(&cmdList));
    cmdList->QueryInterface(IID_PPV_ARGS(&cmdListRT));

    Log("[INFO] Building acceleration structures (2 BLAS + TLAS)...\n");
    D3D12_RESOURCE_DESC asDesc = {}; asDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    asDesc.Height = 1; asDesc.DepthOrArraySize = 1; asDesc.MipLevels = 1;
    asDesc.SampleDesc.Count = 1; asDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    asDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // ===== BLAS 1: Static Geometry =====
    D3D12_RAYTRACING_GEOMETRY_DESC geomStatic = {};
    geomStatic.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geomStatic.Triangles.VertexBuffer.StartAddress = s_vbStatic->GetGPUVirtualAddress();
    geomStatic.Triangles.VertexBuffer.StrideInBytes = sizeof(PTVert);
    geomStatic.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geomStatic.Triangles.VertexCount = s_vertCountStatic;
    geomStatic.Triangles.IndexBuffer = s_ibStatic->GetGPUVirtualAddress();
    geomStatic.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    geomStatic.Triangles.IndexCount = s_indCountStatic;
    geomStatic.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputsStatic = {};
    blasInputsStatic.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputsStatic.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputsStatic.NumDescs = 1;
    blasInputsStatic.pGeometryDescs = &geomStatic;
    blasInputsStatic.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildStatic = {};
    dev12RT->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputsStatic, &prebuildStatic);
    asDesc.Width = prebuildStatic.ResultDataMaxSizeInBytes;
    dev12->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_blasStatic));

    // ===== BLAS 2: Cube Geometry =====
    D3D12_RAYTRACING_GEOMETRY_DESC geomCube = {};
    geomCube.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geomCube.Triangles.VertexBuffer.StartAddress = s_vbCube->GetGPUVirtualAddress();
    geomCube.Triangles.VertexBuffer.StrideInBytes = sizeof(PTVert);
    geomCube.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geomCube.Triangles.VertexCount = s_vertCountCube;
    geomCube.Triangles.IndexBuffer = s_ibCube->GetGPUVirtualAddress();
    geomCube.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    geomCube.Triangles.IndexCount = s_indCountCube;
    geomCube.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputsCube = {};
    blasInputsCube.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputsCube.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputsCube.NumDescs = 1;
    blasInputsCube.pGeometryDescs = &geomCube;
    blasInputsCube.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildCube = {};
    dev12RT->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputsCube, &prebuildCube);
    asDesc.Width = prebuildCube.ResultDataMaxSizeInBytes;
    dev12->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_blasCube));

    // Scratch buffer (large enough for both BLAS and TLAS)
    UINT64 scratchSize = max(prebuildStatic.ScratchDataSizeInBytes, prebuildCube.ScratchDataSizeInBytes);
    scratchSize = max(scratchSize, (UINT64)131072);  // At least 128KB for TLAS
    asDesc.Width = scratchSize;
    dev12->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&s_scratchBuffer));

    // Build BLAS Static
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuildStatic = {};
    blasBuildStatic.Inputs = blasInputsStatic;
    blasBuildStatic.DestAccelerationStructureData = s_blasStatic->GetGPUVirtualAddress();
    blasBuildStatic.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    cmdListRT->BuildRaytracingAccelerationStructure(&blasBuildStatic, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier = {}; uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = s_blasStatic; cmdListRT->ResourceBarrier(1, &uavBarrier);

    // Build BLAS Cube
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuildCube = {};
    blasBuildCube.Inputs = blasInputsCube;
    blasBuildCube.DestAccelerationStructureData = s_blasCube->GetGPUVirtualAddress();
    blasBuildCube.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    cmdListRT->BuildRaytracingAccelerationStructure(&blasBuildCube, 0, nullptr);
    uavBarrier.UAV.pResource = s_blasCube; cmdListRT->ResourceBarrier(1, &uavBarrier);

    // ===== INSTANCE BUFFER (2 instances, persistent mapping) =====
    bufDesc.Width = sizeof(D3D12_RAYTRACING_INSTANCE_DESC) * 2;
    bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_instanceBuffer));
    s_instanceBuffer->Map(0, nullptr, &s_instanceMapped);

    D3D12_RAYTRACING_INSTANCE_DESC* instances = (D3D12_RAYTRACING_INSTANCE_DESC*)s_instanceMapped;
    // Instance 0: Static (identity transform)
    memset(&instances[0], 0, sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
    instances[0].Transform[0][0] = instances[0].Transform[1][1] = instances[0].Transform[2][2] = 1.0f;
    instances[0].InstanceMask = 0xFF;
    instances[0].InstanceID = 0;
    instances[0].AccelerationStructure = s_blasStatic->GetGPUVirtualAddress();
    // Instance 1: Cube (will be updated each frame)
    memset(&instances[1], 0, sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
    instances[1].Transform[0][0] = instances[1].Transform[1][1] = instances[1].Transform[2][2] = 1.0f;
    instances[1].InstanceMask = 0xFF;
    instances[1].InstanceID = 1;
    instances[1].AccelerationStructure = s_blasCube->GetGPUVirtualAddress();
    UpdateCubeTransformPT(0.0f);  // Initial position

    // ===== BUILD TLAS =====
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 2;
    tlasInputs.InstanceDescs = s_instanceBuffer->GetGPUVirtualAddress();
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPrebuild = {};
    dev12RT->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasPrebuild);
    asDesc.Width = tlasPrebuild.ResultDataMaxSizeInBytes;
    dev12->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_tlasBuffer));

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
    tlasBuildDesc.Inputs = tlasInputs;
    tlasBuildDesc.DestAccelerationStructureData = s_tlasBuffer->GetGPUVirtualAddress();
    tlasBuildDesc.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    cmdListRT->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);
    uavBarrier.UAV.pResource = s_tlasBuffer; cmdListRT->ResourceBarrier(1, &uavBarrier);

    // Execute and wait
    cmdList->Close();
    ID3D12CommandList* cmdLists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, cmdLists);
    WaitForGpu();
    Log("[INFO] Acceleration structures built (2 BLAS + TLAS with 2 instances)\n");

    // Update global pointers for shader access
    tlasBuffer = s_tlasBuffer;
    blasBuffer = s_blasStatic;  // For compatibility

    // ===== CREATE PATH TRACING OUTPUT TEXTURE =====
    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = W;
    texDesc.Height = H;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    hr = dev12->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&pathTraceOutput));
    if (FAILED(hr)) { LogHR("CreatePathTraceOutput", hr); return false; }
    Log("[INFO] Path trace output texture created\n");

    // Create denoise temp texture (for ping-pong denoising)
    hr = dev12->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&denoiseTemp));
    if (FAILED(hr)) { LogHR("CreateDenoiseTemp", hr); return false; }
    Log("[INFO] Denoise temp texture created\n");

    // ===== CREATE SRV/UAV HEAP FOR PATH TRACING =====
    // Descriptors: 0=TLAS, 1=Vertices, 2=Indices, 3=PT Output UAV
    //              4=PT Output SRV (for denoise read), 5=DenoiseTemp UAV, 6=DenoiseTemp SRV, 7=PT Output UAV (for denoise write back)
    D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
    srvUavHeapDesc.NumDescriptors = 16;  // Extra for denoise ping-pong
    srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = dev12->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&pathTraceSrvUavHeap));
    if (FAILED(hr)) { LogHR("CreatePathTraceSrvUavHeap", hr); return false; }

    UINT srvUavDescSize = dev12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE heapStart = pathTraceSrvUavHeap->GetCPUDescriptorHandleForHeapStart();

    // Descriptor 0: TLAS SRV (t0)
    Log("[INFO] Creating descriptor 0: TLAS SRV\n");
    D3D12_SHADER_RESOURCE_VIEW_DESC tlasSrvDesc = {};
    tlasSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
    tlasSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    tlasSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    tlasSrvDesc.RaytracingAccelerationStructure.Location = tlasBuffer->GetGPUVirtualAddress();
    dev12->CreateShaderResourceView(nullptr, &tlasSrvDesc, heapStart);

    // Descriptor 1: Vertices SRV (t1) - StructuredBuffer<Vertex> (static geometry only)
    Log("[INFO] Creating descriptor 1: Vertices SRV (stride=%zu, count=%u)\n", sizeof(PTVert), s_vertCountStatic);
    D3D12_SHADER_RESOURCE_VIEW_DESC verticesSrvDesc = {};
    verticesSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
    verticesSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    verticesSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    verticesSrvDesc.Buffer.FirstElement = 0;
    verticesSrvDesc.Buffer.NumElements = s_vertCountStatic;
    verticesSrvDesc.Buffer.StructureByteStride = sizeof(PTVert);
    D3D12_CPU_DESCRIPTOR_HANDLE verticesHandle = heapStart;
    verticesHandle.ptr += srvUavDescSize;
    dev12->CreateShaderResourceView(s_vbStatic, &verticesSrvDesc, verticesHandle);

    // Descriptor 2: Indices SRV (t2) - StructuredBuffer<uint> (static geometry only)
    Log("[INFO] Creating descriptor 2: Indices SRV (count=%u)\n", s_indCountStatic);
    D3D12_SHADER_RESOURCE_VIEW_DESC indicesSrvDesc = {};
    indicesSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
    indicesSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    indicesSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    indicesSrvDesc.Buffer.FirstElement = 0;
    indicesSrvDesc.Buffer.NumElements = s_indCountStatic;
    indicesSrvDesc.Buffer.StructureByteStride = sizeof(UINT);
    D3D12_CPU_DESCRIPTOR_HANDLE indicesHandle = heapStart;
    indicesHandle.ptr += 2 * srvUavDescSize;
    dev12->CreateShaderResourceView(s_ibStatic, &indicesSrvDesc, indicesHandle);

    // Descriptor 3: Output UAV (u0) - RWTexture2D<float4>
    Log("[INFO] Creating descriptor 3: Output UAV\n");
    D3D12_UNORDERED_ACCESS_VIEW_DESC outputUavDesc = {};
    outputUavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    outputUavDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    D3D12_CPU_DESCRIPTOR_HANDLE outputHandle = heapStart;
    outputHandle.ptr += 3 * srvUavDescSize;
    dev12->CreateUnorderedAccessView(pathTraceOutput, nullptr, &outputUavDesc, outputHandle);

    // ===== DENOISE DESCRIPTORS =====
    // Descriptor 4: PT Output as SRV (for denoise to read)
    D3D12_SHADER_RESOURCE_VIEW_DESC texSrvDesc = {};
    texSrvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    texSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    texSrvDesc.Texture2D.MipLevels = 1;
    D3D12_CPU_DESCRIPTOR_HANDLE h4 = heapStart;
    h4.ptr += 4 * srvUavDescSize;
    dev12->CreateShaderResourceView(pathTraceOutput, &texSrvDesc, h4);

    // Descriptor 5: DenoiseTemp as UAV (for denoise to write)
    D3D12_CPU_DESCRIPTOR_HANDLE h5 = heapStart;
    h5.ptr += 5 * srvUavDescSize;
    dev12->CreateUnorderedAccessView(denoiseTemp, nullptr, &outputUavDesc, h5);

    // Descriptor 6: DenoiseTemp as SRV (for second pass to read)
    D3D12_CPU_DESCRIPTOR_HANDLE h6 = heapStart;
    h6.ptr += 6 * srvUavDescSize;
    dev12->CreateShaderResourceView(denoiseTemp, &texSrvDesc, h6);

    // Descriptor 7: PT Output as UAV again (for second pass to write back)
    D3D12_CPU_DESCRIPTOR_HANDLE h7 = heapStart;
    h7.ptr += 7 * srvUavDescSize;
    dev12->CreateUnorderedAccessView(pathTraceOutput, nullptr, &outputUavDesc, h7);

    Log("[INFO] Path tracing and denoise descriptors created\n");

    // ===== CREATE PATH TRACING ROOT SIGNATURE =====
    // Root parameters:
    // 0: CBV (b0) - PathTraceCB
    // 1: Descriptor table (t0: TLAS, t1: Normals, t2: Indices, u0: Output)

    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    // SRVs: t0=TLAS, t1=Vertices, t2=Indices
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    // UAVs: u0=Output
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 3;  // After 3 SRVs

    D3D12_ROOT_PARAMETER rootParams[2] = {};
    // CBV at root parameter 0
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    // Descriptor table at root parameter 1
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[1].DescriptorTable.NumDescriptorRanges = 2;
    rootParams[1].DescriptorTable.pDescriptorRanges = ranges;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = rootParams;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob* sigBlob = nullptr, *errBlob = nullptr;
    hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { Log("[ERROR] PT Root sig: %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        return false;
    }
    hr = dev12->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&pathTraceRootSig));
    sigBlob->Release();
    if (FAILED(hr)) { LogHR("CreatePathTraceRootSig", hr); return false; }
    Log("[INFO] Path tracing root signature created\n");

    // ===== COMPILE PATH TRACING COMPUTE SHADER =====
    Log("[INFO] Compiling path tracing compute shader with DXC...\n");

    if (!g_dxcModule) {
        g_dxcModule = LoadLibraryW(L"dxcompiler.dll");
        if (!g_dxcModule) {
            Log("[ERROR] Failed to load dxcompiler.dll!\n");
            return false;
        }
        g_DxcCreateInstance = (DxcCreateInstanceProc)GetProcAddress(g_dxcModule, "DxcCreateInstance");
        if (!g_DxcCreateInstance) { FreeLibrary(g_dxcModule); g_dxcModule = nullptr; return false; }
        Log("[INFO] DXC loaded\n");
    }

    IDxcUtils* dxcUtils = nullptr;
    IDxcCompiler3* dxcCompiler = nullptr;
    g_DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils));
    g_DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));

    IDxcBlobEncoding* srcBlob = nullptr;
    dxcUtils->CreateBlob(g_ptShaderCode, (UINT)strlen(g_ptShaderCode), CP_UTF8, &srcBlob);
    DxcBuffer srcBuf = { srcBlob->GetBufferPointer(), srcBlob->GetBufferSize(), 0 };

    // Compile with cs_6_5 for RayQuery support
    LPCWSTR csArgs[] = { L"-E", L"PathTraceCS", L"-T", L"cs_6_5", L"-Zi", L"-Od" };

    IDxcResult* csRes = nullptr;
    hr = dxcCompiler->Compile(&srcBuf, csArgs, _countof(csArgs), nullptr, IID_PPV_ARGS(&csRes));

    HRESULT csStatus;
    csRes->GetStatus(&csStatus);

    if (FAILED(csStatus)) {
        IDxcBlobUtf8* err = nullptr;
        csRes->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&err), nullptr);
        Log("[CS ERROR] %s\n", err->GetStringPointer());
        err->Release();
        srcBlob->Release(); csRes->Release(); dxcCompiler->Release(); dxcUtils->Release();
        return false;
    }

    IDxcBlob* csBlob = nullptr;
    csRes->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&csBlob), nullptr);
    Log("[INFO] Path tracing compute shader compiled (size: %zu)\n", csBlob->GetBufferSize());

    srcBlob->Release(); csRes->Release();
    dxcCompiler->Release(); dxcUtils->Release();

    // ===== CREATE PATH TRACING COMPUTE PSO =====
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.pRootSignature = pathTraceRootSig;
    psoDesc.CS = { csBlob->GetBufferPointer(), csBlob->GetBufferSize() };

    hr = dev12->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pathTracePSO));
    csBlob->Release();

    if (FAILED(hr)) {
        LogHR("CreatePathTracePSO", hr);
        // Get detailed error from debug layer
        ID3D12InfoQueue* infoQueue = nullptr;
        if (SUCCEEDED(dev12->QueryInterface(IID_PPV_ARGS(&infoQueue)))) {
            UINT64 numMsgs = infoQueue->GetNumStoredMessages();
            Log("[DEBUG] D3D12 Info Queue has %llu messages:\n", numMsgs);
            for (UINT64 i = 0; i < numMsgs && i < 20; i++) {
                SIZE_T msgLen = 0;
                infoQueue->GetMessage(i, nullptr, &msgLen);
                if (msgLen > 0) {
                    D3D12_MESSAGE* msg = (D3D12_MESSAGE*)malloc(msgLen);
                    if (msg && SUCCEEDED(infoQueue->GetMessage(i, msg, &msgLen))) {
                        Log("[D3D12 %d] %s\n", msg->Severity, msg->pDescription);
                    }
                    free(msg);
                }
            }
            infoQueue->Release();
        }
        return false;
    }
    Log("[INFO] Path tracing compute PSO created\n");

    // ===== CREATE DENOISE ROOT SIGNATURE =====
    // Root params: 0=CBV (DenoiseCB), 1=SRV (input texture), 2=UAV (output texture)
    D3D12_ROOT_PARAMETER denoiseRootParams[3] = {};
    denoiseRootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    denoiseRootParams[0].Descriptor.ShaderRegister = 0;
    denoiseRootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_DESCRIPTOR_RANGE denoiseSrvRange = {};
    denoiseSrvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    denoiseSrvRange.NumDescriptors = 1;
    denoiseSrvRange.BaseShaderRegister = 0;
    denoiseRootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    denoiseRootParams[1].DescriptorTable.NumDescriptorRanges = 1;
    denoiseRootParams[1].DescriptorTable.pDescriptorRanges = &denoiseSrvRange;
    denoiseRootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_DESCRIPTOR_RANGE denoiseUavRange = {};
    denoiseUavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    denoiseUavRange.NumDescriptors = 1;
    denoiseUavRange.BaseShaderRegister = 0;
    denoiseRootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    denoiseRootParams[2].DescriptorTable.NumDescriptorRanges = 1;
    denoiseRootParams[2].DescriptorTable.pDescriptorRanges = &denoiseUavRange;
    denoiseRootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC denoiseRsDesc = {};
    denoiseRsDesc.NumParameters = 3;
    denoiseRsDesc.pParameters = denoiseRootParams;

    sigBlob = nullptr; errBlob = nullptr;
    hr = D3D12SerializeRootSignature(&denoiseRsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { Log("[ERROR] Denoise root sig: %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        return false;
    }
    hr = dev12->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&denoiseRootSig));
    sigBlob->Release();
    if (FAILED(hr)) { LogHR("CreateDenoiseRootSig", hr); return false; }
    Log("[INFO] Denoise root signature created\n");

    // ===== COMPILE DENOISE SHADER =====
    Log("[INFO] Compiling denoise compute shader...\n");
    IDxcBlobEncoding* denoiseSrcBlob = nullptr;
    dxcUtils = nullptr; dxcCompiler = nullptr;
    g_DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils));
    g_DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler));

    dxcUtils->CreateBlob(g_ptDenoiseShaderCode, (UINT)strlen(g_ptDenoiseShaderCode), CP_UTF8, &denoiseSrcBlob);
    DxcBuffer denoiseSrcBuf = { denoiseSrcBlob->GetBufferPointer(), denoiseSrcBlob->GetBufferSize(), 0 };

    LPCWSTR denoiseArgs[] = { L"-E", L"DenoiseCS", L"-T", L"cs_6_0" };
    IDxcResult* denoiseRes = nullptr;
    hr = dxcCompiler->Compile(&denoiseSrcBuf, denoiseArgs, _countof(denoiseArgs), nullptr, IID_PPV_ARGS(&denoiseRes));

    HRESULT denoiseStatus;
    denoiseRes->GetStatus(&denoiseStatus);
    if (FAILED(denoiseStatus)) {
        IDxcBlobUtf8* err = nullptr;
        denoiseRes->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&err), nullptr);
        Log("[DENOISE CS ERROR] %s\n", err->GetStringPointer());
        err->Release();
        denoiseSrcBlob->Release(); denoiseRes->Release(); dxcCompiler->Release(); dxcUtils->Release();
        return false;
    }

    IDxcBlob* denoiseBlob = nullptr;
    denoiseRes->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&denoiseBlob), nullptr);
    Log("[INFO] Denoise shader compiled (size: %zu)\n", denoiseBlob->GetBufferSize());

    denoiseSrcBlob->Release(); denoiseRes->Release();
    dxcCompiler->Release(); dxcUtils->Release();

    // ===== CREATE DENOISE PSO =====
    D3D12_COMPUTE_PIPELINE_STATE_DESC denoisePsoDesc = {};
    denoisePsoDesc.pRootSignature = denoiseRootSig;
    denoisePsoDesc.CS = { denoiseBlob->GetBufferPointer(), denoiseBlob->GetBufferSize() };

    hr = dev12->CreateComputePipelineState(&denoisePsoDesc, IID_PPV_ARGS(&denoisePSO));
    denoiseBlob->Release();
    if (FAILED(hr)) { LogHR("CreateDenoisePSO", hr); return false; }
    Log("[INFO] Denoise PSO created\n");

    // ===== CREATE DENOISE CONSTANT BUFFER =====
    bufDesc.Width = 256;
    hr = dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&denoiseCB));
    if (FAILED(hr)) { LogHR("CreateDenoiseCB", hr); return false; }
    denoiseCB->Map(0, nullptr, &denoiseCBMapped);
    Log("[INFO] Denoise constant buffer created\n");

    // Reset command list for rendering
    cmdAlloc[0]->Reset();
    cmdList->Reset(cmdAlloc[0], nullptr);
    cmdList->Close();

    // Initialize text rendering (shared with base D3D12 renderer)
    if (!InitGPUText12()) {
        Log("[ERROR] Failed to initialize text rendering for Path Tracing!\n");
        return false;
    }

    Log("[INFO] D3D12 + Path Tracing initialization complete\n");
    return true;
}

// ============== RENDER ==============
void RenderD3D12PT()
{
    cmdAlloc[frameIndex]->Reset();
    cmdList->Reset(cmdAlloc[frameIndex], nullptr);  // Start with no PSO - we'll set compute PSO later

    // Update constant buffer
    LARGE_INTEGER nowTime;
    QueryPerformanceCounter(&nowTime);
    float t = (float)(nowTime.QuadPart - g_startTime.QuadPart) / g_perfFreq.QuadPart;

    // ===== UPDATE CUBE TRANSFORM AND REBUILD TLAS =====
    UpdateCubeTransformPT(t);

    // Rebuild TLAS with updated instance transform (using UPDATE flag for fast rebuild)
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 2;
    tlasInputs.InstanceDescs = s_instanceBuffer->GetGPUVirtualAddress();
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
    tlasBuildDesc.Inputs = tlasInputs;
    tlasBuildDesc.DestAccelerationStructureData = s_tlasBuffer->GetGPUVirtualAddress();
    tlasBuildDesc.SourceAccelerationStructureData = s_tlasBuffer->GetGPUVirtualAddress();  // Update in place
    tlasBuildDesc.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    cmdListRT->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = s_tlasBuffer;
    cmdListRT->ResourceBarrier(1, &uavBarrier);

    // Build inverse matrices for camera (looking at room from outside)
    XMMATRIX view = XMMatrixLookAtLH(
        XMVectorSet(0, 0, -3.5f, 1),  // Eye position (further back)
        XMVectorSet(0, 0, 1, 1),      // Look at back wall
        XMVectorSet(0, 1, 0, 0)       // Up
    );
    XMMATRIX proj = XMMatrixPerspectiveFovLH(XM_PI / 3.0f, (float)W / H, 0.1f, 100.0f);  // 60° FOV
    XMMATRIX invView = XMMatrixInverse(nullptr, view);
    XMMATRIX invProj = XMMatrixInverse(nullptr, proj);

    PathTraceCBData cbData;
    XMStoreFloat4x4(&cbData.InvView, XMMatrixTranspose(invView));
    XMStoreFloat4x4(&cbData.InvProj, XMMatrixTranspose(invProj));
    cbData.Time = t;
    cbData.FrameCount = g_frameCount++;
    cbData.Width = W;
    cbData.Height = H;
    memcpy(pathTraceCBMapped, &cbData, sizeof(cbData));

    // Also update regular CB for text rendering
    CB textCbData = { t };
    memcpy(cbMapped12, &textCbData, sizeof(CB));

    // ===== PATH TRACING DISPATCH =====
    cmdList->SetPipelineState(pathTracePSO);
    cmdList->SetComputeRootSignature(pathTraceRootSig);
    cmdList->SetComputeRootConstantBufferView(0, pathTraceCB->GetGPUVirtualAddress());

    ID3D12DescriptorHeap* heaps[] = { pathTraceSrvUavHeap };
    cmdList->SetDescriptorHeaps(1, heaps);
    cmdList->SetComputeRootDescriptorTable(1, pathTraceSrvUavHeap->GetGPUDescriptorHandleForHeapStart());

    // Dispatch compute shader (8x8 thread groups)
    UINT groupsX = (W + 7) / 8;
    UINT groupsY = (H + 7) / 8;
    cmdList->Dispatch(groupsX, groupsY, 1);

    // ===== COPY RAW PATH TRACED OUTPUT TO BACKBUFFER (no denoising) =====
    D3D12_RESOURCE_BARRIER barriers[2] = {};
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = renderTargets12[frameIndex];
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = pathTraceOutput;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(2, barriers);

    // Copy raw path traced output to backbuffer
    cmdList->CopyResource(renderTargets12[frameIndex], pathTraceOutput);

    // Transition for text rendering
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    cmdList->ResourceBarrier(2, barriers);

    // ===== TEXT OVERLAY =====
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap12->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += frameIndex * rtvDescSize;

    // Update text if FPS changed
    if (fps != g_cachedFps || g_textNeedsRebuild) {
        g_cachedFps = fps;
        g_textNeedsRebuild = false;

        static char gpuNameA[128] = {0};
        if (gpuNameA[0] == 0) {
            size_t converted;
            wcstombs_s(&converted, gpuNameA, sizeof(gpuNameA), gpuName.c_str(), _TRUNCATE);
        }

        char infoText[512];
        sprintf_s(infoText,
            "API: D3D12 + Path Tracing\n"
            "GPU: %s\n"
            "FPS: %d\n"
            "Triangles: %u\n"
            "Resolution: %ux%u\n"
            "Rays: 1 SPP | Bounces: 3",
            gpuNameA, fps, totalIndices12 / 3, W, H);

        g_textVertCount = 0;
        DrawTextDirect(infoText, 12.0f, 12.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.5f);
        DrawTextDirect(infoText, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.5f);
        memcpy(textVbMapped12, g_textVerts, g_textVertCount * sizeof(TextVert));
    }

    // Draw text
    if (g_textVertCount > 0) {
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        cmdList->SetPipelineState(textPso);
        cmdList->SetGraphicsRootSignature(textRootSig12);

        ID3D12DescriptorHeap* textHeaps[] = { srvHeap12 };
        cmdList->SetDescriptorHeaps(1, textHeaps);
        cmdList->SetGraphicsRootDescriptorTable(0, srvHeap12->GetGPUDescriptorHandleForHeapStart());

        D3D12_VIEWPORT vp = { 0, 0, (float)W, (float)H, 0, 1 };
        D3D12_RECT scissor = { 0, 0, (LONG)W, (LONG)H };
        cmdList->RSSetViewports(1, &vp);
        cmdList->RSSetScissorRects(1, &scissor);

        cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmdList->IASetVertexBuffers(0, 1, &textVbView12);
        cmdList->IASetIndexBuffer(nullptr);
        cmdList->DrawInstanced(g_textVertCount, 1, 0, 0);
    }

    // Transition to present
    D3D12_RESOURCE_BARRIER presentBarrier = {};
    presentBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    presentBarrier.Transition.pResource = renderTargets12[frameIndex];
    presentBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    presentBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    presentBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &presentBarrier);

    cmdList->Close();
    ID3D12CommandList* cmdLists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, cmdLists);

    UINT presentFlags = g_tearingSupported12 ? DXGI_PRESENT_ALLOW_TEARING : 0;
    swap12->Present(0, presentFlags);
    MoveToNextFrame();
}

// ============== CLEANUP ==============
void CleanupD3D12PT()
{
    WaitForGpu();

    // Path tracing resources
    if (pathTracePSO) pathTracePSO->Release();
    if (pathTraceRootSig) pathTraceRootSig->Release();
    if (pathTraceSrvUavHeap) pathTraceSrvUavHeap->Release();
    if (pathTraceOutput) pathTraceOutput->Release();
    if (pathTraceCB) pathTraceCB->Release();

    // Denoise resources
    if (denoisePSO) denoisePSO->Release();
    if (denoiseRootSig) denoiseRootSig->Release();
    if (denoiseTemp) denoiseTemp->Release();
    if (denoiseCB) denoiseCB->Release();

    // RT resources (local static)
    if (s_instanceBuffer) { s_instanceBuffer->Unmap(0, nullptr); s_instanceBuffer->Release(); s_instanceBuffer = nullptr; }
    if (s_tlasBuffer) { s_tlasBuffer->Release(); s_tlasBuffer = nullptr; }
    if (s_blasStatic) { s_blasStatic->Release(); s_blasStatic = nullptr; }
    if (s_blasCube) { s_blasCube->Release(); s_blasCube = nullptr; }
    if (s_scratchBuffer) { s_scratchBuffer->Release(); s_scratchBuffer = nullptr; }
    if (s_vbStatic) { s_vbStatic->Release(); s_vbStatic = nullptr; }
    if (s_ibStatic) { s_ibStatic->Release(); s_ibStatic = nullptr; }
    if (s_vbCube) { s_vbCube->Release(); s_vbCube = nullptr; }
    if (s_ibCube) { s_ibCube->Release(); s_ibCube = nullptr; }
    s_instanceMapped = nullptr;

    // RT resources (global, for compatibility)
    if (scratchBuffer) scratchBuffer->Release();
    if (instanceBuffer) instanceBuffer->Release();
    if (tlasBuffer) tlasBuffer->Release();
    if (blasBuffer) blasBuffer->Release();
    if (cmdListRT) cmdListRT->Release();
    if (dev12RT) dev12RT->Release();

    // Text resources
    if (textVB12) textVB12->Release();
    if (fontTex12) fontTex12->Release();
    if (textPso) textPso->Release();
    if (textRootSig12) textRootSig12->Release();
    if (srvHeap12) srvHeap12->Release();

    // Main resources
    if (fenceEvent) CloseHandle(fenceEvent);
    if (fence) fence->Release();
    if (cbUpload12) cbUpload12->Release();
    if (ib12) ib12->Release();
    if (vb12) vb12->Release();
    if (cmdList) cmdList->Release();
    for (UINT i = 0; i < FRAME_COUNT; i++) {
        if (cmdAlloc[i]) cmdAlloc[i]->Release();
        if (renderTargets12[i]) renderTargets12[i]->Release();
    }
    if (depthStencil12) depthStencil12->Release();
    if (dsvHeap12) dsvHeap12->Release();
    if (rtvHeap12) rtvHeap12->Release();
    if (swap12) swap12->Release();
    if (cmdQueue) cmdQueue->Release();
    if (dev12) dev12->Release();

    // DXC module
    if (g_dxcModule) {
        FreeLibrary(g_dxcModule);
        g_dxcModule = nullptr;
        g_DxcCreateInstance = nullptr;
    }
}
