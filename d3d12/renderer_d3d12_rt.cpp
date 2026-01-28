// ============== D3D12 + DXR RAY TRACING RENDERER ==============
// Cornell Box with ray-traced shadows, mirror, glass
// Written from scratch - completely independent from base D3D12

#include "../common.h"
#include "d3d12_shared.h"
#include "renderer_d3d12.h"
#include "../shaders/rt_cornell_shaders.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <dxcapi.h>
#include <vector>
#include <string>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;

// ============== SHARED FUNCTION (used by PT renderer too) ==============
bool CheckDXRSupport(IDXGIAdapter1* adapter) {
    (void)adapter;  // Skip RT support check
    return true;
}

// ============== LOCAL TYPES (unique names to avoid ODR violations) ==============
// Material types
enum RTMaterial { RT_MAT_DIFFUSE = 0, RT_MAT_MIRROR = 1, RT_MAT_GLASS = 2, RT_MAT_EMISSIVE = 3 };

// Object IDs
enum RTObjectID {
    RT_OBJ_FLOOR = 0, RT_OBJ_CEILING = 1, RT_OBJ_BACK_WALL = 2,
    RT_OBJ_LEFT_WALL = 3, RT_OBJ_RIGHT_WALL = 4, RT_OBJ_LIGHT = 5,
    RT_OBJ_CUBE = 6, RT_OBJ_MIRROR = 7, RT_OBJ_GLASS = 8, RT_OBJ_SMALL_CUBE = 9,
    RT_OBJ_FRONT_WALL = 10
};

// Vertex structure - UNIQUE NAME to prevent ODR violation with std::vector
#pragma pack(push, 1)
struct RTVert {
    XMFLOAT3 pos;
    XMFLOAT3 norm;
    UINT objectID;
    UINT materialType;
};
#pragma pack(pop)
static_assert(sizeof(RTVert) == 32, "RTVert must be 32 bytes");

// Constant buffer with DXR settings - MATCHES SHADER cbuffer SceneCB
// Feature enable flags are now compile-time #defines, not runtime values
struct alignas(256) RTCB {
    float time;
    float shadowSoftness;
    int shadowSamples;
    int debugMode;      // 0=normal, 1=objID, 2=normals, 3=reflectDir, 4=shadows, 5=UV, 6=depth

    float reflectionStrength;
    float aoRadius;
    float aoStrength;
    int aoSamples;

    int giBounces;
    float giStrength;
    float denoiseBlendFactor;
    int _padding;
};

// ============== SHADER FEATURE FLAGS (for compile-time #defines) ==============
// These determine which #ifdef blocks are active in the shader
struct ShaderFeatures {
    bool useRayQuery;   // SM 6.5 with RayQuery, or SM 6.0 compatible mode
    bool shadows;
    bool softShadows;
    bool rtLighting;
    bool ao;
    bool gi;
    bool reflections;
    bool temporalDenoise;

    bool operator==(const ShaderFeatures& other) const {
        return useRayQuery == other.useRayQuery &&
               shadows == other.shadows &&
               softShadows == other.softShadows &&
               rtLighting == other.rtLighting &&
               ao == other.ao &&
               gi == other.gi &&
               reflections == other.reflections &&
               temporalDenoise == other.temporalDenoise;
    }
    bool operator!=(const ShaderFeatures& other) const { return !(*this == other); }
};

static ShaderFeatures s_compiledFeatures = {};

// ============== LOCAL STATIC RESOURCES ==============
// Device and command objects
static ID3D12Device5* s_device = nullptr;
static ID3D12CommandQueue* s_cmdQueue = nullptr;
static ID3D12CommandAllocator* s_cmdAlloc[3] = {};
static ID3D12GraphicsCommandList4* s_cmdList = nullptr;
static IDXGISwapChain3* s_swapChain = nullptr;

// Render targets
static ID3D12DescriptorHeap* s_rtvHeap = nullptr;
static ID3D12DescriptorHeap* s_dsvHeap = nullptr;
static ID3D12Resource* s_renderTargets[3] = {};
static ID3D12Resource* s_depthStencil = nullptr;
static UINT s_rtvDescSize = 0;
static UINT s_frameIndex = 0;

// Temporal denoising history buffer
static ID3D12Resource* s_historyBuffer = nullptr;
static bool s_historyValid = false;  // First frame has no history

// Synchronization
static ID3D12Fence* s_fence = nullptr;
static UINT64 s_fenceValues[3] = {};
static HANDLE s_fenceEvent = nullptr;

// Geometry buffers (LOCAL - not shared!)
static ID3D12Resource* s_vertexBuffer = nullptr;
static ID3D12Resource* s_indexBuffer = nullptr;
static D3D12_VERTEX_BUFFER_VIEW s_vbView = {};
static D3D12_INDEX_BUFFER_VIEW s_ibView = {};
static UINT s_indexCount = 0;
static UINT s_vertexCount = 0;

// Constant buffer
static ID3D12Resource* s_constantBuffer = nullptr;
static void* s_cbMapped = nullptr;

// Ray tracing resources - Static geometry (room)
static ID3D12Resource* s_blasBufferStatic = nullptr;
static ID3D12Resource* s_vertexBufferStatic = nullptr;
static ID3D12Resource* s_indexBufferStatic = nullptr;
static UINT s_vertexCountStatic = 0;
static UINT s_indexCountStatic = 0;

// Ray tracing resources - Dynamic geometry (cube)
static ID3D12Resource* s_blasBufferCube = nullptr;
static ID3D12Resource* s_vertexBufferCube = nullptr;
static ID3D12Resource* s_indexBufferCube = nullptr;
static UINT s_vertexCountCube = 0;
static UINT s_indexCountCube = 0;
static D3D12_VERTEX_BUFFER_VIEW s_vbViewCube = {};
static D3D12_INDEX_BUFFER_VIEW s_ibViewCube = {};

// TLAS and instances
static ID3D12Resource* s_tlasBuffer = nullptr;
static ID3D12Resource* s_scratchBuffer = nullptr;
static ID3D12Resource* s_instanceBuffer = nullptr;
static void* s_instanceMapped = nullptr;  // Persistent mapping for runtime updates
static UINT64 s_tlasScratchSize = 0;

// Pipeline
static ID3D12RootSignature* s_rootSig = nullptr;
static ID3D12PipelineState* s_pso = nullptr;
static ID3D12DescriptorHeap* s_srvHeap = nullptr;

// Text rendering
static ID3D12RootSignature* s_textRootSig = nullptr;
static ID3D12PipelineState* s_textPso = nullptr;
static ID3D12DescriptorHeap* s_textSrvHeap = nullptr;
static ID3D12Resource* s_fontTexture = nullptr;
static ID3D12Resource* s_textVB = nullptr;
static D3D12_VERTEX_BUFFER_VIEW s_textVBView = {};
static void* s_textVBMapped = nullptr;

// Text vertex cache
static TextVert s_textVerts[6000];
static UINT s_textVertCount = 0;
static int s_cachedFps = -1;

// GPU info
static std::wstring s_gpuName;

// ============== HELPER: Wait for GPU ==============
static void WaitForGpuRT() {
    if (!s_cmdQueue || !s_fence || !s_fenceEvent) return;
    const UINT64 fv = s_fenceValues[s_frameIndex];
    s_cmdQueue->Signal(s_fence, fv);
    if (s_fence->GetCompletedValue() < fv) {
        s_fence->SetEventOnCompletion(fv, s_fenceEvent);
        WaitForSingleObject(s_fenceEvent, INFINITE);
    }
    s_fenceValues[s_frameIndex]++;
}

// ============== HELPER: Move to next frame ==============
static void MoveToNextFrameRT() {
    const UINT64 currentFenceValue = s_fenceValues[s_frameIndex];
    s_cmdQueue->Signal(s_fence, currentFenceValue);
    s_frameIndex = s_swapChain->GetCurrentBackBufferIndex();
    if (s_fence->GetCompletedValue() < s_fenceValues[s_frameIndex]) {
        s_fence->SetEventOnCompletion(s_fenceValues[s_frameIndex], s_fenceEvent);
        WaitForSingleObject(s_fenceEvent, INFINITE);
    }
    s_fenceValues[s_frameIndex] = currentFenceValue + 1;
}

// ============== GEOMETRY BUILDING ==============
static void AddQuad(std::vector<RTVert>& verts, std::vector<UINT>& inds,
    XMFLOAT3 p0, XMFLOAT3 p1, XMFLOAT3 p2, XMFLOAT3 p3,
    XMFLOAT3 normal, UINT objID, UINT matType)
{
    UINT base = (UINT)verts.size();
    RTVert v = {};
    v.norm = normal;
    v.objectID = objID;
    v.materialType = matType;

    v.pos = p0; verts.push_back(v);
    v.pos = p1; verts.push_back(v);
    v.pos = p2; verts.push_back(v);
    v.pos = p3; verts.push_back(v);

    // Two triangles: 0-1-2, 0-2-3
    inds.push_back(base + 0); inds.push_back(base + 1); inds.push_back(base + 2);
    inds.push_back(base + 0); inds.push_back(base + 2); inds.push_back(base + 3);
}

static void AddBox(std::vector<RTVert>& verts, std::vector<UINT>& inds,
    XMFLOAT3 center, XMFLOAT3 halfSize, UINT objID, UINT matType)
{
    float cx = center.x, cy = center.y, cz = center.z;
    float hx = halfSize.x, hy = halfSize.y, hz = halfSize.z;

    // Front face (+Z)
    AddQuad(verts, inds,
        {cx-hx, cy-hy, cz+hz}, {cx+hx, cy-hy, cz+hz},
        {cx+hx, cy+hy, cz+hz}, {cx-hx, cy+hy, cz+hz},
        {0, 0, 1}, objID, matType);
    // Back face (-Z)
    AddQuad(verts, inds,
        {cx+hx, cy-hy, cz-hz}, {cx-hx, cy-hy, cz-hz},
        {cx-hx, cy+hy, cz-hz}, {cx+hx, cy+hy, cz-hz},
        {0, 0, -1}, objID, matType);
    // Right face (+X)
    AddQuad(verts, inds,
        {cx+hx, cy-hy, cz+hz}, {cx+hx, cy-hy, cz-hz},
        {cx+hx, cy+hy, cz-hz}, {cx+hx, cy+hy, cz+hz},
        {1, 0, 0}, objID, matType);
    // Left face (-X)
    AddQuad(verts, inds,
        {cx-hx, cy-hy, cz-hz}, {cx-hx, cy-hy, cz+hz},
        {cx-hx, cy+hy, cz+hz}, {cx-hx, cy+hy, cz-hz},
        {-1, 0, 0}, objID, matType);
    // Top face (+Y)
    AddQuad(verts, inds,
        {cx-hx, cy+hy, cz+hz}, {cx+hx, cy+hy, cz+hz},
        {cx+hx, cy+hy, cz-hz}, {cx-hx, cy+hy, cz-hz},
        {0, 1, 0}, objID, matType);
    // Bottom face (-Y)
    AddQuad(verts, inds,
        {cx-hx, cy-hy, cz-hz}, {cx+hx, cy-hy, cz-hz},
        {cx+hx, cy-hy, cz+hz}, {cx-hx, cy-hy, cz+hz},
        {0, -1, 0}, objID, matType);
}

static void BuildCornellBox(std::vector<RTVert>& verts, std::vector<UINT>& inds) {
    verts.clear();
    inds.clear();
    verts.reserve(150);
    inds.reserve(300);

    const float s = 1.0f; // Room half-size

    // Floor (grey, normal up)
    AddQuad(verts, inds,
        {-s, -s, -s}, {s, -s, -s}, {s, -s, s}, {-s, -s, s},
        {0, 1, 0}, RT_OBJ_FLOOR, RT_MAT_DIFFUSE);

    // Ceiling (white, normal down)
    AddQuad(verts, inds,
        {-s, s, s}, {s, s, s}, {s, s, -s}, {-s, s, -s},
        {0, -1, 0}, RT_OBJ_CEILING, RT_MAT_DIFFUSE);

    // Back wall (grey like floor, normal +Z towards camera)
    AddQuad(verts, inds,
        {-s, -s, s}, {s, -s, s}, {s, s, s}, {-s, s, s},
        {0, 0, -1}, RT_OBJ_BACK_WALL, RT_MAT_DIFFUSE);

    // Left wall (RED, normal +X)
    AddQuad(verts, inds,
        {-s, -s, s}, {-s, s, s}, {-s, s, -s}, {-s, -s, -s},
        {1, 0, 0}, RT_OBJ_LEFT_WALL, RT_MAT_DIFFUSE);

    // Right wall (GREEN, normal -X)
    AddQuad(verts, inds,
        {s, -s, -s}, {s, s, -s}, {s, s, s}, {s, -s, s},
        {-1, 0, 0}, RT_OBJ_RIGHT_WALL, RT_MAT_DIFFUSE);

    // Ceiling light (emissive quad)
    const float ls = 0.3f;
    AddQuad(verts, inds,
        {-ls, s - 0.01f, ls}, {ls, s - 0.01f, ls},
        {ls, s - 0.01f, -ls}, {-ls, s - 0.01f, -ls},
        {0, -1, 0}, RT_OBJ_LIGHT, RT_MAT_EMISSIVE);

    // NOTE: Main rotating cube is now in a separate BLAS for dynamic updates!
    // See BuildDynamicCube() below

    // Mirror at 45-degree angle in back-left corner, facing toward GREEN wall
    // This allows the mirror to reflect the green (right) wall
    const float mh = 0.5f;   // Half height
    const float mw = 0.4f;   // Half width

    // Mirror center position (back-left area of room)
    const float mcx = -0.6f;
    const float mcy = 0.0f;
    const float mcz = 0.6f;

    // 45-degree rotation: tangent = (0.707, 0, 0.707), normal = (0.707, 0, -0.707)
    const float c45 = 0.707f;  // cos(45) = sin(45)

    // Calculate corner positions for angled quad
    // Right direction along mirror surface: (c45, 0, c45)
    AddQuad(verts, inds,
        {mcx - c45*mw, mcy - mh, mcz - c45*mw},  // bottom-left
        {mcx + c45*mw, mcy - mh, mcz + c45*mw},  // bottom-right
        {mcx + c45*mw, mcy + mh, mcz + c45*mw},  // top-right
        {mcx - c45*mw, mcy + mh, mcz - c45*mw},  // top-left
        {c45, 0, -c45}, RT_OBJ_MIRROR, RT_MAT_MIRROR);  // normal toward camera & green wall

    Log("[MIRROR] Angled 45deg at (%.2f, %.2f, %.2f), normal=(%.2f, 0, %.2f)\n", mcx, mcy, mcz, c45, -c45);

    // Small RED cube on the floor (near left wall)
    const float cubeX = -0.5f;
    const float cubeY = -0.85f;
    const float cubeZ = 0.3f;
    AddBox(verts, inds, {cubeX, cubeY, cubeZ}, {0.13f, 0.13f, 0.13f}, RT_OBJ_SMALL_CUBE, RT_MAT_DIFFUSE);

    // Thin glass pane DIRECTLY in front of red cube
    // Glass is just 0.15 units in front of the cube
    const float gz = cubeZ - 0.18f;  // Just in front of cube
    const float gy = cubeY - 0.02f;  // Slightly below cube bottom
    const float gh = 0.35f;          // Covers the cube height
    const float gw = 0.18f;          // Covers the cube width
    // Front face (towards camera, normal -Z)
    AddQuad(verts, inds,
        {cubeX - gw, gy, gz}, {cubeX + gw, gy, gz},
        {cubeX + gw, gy + gh, gz}, {cubeX - gw, gy + gh, gz},
        {0, 0, -1}, RT_OBJ_GLASS, RT_MAT_GLASS);
    // Back face (towards cube, normal +Z)
    AddQuad(verts, inds,
        {cubeX + gw, gy, gz}, {cubeX - gw, gy, gz},
        {cubeX - gw, gy + gh, gz}, {cubeX + gw, gy + gh, gz},
        {0, 0, 1}, RT_OBJ_GLASS, RT_MAT_GLASS);

    // PURPLE front wall - behind camera to block outside light
    // Camera is at z = -2.2, wall at z = -3 (behind camera)
    // Wall faces INTO the room (normal = 0, 0, +1)
    const float fwz = -3.0f;
    const float fws = 2.0f;  // Large wall to cover everything
    AddQuad(verts, inds,
        {-fws, -fws, fwz}, {fws, -fws, fwz},
        {fws, fws, fwz}, {-fws, fws, fwz},
        {0, 0, 1}, RT_OBJ_FRONT_WALL, RT_MAT_DIFFUSE);

    Log("[FRONT_WALL] Purple wall at z=%.2f, behind camera\n", fwz);

    // Log geometry stats
    Log("[GEOMETRY] Static vertices: %zu, Static indices: %zu\n", verts.size(), inds.size());

    // Find and log mirror vertices
    for (size_t i = 0; i < verts.size(); i++) {
        if (verts[i].objectID == RT_OBJ_MIRROR) {
            Log("[GEOMETRY] Mirror vertex %zu: pos=(%.2f,%.2f,%.2f) objID=%u matType=%u\n",
                i, verts[i].pos.x, verts[i].pos.y, verts[i].pos.z,
                verts[i].objectID, verts[i].materialType);
        }
    }
}

// ============== BUILD DYNAMIC CUBE (8 cubes in 2x2x2 like D3D12) ==============
static void BuildDynamicCube(std::vector<RTVert>& verts, std::vector<UINT>& inds) {
    verts.clear();
    inds.clear();
    verts.reserve(24 * 8);
    inds.reserve(36 * 8);

    // 8 small cubes in 2x2x2 arrangement (like D3D12 renderer)
    // Cubes touch each other (spacing = smallSize)
    const float smallSize = 0.11f;  // Half-size of each small cube
    const float spacing = smallSize; // Cubes touch exactly

    int coords[8][3] = {
        {-1, +1, +1}, {+1, +1, +1}, {-1, -1, +1}, {+1, -1, +1},
        {-1, +1, -1}, {+1, +1, -1}, {-1, -1, -1}, {+1, -1, -1},
    };

    for (int c = 0; c < 8; c++) {
        float cx = coords[c][0] * spacing;
        float cy = coords[c][1] * spacing;
        float cz = coords[c][2] * spacing;
        // Use materialType to pass cube index (0-7) for per-cube coloring
        AddBox(verts, inds, {cx, cy, cz}, {smallSize, smallSize, smallSize}, RT_OBJ_CUBE, (UINT)c);
    }

    Log("[GEOMETRY] Dynamic cube (8 cubes): %zu vertices, %zu indices\n", verts.size(), inds.size());
}

// ============== UPDATE CUBE INSTANCE TRANSFORM ==============
// Called each frame to update the cube rotation
static void UpdateCubeTransform(float time) {
    if (!s_instanceMapped) return;

    // Rotation angles (same as D3D12 renderer: Y*1.2, X*0.7)
    float angleY = time * 1.2f;
    float angleX = time * 0.7f;

    // Build rotation matrices
    float cosY = cosf(angleY), sinY = sinf(angleY);
    float cosX = cosf(angleX), sinX = sinf(angleX);

    // Combined rotation: RotY * RotX
    // RotY = | cosY  0  sinY |    RotX = | 1    0     0   |
    //        |  0    1   0   |           | 0  cosX -sinX |
    //        |-sinY  0  cosY |           | 0  sinX  cosX |

    float m00 = cosY;
    float m01 = sinY * sinX;
    float m02 = sinY * cosX;
    float m10 = 0;
    float m11 = cosX;
    float m12 = -sinX;
    float m20 = -sinY;
    float m21 = cosY * sinX;
    float m22 = cosY * cosX;

    // Cube position (where it floats in the scene)
    float tx = 0.15f;
    float ty = 0.15f;
    float tz = 0.2f;

    // Instance 0 = static geometry (identity transform)
    // Instance 1 = dynamic cube (with rotation + translation)
    D3D12_RAYTRACING_INSTANCE_DESC* instances = (D3D12_RAYTRACING_INSTANCE_DESC*)s_instanceMapped;

    // Instance 0: Static geometry (identity, already set)

    // Instance 1: Cube with rotation and translation
    // Transform is 3x4 row-major: rows are X, Y, Z basis vectors + translation
    // IMPORTANT: TLAS applies transform as Matrix * pos (column vector)
    // But shader does pos (row) * Matrix, so we need TRANSPOSE of the rotation!
    instances[1].Transform[0][0] = m00; instances[1].Transform[0][1] = m10; instances[1].Transform[0][2] = m20; instances[1].Transform[0][3] = tx;
    instances[1].Transform[1][0] = m01; instances[1].Transform[1][1] = m11; instances[1].Transform[1][2] = m21; instances[1].Transform[1][3] = ty;
    instances[1].Transform[2][0] = m02; instances[1].Transform[2][1] = m12; instances[1].Transform[2][2] = m22; instances[1].Transform[2][3] = tz;
    // Ensure InstanceID is set correctly (shader uses this to identify cube hits)
    instances[1].InstanceID = 1;
}

// ============== REBUILD TLAS (for dynamic updates) ==============
static void RebuildTLAS() {
    if (!s_cmdList || !s_tlasBuffer || !s_instanceBuffer) return;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 2;  // Static + Cube
    tlasInputs.InstanceDescs = s_instanceBuffer->GetGPUVirtualAddress();
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
    tlasBuildDesc.Inputs = tlasInputs;
    tlasBuildDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    tlasBuildDesc.SourceAccelerationStructureData = s_tlasBuffer->GetGPUVirtualAddress();
    tlasBuildDesc.DestAccelerationStructureData = s_tlasBuffer->GetGPUVirtualAddress();
    tlasBuildDesc.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();

    s_cmdList->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = s_tlasBuffer;
    s_cmdList->ResourceBarrier(1, &uavBarrier);
}

// ============== TEXT DRAWING ==============
static void DrawTextRT(const char* text, float x, float y, float r, float g, float b, float a, float scale) {
    const float charW = 8.0f * scale;
    const float charH = 8.0f * scale;
    float startX = x;

    while (*text && s_textVertCount < 5994) {
        char c = *text++;
        if (c == '\n') { y += charH + 2; x = startX; continue; }
        if (c < 32 || c > 127) c = '?';

        int ci = c - 32;
        int row = ci / 16;
        int col = ci % 16;
        float u0 = col / 16.0f, v0 = row / 6.0f;
        float u1 = (col + 1) / 16.0f, v1 = (row + 1) / 6.0f;

        float x0 = x * 2.0f / W - 1.0f;
        float y0 = 1.0f - y * 2.0f / H;
        float x1 = (x + charW) * 2.0f / W - 1.0f;
        float y1 = 1.0f - (y + charH) * 2.0f / H;

        TextVert* v = &s_textVerts[s_textVertCount];
        v[0] = {x0, y0, u0, v0, r, g, b, a};
        v[1] = {x1, y0, u1, v0, r, g, b, a};
        v[2] = {x0, y1, u0, v1, r, g, b, a};
        v[3] = {x1, y0, u1, v0, r, g, b, a};
        v[4] = {x1, y1, u1, v1, r, g, b, a};
        v[5] = {x0, y1, u0, v1, r, g, b, a};
        s_textVertCount += 6;
        x += charW;
    }
}

// ============== DXC SHADER COMPILATION ==============
// Compiles shader with optional defines (array of L"-D", L"DEFINE_NAME" pairs)
static bool CompileShaderDXC(const char* source, const wchar_t* entry, const wchar_t* target,
                             ID3DBlob** blob, const wchar_t** defines = nullptr, int defineCount = 0) {
    typedef HRESULT(WINAPI* DxcCreateInstanceProc)(REFCLSID, REFIID, LPVOID*);
    HMODULE dxcMod = LoadLibraryW(L"dxcompiler.dll");
    if (!dxcMod) { Log("[ERROR] Cannot load dxcompiler.dll\n"); return false; }

    auto DxcCreate = (DxcCreateInstanceProc)GetProcAddress(dxcMod, "DxcCreateInstance");
    if (!DxcCreate) { FreeLibrary(dxcMod); return false; }

    IDxcUtils* utils = nullptr;
    IDxcCompiler3* compiler = nullptr;
    DxcCreate(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
    DxcCreate(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

    IDxcBlobEncoding* srcBlob = nullptr;
    utils->CreateBlob(source, (UINT)strlen(source), CP_UTF8, &srcBlob);

    DxcBuffer srcBuf = { srcBlob->GetBufferPointer(), srcBlob->GetBufferSize(), CP_UTF8 };

    // Build args array: base args + defines
    // Base: -E entry -T target -O3 (5 args)
    // Each define: -D DEFINE (2 args each)
    const int baseArgCount = 5;
    int totalArgs = baseArgCount + defineCount * 2;
    const wchar_t** args = new const wchar_t*[totalArgs];
    args[0] = L"-E"; args[1] = entry;
    args[2] = L"-T"; args[3] = target;
    args[4] = L"-O3";
    for (int i = 0; i < defineCount; i++) {
        args[baseArgCount + i * 2] = L"-D";
        args[baseArgCount + i * 2 + 1] = defines[i];
    }

    IDxcResult* result = nullptr;
    compiler->Compile(&srcBuf, args, totalArgs, nullptr, IID_PPV_ARGS(&result));
    delete[] args;

    HRESULT status;
    result->GetStatus(&status);
    if (FAILED(status)) {
        IDxcBlobUtf8* err = nullptr;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&err), nullptr);
        if (err) { Log("[SHADER ERROR] %s\n", err->GetStringPointer()); err->Release(); }
        srcBlob->Release(); result->Release(); compiler->Release(); utils->Release();
        FreeLibrary(dxcMod);
        return false;
    }

    IDxcBlob* shaderBlob = nullptr;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr);

    // Copy to ID3DBlob
    D3DCreateBlob(shaderBlob->GetBufferSize(), blob);
    memcpy((*blob)->GetBufferPointer(), shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());

    shaderBlob->Release(); srcBlob->Release(); result->Release();
    compiler->Release(); utils->Release();
    FreeLibrary(dxcMod);
    return true;
}

// Get current shader features from global DXR settings
static ShaderFeatures GetCurrentShaderFeatures() {
    ShaderFeatures f = {};
    f.useRayQuery = g_dxrFeatures.useRayQuery;
    // RayQuery features only active if useRayQuery is enabled
    f.shadows = g_dxrFeatures.rtShadows && g_dxrFeatures.useRayQuery;
    f.softShadows = g_dxrFeatures.rtSoftShadows && g_dxrFeatures.useRayQuery;
    f.ao = g_dxrFeatures.rtAO && g_dxrFeatures.useRayQuery;
    f.gi = g_dxrFeatures.rtGI && g_dxrFeatures.useRayQuery;
    f.reflections = g_dxrFeatures.rtReflections && g_dxrFeatures.useRayQuery;
    // These don't require RayQuery
    f.rtLighting = g_dxrFeatures.rtLighting;
    f.temporalDenoise = g_dxrFeatures.enableTemporalDenoise;
    return f;
}

// Build defines array from shader features - returns count
static int BuildShaderDefines(const ShaderFeatures& f, const wchar_t** defines) {
    int count = 0;
    if (f.useRayQuery)    defines[count++] = L"USE_RAYQUERY";  // Enables SM 6.5 RayQuery code
    if (f.shadows)        defines[count++] = L"FEATURE_SHADOWS";
    if (f.softShadows)    defines[count++] = L"FEATURE_SOFT_SHADOWS";
    if (f.rtLighting)     defines[count++] = L"FEATURE_RT_LIGHTING";
    if (f.ao)             defines[count++] = L"FEATURE_AO";
    if (f.gi)             defines[count++] = L"FEATURE_GI";
    if (f.reflections)    defines[count++] = L"FEATURE_REFLECTIONS";
    if (f.temporalDenoise) defines[count++] = L"FEATURE_TEMPORAL_DENOISE";
    return count;
}

// Recompile shaders with specified feature flags
static bool RecompileShaders(const ShaderFeatures& features) {
    const wchar_t* defines[10];  // Max 8 defines + safety margin
    int defineCount = BuildShaderDefines(features, defines);

    // Select shader model: 6.5 for RayQuery, 6.0 for compatibility
    const wchar_t* vsTarget = features.useRayQuery ? L"vs_6_5" : L"vs_6_0";
    const wchar_t* psTarget = features.useRayQuery ? L"ps_6_5" : L"ps_6_0";

    Log("[INFO] Recompiling shaders (%ls) with features: %s%s%s%s%s%s%s%s\n",
        psTarget,
        features.useRayQuery ? "RAYQUERY " : "",
        features.shadows ? "SHADOWS " : "",
        features.softShadows ? "SOFT_SHADOWS " : "",
        features.rtLighting ? "RT_LIGHTING " : "",
        features.ao ? "AO " : "",
        features.gi ? "GI " : "",
        features.reflections ? "REFLECTIONS " : "",
        features.temporalDenoise ? "TEMPORAL_DENOISE " : "");

    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* psBlob = nullptr;

    if (!CompileShaderDXC(g_rtCornellShaderCode, L"VSMain", vsTarget, &vsBlob, defines, defineCount)) {
        Log("[ERROR] Failed to compile vertex shader\n");
        return false;
    }
    if (!CompileShaderDXC(g_rtCornellShaderCode, L"PSMain", psTarget, &psBlob, defines, defineCount)) {
        vsBlob->Release();
        Log("[ERROR] Failed to compile pixel shader\n");
        return false;
    }

    // Release old PSO
    if (s_pso) { s_pso->Release(); s_pso = nullptr; }

    // Create new PSO with recompiled shaders
    D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"OBJECTID", 0, DXGI_FORMAT_R32_UINT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"MATERIALTYPE", 0, DXGI_FORMAT_R32_UINT, 0, 28, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = { inputLayout, _countof(inputLayout) };
    psoDesc.pRootSignature = s_rootSig;
    psoDesc.VS = { vsBlob->GetBufferPointer(), vsBlob->GetBufferSize() };
    psoDesc.PS = { psBlob->GetBufferPointer(), psBlob->GetBufferSize() };
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    psoDesc.SampleDesc.Count = 1;

    HRESULT hr = s_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&s_pso));
    vsBlob->Release();
    psBlob->Release();

    if (FAILED(hr)) {
        Log("[ERROR] Failed to create PSO after recompile: 0x%08X\n", hr);
        return false;
    }

    s_compiledFeatures = features;
    Log("[INFO] Shaders recompiled successfully\n");
    return true;
}

// ============== INITIALIZATION ==============
bool InitD3D12RT(HWND hwnd) {
    Log("[INFO] Initializing D3D12 + Ray Tracing (from scratch)...\n");

    HRESULT hr;

    // Enable debug layer
    #ifdef _DEBUG
    ID3D12Debug* debug = nullptr;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
        debug->EnableDebugLayer();
        debug->Release();
    }
    #endif

    // Create DXGI factory
    IDXGIFactory6* factory = nullptr;
    hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) { Log("[ERROR] CreateDXGIFactory2 failed\n"); return false; }

    // Find DXR-capable adapter
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; i++)
    {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); continue; }

        // Try to create D3D12 device (skip DXR tier check)
        ID3D12Device5* testDev = nullptr;
        if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&testDev)))) {
            s_device = testDev;
            s_gpuName = desc.Description;
            Log("[INFO] Using GPU: %ls\n", desc.Description);
            break;
        }
        adapter->Release();
        adapter = nullptr;
    }

    if (!s_device) {
        Log("[ERROR] No D3D12 capable GPU found\n");
        factory->Release();
        return false;
    }

    // Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = s_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&s_cmdQueue));
    if (FAILED(hr)) { Log("[ERROR] CreateCommandQueue failed\n"); return false; }

    // Create swap chain
    DXGI_SWAP_CHAIN_DESC1 swapDesc = {};
    swapDesc.Width = W;
    swapDesc.Height = H;
    swapDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapDesc.SampleDesc.Count = 1;
    swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapDesc.BufferCount = 3;
    swapDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

    IDXGISwapChain1* swapChain1 = nullptr;
    hr = factory->CreateSwapChainForHwnd(s_cmdQueue, hwnd, &swapDesc, nullptr, nullptr, &swapChain1);
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    swapChain1->QueryInterface(IID_PPV_ARGS(&s_swapChain));
    swapChain1->Release();
    factory->Release();
    if (adapter) adapter->Release();

    s_frameIndex = s_swapChain->GetCurrentBackBufferIndex();

    // Create RTV heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = 3;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    s_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&s_rtvHeap));
    s_rtvDescSize = s_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Create RTVs
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = s_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 3; i++) {
        s_swapChain->GetBuffer(i, IID_PPV_ARGS(&s_renderTargets[i]));
        s_device->CreateRenderTargetView(s_renderTargets[i], nullptr, rtvHandle);
        rtvHandle.ptr += s_rtvDescSize;
    }

    // Create DSV heap and depth buffer
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    s_device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&s_dsvHeap));

    D3D12_RESOURCE_DESC depthDesc = {};
    depthDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    depthDesc.Width = W;
    depthDesc.Height = H;
    depthDesc.DepthOrArraySize = 1;
    depthDesc.MipLevels = 1;
    depthDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    depthDesc.SampleDesc.Count = 1;
    depthDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;

    D3D12_CLEAR_VALUE clearValue = {};
    clearValue.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    clearValue.DepthStencil.Depth = 1.0f;

    D3D12_HEAP_PROPERTIES defaultHeap = { D3D12_HEAP_TYPE_DEFAULT };
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &depthDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE, &clearValue, IID_PPV_ARGS(&s_depthStencil));

    D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
    dsvDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    s_device->CreateDepthStencilView(s_depthStencil, &dsvDesc, s_dsvHeap->GetCPUDescriptorHandleForHeapStart());

    // Create history buffer for temporal denoising (same format as render target)
    D3D12_RESOURCE_DESC historyDesc = {};
    historyDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    historyDesc.Width = W;
    historyDesc.Height = H;
    historyDesc.DepthOrArraySize = 1;
    historyDesc.MipLevels = 1;
    historyDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    historyDesc.SampleDesc.Count = 1;
    historyDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &historyDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&s_historyBuffer));
    s_historyValid = false;

    // Create command allocators
    for (UINT i = 0; i < 3; i++) {
        s_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s_cmdAlloc[i]));
    }

    // Create fence
    s_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s_fence));
    s_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // ============== BUILD GEOMETRY ==============
    D3D12_HEAP_PROPERTIES uploadHeap = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_RESOURCE_DESC bufDesc = {};
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Height = 1;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    void* mapped;

    // --- STATIC GEOMETRY (room, mirror, walls, etc.) ---
    std::vector<RTVert> vertsStatic;
    std::vector<UINT> indsStatic;
    BuildCornellBox(vertsStatic, indsStatic);
    s_vertexCountStatic = (UINT)vertsStatic.size();
    s_indexCountStatic = (UINT)indsStatic.size();
    Log("[INFO] Static geometry: %u vertices, %u indices\n", s_vertexCountStatic, s_indexCountStatic);

    UINT vbSizeStatic = s_vertexCountStatic * sizeof(RTVert);
    bufDesc.Width = vbSizeStatic;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_vertexBufferStatic));
    s_vertexBufferStatic->Map(0, nullptr, &mapped);
    memcpy(mapped, vertsStatic.data(), vbSizeStatic);
    s_vertexBufferStatic->Unmap(0, nullptr);

    UINT ibSizeStatic = s_indexCountStatic * sizeof(UINT);
    bufDesc.Width = ibSizeStatic;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_indexBufferStatic));
    s_indexBufferStatic->Map(0, nullptr, &mapped);
    memcpy(mapped, indsStatic.data(), ibSizeStatic);
    s_indexBufferStatic->Unmap(0, nullptr);

    // --- DYNAMIC GEOMETRY (cube, at origin) ---
    std::vector<RTVert> vertsCube;
    std::vector<UINT> indsCube;
    BuildDynamicCube(vertsCube, indsCube);
    s_vertexCountCube = (UINT)vertsCube.size();
    s_indexCountCube = (UINT)indsCube.size();
    Log("[INFO] Dynamic cube: %u vertices, %u indices\n", s_vertexCountCube, s_indexCountCube);

    UINT vbSizeCube = s_vertexCountCube * sizeof(RTVert);
    bufDesc.Width = vbSizeCube;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_vertexBufferCube));
    s_vertexBufferCube->Map(0, nullptr, &mapped);
    memcpy(mapped, vertsCube.data(), vbSizeCube);
    s_vertexBufferCube->Unmap(0, nullptr);

    UINT ibSizeCube = s_indexCountCube * sizeof(UINT);
    bufDesc.Width = ibSizeCube;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_indexBufferCube));
    s_indexBufferCube->Map(0, nullptr, &mapped);
    memcpy(mapped, indsCube.data(), ibSizeCube);
    s_indexBufferCube->Unmap(0, nullptr);

    // Static geometry views for rasterization
    s_vertexCount = s_vertexCountStatic;
    s_indexCount = s_indexCountStatic;
    s_vbView.BufferLocation = s_vertexBufferStatic->GetGPUVirtualAddress();
    s_vbView.SizeInBytes = vbSizeStatic;
    s_vbView.StrideInBytes = sizeof(RTVert);
    s_ibView.BufferLocation = s_indexBufferStatic->GetGPUVirtualAddress();
    s_ibView.SizeInBytes = ibSizeStatic;
    s_ibView.Format = DXGI_FORMAT_R32_UINT;

    // Cube geometry views for rasterization (separate draw call)
    s_vbViewCube.BufferLocation = s_vertexBufferCube->GetGPUVirtualAddress();
    s_vbViewCube.SizeInBytes = vbSizeCube;
    s_vbViewCube.StrideInBytes = sizeof(RTVert);
    s_ibViewCube.BufferLocation = s_indexBufferCube->GetGPUVirtualAddress();
    s_ibViewCube.SizeInBytes = ibSizeCube;
    s_ibViewCube.Format = DXGI_FORMAT_R32_UINT;

    // Create constant buffer
    bufDesc.Width = 256;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_constantBuffer));
    s_constantBuffer->Map(0, nullptr, &s_cbMapped);

    // Create command list
    ID3D12GraphicsCommandList* baseCmdList = nullptr;
    s_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_cmdAlloc[0], nullptr, IID_PPV_ARGS(&baseCmdList));
    baseCmdList->QueryInterface(IID_PPV_ARGS(&s_cmdList));
    baseCmdList->Release();

    // ============== BUILD ACCELERATION STRUCTURES ==============
    D3D12_RESOURCE_DESC asDesc = {};
    asDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    asDesc.Height = 1; asDesc.DepthOrArraySize = 1; asDesc.MipLevels = 1;
    asDesc.SampleDesc.Count = 1; asDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    asDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // --- BLAS for STATIC geometry ---
    D3D12_RAYTRACING_GEOMETRY_DESC geomDescStatic = {};
    geomDescStatic.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geomDescStatic.Triangles.VertexBuffer.StartAddress = s_vertexBufferStatic->GetGPUVirtualAddress();
    geomDescStatic.Triangles.VertexBuffer.StrideInBytes = sizeof(RTVert);
    geomDescStatic.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geomDescStatic.Triangles.VertexCount = s_vertexCountStatic;
    geomDescStatic.Triangles.IndexBuffer = s_indexBufferStatic->GetGPUVirtualAddress();
    geomDescStatic.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    geomDescStatic.Triangles.IndexCount = s_indexCountStatic;
    geomDescStatic.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputsStatic = {};
    blasInputsStatic.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputsStatic.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputsStatic.NumDescs = 1;
    blasInputsStatic.pGeometryDescs = &geomDescStatic;
    blasInputsStatic.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildStatic = {};
    s_device->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputsStatic, &blasPrebuildStatic);

    asDesc.Width = blasPrebuildStatic.ResultDataMaxSizeInBytes;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_blasBufferStatic));

    // --- BLAS for CUBE ---
    D3D12_RAYTRACING_GEOMETRY_DESC geomDescCube = {};
    geomDescCube.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geomDescCube.Triangles.VertexBuffer.StartAddress = s_vertexBufferCube->GetGPUVirtualAddress();
    geomDescCube.Triangles.VertexBuffer.StrideInBytes = sizeof(RTVert);
    geomDescCube.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geomDescCube.Triangles.VertexCount = s_vertexCountCube;
    geomDescCube.Triangles.IndexBuffer = s_indexBufferCube->GetGPUVirtualAddress();
    geomDescCube.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    geomDescCube.Triangles.IndexCount = s_indexCountCube;
    geomDescCube.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputsCube = {};
    blasInputsCube.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputsCube.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputsCube.NumDescs = 1;
    blasInputsCube.pGeometryDescs = &geomDescCube;
    blasInputsCube.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildCube = {};
    s_device->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputsCube, &blasPrebuildCube);

    asDesc.Width = blasPrebuildCube.ResultDataMaxSizeInBytes;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_blasBufferCube));

    // --- Scratch buffer (big enough for all builds) ---
    UINT64 scratchSize = max(blasPrebuildStatic.ScratchDataSizeInBytes, blasPrebuildCube.ScratchDataSizeInBytes);
    scratchSize = max(scratchSize, (UINT64)65536);
    s_tlasScratchSize = scratchSize;  // Save for TLAS updates
    asDesc.Width = scratchSize * 2;  // Extra space for TLAS
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&s_scratchBuffer));

    // Build both BLASes
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuildStatic = {};
    blasBuildStatic.Inputs = blasInputsStatic;
    blasBuildStatic.DestAccelerationStructureData = s_blasBufferStatic->GetGPUVirtualAddress();
    blasBuildStatic.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    s_cmdList->BuildRaytracingAccelerationStructure(&blasBuildStatic, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = s_blasBufferStatic;
    s_cmdList->ResourceBarrier(1, &uavBarrier);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuildCube = {};
    blasBuildCube.Inputs = blasInputsCube;
    blasBuildCube.DestAccelerationStructureData = s_blasBufferCube->GetGPUVirtualAddress();
    blasBuildCube.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    s_cmdList->BuildRaytracingAccelerationStructure(&blasBuildCube, 0, nullptr);

    uavBarrier.UAV.pResource = s_blasBufferCube;
    s_cmdList->ResourceBarrier(1, &uavBarrier);

    // --- TLAS with 2 instances (static + cube) ---
    D3D12_RAYTRACING_INSTANCE_DESC instances[2] = {};

    // Instance 0: Static geometry (identity transform)
    instances[0].Transform[0][0] = 1.0f;
    instances[0].Transform[1][1] = 1.0f;
    instances[0].Transform[2][2] = 1.0f;
    instances[0].InstanceID = 0;  // Static geometry
    instances[0].InstanceMask = 0xFF;
    instances[0].AccelerationStructure = s_blasBufferStatic->GetGPUVirtualAddress();

    // Instance 1: Cube (will be updated each frame)
    instances[1].Transform[0][0] = 1.0f;
    instances[1].Transform[1][1] = 1.0f;
    instances[1].Transform[2][2] = 1.0f;
    instances[1].Transform[0][3] = 0.15f;  // Initial position X
    instances[1].Transform[1][3] = 0.15f;  // Initial position Y
    instances[1].Transform[2][3] = 0.2f;   // Initial position Z
    instances[1].InstanceID = 1;  // Cube BLAS - shader uses this to identify cube hits
    instances[1].InstanceMask = 0xFF;
    instances[1].AccelerationStructure = s_blasBufferCube->GetGPUVirtualAddress();

    // Create instance buffer with persistent mapping for runtime updates
    bufDesc.Width = sizeof(instances);
    bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_instanceBuffer));
    s_instanceBuffer->Map(0, nullptr, &s_instanceMapped);  // Keep mapped for runtime updates!
    memcpy(s_instanceMapped, instances, sizeof(instances));

    // TLAS with ALLOW_UPDATE for efficient per-frame rebuilds
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 2;  // Static + Cube
    tlasInputs.InstanceDescs = s_instanceBuffer->GetGPUVirtualAddress();
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
                       D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPrebuild = {};
    s_device->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasPrebuild);

    // Make sure scratch buffer is big enough for TLAS too
    s_tlasScratchSize = max(s_tlasScratchSize, tlasPrebuild.ScratchDataSizeInBytes);

    asDesc.Width = tlasPrebuild.ResultDataMaxSizeInBytes;
    asDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc,
        D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_tlasBuffer));

    // Initial TLAS build
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
    tlasBuildDesc.Inputs = tlasInputs;
    tlasBuildDesc.DestAccelerationStructureData = s_tlasBuffer->GetGPUVirtualAddress();
    tlasBuildDesc.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    s_cmdList->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);

    uavBarrier.UAV.pResource = s_tlasBuffer;
    s_cmdList->ResourceBarrier(1, &uavBarrier);

    Log("[INFO] TLAS built with 2 instances (static + dynamic cube)\n");

    // Execute AS build commands
    s_cmdList->Close();
    ID3D12CommandList* lists[] = { s_cmdList };
    s_cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGpuRT();
    s_cmdAlloc[0]->Reset();
    s_cmdList->Reset(s_cmdAlloc[0], nullptr);

    Log("[INFO] Acceleration structures built\n");

    // ============== SRV HEAP FOR TLAS + HISTORY ==============
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = 2;  // t0: TLAS, t1: History buffer
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    s_device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&s_srvHeap));

    UINT srvDescSize = s_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle = s_srvHeap->GetCPUDescriptorHandleForHeapStart();

    // t0: TLAS
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location = s_tlasBuffer->GetGPUVirtualAddress();
    s_device->CreateShaderResourceView(nullptr, &srvDesc, srvHandle);

    // t1: History buffer for temporal denoising
    srvHandle.ptr += srvDescSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC historySrvDesc = {};
    historySrvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    historySrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    historySrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    historySrvDesc.Texture2D.MipLevels = 1;
    s_device->CreateShaderResourceView(s_historyBuffer, &historySrvDesc, srvHandle);

    // ============== ROOT SIGNATURE ==============
    D3D12_ROOT_PARAMETER rootParams[2] = {};
    // b0: CBV
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[0].Descriptor.ShaderRegister = 0;
    rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    // t0: TLAS, t1: History buffer
    D3D12_DESCRIPTOR_RANGE range = {};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 2;  // t0=TLAS, t1=History
    range.BaseShaderRegister = 0;
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[1].DescriptorTable.pDescriptorRanges = &range;
    rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = rootParams;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ID3DBlob* rsBlob = nullptr, *rsError = nullptr;
    D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError);
    if (rsError) { Log("[ERROR] Root sig: %s\n", (char*)rsError->GetBufferPointer()); rsError->Release(); }
    s_device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&s_rootSig));
    rsBlob->Release();

    // ============== COMPILE SHADERS ==============
    Log("[INFO] Compiling RT shaders with feature defines...\n");
    s_compiledFeatures = GetCurrentShaderFeatures();
    // Don't enable temporal denoise at init - history buffer isn't valid yet
    // It will be enabled on subsequent frames once history is valid
    s_compiledFeatures.temporalDenoise = false;
    const wchar_t* defines[10];
    int defineCount = BuildShaderDefines(s_compiledFeatures, defines);

    // Select shader model: 6.5 for RayQuery, 6.0 for compatibility
    const wchar_t* vsTarget = s_compiledFeatures.useRayQuery ? L"vs_6_5" : L"vs_6_0";
    const wchar_t* psTarget = s_compiledFeatures.useRayQuery ? L"ps_6_5" : L"ps_6_0";

    Log("[INFO] Shader Model: %ls, Features: %s%s%s%s%s%s%s%s\n",
        psTarget,
        s_compiledFeatures.useRayQuery ? "RAYQUERY " : "",
        s_compiledFeatures.shadows ? "SHADOWS " : "",
        s_compiledFeatures.softShadows ? "SOFT_SHADOWS " : "",
        s_compiledFeatures.rtLighting ? "RT_LIGHTING " : "",
        s_compiledFeatures.ao ? "AO " : "",
        s_compiledFeatures.gi ? "GI " : "",
        s_compiledFeatures.reflections ? "REFLECTIONS " : "",
        s_compiledFeatures.temporalDenoise ? "TEMPORAL_DENOISE " : "");

    ID3DBlob* vsBlob = nullptr, *psBlob = nullptr;
    if (!CompileShaderDXC(g_rtCornellShaderCode, L"VSMain", vsTarget, &vsBlob, defines, defineCount)) return false;
    if (!CompileShaderDXC(g_rtCornellShaderCode, L"PSMain", psTarget, &psBlob, defines, defineCount)) { vsBlob->Release(); return false; }
    Log("[INFO] Shaders compiled (VS: %zu, PS: %zu bytes)\n", vsBlob->GetBufferSize(), psBlob->GetBufferSize());

    // ============== PSO ==============
    D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"OBJECTID", 0, DXGI_FORMAT_R32_UINT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"MATERIALTYPE", 0, DXGI_FORMAT_R32_UINT, 0, 28, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = { inputLayout, _countof(inputLayout) };
    psoDesc.pRootSignature = s_rootSig;
    psoDesc.VS = { vsBlob->GetBufferPointer(), vsBlob->GetBufferSize() };
    psoDesc.PS = { psBlob->GetBufferPointer(), psBlob->GetBufferSize() };
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;  // No culling - show both sides
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;
    psoDesc.RasterizerState.DepthClipEnable = TRUE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    psoDesc.SampleDesc.Count = 1;

    hr = s_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&s_pso));
    vsBlob->Release();
    psBlob->Release();
    if (FAILED(hr)) { Log("[ERROR] CreatePSO failed\n"); return false; }

    // ============== TEXT RENDERING SETUP ==============
    // Text root signature
    D3D12_DESCRIPTOR_RANGE texRange = {};
    texRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    texRange.NumDescriptors = 1;
    texRange.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER textParams[1] = {};
    textParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    textParams[0].DescriptorTable.NumDescriptorRanges = 1;
    textParams[0].DescriptorTable.pDescriptorRanges = &texRange;
    textParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.ShaderRegister = 0;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC textRsDesc = {};
    textRsDesc.NumParameters = 1;
    textRsDesc.pParameters = textParams;
    textRsDesc.NumStaticSamplers = 1;
    textRsDesc.pStaticSamplers = &sampler;
    textRsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    D3D12SerializeRootSignature(&textRsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsError);
    s_device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&s_textRootSig));
    rsBlob->Release();

    // Text shaders
    ID3DBlob* textVs = nullptr, *textPs = nullptr;
    D3DCompile(g_textShaderCode, strlen(g_textShaderCode), nullptr, nullptr, nullptr, "TextVS", "vs_5_0", 0, 0, &textVs, nullptr);
    D3DCompile(g_textShaderCode, strlen(g_textShaderCode), nullptr, nullptr, nullptr, "TextPS", "ps_5_0", 0, 0, &textPs, nullptr);

    // Text PSO
    D3D12_INPUT_ELEMENT_DESC textLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC textPsoDesc = {};
    textPsoDesc.InputLayout = { textLayout, _countof(textLayout) };
    textPsoDesc.pRootSignature = s_textRootSig;
    textPsoDesc.VS = { textVs->GetBufferPointer(), textVs->GetBufferSize() };
    textPsoDesc.PS = { textPs->GetBufferPointer(), textPs->GetBufferSize() };
    textPsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    textPsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    textPsoDesc.BlendState.RenderTarget[0].BlendEnable = TRUE;
    textPsoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
    textPsoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    textPsoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    textPsoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    textPsoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    textPsoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    textPsoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    textPsoDesc.SampleMask = UINT_MAX;
    textPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    textPsoDesc.NumRenderTargets = 1;
    textPsoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    textPsoDesc.SampleDesc.Count = 1;

    s_device->CreateGraphicsPipelineState(&textPsoDesc, IID_PPV_ARGS(&s_textPso));
    textVs->Release();
    textPs->Release();

    // Generate font texture from g_font8x8 bitmap data (16x6 characters = 128x48 pixels)
    const int FONT_COLS = 16, FONT_ROWS = 6;
    const int TEX_W = FONT_COLS * 8, TEX_H = FONT_ROWS * 8;
    unsigned char texData[TEX_W * TEX_H];
    memset(texData, 0, sizeof(texData));

    for (int c = 0; c < 96; c++) {
        int col = c % FONT_COLS, row = c / FONT_COLS;
        for (int y = 0; y < 8; y++) {
            unsigned char bits = g_font8x8[c][y];
            for (int x = 0; x < 8; x++) {
                int px = col * 8 + x, py = row * 8 + y;
                texData[py * TEX_W + px] = (bits & (0x80 >> x)) ? 255 : 0;
            }
        }
    }

    // Font texture resource
    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = TEX_W;
    texDesc.Height = TEX_H;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&s_fontTexture));

    // Get proper footprint for upload (handles row pitch alignment)
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT64 uploadSize = 0;
    s_device->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &uploadSize);

    bufDesc.Width = uploadSize;
    bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* uploadBuf = nullptr;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf));
    uploadBuf->Map(0, nullptr, &mapped);
    // Copy row by row using proper aligned pitch
    BYTE* destRow = (BYTE*)mapped + footprint.Offset;
    for (UINT row = 0; row < TEX_H; row++) {
        memcpy(destRow + row * footprint.Footprint.RowPitch, texData + row * TEX_W, TEX_W);
    }
    uploadBuf->Unmap(0, nullptr);

    s_cmdList->Reset(s_cmdAlloc[0], nullptr);
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {}, dstLoc = {};
    srcLoc.pResource = uploadBuf;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint = footprint;
    dstLoc.pResource = s_fontTexture;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    s_cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

    D3D12_RESOURCE_BARRIER texBarrier = {};
    texBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    texBarrier.Transition.pResource = s_fontTexture;
    texBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    texBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    texBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_cmdList->ResourceBarrier(1, &texBarrier);
    s_cmdList->Close();
    s_cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGpuRT();
    uploadBuf->Release();

    // Text SRV heap
    D3D12_DESCRIPTOR_HEAP_DESC textSrvHeapDesc = {};
    textSrvHeapDesc.NumDescriptors = 1;
    textSrvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    textSrvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    s_device->CreateDescriptorHeap(&textSrvHeapDesc, IID_PPV_ARGS(&s_textSrvHeap));

    D3D12_SHADER_RESOURCE_VIEW_DESC texSrvDesc = {};
    texSrvDesc.Format = DXGI_FORMAT_R8_UNORM;
    texSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    texSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    texSrvDesc.Texture2D.MipLevels = 1;
    s_device->CreateShaderResourceView(s_fontTexture, &texSrvDesc, s_textSrvHeap->GetCPUDescriptorHandleForHeapStart());

    // Text vertex buffer
    bufDesc.Width = 6000 * sizeof(TextVert);
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_textVB));
    s_textVB->Map(0, nullptr, &s_textVBMapped);
    s_textVBView.BufferLocation = s_textVB->GetGPUVirtualAddress();
    s_textVBView.SizeInBytes = 6000 * sizeof(TextVert);
    s_textVBView.StrideInBytes = sizeof(TextVert);

    // Reset command list for rendering
    s_cmdAlloc[0]->Reset();
    s_cmdList->Reset(s_cmdAlloc[0], s_pso);
    s_cmdList->Close();

    Log("[INFO] D3D12 + Ray Tracing initialization complete\n");

    return true;
}

// ============== RENDER ==============
void RenderD3D12RT() {
    // Check if shader features changed - recompile if needed
    // MUST be done BEFORE command list reset, because reset uses s_pso
    ShaderFeatures currentFeatures = GetCurrentShaderFeatures();
    // For temporal denoise, only enable in shader if history is valid
    ShaderFeatures effectiveFeatures = currentFeatures;
    effectiveFeatures.temporalDenoise = currentFeatures.temporalDenoise && s_historyValid;
    if (effectiveFeatures != s_compiledFeatures) {
        WaitForGpuRT();
        if (!RecompileShaders(effectiveFeatures)) {
            Log("[ERROR] Shader recompilation failed\n");
        }
    }

    // Reset command allocator and list (uses s_pso which may have been recompiled above)
    s_cmdAlloc[s_frameIndex]->Reset();
    s_cmdList->Reset(s_cmdAlloc[s_frameIndex], s_pso);

    // Frame counter for AA jitter sequence
    static UINT s_rtFrameNumber = 0;
    s_rtFrameNumber++;

    // Update time
    static LARGE_INTEGER startTime = {}, perfFreq = {};
    static float prevTime = 0.0f;
    if (startTime.QuadPart == 0) {
        QueryPerformanceFrequency(&perfFreq);
        QueryPerformanceCounter(&startTime);
    }
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    float time = (float)(now.QuadPart - startTime.QuadPart) / perfFreq.QuadPart;
    float deltaTime = (time - prevTime) * 1000.0f;  // milliseconds
    prevTime = time;
    if (deltaTime < 1.0f) deltaTime = 16.67f;  // Default to 60fps

    // Update constant buffer with DXR settings (parameters only, not enable flags)
    RTCB cb = {};
    cb.time = time;
    cb.shadowSoftness = g_dxrFeatures.shadowSoftness;
    cb.shadowSamples = g_dxrFeatures.softShadowSamples;
    cb.debugMode = g_dxrFeatures.debugMode;
    cb.reflectionStrength = g_dxrFeatures.reflectionStrength;
    cb.aoRadius = g_dxrFeatures.aoRadius;
    cb.aoStrength = g_dxrFeatures.aoStrength;
    cb.aoSamples = g_dxrFeatures.aoSamples;
    cb.giBounces = g_dxrFeatures.giBounces;
    cb.giStrength = g_dxrFeatures.giStrength;
    cb.denoiseBlendFactor = g_dxrFeatures.denoiseBlendFactor;
    memcpy(s_cbMapped, &cb, sizeof(RTCB));

    // Update cube transform and rebuild TLAS for dynamic reflections
    UpdateCubeTransform(time);
    RebuildTLAS();

    // Render resolution and target
    UINT renderW = W;
    UINT renderH = H;
    ID3D12Resource* colorTarget = s_renderTargets[s_frameIndex];
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle, dsvHandle;

    // Use backbuffer
    rtvHandle = s_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += s_frameIndex * s_rtvDescSize;
    dsvHandle = s_dsvHeap->GetCPUDescriptorHandleForHeapStart();

    // Transition backbuffer to render target
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = s_renderTargets[s_frameIndex];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_cmdList->ResourceBarrier(1, &barrier);

    // Clear render targets
    float clearColor[] = { 0.1f, 0.1f, 0.12f, 1.0f };
    s_cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    s_cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    // Set render targets
    s_cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

    // Set root signature and descriptor heap
    s_cmdList->SetGraphicsRootSignature(s_rootSig);

    // IMPORTANT: Ensure history buffer is in correct state before binding
    // On first frame it's in COPY_DEST, must be PIXEL_SHADER_RESOURCE for SRV
    if (!s_historyValid && s_historyBuffer) {
        D3D12_RESOURCE_BARRIER historyBarrier = {};
        historyBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        historyBarrier.Transition.pResource = s_historyBuffer;
        historyBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        historyBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        historyBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        s_cmdList->ResourceBarrier(1, &historyBarrier);
    }

    ID3D12DescriptorHeap* heaps[] = { s_srvHeap };
    s_cmdList->SetDescriptorHeaps(1, heaps);

    // Bind parameters
    s_cmdList->SetGraphicsRootConstantBufferView(0, s_constantBuffer->GetGPUVirtualAddress());
    s_cmdList->SetGraphicsRootDescriptorTable(1, s_srvHeap->GetGPUDescriptorHandleForHeapStart());

    // Set viewport and scissor (render resolution)
    D3D12_VIEWPORT vp = { 0, 0, (float)renderW, (float)renderH, 0, 1 };
    D3D12_RECT scissor = { 0, 0, (LONG)renderW, (LONG)renderH };
    s_cmdList->RSSetViewports(1, &vp);
    s_cmdList->RSSetScissorRects(1, &scissor);

    // Draw static geometry (room)
    s_cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    s_cmdList->IASetVertexBuffers(0, 1, &s_vbView);
    s_cmdList->IASetIndexBuffer(&s_ibView);
    s_cmdList->DrawIndexedInstanced(s_indexCount, 1, 0, 0, 0);

    // Draw dynamic cube (vertex shader applies rotation)
    s_cmdList->IASetVertexBuffers(0, 1, &s_vbViewCube);
    s_cmdList->IASetIndexBuffer(&s_ibViewCube);
    s_cmdList->DrawIndexedInstanced(s_indexCountCube, 1, 0, 0, 0);

    // ===== TEMPORAL DENOISING - Copy current frame to history =====
    // The blending happens in the pixel shader (reads history, blends with current)
    // Here we just copy the final result to history for next frame
    if (g_dxrFeatures.enableTemporalDenoise && s_historyBuffer) {
        D3D12_RESOURCE_BARRIER denoiseBarriers[2] = {};

        // Transition backbuffer to copy source
        denoiseBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        denoiseBarriers[0].Transition.pResource = s_renderTargets[s_frameIndex];
        denoiseBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        denoiseBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        denoiseBarriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        // Transition history from PIXEL_SHADER_RESOURCE to COPY_DEST
        // (Always in PIXEL_SHADER_RESOURCE after rendering due to pre-draw transition)
        denoiseBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        denoiseBarriers[1].Transition.pResource = s_historyBuffer;
        denoiseBarriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        denoiseBarriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        denoiseBarriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

        s_cmdList->ResourceBarrier(2, denoiseBarriers);

        // Copy current frame to history for next frame's blending
        s_cmdList->CopyResource(s_historyBuffer, s_renderTargets[s_frameIndex]);

        // Transition back: backbuffer to render target, history to shader resource
        denoiseBarriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        denoiseBarriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        denoiseBarriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        denoiseBarriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;

        s_cmdList->ResourceBarrier(2, denoiseBarriers);

        s_historyValid = true;
    }

    // ===== TEXT RENDERING =====
    // Cache all settings to detect changes
    static bool s_cachedShadows = false, s_cachedSoftShadows = false;
    static bool s_cachedAO = false, s_cachedGI = false, s_cachedLighting = false;
    bool settingsChanged = (s_cachedShadows != g_dxrFeatures.rtShadows ||
                            s_cachedSoftShadows != g_dxrFeatures.rtSoftShadows || s_cachedAO != g_dxrFeatures.rtAO ||
                            s_cachedGI != g_dxrFeatures.rtGI || s_cachedLighting != g_dxrFeatures.rtLighting);

    if (fps != s_cachedFps || settingsChanged) {
        s_cachedFps = fps;
        s_cachedShadows = g_dxrFeatures.rtShadows;
        s_cachedSoftShadows = g_dxrFeatures.rtSoftShadows;
        s_cachedAO = g_dxrFeatures.rtAO;
        s_cachedGI = g_dxrFeatures.rtGI;
        s_cachedLighting = g_dxrFeatures.rtLighting;
        s_textVertCount = 0;

        char gpuNameA[128] = {};
        size_t converted = 0;
        wcstombs_s(&converted, gpuNameA, 128, s_gpuName.c_str(), 127);

        char buf[256];
        float y = 10.0f;

        // Shadow + Text (basic info)
        sprintf_s(buf, sizeof(buf), "API: D3D12 + DXR 1.1 (RayQuery)");
        DrawTextRT(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawTextRT(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;
        sprintf_s(buf, sizeof(buf), "GPU: %s", gpuNameA); DrawTextRT(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawTextRT(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;
        sprintf_s(buf, sizeof(buf), "FPS: %d", fps); DrawTextRT(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawTextRT(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;
        sprintf_s(buf, sizeof(buf), "Triangles: %u", (s_indexCount + s_indexCountCube) / 3); DrawTextRT(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawTextRT(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;
        sprintf_s(buf, sizeof(buf), "Resolution: %dx%d", W, H);
        DrawTextRT(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawTextRT(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 20;

        // Build enabled features string
        char features[256] = "";
        if (g_dxrFeatures.rtLighting) strcat_s(features, "Spot ");
        if (g_dxrFeatures.rtShadows) strcat_s(features, g_dxrFeatures.rtSoftShadows ? "SoftShadow " : "Shadow ");
        if (g_dxrFeatures.rtAO) strcat_s(features, "AO ");
        if (g_dxrFeatures.rtGI) strcat_s(features, "GI ");
        if (strlen(features) == 0) strcpy_s(features, "None");

        sprintf_s(buf, sizeof(buf), "RT Features: %s", features);
        DrawTextRT(buf, 11, y+1, 0, 0, 0, 1, 1.5f);
        DrawTextRT(buf, 10, y, 0.5f, 1.0f, 0.5f, 1, 1.5f);  // Green tint for features

        memcpy(s_textVBMapped, s_textVerts, s_textVertCount * sizeof(TextVert));
    }

    // Set viewport to display resolution for text
    D3D12_VIEWPORT textVp = { 0, 0, (float)W, (float)H, 0, 1 };
    D3D12_RECT textScissor = { 0, 0, (LONG)W, (LONG)H };
    s_cmdList->RSSetViewports(1, &textVp);
    s_cmdList->RSSetScissorRects(1, &textScissor);

    if (s_textVertCount > 0) {
        s_cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        s_cmdList->SetPipelineState(s_textPso);
        s_cmdList->SetGraphicsRootSignature(s_textRootSig);
        ID3D12DescriptorHeap* textHeaps[] = { s_textSrvHeap };
        s_cmdList->SetDescriptorHeaps(1, textHeaps);
        s_cmdList->SetGraphicsRootDescriptorTable(0, s_textSrvHeap->GetGPUDescriptorHandleForHeapStart());
        s_cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        s_cmdList->IASetVertexBuffers(0, 1, &s_textVBView);
        s_cmdList->DrawInstanced(s_textVertCount, 1, 0, 0);
    }

    // Transition to present
    barrier.Transition.pResource = s_renderTargets[s_frameIndex];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_cmdList->ResourceBarrier(1, &barrier);

    // Execute
    s_cmdList->Close();
    ID3D12CommandList* lists[] = { s_cmdList };
    s_cmdQueue->ExecuteCommandLists(1, lists);

    // Present
    s_swapChain->Present(0, DXGI_PRESENT_ALLOW_TEARING);
    MoveToNextFrameRT();
}

// ============== CLEANUP ==============
void CleanupD3D12RT() {
    WaitForGpuRT();

    // Text rendering
    if (s_textVB) s_textVB->Release();
    if (s_fontTexture) s_fontTexture->Release();
    if (s_textSrvHeap) s_textSrvHeap->Release();
    if (s_textPso) s_textPso->Release();
    if (s_textRootSig) s_textRootSig->Release();

    // Pipeline
    if (s_pso) s_pso->Release();
    if (s_rootSig) s_rootSig->Release();
    if (s_srvHeap) s_srvHeap->Release();

    // Ray tracing - Unmap instance buffer first
    if (s_instanceBuffer && s_instanceMapped) {
        s_instanceBuffer->Unmap(0, nullptr);
        s_instanceMapped = nullptr;
    }

    // Ray tracing resources
    if (s_instanceBuffer) s_instanceBuffer->Release();
    if (s_scratchBuffer) s_scratchBuffer->Release();
    if (s_tlasBuffer) s_tlasBuffer->Release();
    if (s_blasBufferStatic) s_blasBufferStatic->Release();
    if (s_blasBufferCube) s_blasBufferCube->Release();

    // Buffers
    if (s_constantBuffer) s_constantBuffer->Release();
    if (s_indexBufferStatic) s_indexBufferStatic->Release();
    if (s_vertexBufferStatic) s_vertexBufferStatic->Release();
    if (s_indexBufferCube) s_indexBufferCube->Release();
    if (s_vertexBufferCube) s_vertexBufferCube->Release();
    if (s_indexBuffer) s_indexBuffer->Release();
    if (s_vertexBuffer) s_vertexBuffer->Release();

    // Synchronization
    if (s_fenceEvent) CloseHandle(s_fenceEvent);
    if (s_fence) s_fence->Release();
    for (int i = 0; i < 3; i++) if (s_cmdAlloc[i]) s_cmdAlloc[i]->Release();
    if (s_cmdList) s_cmdList->Release();

    // Render targets
    if (s_depthStencil) s_depthStencil->Release();
    if (s_historyBuffer) { s_historyBuffer->Release(); s_historyBuffer = nullptr; }
    s_historyValid = false;
    if (s_dsvHeap) s_dsvHeap->Release();
    for (int i = 0; i < 3; i++) if (s_renderTargets[i]) s_renderTargets[i]->Release();
    if (s_rtvHeap) s_rtvHeap->Release();

    // Core
    if (s_swapChain) s_swapChain->Release();
    if (s_cmdQueue) s_cmdQueue->Release();
    if (s_device) s_device->Release();

    // Reset all pointers
    s_device = nullptr;
    s_cmdQueue = nullptr;
    s_swapChain = nullptr;
    s_frameIndex = 0;
    memset(s_cmdAlloc, 0, sizeof(s_cmdAlloc));
    memset(s_renderTargets, 0, sizeof(s_renderTargets));
    memset(s_fenceValues, 0, sizeof(s_fenceValues));

    Log("[INFO] D3D12 + Ray Tracing cleanup complete\n");
}
