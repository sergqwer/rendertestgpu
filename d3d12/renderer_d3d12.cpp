// ============== D3D12 BASE RENDERER ==============
// Direct3D 12 renderer implementation

#include "../common.h"
#include "d3d12_shared.h"
#include "renderer_d3d12.h"
#include "../shaders/d3d11_shaders.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <vector>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;

// ============== LOCAL STRUCTS ==============
struct Vert {
    XMFLOAT3 p, n;
    UINT cubeID;
};

struct CB {
    float time;
    float _pad[3];
};

// ============== GEOMETRY GENERATION ==============
static void GenRoundedFace(float size, int seg, XMFLOAT3 offset, int faceIdx,
    float edgeRadius[4], UINT cubeID, std::vector<Vert>& verts, std::vector<UINT>& inds)
{
    UINT base = (UINT)verts.size();
    float h = size / 2;

    XMFLOAT3 faceN[6] = {{0,0,1},{0,0,-1},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}};
    XMFLOAT3 faceU[6] = {{-1,0,0},{1,0,0},{0,0,1},{0,0,-1},{1,0,0},{1,0,0}};
    XMFLOAT3 faceV[6] = {{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,0,1},{0,0,-1}};

    XMFLOAT3 fn = faceN[faceIdx], fu = faceU[faceIdx], fv = faceV[faceIdx];

    for (int j = 0; j <= seg; j++) {
        for (int i = 0; i <= seg; i++) {
            float u = (float)i / seg * 2 - 1;
            float vv = (float)j / seg * 2 - 1;

            float px = u * h, py = vv * h;
            float pz = h;
            float nx = 0, ny = 0, nz = 1;

            float rU_raw = (u > 0) ? edgeRadius[0] : edgeRadius[1];
            float rV_raw = (vv > 0) ? edgeRadius[2] : edgeRadius[3];
            float rU = fabsf(rU_raw), rV = fabsf(rV_raw);
            bool outerU = (rU_raw > 0), outerV = (rV_raw > 0);

            if (rU > 0 || rV > 0) {
                float innerU = h - rU, innerV = h - rV;
                float dx = (rU > 0) ? fmaxf(0, fabsf(px) - innerU) : 0;
                float dy = (rV > 0) ? fmaxf(0, fabsf(py) - innerV) : 0;

                if (dx > 0 || dy > 0) {
                    bool isCorner = (dx > 0 && dy > 0);
                    bool sphericalCorner = isCorner && (outerU || outerV);

                    if (sphericalCorner) {
                        float r = fmaxf(rU, rV);
                        float dist = sqrtf(dx*dx + dy*dy);
                        if (dist > r) { dx = dx * r / dist; dy = dy * r / dist; }
                        float curveZ = sqrtf(fmaxf(0, r*r - dx*dx - dy*dy));
                        pz = (h - r) + curveZ;
                        px = (u > 0 ? 1 : -1) * (innerU + dx);
                        py = (vv > 0 ? 1 : -1) * (innerV + dy);
                        nx = (u > 0 ? 1 : -1) * dx / r;
                        ny = (vv > 0 ? 1 : -1) * dy / r;
                        nz = curveZ / r;
                    } else if (isCorner) {
                        if (dx >= dy) {
                            float curveZ = sqrtf(fmaxf(0, rU*rU - dx*dx));
                            pz = (h - rU) + curveZ;
                            px = (u > 0 ? 1 : -1) * (innerU + dx);
                            nx = (u > 0 ? 1 : -1) * dx / rU;
                            nz = curveZ / rU;
                        } else {
                            float curveZ = sqrtf(fmaxf(0, rV*rV - dy*dy));
                            pz = (h - rV) + curveZ;
                            py = (vv > 0 ? 1 : -1) * (innerV + dy);
                            ny = (vv > 0 ? 1 : -1) * dy / rV;
                            nz = curveZ / rV;
                        }
                    } else {
                        float r = (dx > 0) ? rU : rV;
                        float d = (dx > 0) ? dx : dy;
                        float curveZ = sqrtf(fmaxf(0, r*r - d*d));
                        pz = (h - r) + curveZ;
                        if (dx > 0) { px = (u > 0 ? 1 : -1) * (innerU + dx); nx = (u > 0 ? 1 : -1) * dx / r; }
                        else { py = (vv > 0 ? 1 : -1) * (innerV + dy); ny = (vv > 0 ? 1 : -1) * dy / r; }
                        nz = curveZ / r;
                    }
                }
            }

            Vert vert;
            vert.p.x = offset.x + px*fu.x + py*fv.x + pz*fn.x;
            vert.p.y = offset.y + px*fu.y + py*fv.y + pz*fn.y;
            vert.p.z = offset.z + px*fu.z + py*fv.z + pz*fn.z;

            float nnx = nx*fu.x + ny*fv.x + nz*fn.x;
            float nny = nx*fu.y + ny*fv.y + nz*fn.y;
            float nnz = nx*fu.z + ny*fv.z + nz*fn.z;
            float len = sqrtf(nnx*nnx + nny*nny + nnz*nnz);
            if (len < 0.001f) len = 1;
            vert.n = {nnx/len, nny/len, nnz/len};
            vert.cubeID = cubeID;
            verts.push_back(vert);
        }
    }

    for (int j = 0; j < seg; j++) {
        for (int i = 0; i < seg; i++) {
            UINT idx = base + j * (seg + 1) + i;
            inds.push_back(idx); inds.push_back(idx + seg + 1); inds.push_back(idx + 1);
            inds.push_back(idx + 1); inds.push_back(idx + seg + 1); inds.push_back(idx + seg + 2);
        }
    }
}

static void BuildAllGeometry(std::vector<Vert>& verts, std::vector<UINT>& inds)
{
    float cubeSize = 0.95f;
    float outerR = 0.12f, innerR = -0.12f;
    float half = cubeSize / 2;
    int seg = 20;

    int coords[8][3] = {
        {-1, +1, +1}, {+1, +1, +1}, {-1, -1, +1}, {+1, -1, +1},
        {-1, +1, -1}, {+1, +1, -1}, {-1, -1, -1}, {+1, -1, -1},
    };

    for (int c = 0; c < 8; c++) {
        int cx = coords[c][0], cy = coords[c][1], cz = coords[c][2];
        XMFLOAT3 pos = {cx * half, cy * half, cz * half};

        bool renderFace[6] = {(cz > 0), (cz < 0), (cx > 0), (cx < 0), (cy > 0), (cy < 0)};

        for (int f = 0; f < 6; f++) {
            if (!renderFace[f]) continue;

            float er[4];
            switch (f) {
                case 0: er[0] = (cx < 0) ? outerR : innerR; er[1] = (cx > 0) ? outerR : innerR;
                        er[2] = (cy > 0) ? outerR : innerR; er[3] = (cy < 0) ? outerR : innerR; break;
                case 1: er[0] = (cx > 0) ? outerR : innerR; er[1] = (cx < 0) ? outerR : innerR;
                        er[2] = (cy > 0) ? outerR : innerR; er[3] = (cy < 0) ? outerR : innerR; break;
                case 2: er[0] = (cz > 0) ? outerR : innerR; er[1] = (cz < 0) ? outerR : innerR;
                        er[2] = (cy > 0) ? outerR : innerR; er[3] = (cy < 0) ? outerR : innerR; break;
                case 3: er[0] = (cz < 0) ? outerR : innerR; er[1] = (cz > 0) ? outerR : innerR;
                        er[2] = (cy > 0) ? outerR : innerR; er[3] = (cy < 0) ? outerR : innerR; break;
                case 4: er[0] = (cx > 0) ? outerR : innerR; er[1] = (cx < 0) ? outerR : innerR;
                        er[2] = (cz > 0) ? outerR : innerR; er[3] = (cz < 0) ? outerR : innerR; break;
                case 5: er[0] = (cx > 0) ? outerR : innerR; er[1] = (cx < 0) ? outerR : innerR;
                        er[2] = (cz < 0) ? outerR : innerR; er[3] = (cz > 0) ? outerR : innerR; break;
            }
            GenRoundedFace(cubeSize, seg, pos, f, er, c, verts, inds);
        }
    }
}

// ============== SYNCHRONIZATION (non-static, declared in d3d12_shared.h) ==============
void WaitForGpu()
{
    if (!cmdQueue || !fence) return;
    const UINT64 fenceVal = fenceValues[frameIndex];
    cmdQueue->Signal(fence, fenceVal);
    if (fence->GetCompletedValue() < fenceVal) {
        fence->SetEventOnCompletion(fenceVal, fenceEvent);
        WaitForSingleObject(fenceEvent, INFINITE);
    }
    fenceValues[frameIndex]++;
}

void MoveToNextFrame()
{
    const UINT64 currentFenceValue = fenceValues[frameIndex];
    cmdQueue->Signal(fence, currentFenceValue);
    frameIndex = swap12->GetCurrentBackBufferIndex();
    if (fence->GetCompletedValue() < fenceValues[frameIndex]) {
        fence->SetEventOnCompletion(fenceValues[frameIndex], fenceEvent);
        WaitForSingleObject(fenceEvent, INFINITE);
    }
    fenceValues[frameIndex] = currentFenceValue + 1;
}

// ============== TEXT RENDERING ==============
// Non-static - exported for use by PT and DLSS renderers
void DrawTextDirect(const char* text, float x, float y, float r, float g, float b, float a, float scale)
{
    const int FONT_COLS = 16;
    const float CHAR_W = 8.0f * scale, CHAR_H = 8.0f * scale;
    const float LINE_H = CHAR_H * 1.4f;
    const float TEX_W = 128.0f, TEX_H = 48.0f;

    float cx = x, cy = y;

    for (const char* p = text; *p && g_textVertCount < MAX_TEXT_VERTS - 6; p++) {
        if (*p == '\n') { cx = x; cy += LINE_H; continue; }
        if (*p < 32 || *p > 127) continue;

        int idx = *p - 32;
        int col = idx % FONT_COLS, row = idx / FONT_COLS;
        float u0 = col * 8.0f / TEX_W, v0 = row * 8.0f / TEX_H;
        float u1 = u0 + 8.0f / TEX_W, v1 = v0 + 8.0f / TEX_H;

        // Two triangles per character (6 vertices)
        g_textVerts[g_textVertCount++] = {cx, cy, u0, v0, r, g, b, a};
        g_textVerts[g_textVertCount++] = {cx + CHAR_W, cy, u1, v0, r, g, b, a};
        g_textVerts[g_textVertCount++] = {cx, cy + CHAR_H, u0, v1, r, g, b, a};
        g_textVerts[g_textVertCount++] = {cx + CHAR_W, cy, u1, v0, r, g, b, a};
        g_textVerts[g_textVertCount++] = {cx + CHAR_W, cy + CHAR_H, u1, v1, r, g, b, a};
        g_textVerts[g_textVertCount++] = {cx, cy + CHAR_H, u0, v1, r, g, b, a};

        cx += CHAR_W;
    }
}

// Public DrawText12 function (declared in d3d12_shared.h)
void DrawText12(const char* text, float x, float y, float r, float g, float b, float a, float scale)
{
    DrawTextDirect(text, x, y, r, g, b, a, scale);
}

// ============== GPU TEXT INITIALIZATION ==============
// Exported for use by PT and DLSS renderers
bool InitGPUText12()
{
    Log("[INFO] Initializing D3D12 text rendering...\n");
    HRESULT hr;

    // Create SRV heap for font texture (shader visible)
    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = 1;
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = dev12->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&srvHeap12));
    if (FAILED(hr)) { LogHR("CreateSRVHeap", hr); return false; }

    // Text root signature: descriptor table (SRV) + static sampler
    D3D12_DESCRIPTOR_RANGE srvRange = {};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 1;
    srvRange.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER textRootParam = {};
    textRootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    textRootParam.DescriptorTable.NumDescriptorRanges = 1;
    textRootParam.DescriptorTable.pDescriptorRanges = &srvRange;
    textRootParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC staticSampler = {};
    staticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    staticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSampler.ShaderRegister = 0;
    staticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC textRsDesc = {};
    textRsDesc.NumParameters = 1;
    textRsDesc.pParameters = &textRootParam;
    textRsDesc.NumStaticSamplers = 1;
    textRsDesc.pStaticSamplers = &staticSampler;
    textRsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ID3DBlob* sigBlob = nullptr, *errBlob = nullptr;
    hr = D3D12SerializeRootSignature(&textRsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { Log("[ERROR] Text root sig: %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        return false;
    }
    hr = dev12->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&textRootSig12));
    sigBlob->Release();
    if (FAILED(hr)) { LogHR("CreateTextRootSig", hr); return false; }
    Log("[INFO] Text root signature created\n");

    // Compile text shaders
    ID3DBlob* vsBlob = nullptr, *psBlob = nullptr;
    size_t shaderLen = strlen(g_d3d11ShaderCode);
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;

    Log("[INFO] Compiling D3D12 text shaders...\n");
    hr = D3DCompile(g_d3d11ShaderCode, shaderLen, "embedded", nullptr, nullptr, "TextVS", "vs_5_0", flags, 0, &vsBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { Log("[SHADER ERROR] %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        return false;
    }
    hr = D3DCompile(g_d3d11ShaderCode, shaderLen, "embedded", nullptr, nullptr, "TextPS", "ps_5_0", flags, 0, &psBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { Log("[SHADER ERROR] %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        vsBlob->Release(); return false;
    }

    // Text PSO with blending
    D3D12_INPUT_ELEMENT_DESC textLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC textPsoDesc = {};
    textPsoDesc.InputLayout = { textLayout, _countof(textLayout) };
    textPsoDesc.pRootSignature = textRootSig12;
    textPsoDesc.VS = { vsBlob->GetBufferPointer(), vsBlob->GetBufferSize() };
    textPsoDesc.PS = { psBlob->GetBufferPointer(), psBlob->GetBufferSize() };
    textPsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    textPsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    textPsoDesc.RasterizerState.DepthClipEnable = TRUE;
    // Alpha blending
    textPsoDesc.BlendState.RenderTarget[0].BlendEnable = TRUE;
    textPsoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
    textPsoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    textPsoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    textPsoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    textPsoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    textPsoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    textPsoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    textPsoDesc.DepthStencilState.DepthEnable = FALSE;
    textPsoDesc.SampleMask = UINT_MAX;
    textPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    textPsoDesc.NumRenderTargets = 1;
    textPsoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    textPsoDesc.SampleDesc.Count = 1;

    hr = dev12->CreateGraphicsPipelineState(&textPsoDesc, IID_PPV_ARGS(&textPso));
    vsBlob->Release(); psBlob->Release();
    if (FAILED(hr)) { LogHR("CreateTextPSO", hr); return false; }
    Log("[INFO] Text PSO created\n");

    // Create font texture (16x6 characters = 128x48 pixels)
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

    // Create font texture resource
    D3D12_HEAP_PROPERTIES defaultHeap = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = TEX_W;
    texDesc.Height = TEX_H;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    hr = dev12->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&fontTex12));
    if (FAILED(hr)) { LogHR("CreateFontTexture", hr); return false; }

    // Upload texture via upload buffer
    UINT64 uploadSize = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    dev12->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &uploadSize);

    D3D12_HEAP_PROPERTIES uploadHeap = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_RESOURCE_DESC uploadBufDesc = {};
    uploadBufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    uploadBufDesc.Width = uploadSize;
    uploadBufDesc.Height = 1;
    uploadBufDesc.DepthOrArraySize = 1;
    uploadBufDesc.MipLevels = 1;
    uploadBufDesc.SampleDesc.Count = 1;
    uploadBufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource* uploadBuf = nullptr;
    hr = dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &uploadBufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf));
    if (FAILED(hr)) { LogHR("CreateUploadBuffer", hr); return false; }

    // Copy texture data to upload buffer
    void* mapped;
    uploadBuf->Map(0, nullptr, &mapped);
    BYTE* destRow = (BYTE*)mapped + footprint.Offset;
    for (UINT y = 0; y < TEX_H; y++) {
        memcpy(destRow + y * footprint.Footprint.RowPitch, texData + y * TEX_W, TEX_W);
    }
    uploadBuf->Unmap(0, nullptr);

    // Execute copy command
    cmdAlloc[0]->Reset();
    cmdList->Reset(cmdAlloc[0], nullptr);

    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = fontTex12;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.SubresourceIndex = 0;
    src.pResource = uploadBuf;
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = footprint;

    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    // Transition to shader resource
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = fontTex12;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    cmdList->Close();
    ID3D12CommandList* cmdLists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, cmdLists);

    // Wait for upload to complete
    WaitForGpu();
    uploadBuf->Release();

    // Create SRV for font texture
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R8_UNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 1;
    dev12->CreateShaderResourceView(fontTex12, &srvDesc, srvHeap12->GetCPUDescriptorHandleForHeapStart());

    // Create dynamic text vertex buffer (max 1000 characters = 6000 vertices)
    UINT textVbSize = 6000 * sizeof(TextVert);
    D3D12_RESOURCE_DESC bufDesc = {};
    bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Width = textVbSize;
    bufDesc.Height = 1;
    bufDesc.DepthOrArraySize = 1;
    bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    hr = dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&textVB12));
    if (FAILED(hr)) { LogHR("CreateTextVB", hr); return false; }

    textVbView12.BufferLocation = textVB12->GetGPUVirtualAddress();
    textVbView12.SizeInBytes = textVbSize;
    textVbView12.StrideInBytes = sizeof(TextVert);

    // Persistent map - never unmap
    textVB12->Map(0, nullptr, &textVbMapped12);

    Log("[INFO] D3D12 text rendering initialized\n");
    return true;
}

// ============== INITIALIZATION ==============
bool InitD3D12(HWND hwnd)
{
    Log("[INFO] Initializing Direct3D 12...\n");
    HRESULT hr;

    // Get adapter
    IDXGIAdapter1* selectedAdapter = nullptr;
    if (g_settings.selectedGPU >= 0 && g_settings.selectedGPU < (int)g_gpuList.size()) {
        selectedAdapter = g_gpuList[g_settings.selectedGPU].adapter;
        gpuName = g_gpuList[g_settings.selectedGPU].name;
        char gpuNameA[128]; size_t conv;
        wcstombs_s(&conv, gpuNameA, sizeof(gpuNameA), gpuName.c_str(), _TRUNCATE);
        Log("[INFO] Selected GPU: %s\n", gpuNameA);
    }

    // Create device
    hr = D3D12CreateDevice(selectedAdapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev12));
    if (FAILED(hr)) { LogHR("D3D12CreateDevice", hr); return false; }
    Log("[INFO] D3D12 device created\n");

    // Command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = dev12->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&cmdQueue));
    if (FAILED(hr)) { LogHR("CreateCommandQueue", hr); return false; }

    // Swap chain with tearing support check
    IDXGIFactory5* factory5 = nullptr;
    CreateDXGIFactory1(IID_PPV_ARGS(&factory5));

    // Check tearing support
    BOOL tearingSupport = FALSE;
    if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearingSupport, sizeof(tearingSupport)))) {
        g_tearingSupported12 = (tearingSupport == TRUE);
    }
    Log("[INFO] Tearing support: %s\n", g_tearingSupported12 ? "YES" : "NO");

    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.Width = W; scd.Height = H;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = FRAME_COUNT;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scd.Flags = g_tearingSupported12 ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    IDXGISwapChain1* swap1 = nullptr;
    hr = factory5->CreateSwapChainForHwnd(cmdQueue, hwnd, &scd, nullptr, nullptr, &swap1);
    factory5->Release();
    if (FAILED(hr)) { LogHR("CreateSwapChain", hr); return false; }
    swap1->QueryInterface(IID_PPV_ARGS(&swap12));
    swap1->Release();
    frameIndex = swap12->GetCurrentBackBufferIndex();
    Log("[INFO] Swap chain created (BufferCount=%d)\n", FRAME_COUNT);

    // RTV heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = FRAME_COUNT;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    dev12->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap12));
    rtvDescSize = dev12->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // Create RTVs
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap12->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < FRAME_COUNT; i++) {
        swap12->GetBuffer(i, IID_PPV_ARGS(&renderTargets12[i]));
        dev12->CreateRenderTargetView(renderTargets12[i], nullptr, rtvHandle);
        rtvHandle.ptr += rtvDescSize;
    }

    // DSV heap & depth buffer
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dev12->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap12));

    D3D12_HEAP_PROPERTIES heapProps = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC dsDesc = {};
    dsDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    dsDesc.Width = W; dsDesc.Height = H; dsDesc.DepthOrArraySize = 1;
    dsDesc.MipLevels = 1; dsDesc.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    dsDesc.SampleDesc.Count = 1;
    dsDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clearVal = {}; clearVal.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; clearVal.DepthStencil.Depth = 1.0f;
    dev12->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &dsDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clearVal, IID_PPV_ARGS(&depthStencil12));
    dev12->CreateDepthStencilView(depthStencil12, nullptr, dsvHeap12->GetCPUDescriptorHandleForHeapStart());

    // Command allocators
    for (UINT i = 0; i < FRAME_COUNT; i++) {
        dev12->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&cmdAlloc[i]));
    }

    // Fence
    dev12->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    fenceValues[0] = fenceValues[1] = fenceValues[2] = 1;
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // Root signature (simple: 1 CBV at b0)
    D3D12_ROOT_PARAMETER rootParam = {};
    rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParam.Descriptor.ShaderRegister = 0;
    rootParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &rootParam;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    ID3DBlob* sigBlob = nullptr, *errBlob = nullptr;
    hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sigBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { Log("[ERROR] Root sig: %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        return false;
    }
    dev12->CreateRootSignature(0, sigBlob->GetBufferPointer(), sigBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig));
    sigBlob->Release();
    Log("[INFO] Root signature created\n");

    // Compile shaders
    ID3DBlob* vsBlob = nullptr, *psBlob = nullptr;
    size_t shaderLen = strlen(g_d3d11ShaderCode);
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;

    Log("[INFO] Compiling D3D12 shaders...\n");
    hr = D3DCompile(g_d3d11ShaderCode, shaderLen, "embedded", nullptr, nullptr, "VS", "vs_5_0", flags, 0, &vsBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { Log("[SHADER ERROR] %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        return false;
    }
    hr = D3DCompile(g_d3d11ShaderCode, shaderLen, "embedded", nullptr, nullptr, "PS", "ps_5_0", flags, 0, &psBlob, &errBlob);
    if (FAILED(hr)) {
        if (errBlob) { Log("[SHADER ERROR] %s\n", (char*)errBlob->GetBufferPointer()); errBlob->Release(); }
        vsBlob->Release(); return false;
    }

    // PSO
    D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"CUBEID", 0, DXGI_FORMAT_R32_UINT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = { inputLayout, _countof(inputLayout) };
    psoDesc.pRootSignature = rootSig;
    psoDesc.VS = { vsBlob->GetBufferPointer(), vsBlob->GetBufferSize() };
    psoDesc.PS = { psBlob->GetBufferPointer(), psBlob->GetBufferSize() };
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
    psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
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

    hr = dev12->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso));
    vsBlob->Release(); psBlob->Release();
    if (FAILED(hr)) { LogHR("CreatePSO", hr); return false; }
    Log("[INFO] PSO created\n");

    // Command list
    dev12->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, cmdAlloc[0], pso, IID_PPV_ARGS(&cmdList));
    cmdList->Close();

    // Build geometry and upload
    std::vector<Vert> verts;
    std::vector<UINT> inds;
    BuildAllGeometry(verts, inds);
    totalIndices12 = (UINT)inds.size();
    totalVertices12 = (UINT)verts.size();

    // Upload VB
    UINT vbSize = (UINT)(verts.size() * sizeof(Vert));
    D3D12_HEAP_PROPERTIES uploadHeap = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_RESOURCE_DESC bufDesc = {}; bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Width = vbSize; bufDesc.Height = 1; bufDesc.DepthOrArraySize = 1; bufDesc.MipLevels = 1;
    bufDesc.SampleDesc.Count = 1; bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vb12));
    void* mapped; vb12->Map(0, nullptr, &mapped);
    memcpy(mapped, verts.data(), vbSize);
    vb12->Unmap(0, nullptr);
    vbView12.BufferLocation = vb12->GetGPUVirtualAddress();
    vbView12.SizeInBytes = vbSize;
    vbView12.StrideInBytes = sizeof(Vert);

    // Upload IB
    UINT ibSize = (UINT)(inds.size() * sizeof(UINT));
    bufDesc.Width = ibSize;
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&ib12));
    ib12->Map(0, nullptr, &mapped);
    memcpy(mapped, inds.data(), ibSize);
    ib12->Unmap(0, nullptr);
    ibView12.BufferLocation = ib12->GetGPUVirtualAddress();
    ibView12.SizeInBytes = ibSize;
    ibView12.Format = DXGI_FORMAT_R32_UINT;

    // Upload CB with persistent mapping
    bufDesc.Width = 256; // Aligned
    dev12->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cbUpload12));
    cbUpload12->Map(0, nullptr, &cbMapped12); // Persistent map - never unmap

    // Initialize text rendering
    if (!InitGPUText12()) {
        Log("[WARN] Text rendering initialization failed, continuing without text\n");
    }

    Log("[INFO] D3D12 initialization complete\n");
    return true;
}

// ============== RENDERING ==============
void RenderD3D12()
{
    cmdAlloc[frameIndex]->Reset();
    cmdList->Reset(cmdAlloc[frameIndex], pso);

    // Update CB using persistent mapping (no Map/Unmap overhead)
    LARGE_INTEGER nowTime;
    QueryPerformanceCounter(&nowTime);
    float t = (float)(nowTime.QuadPart - g_startTime.QuadPart) / g_perfFreq.QuadPart;
    CB cbData = { t };
    memcpy(cbMapped12, &cbData, sizeof(CB));

    // Transition to render target
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = renderTargets12[frameIndex];
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmdList->ResourceBarrier(1, &barrier);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap12->GetCPUDescriptorHandleForHeapStart();
    rtvHandle.ptr += frameIndex * rtvDescSize;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap12->GetCPUDescriptorHandleForHeapStart();

    float gray[] = { 0.5f, 0.5f, 0.5f, 1.0f };
    cmdList->ClearRenderTargetView(rtvHandle, gray, 0, nullptr);
    cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
    cmdList->SetGraphicsRootSignature(rootSig);
    cmdList->SetGraphicsRootConstantBufferView(0, cbUpload12->GetGPUVirtualAddress());

    D3D12_VIEWPORT vp = { 0, 0, (float)W, (float)H, 0, 1 };
    D3D12_RECT scissor = { 0, 0, (LONG)W, (LONG)H };
    cmdList->RSSetViewports(1, &vp);
    cmdList->RSSetScissorRects(1, &scissor);

    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmdList->IASetVertexBuffers(0, 1, &vbView12);
    cmdList->IASetIndexBuffer(&ibView12);
    cmdList->DrawIndexedInstanced(totalIndices12, 1, 0, 0, 0);

    // ============== TEXT RENDERING (GPU cached) ==============
    // Only rebuild text when FPS changes (once per second) - NOT every frame!
    if (fps != g_cachedFps || g_textNeedsRebuild) {
        g_cachedFps = fps;
        g_textNeedsRebuild = false;

        // Build info text (only when changed)
        static char gpuNameA[128] = {0};
        if (gpuNameA[0] == 0) {
            size_t converted;
            wcstombs_s(&converted, gpuNameA, sizeof(gpuNameA), gpuName.c_str(), _TRUNCATE);
        }

        char infoText[512];
        sprintf_s(infoText,
            "API: Direct3D 12\n"
            "GPU: %s\n"
            "FPS: %d\n"
            "Triangles: %u\n"
            "Resolution: %ux%u",
            gpuNameA, fps, totalIndices12 / 3, W, H);

        // Build text vertices (only when changed)
        g_textVertCount = 0;
        DrawTextDirect(infoText, 12.0f, 12.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.5f); // Shadow
        DrawTextDirect(infoText, 10.0f, 10.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.5f); // Main text

        // Upload to GPU (only when changed)
        memcpy(textVbMapped12, g_textVerts, g_textVertCount * sizeof(TextVert));
    }

    // Always draw text (uses cached vertex data)
    if (g_textVertCount > 0 && textPso && textRootSig12 && srvHeap12) {
        cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        cmdList->SetPipelineState(textPso);
        cmdList->SetGraphicsRootSignature(textRootSig12);

        ID3D12DescriptorHeap* heaps[] = { srvHeap12 };
        cmdList->SetDescriptorHeaps(1, heaps);
        cmdList->SetGraphicsRootDescriptorTable(0, srvHeap12->GetGPUDescriptorHandleForHeapStart());

        cmdList->IASetVertexBuffers(0, 1, &textVbView12);
        cmdList->IASetIndexBuffer(nullptr);
        cmdList->DrawInstanced(g_textVertCount, 1, 0, 0);
    }

    // Transition to present
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    cmdList->ResourceBarrier(1, &barrier);

    cmdList->Close();
    ID3D12CommandList* cmdLists[] = { cmdList };
    cmdQueue->ExecuteCommandLists(1, cmdLists);

    UINT presentFlags = g_tearingSupported12 ? DXGI_PRESENT_ALLOW_TEARING : 0;
    swap12->Present(0, presentFlags);
    MoveToNextFrame();
}

// ============== CLEANUP ==============
void CleanupD3D12()
{
    WaitForGpu();
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
    if (pso) pso->Release();
    if (rootSig) rootSig->Release();
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
}
