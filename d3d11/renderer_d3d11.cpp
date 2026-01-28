// ============== D3D11 RENDERER IMPLEMENTATION ==============

#include <d3d11_1.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <DirectXMath.h>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")

#include "../common.h"
#include "../shaders/d3d11_shaders.h"

using namespace DirectX;

// ============== LOCAL TYPES ==============

// Vertex with cube ID for GPU color lookup
struct Vert {
    XMFLOAT3 p, n;
    UINT cubeID;
};

// Constant buffer - ONLY dynamic data (everything else in shader)
struct CB {
    float time;
    float _pad[3];
};

// ============== D3D11 GLOBALS ==============

static ID3D11Device* dev = nullptr;
static ID3D11DeviceContext* ctx = nullptr;
static IDXGISwapChain* swap = nullptr;
static ID3D11RenderTargetView* rtv = nullptr;
static ID3D11DepthStencilView* dsv = nullptr;
static ID3D11VertexShader* vs = nullptr;
static ID3D11PixelShader* ps = nullptr;
static ID3D11InputLayout* il = nullptr;
static ID3D11Buffer* vb = nullptr, *ib = nullptr, *cbuf = nullptr;
static UINT totalIndices = 0;
static UINT totalVertices = 0;

// GPU text rendering (D3D11)
static ID3D11VertexShader* textVS = nullptr;
static ID3D11PixelShader* textPS = nullptr;
static ID3D11InputLayout* textIL = nullptr;
static ID3D11Buffer* textVB = nullptr;
static ID3D11Texture2D* fontTex = nullptr;
static ID3D11ShaderResourceView* fontSRV = nullptr;
static ID3D11SamplerState* fontSampler = nullptr;
static ID3D11BlendState* textBlend = nullptr;

// ============== FORWARD DECLARATIONS ==============
static bool InitShaders();
static bool InitGPUText();

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

// ============== TEXT RENDERING ==============

static void DrawTextRaw(const char* text, float x, float y, float r, float g, float b, float a, float scale, std::vector<TextVert>& verts)
{
    const int FONT_COLS = 16;
    const float CHAR_W = 8.0f * scale, CHAR_H = 8.0f * scale;
    const float LINE_H = CHAR_H * 1.4f;  // 40% extra line spacing
    const float TEX_W = 128.0f, TEX_H = 48.0f;

    float cx = x, cy = y;

    for (const char* p = text; *p; p++) {
        if (*p == '\n') { cx = x; cy += LINE_H; continue; }
        if (*p < 32 || *p > 127) continue;

        int idx = *p - 32;
        int col = idx % FONT_COLS, row = idx / FONT_COLS;
        float u0 = col * 8.0f / TEX_W, v0 = row * 8.0f / TEX_H;
        float u1 = u0 + 8.0f / TEX_W, v1 = v0 + 8.0f / TEX_H;

        // Two triangles per character
        verts.push_back({cx, cy, u0, v0, r, g, b, a});
        verts.push_back({cx + CHAR_W, cy, u1, v0, r, g, b, a});
        verts.push_back({cx, cy + CHAR_H, u0, v1, r, g, b, a});
        verts.push_back({cx + CHAR_W, cy, u1, v0, r, g, b, a});
        verts.push_back({cx + CHAR_W, cy + CHAR_H, u1, v1, r, g, b, a});
        verts.push_back({cx, cy + CHAR_H, u0, v1, r, g, b, a});

        cx += CHAR_W;
    }
}

static void DrawTextWithShadow(const char* text, float x, float y, float r, float g, float b, float scale = 2.0f)
{
    std::vector<TextVert> verts;

    // Shadow (dark, offset)
    float shadowOff = scale * 1.5f;
    DrawTextRaw(text, x + shadowOff, y + shadowOff, 0.0f, 0.0f, 0.0f, 0.8f, scale, verts);

    // Main text
    DrawTextRaw(text, x, y, r, g, b, 1.0f, scale, verts);

    if (verts.empty()) return;

    // Update vertex buffer
    D3D11_MAPPED_SUBRESOURCE m;
    ctx->Map(textVB, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    memcpy(m.pData, verts.data(), verts.size() * sizeof(TextVert));
    ctx->Unmap(textVB, 0);

    // Save state
    ID3D11BlendState* oldBlend = nullptr;
    float oldFactor[4]; UINT oldMask;
    ctx->OMGetBlendState(&oldBlend, oldFactor, &oldMask);

    // Set text rendering state
    ctx->OMSetBlendState(textBlend, nullptr, 0xFFFFFFFF);
    ctx->IASetInputLayout(textIL);
    UINT stride = sizeof(TextVert), offset = 0;
    ctx->IASetVertexBuffers(0, 1, &textVB, &stride, &offset);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(textVS, nullptr, 0);
    ctx->PSSetShader(textPS, nullptr, 0);
    ctx->PSSetShaderResources(0, 1, &fontSRV);
    ctx->PSSetSamplers(0, 1, &fontSampler);

    ctx->Draw((UINT)verts.size(), 0);

    // Restore state
    ctx->OMSetBlendState(oldBlend, oldFactor, oldMask);
    if (oldBlend) oldBlend->Release();
}

// ============== INITIALIZATION ==============

bool InitD3D11(HWND hwnd)
{
    Log("[INFO] Initializing Direct3D 11...\n");

    // Use GPU selected from settings dialog
    IDXGIAdapter1* selectedAdapter = nullptr;
    if (g_settings.selectedGPU >= 0 && g_settings.selectedGPU < (int)g_gpuList.size()) {
        selectedAdapter = g_gpuList[g_settings.selectedGPU].adapter;
        gpuName = g_gpuList[g_settings.selectedGPU].name;
        char gpuNameA[128];
        size_t conv;
        wcstombs_s(&conv, gpuNameA, sizeof(gpuNameA), gpuName.c_str(), _TRUNCATE);
        Log("[INFO] Selected GPU: %s\n", gpuNameA);
    }

    // Create device first (without swap chain)
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL featureLevel;
    UINT createFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    Log("[INFO] Creating D3D11 device...\n");
    HRESULT hr = D3D11CreateDevice(
        selectedAdapter,
        D3D_DRIVER_TYPE_UNKNOWN,
        nullptr,
        createFlags,
        featureLevels,
        _countof(featureLevels),
        D3D11_SDK_VERSION,
        &dev,
        &featureLevel,
        &ctx
    );

    if (FAILED(hr)) {
        LogHR("D3D11CreateDevice", hr);
        return false;
    }
    Log("[INFO] D3D11 device created. Feature level: 0x%X\n", featureLevel);

    // Verify actual GPU used
    IDXGIDevice* dxgiDev = nullptr;
    dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDev);
    if (dxgiDev) {
        IDXGIAdapter* actualAdapter = nullptr;
        dxgiDev->GetAdapter(&actualAdapter);
        if (actualAdapter) {
            DXGI_ADAPTER_DESC actualDesc;
            actualAdapter->GetDesc(&actualDesc);
            gpuName = actualDesc.Description;  // Show ACTUAL GPU
            actualAdapter->Release();
        }
    }

    // Get DXGI factory from device (ensures same factory used)
    IDXGIFactory2* factory2 = nullptr;
    if (dxgiDev) {
        IDXGIAdapter* adapter = nullptr;
        dxgiDev->GetAdapter(&adapter);
        if (adapter) {
            adapter->GetParent(__uuidof(IDXGIFactory2), (void**)&factory2);
            adapter->Release();
        }
        dxgiDev->Release();
    }

    if (!factory2) {
        Log("[ERROR] Failed to get DXGI Factory2 from adapter\n");
        return false;
    }

    // Create swap chain using FLIP model (better for hybrid GPU)
    DXGI_SWAP_CHAIN_DESC1 sd = {};
    sd.Width = W;
    sd.Height = H;
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = 2;  // Flip model requires at least 2
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;  // Modern flip model
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

    IDXGISwapChain1* swap1 = nullptr;
    hr = factory2->CreateSwapChainForHwnd(dev, hwnd, &sd, nullptr, nullptr, &swap1);
    factory2->Release();

    if (FAILED(hr)) {
        // Fallback: try without tearing flag
        sd.Flags = 0;
        IDXGIFactory2* factory2b = nullptr;
        IDXGIDevice* dxgiDev2 = nullptr;
        dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDev2);
        if (dxgiDev2) {
            IDXGIAdapter* adapter = nullptr;
            dxgiDev2->GetAdapter(&adapter);
            if (adapter) {
                adapter->GetParent(__uuidof(IDXGIFactory2), (void**)&factory2b);
                adapter->Release();
            }
            dxgiDev2->Release();
        }
        if (factory2b) {
            hr = factory2b->CreateSwapChainForHwnd(dev, hwnd, &sd, nullptr, nullptr, &swap1);
            factory2b->Release();
        }

        if (FAILED(hr)) {
            LogHR("CreateSwapChainForHwnd", hr);
            return false;
        }
        Log("[INFO] Swap chain created (fallback without tearing)\n");
    } else {
        Log("[INFO] Swap chain created with tearing support\n");
    }

    swap1->QueryInterface(__uuidof(IDXGISwapChain), (void**)&swap);
    swap1->Release();

    // Check if tearing (no VSync) is supported
    IDXGIFactory5* factory5 = nullptr;
    IDXGIDevice* dxgiDevTear = nullptr;
    dev->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevTear);
    if (dxgiDevTear) {
        IDXGIAdapter* adapterTear = nullptr;
        dxgiDevTear->GetAdapter(&adapterTear);
        if (adapterTear) {
            adapterTear->GetParent(__uuidof(IDXGIFactory5), (void**)&factory5);
            adapterTear->Release();
        }
        dxgiDevTear->Release();
    }
    if (factory5) {
        BOOL allowTearing = FALSE;
        if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing)))) {
            g_tearingSupported = (allowTearing == TRUE);
        }
        factory5->Release();
    }

    ID3D11Texture2D* bb; swap->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&bb);
    dev->CreateRenderTargetView(bb, 0, &rtv); bb->Release();

    D3D11_TEXTURE2D_DESC td = {}; td.Width = W; td.Height = H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT; td.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    ID3D11Texture2D* ds; dev->CreateTexture2D(&td, 0, &ds);
    dev->CreateDepthStencilView(ds, 0, &dsv); ds->Release();

    ctx->OMSetRenderTargets(1, &rtv, dsv);
    D3D11_VIEWPORT vp = {0, 0, (float)W, (float)H, 0, 1};
    ctx->RSSetViewports(1, &vp);

    // Initialize shaders
    if (!InitShaders()) return false;
    if (!InitGPUText()) return false;

    return true;
}

static bool InitShaders()
{
    ID3DBlob* vsB = nullptr, *psB = nullptr, *err = nullptr;
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
    size_t shaderLen = strlen(g_d3d11ShaderCode);
    HRESULT hr;

    Log("[INFO] Compiling vertex shader VS...\n");
    hr = D3DCompile(g_d3d11ShaderCode, shaderLen, "embedded", nullptr, nullptr, "VS", "vs_5_0", flags, 0, &vsB, &err);
    if (FAILED(hr)) {
        LogHR("D3DCompile VS", hr);
        if (err) { Log("[SHADER ERROR] %s\n", (char*)err->GetBufferPointer()); err->Release(); }
        return false;
    }

    Log("[INFO] Compiling pixel shader PS...\n");
    hr = D3DCompile(g_d3d11ShaderCode, shaderLen, "embedded", nullptr, nullptr, "PS", "ps_5_0", flags, 0, &psB, &err);
    if (FAILED(hr)) {
        LogHR("D3DCompile PS", hr);
        if (err) { Log("[SHADER ERROR] %s\n", (char*)err->GetBufferPointer()); err->Release(); }
        vsB->Release();
        return false;
    }

    dev->CreateVertexShader(vsB->GetBufferPointer(), vsB->GetBufferSize(), 0, &vs);
    dev->CreatePixelShader(psB->GetBufferPointer(), psB->GetBufferSize(), 0, &ps);

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"CUBEID", 0, DXGI_FORMAT_R32_UINT, 0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    dev->CreateInputLayout(layout, 3, vsB->GetBufferPointer(), vsB->GetBufferSize(), &il);
    vsB->Release(); psB->Release();

    std::vector<Vert> verts;
    std::vector<UINT> inds;
    BuildAllGeometry(verts, inds);
    totalIndices = (UINT)inds.size();

    D3D11_BUFFER_DESC bd = {}; bd.Usage = D3D11_USAGE_IMMUTABLE;
    bd.ByteWidth = (UINT)(verts.size() * sizeof(Vert)); bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA init = {verts.data()};
    dev->CreateBuffer(&bd, &init, &vb);

    bd.ByteWidth = (UINT)(inds.size() * sizeof(UINT)); bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    init.pSysMem = inds.data(); dev->CreateBuffer(&bd, &init, &ib);

    bd.ByteWidth = sizeof(CB); bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.Usage = D3D11_USAGE_DYNAMIC; bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    dev->CreateBuffer(&bd, 0, &cbuf);

    return true;
}

static bool InitGPUText()
{
    ID3DBlob* vsB = nullptr, *psB = nullptr, *err = nullptr;
    UINT flags = D3DCOMPILE_ENABLE_STRICTNESS;
    size_t shaderLen = strlen(g_d3d11ShaderCode);
    HRESULT hr;

    Log("[INFO] Compiling text vertex shader TextVS...\n");
    hr = D3DCompile(g_d3d11ShaderCode, shaderLen, "embedded", nullptr, nullptr, "TextVS", "vs_5_0", flags, 0, &vsB, &err);
    if (FAILED(hr)) {
        LogHR("D3DCompile TextVS", hr);
        if (err) { Log("[SHADER ERROR] %s\n", (char*)err->GetBufferPointer()); err->Release(); }
        return false;
    }

    Log("[INFO] Compiling text pixel shader TextPS...\n");
    hr = D3DCompile(g_d3d11ShaderCode, shaderLen, "embedded", nullptr, nullptr, "TextPS", "ps_5_0", flags, 0, &psB, &err);
    if (FAILED(hr)) {
        LogHR("D3DCompile TextPS", hr);
        if (err) { Log("[SHADER ERROR] %s\n", (char*)err->GetBufferPointer()); err->Release(); }
        vsB->Release();
        return false;
    }

    dev->CreateVertexShader(vsB->GetBufferPointer(), vsB->GetBufferSize(), 0, &textVS);
    dev->CreatePixelShader(psB->GetBufferPointer(), psB->GetBufferSize(), 0, &textPS);

    D3D11_INPUT_ELEMENT_DESC textLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0},
    };
    dev->CreateInputLayout(textLayout, 3, vsB->GetBufferPointer(), vsB->GetBufferSize(), &textIL);
    vsB->Release(); psB->Release();

    // Create font texture (16x6 characters = 128x48 pixels for ASCII 32-127)
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

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = TEX_W; td.Height = TEX_H; td.MipLevels = 1; td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R8_UNORM; td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_IMMUTABLE; td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initData = {texData, TEX_W, 0};
    dev->CreateTexture2D(&td, &initData, &fontTex);
    dev->CreateShaderResourceView(fontTex, nullptr, &fontSRV);

    // Sampler
    D3D11_SAMPLER_DESC sampDesc = {};
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    sampDesc.AddressU = sampDesc.AddressV = sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    dev->CreateSamplerState(&sampDesc, &fontSampler);

    // Blend state for text
    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    dev->CreateBlendState(&blendDesc, &textBlend);

    // Dynamic vertex buffer for text (max 1000 characters = 6000 vertices)
    D3D11_BUFFER_DESC vbd = {};
    vbd.ByteWidth = 6000 * sizeof(TextVert);
    vbd.Usage = D3D11_USAGE_DYNAMIC;
    vbd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    dev->CreateBuffer(&vbd, nullptr, &textVB);

    return true;
}

// ============== RENDERING ==============

void RenderD3D11()
{
    ctx->OMSetRenderTargets(1, &rtv, dsv);

    float gray[] = {0.5f, 0.5f, 0.5f, 1};
    ctx->ClearRenderTargetView(rtv, gray);
    ctx->ClearDepthStencilView(dsv, D3D11_CLEAR_DEPTH, 1, 0);

    ctx->IASetInputLayout(il);
    UINT stride = sizeof(Vert), off = 0;
    ctx->IASetVertexBuffers(0, 1, &vb, &stride, &off);
    ctx->IASetIndexBuffer(ib, DXGI_FORMAT_R32_UINT, 0);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(vs, 0, 0); ctx->PSSetShader(ps, 0, 0);
    ctx->VSSetConstantBuffers(0, 1, &cbuf);

    // Update time for rotation
    LARGE_INTEGER nowTime;
    QueryPerformanceCounter(&nowTime);
    float t = (float)(nowTime.QuadPart - g_startTime.QuadPart) / g_perfFreq.QuadPart;
    D3D11_MAPPED_SUBRESOURCE m;
    ctx->Map(cbuf, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    ((CB*)m.pData)->time = t;
    ctx->Unmap(cbuf, 0);

    ctx->DrawIndexed(totalIndices, 0, 0);

    // GPU-based text rendering (no CPU-GPU sync issues)
    ctx->OMSetRenderTargets(1, &rtv, nullptr); // Disable depth for text

    // Convert GPU name to ASCII
    char gpuNameA[128];
    size_t converted;
    wcstombs_s(&converted, gpuNameA, sizeof(gpuNameA), gpuName.c_str(), _TRUNCATE);

    // Build info text
    char infoText[512];
    sprintf_s(infoText,
        "API: Direct3D 11\n"
        "GPU: %s\n"
        "FPS: %d\n"
        "Triangles: %u\n"
        "Resolution: %ux%u",
        gpuNameA, fps, totalIndices / 3, W, H);

    // White text with shadow for better readability
    DrawTextWithShadow(infoText, 10, 10, 1.0f, 1.0f, 1.0f, 1.5f);

    // Present with tearing (no VSync) if supported
    UINT presentFlags = g_tearingSupported ? DXGI_PRESENT_ALLOW_TEARING : 0;
    swap->Present(0, presentFlags);
}

// ============== CLEANUP ==============

void CleanupD3D11()
{
    // GPU text resources
    if (textBlend) textBlend->Release();
    if (fontSampler) fontSampler->Release();
    if (fontSRV) fontSRV->Release();
    if (fontTex) fontTex->Release();
    if (textVB) textVB->Release();
    if (textIL) textIL->Release();
    if (textPS) textPS->Release();
    if (textVS) textVS->Release();

    // Main resources
    if (cbuf) cbuf->Release(); if (ib) ib->Release(); if (vb) vb->Release();
    if (il) il->Release(); if (ps) ps->Release(); if (vs) vs->Release();
    if (dsv) dsv->Release(); if (rtv) rtv->Release();
    if (swap) swap->Release(); if (ctx) ctx->Release(); if (dev) dev->Release();
}
