// ============== MAIN ENTRY POINT ==============
// RenderTestGPU - Multi-API GPU Renderer Test Application
// Supports: D3D11, D3D12, D3D12+RT, D3D12+PT, D3D12+DLSS, OpenGL, Vulkan

#include "common.h"

// Include renderer headers
#include "d3d11/renderer_d3d11.h"
#include "d3d12/renderer_d3d12.h"
#include "opengl/renderer_opengl.h"
#include "vulkan/renderer_vulkan.h"
#include "vulkan/renderer_vulkan_rt.h"
#include "vulkan/renderer_vulkan_rq.h"

#pragma comment(lib, "comctl32.lib")

// Force discrete GPU on hybrid systems
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) DWORD AmdPowerXpressRequestHighPerformance = 0x00000001;
}

// ============== GLOBAL DEFINITIONS ==============
const UINT W = 640;
const UINT H = 480;

std::vector<GPUInfo> g_gpuList;
Settings g_settings;
HINSTANCE g_hInstance;
bool g_tearingSupported = false;
std::wstring gpuName;
int fps = 0;
LARGE_INTEGER g_startTime, g_perfFreq;
HWND g_hMainWnd = nullptr;
static HWND g_hSettingsDlg = nullptr;
static bool g_settingsAccepted = false;
static bool g_settingsDlgClosed = false;
static bool g_inSizeMove = false;

// Vulkan text state
bool g_vkTextInitialized = false;

// ============== COMMAND LINE PARSING ==============
struct CmdLineArgs {
    RendererType renderer = RENDERER_D3D11;
    int gpuIndex = 0;
    bool hasRenderer = false;
    bool hasGpu = false;
    bool skipDialogs = false;
};

static CmdLineArgs g_cmdArgs;

static bool ParseRendererType(const char* str, RendererType& out) {
    if (_stricmp(str, "d3d11") == 0) { out = RENDERER_D3D11; return true; }
    if (_stricmp(str, "d3d12") == 0) { out = RENDERER_D3D12; return true; }
    if (_stricmp(str, "d3d12_dxr10") == 0 || _stricmp(str, "dxr10") == 0) { out = RENDERER_D3D12_DXR10; return true; }
    if (_stricmp(str, "d3d12_rt") == 0 || _stricmp(str, "dxr11") == 0) { out = RENDERER_D3D12_RT; return true; }
    if (_stricmp(str, "d3d12_pt") == 0 || _stricmp(str, "pt") == 0) { out = RENDERER_D3D12_PT; return true; }
    if (_stricmp(str, "d3d12_pt_dlss") == 0 || _stricmp(str, "dlss") == 0) { out = RENDERER_D3D12_PT_DLSS; return true; }
    if (_stricmp(str, "opengl") == 0 || _stricmp(str, "gl") == 0) { out = RENDERER_OPENGL; return true; }
    if (_stricmp(str, "vulkan") == 0 || _stricmp(str, "vk") == 0) { out = RENDERER_VULKAN; return true; }
    if (_stricmp(str, "vulkan_rt") == 0 || _stricmp(str, "vk_rt") == 0) { out = RENDERER_VULKAN_RT; return true; }
    if (_stricmp(str, "vulkan_rq") == 0 || _stricmp(str, "vk_rq") == 0) { out = RENDERER_VULKAN_RQ; return true; }
    return false;
}

static void ParseCommandLine(LPSTR cmdLine) {
    if (!cmdLine || !*cmdLine) return;

    char* cmd = _strdup(cmdLine);
    char* context = nullptr;
    char* token = strtok_s(cmd, " ", &context);

    while (token) {
        // --renderer=vulkan_rt or -r vulkan_rt
        if (strncmp(token, "--renderer=", 11) == 0) {
            if (ParseRendererType(token + 11, g_cmdArgs.renderer)) {
                g_cmdArgs.hasRenderer = true;
                g_cmdArgs.skipDialogs = true;
            }
        }
        else if (strcmp(token, "-r") == 0 || strcmp(token, "--renderer") == 0) {
            token = strtok_s(nullptr, " ", &context);
            if (token && ParseRendererType(token, g_cmdArgs.renderer)) {
                g_cmdArgs.hasRenderer = true;
                g_cmdArgs.skipDialogs = true;
            }
            continue;
        }
        // --gpu=0 or -g 0
        else if (strncmp(token, "--gpu=", 6) == 0) {
            g_cmdArgs.gpuIndex = atoi(token + 6);
            g_cmdArgs.hasGpu = true;
        }
        else if (strcmp(token, "-g") == 0 || strcmp(token, "--gpu") == 0) {
            token = strtok_s(nullptr, " ", &context);
            if (token) {
                g_cmdArgs.gpuIndex = atoi(token);
                g_cmdArgs.hasGpu = true;
            }
            continue;
        }
        // --help or -h
        else if (strcmp(token, "--help") == 0 || strcmp(token, "-h") == 0) {
            MessageBoxA(0,
                "RenderTestGPU - Command Line Options:\n\n"
                "  --renderer=<type> or -r <type>\n"
                "    Renderer types:\n"
                "      d3d11, d3d12, d3d12_dxr10, d3d12_rt,\n"
                "      d3d12_pt, d3d12_pt_dlss, opengl, vulkan, vulkan_rt, vulkan_rq\n"
                "    Short aliases: dxr10, dxr11, pt, dlss, gl, vk, vk_rt, vk_rq\n\n"
                "  --gpu=<index> or -g <index>\n"
                "    GPU index (0 = first GPU)\n\n"
                "Examples:\n"
                "  rendertestgpu.exe --renderer=vulkan_rt\n"
                "  rendertestgpu.exe -r vk_rt -g 0\n",
                "Help", MB_OK);
            free(cmd);
            exit(0);
        }
        token = strtok_s(nullptr, " ", &context);
    }
    free(cmd);
}

// ============== LOGGING ==============
static FILE* g_logFile = nullptr;
static wchar_t g_logPath[MAX_PATH] = {0};

void InitLog()
{
    wchar_t exePath[MAX_PATH];
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    wcscpy_s(g_logPath, exePath);
    wchar_t* dot = wcsrchr(g_logPath, L'.');
    if (dot) wcscpy_s(dot, 20, L"_error.log");
    else wcscat_s(g_logPath, L"_error.log");
}

void Log(const char* fmt, ...)
{
    if (!g_logFile) {
        _wfopen_s(&g_logFile, g_logPath, L"a");
        if (!g_logFile) return;
        time_t now = time(nullptr);
        tm t; localtime_s(&t, &now);
        fprintf(g_logFile, "\n========== %04d-%02d-%02d %02d:%02d:%02d ==========\n",
            t.tm_year+1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
    }
    va_list args;
    va_start(args, fmt);
    vfprintf(g_logFile, fmt, args);
    va_end(args);
    fflush(g_logFile);
}

void LogHR(const char* operation, HRESULT hr)
{
    char msg[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, hr, 0, msg, sizeof(msg), nullptr);
    Log("[ERROR] %s failed: 0x%08X - %s\n", operation, hr, msg);
}

void CloseLog()
{
    if (g_logFile) { fclose(g_logFile); g_logFile = nullptr; }
}

// ============== 8x8 BITMAP FONT DATA ==============
const unsigned char g_font8x8[96][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // Space
    {0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x00}, // !
    {0x6C,0x6C,0x24,0x00,0x00,0x00,0x00,0x00}, // "
    {0x6C,0x6C,0xFE,0x6C,0xFE,0x6C,0x6C,0x00}, // #
    {0x18,0x3E,0x60,0x3C,0x06,0x7C,0x18,0x00}, // $
    {0x00,0x66,0xAC,0xD8,0x36,0x6A,0xCC,0x00}, // %
    {0x38,0x6C,0x68,0x76,0xDC,0xCE,0x7B,0x00}, // &
    {0x18,0x18,0x30,0x00,0x00,0x00,0x00,0x00}, // '
    {0x0C,0x18,0x30,0x30,0x30,0x18,0x0C,0x00}, // (
    {0x30,0x18,0x0C,0x0C,0x0C,0x18,0x30,0x00}, // )
    {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00}, // *
    {0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00}, // +
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x30}, // ,
    {0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00}, // -
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00}, // .
    {0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00}, // /
    {0x7C,0xCE,0xDE,0xF6,0xE6,0xC6,0x7C,0x00}, // 0
    {0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0x00}, // 1
    {0x7C,0xC6,0x06,0x7C,0xC0,0xC0,0xFE,0x00}, // 2
    {0xFC,0x06,0x06,0x3C,0x06,0x06,0xFC,0x00}, // 3
    {0x0C,0xCC,0xCC,0xCC,0xFE,0x0C,0x0C,0x00}, // 4
    {0xFE,0xC0,0xFC,0x06,0x06,0xC6,0x7C,0x00}, // 5
    {0x7C,0xC0,0xC0,0xFC,0xC6,0xC6,0x7C,0x00}, // 6
    {0xFE,0x06,0x06,0x0C,0x18,0x18,0x18,0x00}, // 7
    {0x7C,0xC6,0xC6,0x7C,0xC6,0xC6,0x7C,0x00}, // 8
    {0x7C,0xC6,0xC6,0x7E,0x06,0x06,0x7C,0x00}, // 9
    {0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x00}, // :
    {0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x30}, // ;
    {0x0C,0x18,0x30,0x60,0x30,0x18,0x0C,0x00}, // <
    {0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00}, // =
    {0x30,0x18,0x0C,0x06,0x0C,0x18,0x30,0x00}, // >
    {0x3C,0x66,0x06,0x1C,0x18,0x00,0x18,0x00}, // ?
    {0x7C,0xC6,0xDE,0xDE,0xDE,0xC0,0x7C,0x00}, // @
    {0x38,0x6C,0xC6,0xC6,0xFE,0xC6,0xC6,0x00}, // A
    {0xFC,0xC6,0xC6,0xFC,0xC6,0xC6,0xFC,0x00}, // B
    {0x7C,0xC6,0xC0,0xC0,0xC0,0xC6,0x7C,0x00}, // C
    {0xF8,0xCC,0xC6,0xC6,0xC6,0xCC,0xF8,0x00}, // D
    {0xFE,0xC0,0xC0,0xF8,0xC0,0xC0,0xFE,0x00}, // E
    {0xFE,0xC0,0xC0,0xF8,0xC0,0xC0,0xC0,0x00}, // F
    {0x7C,0xC6,0xC0,0xCE,0xC6,0xC6,0x7E,0x00}, // G
    {0xC6,0xC6,0xC6,0xFE,0xC6,0xC6,0xC6,0x00}, // H
    {0x7E,0x18,0x18,0x18,0x18,0x18,0x7E,0x00}, // I
    {0x06,0x06,0x06,0x06,0x06,0xC6,0x7C,0x00}, // J
    {0xC6,0xCC,0xD8,0xF0,0xD8,0xCC,0xC6,0x00}, // K
    {0xC0,0xC0,0xC0,0xC0,0xC0,0xC0,0xFE,0x00}, // L
    {0xC6,0xEE,0xFE,0xD6,0xC6,0xC6,0xC6,0x00}, // M
    {0xC6,0xE6,0xF6,0xDE,0xCE,0xC6,0xC6,0x00}, // N
    {0x7C,0xC6,0xC6,0xC6,0xC6,0xC6,0x7C,0x00}, // O
    {0xFC,0xC6,0xC6,0xFC,0xC0,0xC0,0xC0,0x00}, // P
    {0x7C,0xC6,0xC6,0xC6,0xD6,0xDE,0x7C,0x06}, // Q
    {0xFC,0xC6,0xC6,0xFC,0xD8,0xCC,0xC6,0x00}, // R
    {0x7C,0xC6,0xC0,0x7C,0x06,0xC6,0x7C,0x00}, // S
    {0xFF,0x18,0x18,0x18,0x18,0x18,0x18,0x00}, // T
    {0xC6,0xC6,0xC6,0xC6,0xC6,0xC6,0xFE,0x00}, // U
    {0xC6,0xC6,0xC6,0xC6,0xC6,0x6C,0x38,0x00}, // V
    {0xC6,0xC6,0xC6,0xD6,0xFE,0xEE,0xC6,0x00}, // W
    {0xC6,0xC6,0x6C,0x38,0x6C,0xC6,0xC6,0x00}, // X
    {0xC3,0xC3,0x66,0x3C,0x18,0x18,0x18,0x00}, // Y
    {0xFE,0x06,0x0C,0x18,0x30,0x60,0xFE,0x00}, // Z
    {0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00}, // [
    {0xC0,0x60,0x30,0x18,0x0C,0x06,0x02,0x00}, // backslash
    {0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00}, // ]
    {0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00}, // ^
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF}, // _
    {0x18,0x18,0x0C,0x00,0x00,0x00,0x00,0x00}, // `
    {0x00,0x00,0x7C,0x06,0x7E,0xC6,0x7E,0x00}, // a
    {0xC0,0xC0,0xFC,0xC6,0xC6,0xC6,0xFC,0x00}, // b
    {0x00,0x00,0x7C,0xC6,0xC0,0xC6,0x7C,0x00}, // c
    {0x06,0x06,0x7E,0xC6,0xC6,0xC6,0x7E,0x00}, // d
    {0x00,0x00,0x7C,0xC6,0xFE,0xC0,0x7C,0x00}, // e
    {0x1C,0x30,0x30,0x7C,0x30,0x30,0x30,0x00}, // f
    {0x00,0x00,0x7E,0xC6,0xC6,0x7E,0x06,0x7C}, // g
    {0xC0,0xC0,0xFC,0xC6,0xC6,0xC6,0xC6,0x00}, // h
    {0x18,0x00,0x38,0x18,0x18,0x18,0x3C,0x00}, // i
    {0x06,0x00,0x0E,0x06,0x06,0x06,0xC6,0x7C}, // j
    {0xC0,0xC0,0xCC,0xD8,0xF0,0xD8,0xCC,0x00}, // k
    {0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00}, // l
    {0x00,0x00,0xCC,0xFE,0xD6,0xC6,0xC6,0x00}, // m
    {0x00,0x00,0xFC,0xC6,0xC6,0xC6,0xC6,0x00}, // n
    {0x00,0x00,0x7C,0xC6,0xC6,0xC6,0x7C,0x00}, // o
    {0x00,0x00,0xFC,0xC6,0xC6,0xFC,0xC0,0xC0}, // p
    {0x00,0x00,0x7E,0xC6,0xC6,0x7E,0x06,0x06}, // q
    {0x00,0x00,0xDC,0xE6,0xC0,0xC0,0xC0,0x00}, // r
    {0x00,0x00,0x7E,0xC0,0x7C,0x06,0xFC,0x00}, // s
    {0x30,0x30,0x7C,0x30,0x30,0x30,0x1C,0x00}, // t
    {0x00,0x00,0xC6,0xC6,0xC6,0xC6,0x7E,0x00}, // u
    {0x00,0x00,0xC6,0xC6,0xC6,0x6C,0x38,0x00}, // v
    {0x00,0x00,0xC6,0xC6,0xD6,0xFE,0x6C,0x00}, // w
    {0x00,0x00,0xC6,0x6C,0x38,0x6C,0xC6,0x00}, // x
    {0x00,0x00,0xC6,0xC6,0xC6,0x7E,0x06,0x7C}, // y
    {0x00,0x00,0xFE,0x0C,0x38,0x60,0xFE,0x00}, // z
    {0x0E,0x18,0x18,0x70,0x18,0x18,0x0E,0x00}, // {
    {0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00}, // |
    {0x70,0x18,0x18,0x0E,0x18,0x18,0x70,0x00}, // }
    {0x72,0x9C,0x00,0x00,0x00,0x00,0x00,0x00}, // ~
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // DEL
};

// ============== GPU ENUMERATION ==============
void EnumerateGPUs()
{
    IDXGIFactory6* factory6 = nullptr;
    IDXGIFactory1* factory1 = nullptr;
    HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory6), (void**)&factory6);

    if (SUCCEEDED(hr) && factory6) {
        IDXGIAdapter1* adapter = nullptr;
        UINT i = 0;
        while (SUCCEEDED(factory6->EnumAdapterByGpuPreference(i++,
            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            __uuidof(IDXGIAdapter1), (void**)&adapter)))
        {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); continue; }
            GPUInfo info;
            info.name = desc.Description;
            info.adapter = adapter;
            info.vram = desc.DedicatedVideoMemory;
            g_gpuList.push_back(info);
        }
        factory6->Release();
    } else {
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory1);
        IDXGIAdapter1* adapter = nullptr;
        UINT i = 0;
        while (factory1->EnumAdapters1(i++, &adapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); continue; }
            GPUInfo info;
            info.name = desc.Description;
            info.adapter = adapter;
            info.vram = desc.DedicatedVideoMemory;
            g_gpuList.push_back(info);
        }
        factory1->Release();
    }
}

void FreeGPUList()
{
    for (auto& gpu : g_gpuList) { if (gpu.adapter) gpu.adapter->Release(); }
    g_gpuList.clear();
}

// ============== SETTINGS DIALOG ==============
#define IDC_GPU_COMBO    1001
#define IDC_START_BTN    1002
#define IDC_GPU_LABEL    1003
#define IDC_API_LABEL    1004
#define IDC_API_COMBO    1005

// ============== DXR 1.1 SETTINGS DIALOG ==============
#define IDC_DXR_LIGHTING      2000
#define IDC_DXR_SHADOWS       2001
#define IDC_DXR_SOFT_SHADOWS  2002
#define IDC_DXR_REFLECTIONS   2003
#define IDC_DXR_AO            2004
#define IDC_DXR_GI            2005
#define IDC_DXR_SHADOW_SAMPLES 2006
#define IDC_DXR_AO_SAMPLES    2007
#define IDC_DXR_GI_BOUNCES    2008
#define IDC_DXR_CONTINUE      2009
#define IDC_DXR_TEMPORAL_DENOISE  2010
#define IDC_DXR_DENOISE_BLEND     2011

static HWND g_hDxrSettingsDlg = nullptr;
static bool g_dxrSettingsAccepted = false;
static bool g_dxrSettingsDlgClosed = false;

// ============== DXR 1.0 SETTINGS DIALOG ==============
#define IDC_DXR10_SPOTLIGHT       3000
#define IDC_DXR10_SHADOWS         3001
#define IDC_DXR10_SHADOW_SAMPLES  3002
#define IDC_DXR10_LIGHT_RADIUS    3003
#define IDC_DXR10_AO              3004
#define IDC_DXR10_AO_SAMPLES      3005
#define IDC_DXR10_AO_RADIUS       3006
#define IDC_DXR10_GI              3007
#define IDC_DXR10_REFLECTIONS     3008
#define IDC_DXR10_GLASS           3009
#define IDC_DXR10_CONTINUE        3010

static HWND g_hDxr10SettingsDlg = nullptr;
static bool g_dxr10SettingsAccepted = false;
static bool g_dxr10SettingsDlgClosed = false;

// ============== VULKAN RT SETTINGS DIALOG ==============
#define IDC_VKRT_SPOTLIGHT       4000
#define IDC_VKRT_SHADOWS         4001
#define IDC_VKRT_SHADOW_SAMPLES  4002
#define IDC_VKRT_LIGHT_RADIUS    4003
#define IDC_VKRT_AO              4004
#define IDC_VKRT_AO_SAMPLES      4005
#define IDC_VKRT_AO_RADIUS       4006
#define IDC_VKRT_GI              4007
#define IDC_VKRT_REFLECTIONS     4008
#define IDC_VKRT_GLASS           4009
#define IDC_VKRT_CONTINUE        4010

static HWND g_hVkRTSettingsDlg = nullptr;
static bool g_vkrtSettingsAccepted = false;
static bool g_vkrtSettingsDlgClosed = false;

// Forward declaration
#include "d3d12/d3d12_shared.h"

static void UpdateDxrControlStates(HWND hwnd)
{
    // Soft shadows require base shadows
    BOOL shadowsEnabled = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_SHADOWS), BM_GETCHECK, 0, 0) == BST_CHECKED);
    EnableWindow(GetDlgItem(hwnd, IDC_DXR_SOFT_SHADOWS), shadowsEnabled);
    EnableWindow(GetDlgItem(hwnd, IDC_DXR_SHADOW_SAMPLES), shadowsEnabled &&
        (SendMessageW(GetDlgItem(hwnd, IDC_DXR_SOFT_SHADOWS), BM_GETCHECK, 0, 0) == BST_CHECKED));

    // AO samples only if AO enabled
    BOOL aoEnabled = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_AO), BM_GETCHECK, 0, 0) == BST_CHECKED);
    EnableWindow(GetDlgItem(hwnd, IDC_DXR_AO_SAMPLES), aoEnabled);

    // GI bounces only if GI enabled
    BOOL giEnabled = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_GI), BM_GETCHECK, 0, 0) == BST_CHECKED);
    EnableWindow(GetDlgItem(hwnd, IDC_DXR_GI_BOUNCES), giEnabled);
}

static LRESULT CALLBACK DxrSettingsDlgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_CREATE: {
        int y = 15;
        int checkSpacing = 22;
        int descSpacing = 18;

        // Title
        CreateWindowW(L"STATIC", L"DXR Feature Settings", WS_CHILD | WS_VISIBLE | SS_CENTER,
            10, y, 400, 20, hwnd, nullptr, g_hInstance, nullptr);
        y += 30;

        // RT Lighting checkbox (Spotlight + GI model)
        HWND hLighting = CreateWindowW(L"BUTTON", L"RT Lighting (Spotlight + Color Bleeding)",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 280, 20, hwnd, (HMENU)IDC_DXR_LIGHTING, g_hInstance, nullptr);
        SendMessageW(hLighting, BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing;
        CreateWindowW(L"STATIC", L"Spotlight cone lighting with wall color bleeding (GI approximation)",
            WS_CHILD | WS_VISIBLE, 40, y, 380, 16, hwnd, nullptr, g_hInstance, nullptr);
        y += descSpacing + 8;

        // RT Shadows checkbox
        HWND hShadows = CreateWindowW(L"BUTTON", L"Ray-Traced Shadows",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 200, 20, hwnd, (HMENU)IDC_DXR_SHADOWS, g_hInstance, nullptr);
        SendMessageW(hShadows, BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing;
        CreateWindowW(L"STATIC", L"Traces rays to light to determine shadow visibility on all surfaces",
            WS_CHILD | WS_VISIBLE, 40, y, 380, 16, hwnd, nullptr, g_hInstance, nullptr);
        y += descSpacing + 8;

        // Soft Shadows checkbox + samples combo (enabled by default)
        HWND hSoftShadows = CreateWindowW(L"BUTTON", L"  Soft Shadows",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 40, y, 130, 20, hwnd, (HMENU)IDC_DXR_SOFT_SHADOWS, g_hInstance, nullptr);
        SendMessageW(hSoftShadows, BM_SETCHECK, BST_CHECKED, 0);
        CreateWindowW(L"STATIC", L"Samples:", WS_CHILD | WS_VISIBLE, 200, y+2, 55, 20, hwnd, nullptr, g_hInstance, nullptr);
        HWND hShadowSamples = CreateWindowW(L"COMBOBOX", nullptr,
            WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 260, y, 60, 100, hwnd, (HMENU)IDC_DXR_SHADOW_SAMPLES, g_hInstance, nullptr);
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"4");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"8");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"16");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"32");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"64");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"128");
        SendMessageW(hShadowSamples, CB_SETCURSEL, 2, 0); // Default 16
        y += checkSpacing;
        CreateWindowW(L"STATIC", L"Multiple rays for smooth shadow edges (more samples = softer)",
            WS_CHILD | WS_VISIBLE, 60, y, 360, 16, hwnd, nullptr, g_hInstance, nullptr);
        y += descSpacing + 8;

        // Reflections checkbox (enabled by default)
        HWND hReflections = CreateWindowW(L"BUTTON", L"Ray-Traced Reflections",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 200, 20, hwnd, (HMENU)IDC_DXR_REFLECTIONS, g_hInstance, nullptr);
        SendMessageW(hReflections, BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing;
        CreateWindowW(L"STATIC", L"Mirror shows real scene reflection; glass shows objects behind it",
            WS_CHILD | WS_VISIBLE, 40, y, 380, 16, hwnd, nullptr, g_hInstance, nullptr);
        y += descSpacing + 8;

        // AO checkbox + samples (ENABLED by default)
        HWND hAO = CreateWindowW(L"BUTTON", L"Ray-Traced Ambient Occlusion",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 220, 20, hwnd, (HMENU)IDC_DXR_AO, g_hInstance, nullptr);
        SendMessageW(hAO, BM_SETCHECK, BST_CHECKED, 0);
        HWND hAOSamples = CreateWindowW(L"COMBOBOX", nullptr,
            WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 280, y, 60, 100, hwnd, (HMENU)IDC_DXR_AO_SAMPLES, g_hInstance, nullptr);
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"4");
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"8");
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"16");
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"32");
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"64");
        SendMessageW(hAOSamples, CB_SETCURSEL, 2, 0); // Default 16
        y += checkSpacing;
        CreateWindowW(L"STATIC", L"Darkens corners and crevices where light is occluded",
            WS_CHILD | WS_VISIBLE, 40, y, 380, 16, hwnd, nullptr, g_hInstance, nullptr);
        y += descSpacing + 8;

        // GI checkbox + bounces (ENABLED by default)
        HWND hGI = CreateWindowW(L"BUTTON", L"Global Illumination",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 160, 20, hwnd, (HMENU)IDC_DXR_GI, g_hInstance, nullptr);
        SendMessageW(hGI, BM_SETCHECK, BST_CHECKED, 0);
        CreateWindowW(L"STATIC", L"Bounces:", WS_CHILD | WS_VISIBLE, 200, y+2, 55, 20, hwnd, nullptr, g_hInstance, nullptr);
        HWND hGIBounces = CreateWindowW(L"COMBOBOX", nullptr,
            WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 260, y, 60, 100, hwnd, (HMENU)IDC_DXR_GI_BOUNCES, g_hInstance, nullptr);
        SendMessageW(hGIBounces, CB_ADDSTRING, 0, (LPARAM)L"1");
        SendMessageW(hGIBounces, CB_ADDSTRING, 0, (LPARAM)L"2");
        SendMessageW(hGIBounces, CB_ADDSTRING, 0, (LPARAM)L"3");
        SendMessageW(hGIBounces, CB_ADDSTRING, 0, (LPARAM)L"4");
        SendMessageW(hGIBounces, CB_ADDSTRING, 0, (LPARAM)L"5");
        SendMessageW(hGIBounces, CB_SETCURSEL, 1, 0); // Default 2
        y += checkSpacing;
        CreateWindowW(L"STATIC", L"Color bleeding from colored walls onto other surfaces",
            WS_CHILD | WS_VISIBLE, 40, y, 380, 16, hwnd, nullptr, g_hInstance, nullptr);
        y += descSpacing + 15;

        // Separator line
        CreateWindowW(L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_ETCHEDHORZ,
            20, y, 380, 2, hwnd, nullptr, g_hInstance, nullptr);
        y += 10;

        // Temporal Denoising checkbox + blend factor
        HWND hDenoise = CreateWindowW(L"BUTTON", L"Temporal Denoising",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 160, 20, hwnd, (HMENU)IDC_DXR_TEMPORAL_DENOISE, g_hInstance, nullptr);
        SendMessageW(hDenoise, BM_SETCHECK, BST_CHECKED, 0);  // ON by default
        CreateWindowW(L"STATIC", L"Blend:", WS_CHILD | WS_VISIBLE, 190, y+2, 45, 20, hwnd, nullptr, g_hInstance, nullptr);
        HWND hDenoiseBlend = CreateWindowW(L"COMBOBOX", nullptr,
            WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST, 240, y, 80, 150, hwnd, (HMENU)IDC_DXR_DENOISE_BLEND, g_hInstance, nullptr);
        SendMessageW(hDenoiseBlend, CB_ADDSTRING, 0, (LPARAM)L"50%");
        SendMessageW(hDenoiseBlend, CB_ADDSTRING, 0, (LPARAM)L"70%");
        SendMessageW(hDenoiseBlend, CB_ADDSTRING, 0, (LPARAM)L"80%");
        SendMessageW(hDenoiseBlend, CB_ADDSTRING, 0, (LPARAM)L"90%");
        SendMessageW(hDenoiseBlend, CB_ADDSTRING, 0, (LPARAM)L"95%");
        SendMessageW(hDenoiseBlend, CB_ADDSTRING, 0, (LPARAM)L"98%");
        SendMessageW(hDenoiseBlend, CB_SETCURSEL, 3, 0); // Default 90%
        y += checkSpacing;
        CreateWindowW(L"STATIC", L"Smooths RT noise by blending frames (higher = smoother but ghosting)",
            WS_CHILD | WS_VISIBLE, 40, y, 420, 16, hwnd, nullptr, g_hInstance, nullptr);
        y += descSpacing + 15;

        // Info text
        CreateWindowW(L"STATIC", L"Disabled features use rasterization fallback",
            WS_CHILD | WS_VISIBLE | SS_CENTER, 20, y, 440, 20, hwnd, nullptr, g_hInstance, nullptr);
        y += 30;

        // Continue button
        CreateWindowW(L"BUTTON", L"Continue", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
            140, y, 100, 30, hwnd, (HMENU)IDC_DXR_CONTINUE, g_hInstance, nullptr);

        // Initial state update
        UpdateDxrControlStates(hwnd);
        return 0;
    }
    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_DXR_CONTINUE) {
            // Save settings
            g_dxrFeatures.SetDefaults();

            g_dxrFeatures.rtLighting = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_LIGHTING), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxrFeatures.rtShadows = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_SHADOWS), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxrFeatures.rtSoftShadows = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_SOFT_SHADOWS), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxrFeatures.rtReflections = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_REFLECTIONS), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxrFeatures.rtAO = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_AO), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxrFeatures.rtGI = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_GI), BM_GETCHECK, 0, 0) == BST_CHECKED);

            // Get sample counts
            int shadowSampleIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_DXR_SHADOW_SAMPLES), CB_GETCURSEL, 0, 0);
            int shadowSamples[] = {4, 8, 16, 32, 64, 128};
            g_dxrFeatures.softShadowSamples = shadowSamples[shadowSampleIdx];

            int aoSampleIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_DXR_AO_SAMPLES), CB_GETCURSEL, 0, 0);
            int aoSamples[] = {4, 8, 16, 32, 64};
            g_dxrFeatures.aoSamples = aoSamples[aoSampleIdx];

            int giBouncesIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_DXR_GI_BOUNCES), CB_GETCURSEL, 0, 0);
            g_dxrFeatures.giBounces = giBouncesIdx + 1;

            // Temporal denoising settings
            g_dxrFeatures.enableTemporalDenoise = (SendMessageW(GetDlgItem(hwnd, IDC_DXR_TEMPORAL_DENOISE), BM_GETCHECK, 0, 0) == BST_CHECKED);
            int denoiseBlendIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_DXR_DENOISE_BLEND), CB_GETCURSEL, 0, 0);
            float blendFactors[] = {0.5f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f};
            g_dxrFeatures.denoiseBlendFactor = blendFactors[denoiseBlendIdx];

            g_dxrSettingsAccepted = true;
            DestroyWindow(hwnd);
        }
        // Update control states when checkboxes change
        else if (HIWORD(wParam) == BN_CLICKED) {
            UpdateDxrControlStates(hwnd);
        }
        return 0;
    case WM_CLOSE: DestroyWindow(hwnd); return 0;
    case WM_DESTROY: g_dxrSettingsDlgClosed = true; return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static bool ShowDxrSettingsDialog()
{
    g_dxrSettingsDlgClosed = false;
    g_dxrSettingsAccepted = false;

    WNDCLASSW wc = {};
    wc.lpfnWndProc = DxrSettingsDlgProc;
    wc.hInstance = g_hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"DxrSettingsDialog";
    RegisterClassW(&wc);

    g_hDxrSettingsDlg = CreateWindowW(L"DxrSettingsDialog", L"DXR Settings",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT,
        500, 620, nullptr, nullptr, g_hInstance, nullptr);
    ShowWindow(g_hDxrSettingsDlg, SW_SHOW);
    UpdateWindow(g_hDxrSettingsDlg);

    MSG msg;
    while (!g_dxrSettingsDlgClosed && GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return g_dxrSettingsAccepted;
}

// ============== DXR 1.0 SETTINGS DIALOG PROCEDURE ==============
static void UpdateDxr10ControlStates(HWND hwnd)
{
    // Shadow samples and light radius require soft shadows
    BOOL shadowsEnabled = (SendMessageW(GetDlgItem(hwnd, IDC_DXR10_SHADOWS), BM_GETCHECK, 0, 0) == BST_CHECKED);
    EnableWindow(GetDlgItem(hwnd, IDC_DXR10_SHADOW_SAMPLES), shadowsEnabled);
    EnableWindow(GetDlgItem(hwnd, IDC_DXR10_LIGHT_RADIUS), shadowsEnabled);

    // AO samples and radius require AO enabled
    BOOL aoEnabled = (SendMessageW(GetDlgItem(hwnd, IDC_DXR10_AO), BM_GETCHECK, 0, 0) == BST_CHECKED);
    EnableWindow(GetDlgItem(hwnd, IDC_DXR10_AO_SAMPLES), aoEnabled);
    EnableWindow(GetDlgItem(hwnd, IDC_DXR10_AO_RADIUS), aoEnabled);
}

static LRESULT CALLBACK Dxr10SettingsDlgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_CREATE: {
        int y = 20;
        const int checkSpacing = 25;
        const int descSpacing = 14;

        // Title
        CreateWindowW(L"STATIC", L"DXR 1.0 Ray Tracing Settings", WS_CHILD | WS_VISIBLE | SS_CENTER,
            20, y, 360, 24, hwnd, nullptr, g_hInstance, nullptr);
        y += 35;

        // Spotlight
        HWND hSpot = CreateWindowW(L"BUTTON", L"Spotlight Lighting (cone light)",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 300, 20, hwnd, (HMENU)IDC_DXR10_SPOTLIGHT, g_hInstance, nullptr);
        SendMessageW(hSpot, BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing;

        // Soft Shadows
        CreateWindowW(L"BUTTON", L"Soft Shadows",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 140, 20, hwnd, (HMENU)IDC_DXR10_SHADOWS, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_DXR10_SHADOWS), BM_SETCHECK, BST_CHECKED, 0);
        CreateWindowW(L"STATIC", L"Samples:", WS_CHILD | WS_VISIBLE, 180, y+2, 50, 18, hwnd, nullptr, g_hInstance, nullptr);
        HWND hShadowSamples = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
            240, y, 60, 100, hwnd, (HMENU)IDC_DXR10_SHADOW_SAMPLES, g_hInstance, nullptr);
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"1");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"4");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"8");
        SendMessageW(hShadowSamples, CB_SETCURSEL, 1, 0);  // Default: 4
        y += checkSpacing;

        // Light radius (for soft shadows)
        CreateWindowW(L"STATIC", L"Light Radius:", WS_CHILD | WS_VISIBLE, 40, y+2, 80, 18, hwnd, nullptr, g_hInstance, nullptr);
        HWND hLightRadius = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
            130, y, 80, 100, hwnd, (HMENU)IDC_DXR10_LIGHT_RADIUS, g_hInstance, nullptr);
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.05");
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.1");
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.15");
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.2");
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.3");
        SendMessageW(hLightRadius, CB_SETCURSEL, 2, 0);  // Default: 0.15
        y += checkSpacing + 10;

        // Ambient Occlusion
        CreateWindowW(L"BUTTON", L"Ambient Occlusion",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 150, 20, hwnd, (HMENU)IDC_DXR10_AO, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_DXR10_AO), BM_SETCHECK, BST_CHECKED, 0);
        CreateWindowW(L"STATIC", L"Samples:", WS_CHILD | WS_VISIBLE, 180, y+2, 50, 18, hwnd, nullptr, g_hInstance, nullptr);
        HWND hAOSamples = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
            240, y, 60, 100, hwnd, (HMENU)IDC_DXR10_AO_SAMPLES, g_hInstance, nullptr);
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"1");
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"3");
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"5");
        SendMessageW(hAOSamples, CB_SETCURSEL, 1, 0);  // Default: 3
        y += checkSpacing;

        // AO radius
        CreateWindowW(L"STATIC", L"AO Radius:", WS_CHILD | WS_VISIBLE, 40, y+2, 70, 18, hwnd, nullptr, g_hInstance, nullptr);
        HWND hAORadius = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
            130, y, 80, 100, hwnd, (HMENU)IDC_DXR10_AO_RADIUS, g_hInstance, nullptr);
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"0.1");
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"0.2");
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"0.3");
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"0.5");
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"1.0");
        SendMessageW(hAORadius, CB_SETCURSEL, 2, 0);  // Default: 0.3
        y += checkSpacing + 10;

        // Global Illumination
        CreateWindowW(L"BUTTON", L"Global Illumination (1 bounce)",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 250, 20, hwnd, (HMENU)IDC_DXR10_GI, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_DXR10_GI), BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing + 10;

        // Reflections
        CreateWindowW(L"BUTTON", L"Mirror Reflections",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 200, 20, hwnd, (HMENU)IDC_DXR10_REFLECTIONS, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_DXR10_REFLECTIONS), BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing;

        // Glass Refraction
        CreateWindowW(L"BUTTON", L"Glass Refraction (fresnel)",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 250, 20, hwnd, (HMENU)IDC_DXR10_GLASS, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_DXR10_GLASS), BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing + 15;

        // Info text
        CreateWindowW(L"STATIC", L"Disabled features reduce shader complexity",
            WS_CHILD | WS_VISIBLE | SS_CENTER, 20, y, 360, 20, hwnd, nullptr, g_hInstance, nullptr);
        y += 30;

        // Continue button
        CreateWindowW(L"BUTTON", L"Continue", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
            140, y, 100, 30, hwnd, (HMENU)IDC_DXR10_CONTINUE, g_hInstance, nullptr);

        // Initial state update
        UpdateDxr10ControlStates(hwnd);
        return 0;
    }
    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_DXR10_CONTINUE) {
            // Save settings to g_dxr10Features
            g_dxr10Features.SetDefaults();

            g_dxr10Features.spotlight = (SendMessageW(GetDlgItem(hwnd, IDC_DXR10_SPOTLIGHT), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxr10Features.softShadows = (SendMessageW(GetDlgItem(hwnd, IDC_DXR10_SHADOWS), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxr10Features.ambientOcclusion = (SendMessageW(GetDlgItem(hwnd, IDC_DXR10_AO), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxr10Features.globalIllum = (SendMessageW(GetDlgItem(hwnd, IDC_DXR10_GI), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxr10Features.reflections = (SendMessageW(GetDlgItem(hwnd, IDC_DXR10_REFLECTIONS), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_dxr10Features.glassRefraction = (SendMessageW(GetDlgItem(hwnd, IDC_DXR10_GLASS), BM_GETCHECK, 0, 0) == BST_CHECKED);

            // Get sample counts and radii
            int shadowSampleIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_DXR10_SHADOW_SAMPLES), CB_GETCURSEL, 0, 0);
            int shadowSamples[] = {1, 4, 8};
            g_dxr10Features.shadowSamples = shadowSamples[shadowSampleIdx];

            int lightRadiusIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_DXR10_LIGHT_RADIUS), CB_GETCURSEL, 0, 0);
            float lightRadii[] = {0.05f, 0.1f, 0.15f, 0.2f, 0.3f};
            g_dxr10Features.lightRadius = lightRadii[lightRadiusIdx];

            int aoSampleIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_DXR10_AO_SAMPLES), CB_GETCURSEL, 0, 0);
            int aoSamples[] = {1, 3, 5};
            g_dxr10Features.aoSamples = aoSamples[aoSampleIdx];

            int aoRadiusIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_DXR10_AO_RADIUS), CB_GETCURSEL, 0, 0);
            float aoRadii[] = {0.1f, 0.2f, 0.3f, 0.5f, 1.0f};
            g_dxr10Features.aoRadius = aoRadii[aoRadiusIdx];

            g_dxr10SettingsAccepted = true;
            DestroyWindow(hwnd);
        }
        else if (HIWORD(wParam) == BN_CLICKED) {
            UpdateDxr10ControlStates(hwnd);
        }
        return 0;
    case WM_CLOSE: DestroyWindow(hwnd); return 0;
    case WM_DESTROY: g_dxr10SettingsDlgClosed = true; return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static bool ShowDxr10SettingsDialog()
{
    g_dxr10SettingsDlgClosed = false;
    g_dxr10SettingsAccepted = false;

    WNDCLASSW wc = {};
    wc.lpfnWndProc = Dxr10SettingsDlgProc;
    wc.hInstance = g_hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"Dxr10SettingsDialog";
    RegisterClassW(&wc);

    g_hDxr10SettingsDlg = CreateWindowW(L"Dxr10SettingsDialog", L"DXR 1.0 Settings",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT,
        420, 420, nullptr, nullptr, g_hInstance, nullptr);
    ShowWindow(g_hDxr10SettingsDlg, SW_SHOW);
    UpdateWindow(g_hDxr10SettingsDlg);

    MSG msg;
    while (!g_dxr10SettingsDlgClosed && GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return g_dxr10SettingsAccepted;
}

// ============== VULKAN RT SETTINGS DIALOG PROCEDURE ==============
static void UpdateVkRTControlStates(HWND hwnd)
{
    BOOL shadowsEnabled = (SendMessageW(GetDlgItem(hwnd, IDC_VKRT_SHADOWS), BM_GETCHECK, 0, 0) == BST_CHECKED);
    EnableWindow(GetDlgItem(hwnd, IDC_VKRT_SHADOW_SAMPLES), shadowsEnabled);
    EnableWindow(GetDlgItem(hwnd, IDC_VKRT_LIGHT_RADIUS), shadowsEnabled);

    BOOL aoEnabled = (SendMessageW(GetDlgItem(hwnd, IDC_VKRT_AO), BM_GETCHECK, 0, 0) == BST_CHECKED);
    EnableWindow(GetDlgItem(hwnd, IDC_VKRT_AO_SAMPLES), aoEnabled);
    EnableWindow(GetDlgItem(hwnd, IDC_VKRT_AO_RADIUS), aoEnabled);
}

static LRESULT CALLBACK VkRTSettingsDlgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_CREATE: {
        int y = 20;
        const int checkSpacing = 25;

        CreateWindowW(L"STATIC", L"Vulkan Ray Tracing Settings", WS_CHILD | WS_VISIBLE | SS_CENTER,
            20, y, 360, 24, hwnd, nullptr, g_hInstance, nullptr);
        y += 35;

        HWND hSpot = CreateWindowW(L"BUTTON", L"Spotlight Lighting (cone light)",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 300, 20, hwnd, (HMENU)IDC_VKRT_SPOTLIGHT, g_hInstance, nullptr);
        SendMessageW(hSpot, BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing;

        CreateWindowW(L"BUTTON", L"Soft Shadows",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 140, 20, hwnd, (HMENU)IDC_VKRT_SHADOWS, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_VKRT_SHADOWS), BM_SETCHECK, BST_CHECKED, 0);
        CreateWindowW(L"STATIC", L"Samples:", WS_CHILD | WS_VISIBLE, 180, y+2, 50, 18, hwnd, nullptr, g_hInstance, nullptr);
        HWND hShadowSamples = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
            240, y, 60, 100, hwnd, (HMENU)IDC_VKRT_SHADOW_SAMPLES, g_hInstance, nullptr);
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"1");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"4");
        SendMessageW(hShadowSamples, CB_ADDSTRING, 0, (LPARAM)L"8");
        SendMessageW(hShadowSamples, CB_SETCURSEL, 1, 0);
        y += checkSpacing;

        CreateWindowW(L"STATIC", L"Light Radius:", WS_CHILD | WS_VISIBLE, 40, y+2, 80, 18, hwnd, nullptr, g_hInstance, nullptr);
        HWND hLightRadius = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
            130, y, 80, 100, hwnd, (HMENU)IDC_VKRT_LIGHT_RADIUS, g_hInstance, nullptr);
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.05");
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.1");
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.15");
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.2");
        SendMessageW(hLightRadius, CB_ADDSTRING, 0, (LPARAM)L"0.3");
        SendMessageW(hLightRadius, CB_SETCURSEL, 2, 0);
        y += checkSpacing + 10;

        CreateWindowW(L"BUTTON", L"Ambient Occlusion",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 150, 20, hwnd, (HMENU)IDC_VKRT_AO, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_VKRT_AO), BM_SETCHECK, BST_CHECKED, 0);
        CreateWindowW(L"STATIC", L"Samples:", WS_CHILD | WS_VISIBLE, 180, y+2, 50, 18, hwnd, nullptr, g_hInstance, nullptr);
        HWND hAOSamples = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
            240, y, 60, 100, hwnd, (HMENU)IDC_VKRT_AO_SAMPLES, g_hInstance, nullptr);
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"1");
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"3");
        SendMessageW(hAOSamples, CB_ADDSTRING, 0, (LPARAM)L"5");
        SendMessageW(hAOSamples, CB_SETCURSEL, 1, 0);
        y += checkSpacing;

        CreateWindowW(L"STATIC", L"AO Radius:", WS_CHILD | WS_VISIBLE, 40, y+2, 70, 18, hwnd, nullptr, g_hInstance, nullptr);
        HWND hAORadius = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST,
            130, y, 80, 100, hwnd, (HMENU)IDC_VKRT_AO_RADIUS, g_hInstance, nullptr);
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"0.1");
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"0.2");
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"0.3");
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"0.5");
        SendMessageW(hAORadius, CB_ADDSTRING, 0, (LPARAM)L"1.0");
        SendMessageW(hAORadius, CB_SETCURSEL, 2, 0);
        y += checkSpacing + 10;

        CreateWindowW(L"BUTTON", L"Global Illumination (1 bounce)",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 250, 20, hwnd, (HMENU)IDC_VKRT_GI, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_VKRT_GI), BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing + 10;

        CreateWindowW(L"BUTTON", L"Mirror Reflections",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 200, 20, hwnd, (HMENU)IDC_VKRT_REFLECTIONS, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_VKRT_REFLECTIONS), BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing;

        CreateWindowW(L"BUTTON", L"Glass Refraction (fresnel)",
            WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX, 20, y, 250, 20, hwnd, (HMENU)IDC_VKRT_GLASS, g_hInstance, nullptr);
        SendMessageW(GetDlgItem(hwnd, IDC_VKRT_GLASS), BM_SETCHECK, BST_CHECKED, 0);
        y += checkSpacing + 15;

        CreateWindowW(L"STATIC", L"Disabled features reduce shader complexity",
            WS_CHILD | WS_VISIBLE | SS_CENTER, 20, y, 360, 20, hwnd, nullptr, g_hInstance, nullptr);
        y += 30;

        CreateWindowW(L"BUTTON", L"Continue", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
            140, y, 100, 30, hwnd, (HMENU)IDC_VKRT_CONTINUE, g_hInstance, nullptr);

        UpdateVkRTControlStates(hwnd);
        return 0;
    }
    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_VKRT_CONTINUE) {
            g_vulkanRTFeatures.SetDefaults();

            g_vulkanRTFeatures.spotlight = (SendMessageW(GetDlgItem(hwnd, IDC_VKRT_SPOTLIGHT), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_vulkanRTFeatures.softShadows = (SendMessageW(GetDlgItem(hwnd, IDC_VKRT_SHADOWS), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_vulkanRTFeatures.ambientOcclusion = (SendMessageW(GetDlgItem(hwnd, IDC_VKRT_AO), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_vulkanRTFeatures.globalIllum = (SendMessageW(GetDlgItem(hwnd, IDC_VKRT_GI), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_vulkanRTFeatures.reflections = (SendMessageW(GetDlgItem(hwnd, IDC_VKRT_REFLECTIONS), BM_GETCHECK, 0, 0) == BST_CHECKED);
            g_vulkanRTFeatures.glassRefraction = (SendMessageW(GetDlgItem(hwnd, IDC_VKRT_GLASS), BM_GETCHECK, 0, 0) == BST_CHECKED);

            int shadowSampleIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_VKRT_SHADOW_SAMPLES), CB_GETCURSEL, 0, 0);
            int shadowSamples[] = {1, 4, 8};
            g_vulkanRTFeatures.shadowSamples = shadowSamples[shadowSampleIdx];

            int lightRadiusIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_VKRT_LIGHT_RADIUS), CB_GETCURSEL, 0, 0);
            float lightRadii[] = {0.05f, 0.1f, 0.15f, 0.2f, 0.3f};
            g_vulkanRTFeatures.lightRadius = lightRadii[lightRadiusIdx];

            int aoSampleIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_VKRT_AO_SAMPLES), CB_GETCURSEL, 0, 0);
            int aoSamples[] = {1, 3, 5};
            g_vulkanRTFeatures.aoSamples = aoSamples[aoSampleIdx];

            int aoRadiusIdx = (int)SendMessageW(GetDlgItem(hwnd, IDC_VKRT_AO_RADIUS), CB_GETCURSEL, 0, 0);
            float aoRadii[] = {0.1f, 0.2f, 0.3f, 0.5f, 1.0f};
            g_vulkanRTFeatures.aoRadius = aoRadii[aoRadiusIdx];

            g_vkrtSettingsAccepted = true;
            DestroyWindow(hwnd);
        }
        else if (HIWORD(wParam) == BN_CLICKED) {
            UpdateVkRTControlStates(hwnd);
        }
        return 0;
    case WM_CLOSE: DestroyWindow(hwnd); return 0;
    case WM_DESTROY: g_vkrtSettingsDlgClosed = true; return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static bool ShowVulkanRTSettingsDialog()
{
    g_vkrtSettingsDlgClosed = false;
    g_vkrtSettingsAccepted = false;

    WNDCLASSW wc = {};
    wc.lpfnWndProc = VkRTSettingsDlgProc;
    wc.hInstance = g_hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"VkRTSettingsDialog";
    RegisterClassW(&wc);

    g_hVkRTSettingsDlg = CreateWindowW(L"VkRTSettingsDialog", L"Vulkan RT Settings",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT,
        420, 420, nullptr, nullptr, g_hInstance, nullptr);
    ShowWindow(g_hVkRTSettingsDlg, SW_SHOW);
    UpdateWindow(g_hVkRTSettingsDlg);

    MSG msg;
    while (!g_vkrtSettingsDlgClosed && GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return g_vkrtSettingsAccepted;
}

static LRESULT CALLBACK SettingsDlgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg) {
    case WM_CREATE: {
        CreateWindowW(L"STATIC", L"Renderer:", WS_CHILD | WS_VISIBLE, 20, 20, 70, 20, hwnd, (HMENU)IDC_API_LABEL, g_hInstance, nullptr);
        HWND hApiCombo = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
            100, 18, 280, 300, hwnd, (HMENU)IDC_API_COMBO, g_hInstance, nullptr);
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Direct3D 11");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Direct3D 12");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Direct3D 12 + DXR 1.0 (TraceRay)");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Direct3D 12 + DXR 1.1 (RayQuery)");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Direct3D 12 + Path Tracing");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Direct3D 12 + PT + DLSS RR");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"OpenGL");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Vulkan");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Vulkan + RT (VK_KHR_ray_tracing_pipeline)");
        SendMessageW(hApiCombo, CB_ADDSTRING, 0, (LPARAM)L"Vulkan + RayQuery (VK_KHR_ray_query)");
        SendMessageW(hApiCombo, CB_SETCURSEL, 0, 0);

        CreateWindowW(L"STATIC", L"GPU:", WS_CHILD | WS_VISIBLE, 20, 50, 70, 20, hwnd, (HMENU)IDC_GPU_LABEL, g_hInstance, nullptr);
        HWND hGpuCombo = CreateWindowW(L"COMBOBOX", nullptr, WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_VSCROLL,
            20, 75, 340, 200, hwnd, (HMENU)IDC_GPU_COMBO, g_hInstance, nullptr);
        for (size_t i = 0; i < g_gpuList.size(); i++) {
            wchar_t buf[256];
            SIZE_T vramMB = g_gpuList[i].vram / (1024 * 1024);
            swprintf_s(buf, L"%s (%zu MB)", g_gpuList[i].name.c_str(), vramMB);
            SendMessageW(hGpuCombo, CB_ADDSTRING, 0, (LPARAM)buf);
        }
        SendMessageW(hGpuCombo, CB_SETCURSEL, 0, 0);

        CreateWindowW(L"BUTTON", L"Start", WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
            140, 115, 100, 30, hwnd, (HMENU)IDC_START_BTN, g_hInstance, nullptr);
        return 0;
    }
    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_START_BTN) {
            g_settings.selectedGPU = (int)SendMessageW(GetDlgItem(hwnd, IDC_GPU_COMBO), CB_GETCURSEL, 0, 0);
            int apiSel = (int)SendMessageW(GetDlgItem(hwnd, IDC_API_COMBO), CB_GETCURSEL, 0, 0);
            g_settings.renderer = (RendererType)apiSel;
            g_settingsAccepted = true;
            DestroyWindow(hwnd);
        }
        return 0;
    case WM_CLOSE: DestroyWindow(hwnd); return 0;
    case WM_DESTROY: g_settingsDlgClosed = true; return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static bool ShowSettingsDialog()
{
    g_settingsDlgClosed = false;
    WNDCLASSW wc = {};
    wc.lpfnWndProc = SettingsDlgProc;
    wc.hInstance = g_hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"SettingsDialog";
    RegisterClassW(&wc);

    g_hSettingsDlg = CreateWindowW(L"SettingsDialog", L"RenderTestGPU - Settings",
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, CW_USEDEFAULT,
        420, 200, nullptr, nullptr, g_hInstance, nullptr);
    ShowWindow(g_hSettingsDlg, SW_SHOW);
    UpdateWindow(g_hSettingsDlg);

    MSG msg;
    while (!g_settingsDlgClosed && GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return g_settingsAccepted;
}

// ============== WINDOW PROCEDURE ==============
static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l)
{
    switch (m) {
    case WM_CLOSE:
        if (h == g_hMainWnd) DestroyWindow(h);
        return 0;
    case WM_DESTROY:
        if (h == g_hMainWnd) PostQuitMessage(0);
        return 0;
    case WM_KEYDOWN:
        if (w == VK_ESCAPE) PostQuitMessage(0);
        // Debug mode keys 0-6 (for both DXR 1.0 and 1.1 renderers)
        if (g_settings.renderer == RENDERER_D3D12_DXR10 || g_settings.renderer == RENDERER_D3D12_RT) {
            if (w >= '0' && w <= '6') {
                g_dxrFeatures.debugMode = (int)(w - '0');
                // Show debug mode name in title
                const char* modeNames[] = {"Normal", "Object IDs", "Normals", "Reflect Dir", "Shadows", "World Pos", "Depth"};
                const char* dxrVersion = (g_settings.renderer == RENDERER_D3D12_DXR10) ? "1.0" : "1.1";
                char title[128];
                sprintf_s(title, "RenderTestGPU - D3D12 + DXR %s [Debug: %s]", dxrVersion, modeNames[g_dxrFeatures.debugMode]);
                SetWindowTextA(h, title);
            }
        }
        break;
    case WM_ENTERSIZEMOVE:
        g_inSizeMove = true;
        SetTimer(h, 1, 1, nullptr);
        break;
    case WM_EXITSIZEMOVE:
        g_inSizeMove = false;
        KillTimer(h, 1);
        break;
    case WM_TIMER:
        if (w == 1 && g_inSizeMove) {
            switch (g_settings.renderer) {
            case RENDERER_D3D12_PT_DLSS: RenderD3D12PT_DLSS(); break;
            case RENDERER_D3D12_PT: RenderD3D12PT(); break;
            case RENDERER_D3D12_RT: RenderD3D12RT(); break;
            case RENDERER_D3D12_DXR10: RenderD3D12DXR10(); break;
            case RENDERER_D3D12: RenderD3D12(); break;
            case RENDERER_OPENGL: RenderOpenGL(); break;
            case RENDERER_VULKAN: RenderVulkan(); break;
            case RENDERER_VULKAN_RT: RenderVulkanRT(); break;
            case RENDERER_VULKAN_RQ: RenderVulkanRQ(); break;
            default: RenderD3D11(); break;
            }
        }
        break;
    }
    return DefWindowProc(h, m, w, l);
}

// ============== MAIN ENTRY POINT ==============
int WINAPI WinMain(HINSTANCE hI, HINSTANCE, LPSTR cmdLine, int)
{
    g_hInstance = hI;
    InitLog();

    // Parse command line arguments first
    ParseCommandLine(cmdLine);

    EnumerateGPUs();

    if (g_gpuList.empty()) {
        Log("[FATAL] No compatible GPU found!\n");
        MessageBoxW(0, L"No compatible GPU found!", L"Error", MB_OK);
        CloseLog();
        return 1;
    }

    // If command line specifies renderer, skip dialogs and use defaults
    if (g_cmdArgs.skipDialogs) {
        g_settings.renderer = g_cmdArgs.renderer;
        g_settings.selectedGPU = g_cmdArgs.hasGpu ? g_cmdArgs.gpuIndex : 0;

        // Validate GPU index
        if (g_settings.selectedGPU >= (int)g_gpuList.size()) {
            Log("[WARN] GPU index %d out of range, using 0\n", g_settings.selectedGPU);
            g_settings.selectedGPU = 0;
        }

        // Use default settings for RT renderers
        g_dxrFeatures.SetDefaults();
        g_dxr10Features.SetDefaults();
        g_vulkanRTFeatures.SetDefaults();

        Log("[INFO] Command line mode: renderer=%d gpu=%d\n",
            (int)g_settings.renderer, g_settings.selectedGPU);
    }
    else {
        // Show settings dialog
        if (!ShowSettingsDialog()) {
            FreeGPUList();
            CloseLog();
            return 0;
        }

        // Show DXR settings dialog if DXR renderer selected
        if (g_settings.renderer == RENDERER_D3D12_DXR10) {
            // DXR 1.0 uses separate dialog with different features
            { MSG tmpMsg; while (PeekMessage(&tmpMsg, nullptr, WM_QUIT, WM_QUIT, PM_REMOVE)) {} }
            if (!ShowDxr10SettingsDialog()) {
                FreeGPUList();
                CloseLog();
                return 0;
            }
        } else if (g_settings.renderer == RENDERER_D3D12_RT) {
            // DXR 1.1 (RayQuery) dialog
            { MSG tmpMsg; while (PeekMessage(&tmpMsg, nullptr, WM_QUIT, WM_QUIT, PM_REMOVE)) {} }
            if (!ShowDxrSettingsDialog()) {
                FreeGPUList();
                CloseLog();
                return 0;
            }
        } else if (g_settings.renderer == RENDERER_VULKAN_RT || g_settings.renderer == RENDERER_VULKAN_RQ) {
            // Vulkan RT/RQ settings dialog (same features)
            { MSG tmpMsg; while (PeekMessage(&tmpMsg, nullptr, WM_QUIT, WM_QUIT, PM_REMOVE)) {} }
            if (!ShowVulkanRTSettingsDialog()) {
                FreeGPUList();
                CloseLog();
                return 0;
            }
        } else {
            // Initialize defaults for non-RT renderers
            g_dxrFeatures.SetDefaults();
            g_dxr10Features.SetDefaults();
            g_vulkanRTFeatures.SetDefaults();
        }
    }

    { MSG tmpMsg; while (PeekMessage(&tmpMsg, nullptr, WM_QUIT, WM_QUIT, PM_REMOVE)) {} }

    const char* windowTitle = "RenderTestGPU - Direct3D 11";
    switch (g_settings.renderer) {
    case RENDERER_D3D12: windowTitle = "RenderTestGPU - Direct3D 12"; break;
    case RENDERER_D3D12_DXR10: windowTitle = "RenderTestGPU - D3D12 + DXR 1.0"; break;
    case RENDERER_D3D12_RT: windowTitle = "RenderTestGPU - D3D12 + DXR 1.1"; break;
    case RENDERER_D3D12_PT: windowTitle = "RenderTestGPU - Direct3D 12 + Path Tracing"; break;
    case RENDERER_D3D12_PT_DLSS: windowTitle = "RenderTestGPU - D3D12 + PT + DLSS RR"; break;
    case RENDERER_OPENGL: windowTitle = "RenderTestGPU - OpenGL"; break;
    case RENDERER_VULKAN: windowTitle = "RenderTestGPU - Vulkan"; break;
    case RENDERER_VULKAN_RT: windowTitle = "RenderTestGPU - Vulkan + RT"; break;
    case RENDERER_VULKAN_RQ: windowTitle = "RenderTestGPU - Vulkan + RayQuery"; break;
    }

    WNDCLASS wc = {0, WndProc, 0, 0, hI, 0, LoadCursor(0, IDC_ARROW), 0, 0, "RenderTestGPU"};
    RegisterClass(&wc);
    RECT r = {0, 0, (LONG)W, (LONG)H}; AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindow("RenderTestGPU", windowTitle, WS_OVERLAPPEDWINDOW, 100, 100,
        r.right - r.left, r.bottom - r.top, 0, 0, hI, 0);
    g_hMainWnd = hwnd;
    if (!hwnd) { FreeGPUList(); CloseLog(); return 1; }
    ShowWindow(hwnd, SW_SHOW);

    QueryPerformanceFrequency(&g_perfFreq);
    QueryPerformanceCounter(&g_startTime);

    // Initialize selected renderer
    bool initOK = false;
    switch (g_settings.renderer) {
    case RENDERER_D3D12_PT_DLSS:
        initOK = InitD3D12PT_DLSS(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init D3D12+DLSS!", L"Error", MB_OK); }
        break;
    case RENDERER_D3D12_PT:
        initOK = InitD3D12PT(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init D3D12+PT!", L"Error", MB_OK); }
        break;
    case RENDERER_D3D12_RT:
        initOK = InitD3D12RT(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init D3D12+DXR1.1!", L"Error", MB_OK); }
        break;
    case RENDERER_D3D12_DXR10:
        initOK = InitD3D12DXR10(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init D3D12+DXR1.0!", L"Error", MB_OK); }
        break;
    case RENDERER_D3D12:
        initOK = InitD3D12(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init D3D12!", L"Error", MB_OK); }
        break;
    case RENDERER_OPENGL:
        initOK = InitOpenGL(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init OpenGL!", L"Error", MB_OK); }
        break;
    case RENDERER_VULKAN:
        initOK = InitVulkan(hwnd);
        if (initOK && InitVulkanText()) g_vkTextInitialized = true;
        break;
    case RENDERER_VULKAN_RT:
        initOK = InitVulkanRT(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init Vulkan RT!", L"Error", MB_OK); }
        break;
    case RENDERER_VULKAN_RQ:
        initOK = InitVulkanRQ(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init Vulkan RayQuery!", L"Error", MB_OK); }
        break;
    default:
        initOK = InitD3D11(hwnd);
        if (!initOK) { MessageBoxW(0, L"Failed to init D3D11!", L"Error", MB_OK); }
        break;
    }

    if (!initOK) {
        FreeGPUList();
        CloseLog();
        return 1;
    }

    LARGE_INTEGER freq, lastTime, nowTime;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&lastTime);
    UINT64 frames = 0;

    MSG msg = {};
    while (msg.message != WM_QUIT) {
        while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            if (msg.message == WM_QUIT) break;
        }
        if (msg.message == WM_QUIT) break;

        switch (g_settings.renderer) {
        case RENDERER_D3D12_PT_DLSS: RenderD3D12PT_DLSS(); break;
        case RENDERER_D3D12_PT: RenderD3D12PT(); break;
        case RENDERER_D3D12_RT: RenderD3D12RT(); break;
        case RENDERER_D3D12_DXR10: RenderD3D12DXR10(); break;
        case RENDERER_D3D12: RenderD3D12(); break;
        case RENDERER_OPENGL: RenderOpenGL(); break;
        case RENDERER_VULKAN: RenderVulkan(); break;
        case RENDERER_VULKAN_RT: RenderVulkanRT(); break;
        case RENDERER_VULKAN_RQ: RenderVulkanRQ(); break;
        default: RenderD3D11(); break;
        }
        frames++;

        QueryPerformanceCounter(&nowTime);
        double elapsed = (double)(nowTime.QuadPart - lastTime.QuadPart) / freq.QuadPart;
        if (elapsed >= 1.0) {
            fps = (int)(frames / elapsed);
            frames = 0;
            lastTime = nowTime;
        }
    }

    // Cleanup
    switch (g_settings.renderer) {
    case RENDERER_D3D12_PT_DLSS: CleanupD3D12PT_DLSS(); break;
    case RENDERER_D3D12_PT: CleanupD3D12PT(); break;
    case RENDERER_D3D12_RT: CleanupD3D12RT(); break;
    case RENDERER_D3D12_DXR10: CleanupD3D12DXR10(); break;
    case RENDERER_D3D12: CleanupD3D12(); break;
    case RENDERER_OPENGL: CleanupOpenGL(); break;
    case RENDERER_VULKAN: CleanupVulkan(); break;
    case RENDERER_VULKAN_RT: CleanupVulkanRT(); break;
    case RENDERER_VULKAN_RQ: CleanupVulkanRQ(); break;
    default: CleanupD3D11(); break;
    }

    FreeGPUList();
    CloseLog();
    return 0;
}
