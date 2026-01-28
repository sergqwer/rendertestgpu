#pragma once
// ============== D3D12 RENDERERS ==============

#include <Windows.h>

// D3D12 Base Renderer
bool InitD3D12(HWND hwnd);
void RenderD3D12();
void CleanupD3D12();

// D3D12 + DXR 1.0 (TraceRay with raygen/hit/miss shaders)
bool InitD3D12DXR10(HWND hwnd);
void RenderD3D12DXR10();
void CleanupD3D12DXR10();

// D3D12 + DXR 1.1 (Inline RayQuery in pixel shader)
bool InitD3D12RT(HWND hwnd);
void RenderD3D12RT();
void CleanupD3D12RT();

// D3D12 + Path Tracing
bool InitD3D12PT(HWND hwnd);
void RenderD3D12PT();
void CleanupD3D12PT();

// D3D12 + Path Tracing + DLSS Ray Reconstruction
bool InitD3D12PT_DLSS(HWND hwnd);
void RenderD3D12PT_DLSS();
void CleanupD3D12PT_DLSS();
