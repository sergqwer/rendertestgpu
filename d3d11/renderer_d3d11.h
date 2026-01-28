#pragma once
// ============== D3D11 RENDERER ==============

#include <Windows.h>

// D3D11 Renderer API
bool InitD3D11(HWND hwnd);
void RenderD3D11();
void CleanupD3D11();
