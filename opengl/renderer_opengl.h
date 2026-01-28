#pragma once
// ============== OPENGL RENDERER ==============

#include <Windows.h>

// OpenGL Renderer API
bool InitOpenGL(HWND hwnd);
void RenderOpenGL();
void CleanupOpenGL();
