#pragma once
// ============== VULKAN RAY TRACING RENDERER ==============
// Uses VK_KHR_ray_tracing_pipeline extension

#include <Windows.h>

// Vulkan RT Renderer API
bool InitVulkanRT(HWND hwnd);
void RenderVulkanRT();
void CleanupVulkanRT();
