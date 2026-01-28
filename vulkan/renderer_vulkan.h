#pragma once
// ============== VULKAN RENDERER ==============

#include <Windows.h>

// Vulkan Renderer API
bool InitVulkan(HWND hwnd);
bool InitVulkanText();
void RenderVulkan();
void CleanupVulkan();

// Text initialization state (set by main after calling InitVulkanText)
extern bool g_vkTextInitialized;
