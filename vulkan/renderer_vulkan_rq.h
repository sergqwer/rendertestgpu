#pragma once
// ============== VULKAN RAYQUERY RENDERER ==============
// Uses VK_KHR_ray_query extension (inline ray tracing in compute shader)
// Simpler than VK_KHR_ray_tracing_pipeline - no SBT needed

#include <Windows.h>

// Vulkan RayQuery Renderer API
bool InitVulkanRQ(HWND hwnd);
void RenderVulkanRQ();
void CleanupVulkanRQ();
