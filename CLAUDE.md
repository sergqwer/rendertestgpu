# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
RenderTestGPU is a multi-API graphics test application that renders the same scene (rotating colored cubes with lighting) across different graphics APIs. Users can switch between renderers at runtime via a startup dialog.

## Building
```bash
# From rendertestgpu directory
rebuild.bat          # Full rebuild
build_release.bat    # Release build only
```
Output: `bin/Release/rendertestgpu.exe`

Requires Visual Studio 2019+ (v142 toolset) with:
- Windows SDK 10.0 (D3D11, D3D12)
- Vulkan headers (included in `vulkan/` directory)
- NVIDIA DLSS SDK (optional, DLLs in `bin/Release/`)
- DXC compiler (`dxcompiler.dll`, `dxil.dll`) for runtime shader compilation

## Supported Renderers
| Renderer | File | DXR/RT Version |
|----------|------|----------------|
| Direct3D 11 | `d3d11/renderer_d3d11.cpp` | - |
| Direct3D 12 | `d3d12/renderer_d3d12.cpp` | - |
| D3D12 + DXR 1.0 | `d3d12/renderer_d3d12_dxr10.cpp` | TraceRay (raygen/hit/miss shaders) |
| D3D12 + DXR 1.1 | `d3d12/renderer_d3d12_rt.cpp` | Inline RayQuery in pixel shader |
| D3D12 + Path Tracing | `d3d12/renderer_d3d12_pt.cpp` | Compute shader with RayQuery |
| D3D12 + DLSS | `d3d12/renderer_d3d12_dlss.cpp` | Path tracing + DLSS Ray Reconstruction |
| OpenGL | `opengl/renderer_opengl.cpp` | - |
| Vulkan | `vulkan/renderer_vulkan.cpp` | - |
| Vulkan + RT | `vulkan/renderer_vulkan_rt.cpp` | VK_KHR_ray_tracing_pipeline |

## Architecture

### Renderer Interface Pattern
Every renderer follows the same three-function interface:
```cpp
bool Init<API>(HWND hwnd);   // Create device, swapchain, pipelines, geometry
void Render<API>();          // Per-frame rendering
void Cleanup<API>();         // Release all resources
```

### D3D12 Shared Resources
D3D12 renderers share common state via `d3d12/d3d12_shared.h` and `d3d12/d3d12_globals.cpp`:
- Device, command queue, swap chain, fence synchronization
- Text rendering pipeline (`textPso`, `textRootSig12`, `fontTex12`)
- Call `InitGPUText12()` after device creation but before render loop

Feature flags are configured via structs:
- `DXRFeatures g_dxrFeatures` - DXR 1.1 settings (shadows, AO, GI, reflections)
- `DXR10Features g_dxr10Features` - DXR 1.0 settings
- `VulkanRTFeatures g_vulkanRTFeatures` - Vulkan RT settings

### Shader Organization
| Header | Purpose |
|--------|---------|
| `shaders/d3d11_shaders.h` | HLSL for D3D11/D3D12 rasterization |
| `shaders/d3d12_rt_shaders.h` | DXR 1.1 ray tracing shaders |
| `shaders/d3d12_pt_shaders.h` | Path tracing compute shaders |
| `shaders/d3d12_denoise_shaders.h` | Temporal/spatial denoising |
| `vulkan/vulkan_shaders.h` | Pre-compiled SPIR-V bytecode |
| `vulkan/vulkan_rt_spirv.h` | Vulkan RT SPIR-V shaders |

### Text Rendering
All renderers implement GPU-accelerated text using an 8x8 bitmap font defined in `main.cpp`:
- Font data: `g_font8x8[96][8]` - ASCII 32-127
- Texture: 128x48 pixels (16 chars × 6 rows)
- Vertex format: `TextVert {float2 pos, float2 uv, float4 color}`
- Text positioned at (10, 10) with 1.5× scale, white with black shadow

## Key Technical Details

### Lighting Model
- World space light direction: `normalize(0.2, 1.0, 0.3)`
- Diffuse formula: `max(dot(N, L), 0) * 0.65 + 0.35` (includes ambient)
- D3D11/D3D12: Transform normals to world space in vertex shader
- Vulkan: Transform light to object space on CPU (`lightObj = rot^T * lightWorld`)

### SPIR-V Notes (Vulkan)
- IDs must be globally unique; `Bound` in header = max ID + 1
- Use `OpConstantComposite` for module-level constants
- Fragment shaders require `OpExecutionMode OriginUpperLeft`

### Debug Modes (DXR renderers)
Press 0-6 during runtime to visualize:
0. Normal rendering
1. Object IDs (color-coded)
2. Normals
3. Reflection directions
4. Shadow visibility
5. World position
6. Depth

## Common Issues

### Vulkan text not rendering
1. Verify SPIR-V shader validity (Bound value, unique IDs)
2. Check descriptor set layout matches shader bindings
3. Ensure descriptor set bound before draw call
4. Text must render in same render pass as 3D (separate pass clears buffer)

### D3D12 text not showing
Call `InitGPUText12()` after device/command queue creation, before main loop.

### Lighting appears to rotate with object
Transform light direction to object space: `lightObj = transpose(rotationMatrix) * lightWorld`

### DXR initialization fails
Check `CheckDXRSupport()` return value - requires D3D12 device with DXR Tier 1.0+ support.
