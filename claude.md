# RenderTestGPU - Multi-API Graphics Renderer

## Overview
A test application demonstrating multiple graphics APIs rendering the same scene (rotating colored cubes with lighting). Supports switching between renderers at runtime via dropdown menu.

## Supported Renderers
| Renderer | File | Description |
|----------|------|-------------|
| Direct3D 11 | `d3d11/renderer_d3d11.cpp` | Base D3D11 renderer |
| Direct3D 12 | `d3d12/renderer_d3d12.cpp` | Base D3D12 renderer |
| Direct3D 12 + Ray Tracing | `d3d12/renderer_d3d12_rt.cpp` | DXR hardware ray tracing |
| Direct3D 12 + Path Tracing | `d3d12/renderer_d3d12_pt.cpp` | Compute shader path tracing with RayQuery |
| Direct3D 12 + DLSS | `d3d12/renderer_d3d12_dlss.cpp` | NVIDIA DLSS integration |
| OpenGL | `opengl/renderer_opengl.cpp` | OpenGL 4.x renderer |
| Vulkan | `vulkan/renderer_vulkan.cpp` | Vulkan renderer |

## Directory Structure
```
rendertestgpu/
├── main.cpp                    # Window creation, message loop, renderer selection
├── common.h                    # Shared types (TextVert, font data extern)
├── shaders/
│   └── d3d11_shaders.h        # HLSL shaders for D3D11/D3D12
├── d3d11/
│   └── renderer_d3d11.cpp     # D3D11 implementation
├── d3d12/
│   ├── d3d12_shared.h         # Shared D3D12 globals declarations
│   ├── d3d12_globals.cpp      # D3D12 global variable definitions
│   ├── renderer_d3d12.cpp     # Base D3D12
│   ├── renderer_d3d12_rt.cpp  # DXR ray tracing
│   ├── renderer_d3d12_pt.cpp  # Path tracing (compute)
│   └── renderer_d3d12_dlss.cpp# DLSS integration
├── opengl/
│   └── renderer_opengl.cpp    # OpenGL implementation
├── vulkan/
│   ├── renderer_vulkan.cpp    # Vulkan implementation
│   ├── renderer_vulkan.h      # Vulkan function declarations
│   └── vulkan_shaders.h       # Pre-compiled SPIR-V shaders
└── bin/Release/
    └── rendertestgpu.exe      # Output executable
```

## Building
```bash
# From rendertestgpu directory
rebuild.bat          # Full rebuild
build_release.bat    # Release build only
```
Requires Visual Studio 2019+ with:
- Windows SDK (D3D11, D3D12)
- Vulkan SDK
- NVIDIA DLSS SDK (optional, for DLSS renderer)

## Key Technical Details

### Text Rendering
All renderers implement GPU-accelerated text rendering using an 8x8 bitmap font:
- Font texture: 128x48 (16 chars x 6 rows = 96 ASCII chars)
- Vertex format: `{float2 pos, float2 uv, float4 color}`
- Alpha blending with font texture mask

### Vulkan SPIR-V Shaders
Vulkan shaders are embedded as pre-compiled SPIR-V bytecode in `vulkan_shaders.h`:
- **3D shaders**: Position transform with MVP, lighting calculation
- **Text shaders**: UV passthrough, texture sampling with color multiply

Important SPIR-V notes:
- IDs must be unique across the entire module
- `Bound` in header = max ID + 1
- Use `OpConstantComposite` for module-level constants, `OpCompositeConstruct` for function-level
- Fragment shader needs `OpExecutionMode OriginUpperLeft`

### Lighting
- Light direction: `normalize(0.2, 1.0, 0.3)` in world space
- D3D11/D3D12: Normals transformed to world space in vertex shader
- Vulkan: Light transformed to object space on CPU (inverse rotation)
- Diffuse: `max(dot(N, L), 0) * 0.65 + 0.35` (ambient term)

### D3D12 Shared Resources
D3D12 renderers share common resources via `d3d12_shared.h`:
- Device, command queue, swap chain
- Text rendering pipeline and font texture
- Use `InitGPUText12()` after device creation

## Common Issues

### Vulkan text not rendering
1. Check SPIR-V shader validity (Bound value, unique IDs)
2. Verify descriptor set layout matches shader bindings
3. Ensure descriptor set is bound before draw call
4. Text must render in same render pass as 3D (separate pass clears buffer)

### Lighting rotates with object
Transform light direction to object space: `lightObj = rot^T * lightWorld`
(For orthogonal matrices, inverse = transpose)

### D3D12 text not showing
Call `InitGPUText12()` after creating device and command queue but before main loop.

## Text Display Format
```
API: [Renderer Name]
GPU: [GPU Name]
FPS: [Frame Rate]
Triangles: [Count]
Resolution: [Width]x[Height]
```
Position: top-left (10, 10), white text with black shadow, scale 1.5x
