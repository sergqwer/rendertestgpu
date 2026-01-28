// ============== D3D12 + DXR 1.0 RENDERER ==============
// Full ray tracing pipeline using TraceRay() with raygen/closesthit/miss shaders
// Uses DispatchRays() - compatible with DXR 1.0 GPUs (SM 6.3)
// Scene matches DXR 1.1 renderer exactly

#include "../common.h"
#include "d3d12_shared.h"
#include "renderer_d3d12.h"

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <dxcapi.h>
#include <vector>
#include <string>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;

// ============== DXR 1.0 SHADER CODE (lib_6_3) ==============
// Split into multiple strings to avoid MSVC string length limit
// Uses #ifdef for conditional feature compilation:
//   FEATURE_SPOTLIGHT    - Cone light with falloff
//   FEATURE_SOFT_SHADOWS - Multiple shadow samples
//   FEATURE_AO           - Ambient occlusion
//   FEATURE_GI           - Global illumination (1 bounce)
//   FEATURE_REFLECTIONS  - Mirror reflections
//   FEATURE_GLASS        - Glass transparency with fresnel

static const char* g_dxr10ShaderPart1 = R"HLSL(
// ============== RAYTRACING SHADER LIBRARY (lib_6_3) ==============
// Cornell Box scene - matches DXR 1.1 exactly
// Uses InstanceIndex() and PrimitiveIndex() ranges - no vertex buffer access
// Conditional compilation via #ifdef FEATURE_*

// Output UAV
RWTexture2D<float4> OutputUAV : register(u0);

// Acceleration structure
RaytracingAccelerationStructure Scene : register(t0);

// Constant buffer - parameters from CPU
cbuffer SceneCB : register(b0) {
    float Time;
    float3 LightPos;
    float LightRadius;       // For soft shadows
    uint FrameCount;
    int ShadowSamples;       // 1, 4, 8
    int AOSamples;           // 1, 3, 5
    float AORadius;          // 0.1 - 1.0
};

// Hardcoded camera (same as DXR 1.1)
static const float3 CameraPos = float3(0, 0, -2.2);
static const float3 LightPosDefault = float3(0, 0.92, 0);

#ifdef FEATURE_SPOTLIGHT
// Spotlight parameters - creates visible cone of light on floor
static const float3 SpotlightDir = normalize(float3(0, -1, 0.15));  // Points down with slight forward tilt
static const float SpotInnerCos = 0.85;  // ~32 degrees - full intensity
static const float SpotOuterCos = 0.5;   // ~60 degrees - falloff edge
#endif

// Ray payload
struct RayPayload {
    float3 color;
    float hitT;
    float3 normal;
    float3 hitPos;
    uint objectID;
    uint cubeIndex;    // For rotating cubes
    uint materialType;
    bool hit;
};

// Shadow payload
struct ShadowPayload {
    bool inShadow;
};

// ============== MATERIAL TYPES ==============
#define MAT_DIFFUSE  0
#define MAT_MIRROR   1
#define MAT_GLASS    2
#define MAT_EMISSIVE 3

// ============== OBJECT IDs ==============
#define OBJ_FLOOR      0
#define OBJ_CEILING    1
#define OBJ_BACK_WALL  2
#define OBJ_LEFT_WALL  3
#define OBJ_RIGHT_WALL 4
#define OBJ_LIGHT      5
#define OBJ_CUBE       6
#define OBJ_MIRROR     7
#define OBJ_GLASS      8
#define OBJ_SMALL_CUBE 9
#define OBJ_FRONT_WALL 10

// ============== INSTANCE IDs ==============
#define INSTANCE_STATIC 0
#define INSTANCE_CUBES  1

// ============== SCENE COLORS (MUST MATCH DXR 1.1) ==============
static const float3 Colors[11] = {
    float3(0.7, 0.7, 0.7),    // 0: Floor - grey
    float3(0.9, 0.9, 0.9),    // 1: Ceiling - white
    float3(0.7, 0.7, 0.7),    // 2: Back wall - grey
    float3(0.75, 0.15, 0.15), // 3: Left wall - RED
    float3(0.15, 0.75, 0.15), // 4: Right wall - GREEN
    float3(15.0, 14.0, 12.0), // 5: Light - bright emissive
    float3(0.9, 0.6, 0.2),    // 6: Cube - orange (fallback)
    float3(0.95, 0.95, 0.95), // 7: Mirror - neutral
    float3(0.9, 0.95, 1.0),   // 8: Glass - slight blue tint
    float3(0.9, 0.15, 0.1),   // 9: Small cube - RED
    float3(0.5, 0.15, 0.7)    // 10: Front wall - PURPLE
};

// ============== CUBE COLORS (8 cubes, brighter and more saturated) ==============
static const float3 CubeColors[8] = {
    float3(1.0, 0.15, 0.1),   // 0: Bright Red
    float3(0.1, 0.9, 0.2),    // 1: Bright Green
    float3(0.1, 0.4, 1.0),    // 2: Bright Blue
    float3(1.0, 0.95, 0.1),   // 3: Bright Yellow
    float3(1.0, 0.95, 0.1),   // 4: Bright Yellow
    float3(0.1, 0.4, 1.0),    // 5: Bright Blue
    float3(0.1, 0.9, 0.2),    // 6: Bright Green
    float3(1.0, 0.15, 0.1)    // 7: Bright Red
};

// ============== STATIC GEOMETRY PRIMITIVE RANGES ==============
// Order: floor(2), ceiling(2), back_wall(2), left_wall(2), right_wall(2),
//        light(2), mirror(2), small_cube(12), glass(4), front_wall(2)
void GetStaticObjectInfo(uint primID, out uint objID, out uint matType, out float3 normal) {
    matType = MAT_DIFFUSE;
    if (primID < 2) { objID = OBJ_FLOOR; normal = float3(0, 1, 0); }
    else if (primID < 4) { objID = OBJ_CEILING; normal = float3(0, -1, 0); }
    else if (primID < 6) { objID = OBJ_BACK_WALL; normal = float3(0, 0, -1); }
    else if (primID < 8) { objID = OBJ_LEFT_WALL; normal = float3(1, 0, 0); }
    else if (primID < 10) { objID = OBJ_RIGHT_WALL; normal = float3(-1, 0, 0); }
    else if (primID < 12) { objID = OBJ_LIGHT; normal = float3(0, -1, 0); matType = MAT_EMISSIVE; }
    else if (primID < 14) { objID = OBJ_MIRROR; normal = normalize(float3(0.707, 0, -0.707)); matType = MAT_MIRROR; }
    else if (primID < 26) {
        objID = OBJ_SMALL_CUBE;
        // Get face normal for small cube (12 triangles = 6 faces)
        uint faceIdx = (primID - 14) / 2;
        if (faceIdx == 0) normal = float3(0, 0, 1);      // front
        else if (faceIdx == 1) normal = float3(0, 0, -1); // back
        else if (faceIdx == 2) normal = float3(1, 0, 0);  // right
        else if (faceIdx == 3) normal = float3(-1, 0, 0); // left
        else if (faceIdx == 4) normal = float3(0, 1, 0);  // top
        else normal = float3(0, -1, 0);                   // bottom
    }
    else if (primID < 30) { objID = OBJ_GLASS; normal = float3(0, 0, -1); matType = MAT_GLASS; }
    else { objID = OBJ_FRONT_WALL; normal = float3(0, 0, 1); }
}

// ============== ROTATION MATRICES ==============
float3x3 RotateY(float angle) {
    float c = cos(angle), s = sin(angle);
    return float3x3(c, 0, s, 0, 1, 0, -s, 0, c);
}
float3x3 RotateX(float angle) {
    float c = cos(angle), s = sin(angle);
    return float3x3(1, 0, 0, 0, c, -s, 0, s, c);
}

// Get cube face normal in WORLD SPACE (after rotation)
float3 GetCubeFaceNormal(uint primID) {
    // Each cube has 12 triangles (6 faces * 2)
    uint localPrim = primID % 12;
    uint faceIdx = localPrim / 2;
    float3 localNormal;
    if (faceIdx == 0) localNormal = float3(0, 0, 1);      // front
    else if (faceIdx == 1) localNormal = float3(0, 0, -1); // back
    else if (faceIdx == 2) localNormal = float3(1, 0, 0);  // right
    else if (faceIdx == 3) localNormal = float3(-1, 0, 0); // left
    else if (faceIdx == 4) localNormal = float3(0, 1, 0);  // top
    else localNormal = float3(0, -1, 0);                   // bottom

    // Transform to world space using current rotation
    float angleY = Time * 1.2;
    float angleX = Time * 0.7;
    float3x3 rot = mul(RotateY(angleY), RotateX(angleX));
    return normalize(mul(localNormal, rot));
}

// Get color for object
float3 GetObjectColor(uint objID, uint cubeIndex) {
    if (objID == OBJ_CUBE) {
        return CubeColors[min(cubeIndex, 7u)];
    }
    return Colors[min(objID, 10u)];
}

// ============== RANDOM NUMBER GENERATOR ==============
uint WangHash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float Random(inout uint seed) {
    seed = WangHash(seed);
    return float(seed) / 4294967295.0;
}

float3 RandomInDisk(inout uint seed) {
    float r = sqrt(Random(seed));
    float theta = 6.28318530718 * Random(seed);
    return float3(r * cos(theta), 0, r * sin(theta));
}

float3 RandomInHemisphere(float3 normal, inout uint seed) {
    float u1 = Random(seed);
    float u2 = Random(seed);
    float r = sqrt(u1);
    float theta = 6.28318530718 * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - u1);
    float3 up = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

#ifdef FEATURE_SPOTLIGHT
// Spotlight cone attenuation - creates visible light cone on floor
float SpotlightAttenuation(float3 lightToPoint) {
    float3 L = normalize(lightToPoint);
    float cosAngle = dot(L, SpotlightDir);
    return saturate((cosAngle - SpotOuterCos) / (SpotInnerCos - SpotOuterCos));
}
#endif

)HLSL";

static const char* g_dxr10ShaderPart2 = R"HLSL(

// ============== RAY GENERATION SHADER ==============
[shader("raygeneration")]
void RayGen() {
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    // Generate ray from camera
    float2 uv = (float2(launchIndex) + 0.5) / float2(launchDim);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float aspectRatio = float(launchDim.x) / float(launchDim.y);
    float tanHalfFovY = 1.0 / 1.73;
    float tanHalfFovX = tanHalfFovY * aspectRatio;

    float3 rayDir = normalize(float3(ndc.x * tanHalfFovX, ndc.y * tanHalfFovY, 1.0));
    float3 rayOrigin = CameraPos;

    // Trace primary ray
    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDir;
    ray.TMin = 0.001;
    ray.TMax = 1000.0;

    RayPayload payload;
    payload.color = float3(0, 0, 0);
    payload.hitT = -1;
    payload.hit = false;
    payload.cubeIndex = 0;

    TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);

    float3 finalColor = float3(0.05, 0.05, 0.08);  // Background

    if (payload.hit) {
        float3 hitPos = payload.hitPos;
        float3 normal = payload.normal;
        uint objID = payload.objectID;
        uint matType = payload.materialType;
        uint cubeIdx = payload.cubeIndex;
        float3 baseColor = GetObjectColor(objID, cubeIdx);

        // Emissive light
        if (objID == OBJ_LIGHT) {
            finalColor = float3(1.0, 0.98, 0.9);
        }
#ifdef FEATURE_REFLECTIONS
        // Mirror reflection
        else if (matType == MAT_MIRROR) {
            float3 reflectDir = reflect(rayDir, normal);
            RayDesc reflectRay;
            reflectRay.Origin = hitPos + normal * 0.002;
            reflectRay.Direction = reflectDir;
            reflectRay.TMin = 0.001;
            reflectRay.TMax = 100.0;

            RayPayload reflectPayload;
            reflectPayload.hit = false;
            reflectPayload.cubeIndex = 0;
            TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, reflectRay, reflectPayload);

            if (reflectPayload.hit) {
                float3 reflColor = GetObjectColor(reflectPayload.objectID, reflectPayload.cubeIndex);
                if (reflectPayload.objectID == OBJ_LIGHT) {
                    reflColor = float3(1.0, 0.98, 0.9);
                } else {
                    float3 toLight = normalize(LightPos - reflectPayload.hitPos);
                    float NdotL = max(dot(reflectPayload.normal, toLight), 0.0);
#ifdef FEATURE_SPOTLIGHT
                    float reflSpot = SpotlightAttenuation(reflectPayload.hitPos - LightPos);
                    reflColor *= (0.15 + NdotL * reflSpot * 0.85);
#else
                    reflColor *= (0.25 + NdotL * 0.75);
#endif
                }
                finalColor = lerp(baseColor * 0.1, reflColor, 0.9);
            } else {
                finalColor = baseColor * 0.3;
            }
        }
#endif
#ifdef FEATURE_GLASS
        // Glass transparency
        else if (matType == MAT_GLASS) {
            RayDesc throughRay;
            throughRay.Origin = hitPos + rayDir * 0.01;
            throughRay.Direction = rayDir;
            throughRay.TMin = 0.001;
            throughRay.TMax = 100.0;

            RayPayload throughPayload;
            throughPayload.hit = false;
            throughPayload.cubeIndex = 0;
            TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, throughRay, throughPayload);

            float3 behindColor = float3(0.05, 0.05, 0.08);
            if (throughPayload.hit) {
                behindColor = GetObjectColor(throughPayload.objectID, throughPayload.cubeIndex);
                if (throughPayload.objectID != OBJ_LIGHT) {
                    float3 toLight = normalize(LightPos - throughPayload.hitPos);
                    float NdotL = max(dot(throughPayload.normal, toLight), 0.0);
#ifdef FEATURE_SPOTLIGHT
                    float glassSpot = SpotlightAttenuation(throughPayload.hitPos - LightPos);
                    behindColor *= (0.2 + NdotL * glassSpot * 0.8);
#else
                    behindColor *= (0.3 + NdotL * 0.7);
#endif
                }
            }
            float fresnel = pow(1.0 - abs(dot(-rayDir, normal)), 3.0);
            float3 glassTint = float3(0.95, 0.97, 1.0);
            finalColor = behindColor * glassTint * (1.0 - fresnel * 0.3);
        }
#endif
        // Diffuse surfaces
        else {
            uint seed = launchIndex.x + launchIndex.y * 1920 + FrameCount * 1920 * 1080;

            float3 toLight = normalize(LightPos - hitPos);
            float NdotL = max(dot(normal, toLight), 0.0);
            float lightDist = length(LightPos - hitPos);

            // ============== SHADOWS ==============
            float shadow = 1.0;
#ifdef FEATURE_SOFT_SHADOWS
            // Soft shadows with multiple samples
            shadow = 0.0;
            int shadowSamples = max(ShadowSamples, 1);
            for (int s = 0; s < shadowSamples; s++) {
                float3 jitter = RandomInDisk(seed) * LightRadius;
                float3 targetPos = LightPos + jitter;
                float3 toJitteredLight = targetPos - hitPos;
                float jitteredDist = length(toJitteredLight);
                float3 jitteredDir = toJitteredLight / jitteredDist;

                RayDesc shadowRay;
                shadowRay.Origin = hitPos + normal * 0.002;
                shadowRay.Direction = jitteredDir;
                shadowRay.TMin = 0.001;
                shadowRay.TMax = jitteredDist - 0.01;

                ShadowPayload shadowPayload;
                shadowPayload.inShadow = false;
                TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                         0xFF, 1, 1, 1, shadowRay, shadowPayload);

                if (!shadowPayload.inShadow) shadow += 1.0;
            }
            shadow /= float(shadowSamples);
#else
            // Single hard shadow ray
            {
                RayDesc shadowRay;
                shadowRay.Origin = hitPos + normal * 0.002;
                shadowRay.Direction = toLight;
                shadowRay.TMin = 0.001;
                shadowRay.TMax = lightDist - 0.01;

                ShadowPayload shadowPayload;
                shadowPayload.inShadow = false;
                TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                         0xFF, 1, 1, 1, shadowRay, shadowPayload);
                shadow = shadowPayload.inShadow ? 0.0 : 1.0;
            }
#endif

            // ============== AMBIENT OCCLUSION ==============
            float ao = 1.0;
#ifdef FEATURE_AO
            ao = 0.0;
            int aoSamples = max(AOSamples, 1);
            for (int a = 0; a < aoSamples; a++) {
                float3 aoDir = RandomInHemisphere(normal, seed);

                RayDesc aoRay;
                aoRay.Origin = hitPos + normal * 0.002;
                aoRay.Direction = aoDir;
                aoRay.TMin = 0.001;
                aoRay.TMax = AORadius;

                ShadowPayload aoPayload;
                aoPayload.inShadow = false;
                TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
                         0xFF, 1, 1, 1, aoRay, aoPayload);

                if (!aoPayload.inShadow) ao += 1.0;
            }
            ao /= float(aoSamples);
#endif

            // ============== GLOBAL ILLUMINATION ==============
            float3 gi = float3(0, 0, 0);
#ifdef FEATURE_GI
            float3 giDir = RandomInHemisphere(normal, seed);

            RayDesc giRay;
            giRay.Origin = hitPos + normal * 0.002;
            giRay.Direction = giDir;
            giRay.TMin = 0.001;
            giRay.TMax = 10.0;

            RayPayload giPayload;
            giPayload.hit = false;
            giPayload.cubeIndex = 0;
            TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, giRay, giPayload);

            if (giPayload.hit && giPayload.objectID != OBJ_LIGHT) {
                float3 giColor = GetObjectColor(giPayload.objectID, giPayload.cubeIndex);
                float giNdotL = max(dot(giPayload.normal, -giDir), 0.0);
                gi = giColor * giNdotL * 0.3;
            }
#endif

            // ============== FINAL LIGHTING ==============
            float distAtten = 2.5 / (1.0 + lightDist * lightDist * 0.08);

#ifdef FEATURE_SPOTLIGHT
            float spotAtten = SpotlightAttenuation(hitPos - LightPos);
            float totalAtten = distAtten * spotAtten;
            float3 ambient = baseColor * 0.08 * ao;
#else
            float totalAtten = distAtten;
            float3 ambient = baseColor * 0.15 * ao;
#endif

            float3 diffuse = baseColor * NdotL * shadow * totalAtten;
            float3 indirect = baseColor * gi;
            finalColor = ambient + diffuse + indirect;

            // Tone mapping
            finalColor = finalColor / (finalColor + 1.0);
        }
    }

    // Gamma correction
    finalColor = pow(max(finalColor, 0.0), 1.0 / 2.2);

    OutputUAV[launchIndex] = float4(finalColor, 1.0);
}

)HLSL";

static const char* g_dxr10ShaderPart3 = R"HLSL(

// ============== CLOSEST HIT SHADER ==============
[shader("closesthit")]
void ClosestHit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    uint primIdx = PrimitiveIndex();
    // Use InstanceIndex() instead of InstanceID() - InstanceIndex returns 0-based TLAS index
    // InstanceID() returns user-defined value which we never set, so it's always 0!
    uint instIdx = InstanceIndex();

    payload.hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.hitT = RayTCurrent();
    payload.hit = true;
    payload.cubeIndex = 0;

    if (instIdx == INSTANCE_STATIC) {  // Instance 0 = static geometry
        // Static geometry - use primitive ranges to determine object
        GetStaticObjectInfo(primIdx, payload.objectID, payload.materialType, payload.normal);
    } else {
        // Dynamic cubes - Instance 1
        payload.objectID = OBJ_CUBE;
        payload.materialType = MAT_DIFFUSE;
        // Each of 8 cubes has 12 triangles
        payload.cubeIndex = primIdx / 12;
        // Get rotated normal
        payload.normal = GetCubeFaceNormal(primIdx);
    }
}

// ============== MISS SHADER ==============
[shader("miss")]
void Miss(inout RayPayload payload) {
    payload.hit = false;
    payload.color = float3(0.05, 0.05, 0.08);
}

// ============== SHADOW CLOSEST HIT ==============
[shader("closesthit")]
void ShadowHit(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attribs) {
    payload.inShadow = true;
}

// ============== SHADOW MISS ==============
[shader("miss")]
void ShadowMiss(inout ShadowPayload payload) {
    payload.inShadow = false;
}

)HLSL";

// ============== LOCAL TYPES ==============
enum DXR10Material { DXR10_MAT_DIFFUSE = 0, DXR10_MAT_MIRROR = 1, DXR10_MAT_GLASS = 2, DXR10_MAT_EMISSIVE = 3 };
enum DXR10ObjectID {
    DXR10_OBJ_FLOOR = 0, DXR10_OBJ_CEILING = 1, DXR10_OBJ_BACK_WALL = 2,
    DXR10_OBJ_LEFT_WALL = 3, DXR10_OBJ_RIGHT_WALL = 4, DXR10_OBJ_LIGHT = 5,
    DXR10_OBJ_CUBE = 6, DXR10_OBJ_MIRROR = 7, DXR10_OBJ_GLASS = 8, DXR10_OBJ_SMALL_CUBE = 9,
    DXR10_OBJ_FRONT_WALL = 10
};

#pragma pack(push, 1)
struct DXR10Vert {
    XMFLOAT3 pos;
    XMFLOAT3 norm;
    UINT objectID;
    UINT materialType;
};
#pragma pack(pop)
static_assert(sizeof(DXR10Vert) == 32, "DXR10Vert must be 32 bytes");

struct alignas(256) DXR10CB {
    float time;
    XMFLOAT3 lightPos;
    float lightRadius;       // For soft shadows
    UINT frameCount;
    INT shadowSamples;       // 1, 4, 8
    INT aoSamples;           // 1, 3, 5
    float aoRadius;          // 0.1 - 1.0
};

// ============== FEATURE TRACKING ==============
static DXR10Features s_compiledFeatures = {};  // Currently compiled features

// ============== LOCAL STATIC RESOURCES ==============
static ID3D12Device5* s_device = nullptr;
static ID3D12CommandQueue* s_cmdQueue = nullptr;
static ID3D12CommandAllocator* s_cmdAlloc[3] = {};
static ID3D12GraphicsCommandList4* s_cmdList = nullptr;
static IDXGISwapChain3* s_swapChain = nullptr;

static ID3D12DescriptorHeap* s_rtvHeap = nullptr;
static ID3D12Resource* s_renderTargets[3] = {};
static UINT s_rtvDescSize = 0;
static UINT s_frameIndex = 0;

static ID3D12Fence* s_fence = nullptr;
static UINT64 s_fenceValues[3] = {};
static HANDLE s_fenceEvent = nullptr;

// RT resources
static ID3D12Resource* s_outputUAV = nullptr;
static ID3D12Resource* s_blasStatic = nullptr;
static ID3D12Resource* s_blasCube = nullptr;
static ID3D12Resource* s_tlas = nullptr;
static ID3D12Resource* s_scratchBuffer = nullptr;
static ID3D12Resource* s_instanceBuffer = nullptr;
static void* s_instanceMapped = nullptr;

static ID3D12Resource* s_vertexBufferStatic = nullptr;
static ID3D12Resource* s_indexBufferStatic = nullptr;
static ID3D12Resource* s_vertexBufferCube = nullptr;
static ID3D12Resource* s_indexBufferCube = nullptr;
static UINT s_vertexCountStatic = 0, s_indexCountStatic = 0;
static UINT s_vertexCountCube = 0, s_indexCountCube = 0;

static ID3D12Resource* s_constantBuffer = nullptr;
static void* s_cbMapped = nullptr;

// RT pipeline
static ID3D12StateObject* s_rtPSO = nullptr;
static ID3D12StateObjectProperties* s_rtPSOProps = nullptr;
static ID3D12RootSignature* s_globalRootSig = nullptr;
static ID3D12DescriptorHeap* s_srvUavHeap = nullptr;

// Shader tables
static ID3D12Resource* s_rayGenTable = nullptr;
static ID3D12Resource* s_missTable = nullptr;
static ID3D12Resource* s_hitGroupTable = nullptr;
static UINT64 s_rayGenRecordSize = 0;
static UINT64 s_missRecordSize = 0;
static UINT64 s_hitGroupRecordSize = 0;

// Text rendering
static ID3D12RootSignature* s_textRootSig = nullptr;
static ID3D12PipelineState* s_textPso = nullptr;
static ID3D12DescriptorHeap* s_textSrvHeap = nullptr;
static ID3D12Resource* s_fontTexture = nullptr;
static ID3D12Resource* s_textVB = nullptr;
static D3D12_VERTEX_BUFFER_VIEW s_textVBView = {};
static void* s_textVBMapped = nullptr;
static TextVert s_textVerts[6000];
static UINT s_textVertCount = 0;
static int s_cachedFps = -1;
static std::wstring s_gpuName;

// ============== HELPERS ==============
static void WaitForGpu10() {
    if (!s_cmdQueue || !s_fence || !s_fenceEvent) return;
    const UINT64 fv = s_fenceValues[s_frameIndex];
    s_cmdQueue->Signal(s_fence, fv);
    if (s_fence->GetCompletedValue() < fv) {
        s_fence->SetEventOnCompletion(fv, s_fenceEvent);
        WaitForSingleObject(s_fenceEvent, INFINITE);
    }
    s_fenceValues[s_frameIndex]++;
}

static void MoveToNextFrame10() {
    const UINT64 currentFenceValue = s_fenceValues[s_frameIndex];
    s_cmdQueue->Signal(s_fence, currentFenceValue);
    s_frameIndex = s_swapChain->GetCurrentBackBufferIndex();
    if (s_fence->GetCompletedValue() < s_fenceValues[s_frameIndex]) {
        s_fence->SetEventOnCompletion(s_fenceValues[s_frameIndex], s_fenceEvent);
        WaitForSingleObject(s_fenceEvent, INFINITE);
    }
    s_fenceValues[s_frameIndex] = currentFenceValue + 1;
}

// ============== GEOMETRY (copied from DXR 1.1) ==============
static void AddQuad(std::vector<DXR10Vert>& verts, std::vector<UINT>& inds,
    XMFLOAT3 p0, XMFLOAT3 p1, XMFLOAT3 p2, XMFLOAT3 p3, XMFLOAT3 normal, UINT objID, UINT matType) {
    UINT base = (UINT)verts.size();
    DXR10Vert v = {}; v.norm = normal; v.objectID = objID; v.materialType = matType;
    v.pos = p0; verts.push_back(v);
    v.pos = p1; verts.push_back(v);
    v.pos = p2; verts.push_back(v);
    v.pos = p3; verts.push_back(v);
    inds.push_back(base + 0); inds.push_back(base + 1); inds.push_back(base + 2);
    inds.push_back(base + 0); inds.push_back(base + 2); inds.push_back(base + 3);
}

static void AddBox(std::vector<DXR10Vert>& verts, std::vector<UINT>& inds,
    XMFLOAT3 center, XMFLOAT3 halfSize, UINT objID, UINT matType) {
    float cx = center.x, cy = center.y, cz = center.z;
    float hx = halfSize.x, hy = halfSize.y, hz = halfSize.z;
    AddQuad(verts, inds, {cx-hx, cy-hy, cz+hz}, {cx+hx, cy-hy, cz+hz}, {cx+hx, cy+hy, cz+hz}, {cx-hx, cy+hy, cz+hz}, {0, 0, 1}, objID, matType);
    AddQuad(verts, inds, {cx+hx, cy-hy, cz-hz}, {cx-hx, cy-hy, cz-hz}, {cx-hx, cy+hy, cz-hz}, {cx+hx, cy+hy, cz-hz}, {0, 0, -1}, objID, matType);
    AddQuad(verts, inds, {cx+hx, cy-hy, cz+hz}, {cx+hx, cy-hy, cz-hz}, {cx+hx, cy+hy, cz-hz}, {cx+hx, cy+hy, cz+hz}, {1, 0, 0}, objID, matType);
    AddQuad(verts, inds, {cx-hx, cy-hy, cz-hz}, {cx-hx, cy-hy, cz+hz}, {cx-hx, cy+hy, cz+hz}, {cx-hx, cy+hy, cz-hz}, {-1, 0, 0}, objID, matType);
    AddQuad(verts, inds, {cx-hx, cy+hy, cz+hz}, {cx+hx, cy+hy, cz+hz}, {cx+hx, cy+hy, cz-hz}, {cx-hx, cy+hy, cz-hz}, {0, 1, 0}, objID, matType);
    AddQuad(verts, inds, {cx-hx, cy-hy, cz-hz}, {cx+hx, cy-hy, cz-hz}, {cx+hx, cy-hy, cz+hz}, {cx-hx, cy-hy, cz+hz}, {0, -1, 0}, objID, matType);
}

static void BuildCornellBox10(std::vector<DXR10Vert>& verts, std::vector<UINT>& inds) {
    verts.clear(); inds.clear();
    const float s = 1.0f;
    // Floor
    AddQuad(verts, inds, {-s, -s, -s}, {s, -s, -s}, {s, -s, s}, {-s, -s, s}, {0, 1, 0}, DXR10_OBJ_FLOOR, DXR10_MAT_DIFFUSE);
    // Ceiling
    AddQuad(verts, inds, {-s, s, s}, {s, s, s}, {s, s, -s}, {-s, s, -s}, {0, -1, 0}, DXR10_OBJ_CEILING, DXR10_MAT_DIFFUSE);
    // Back wall
    AddQuad(verts, inds, {-s, -s, s}, {s, -s, s}, {s, s, s}, {-s, s, s}, {0, 0, -1}, DXR10_OBJ_BACK_WALL, DXR10_MAT_DIFFUSE);
    // Left wall (RED)
    AddQuad(verts, inds, {-s, -s, s}, {-s, s, s}, {-s, s, -s}, {-s, -s, -s}, {1, 0, 0}, DXR10_OBJ_LEFT_WALL, DXR10_MAT_DIFFUSE);
    // Right wall (GREEN)
    AddQuad(verts, inds, {s, -s, -s}, {s, s, -s}, {s, s, s}, {s, -s, s}, {-1, 0, 0}, DXR10_OBJ_RIGHT_WALL, DXR10_MAT_DIFFUSE);
    // Ceiling light
    const float ls = 0.3f;
    AddQuad(verts, inds, {-ls, s - 0.01f, ls}, {ls, s - 0.01f, ls}, {ls, s - 0.01f, -ls}, {-ls, s - 0.01f, -ls}, {0, -1, 0}, DXR10_OBJ_LIGHT, DXR10_MAT_EMISSIVE);
    // Mirror at 45 degrees
    const float mh = 0.5f, mw = 0.4f, mcx = -0.6f, mcy = 0.0f, mcz = 0.6f, c45 = 0.707f;
    AddQuad(verts, inds, {mcx - c45*mw, mcy - mh, mcz - c45*mw}, {mcx + c45*mw, mcy - mh, mcz + c45*mw},
        {mcx + c45*mw, mcy + mh, mcz + c45*mw}, {mcx - c45*mw, mcy + mh, mcz - c45*mw}, {c45, 0, -c45}, DXR10_OBJ_MIRROR, DXR10_MAT_MIRROR);
    // Small RED cube
    const float cubeX = -0.5f, cubeY = -0.85f, cubeZ = 0.3f;
    AddBox(verts, inds, {cubeX, cubeY, cubeZ}, {0.13f, 0.13f, 0.13f}, DXR10_OBJ_SMALL_CUBE, DXR10_MAT_DIFFUSE);
    // Glass pane
    const float gz = cubeZ - 0.18f, gy = cubeY - 0.02f, gh = 0.35f, gw = 0.18f;
    AddQuad(verts, inds, {cubeX - gw, gy, gz}, {cubeX + gw, gy, gz}, {cubeX + gw, gy + gh, gz}, {cubeX - gw, gy + gh, gz}, {0, 0, -1}, DXR10_OBJ_GLASS, DXR10_MAT_GLASS);
    AddQuad(verts, inds, {cubeX + gw, gy, gz}, {cubeX - gw, gy, gz}, {cubeX - gw, gy + gh, gz}, {cubeX + gw, gy + gh, gz}, {0, 0, 1}, DXR10_OBJ_GLASS, DXR10_MAT_GLASS);
    // Purple front wall (behind camera)
    const float fwz = -3.0f, fws = 2.0f;
    AddQuad(verts, inds, {-fws, -fws, fwz}, {fws, -fws, fwz}, {fws, fws, fwz}, {-fws, fws, fwz}, {0, 0, 1}, DXR10_OBJ_FRONT_WALL, DXR10_MAT_DIFFUSE);
}

static void BuildDynamicCube10(std::vector<DXR10Vert>& verts, std::vector<UINT>& inds) {
    verts.clear(); inds.clear();
    const float smallSize = 0.11f, spacing = smallSize;
    int coords[8][3] = {{-1,+1,+1},{+1,+1,+1},{-1,-1,+1},{+1,-1,+1},{-1,+1,-1},{+1,+1,-1},{-1,-1,-1},{+1,-1,-1}};
    for (int c = 0; c < 8; c++) {
        float cx = coords[c][0] * spacing, cy = coords[c][1] * spacing, cz = coords[c][2] * spacing;
        AddBox(verts, inds, {cx, cy, cz}, {smallSize, smallSize, smallSize}, DXR10_OBJ_CUBE, (UINT)c);
    }
}

static void UpdateCubeTransform10(float time) {
    if (!s_instanceMapped) return;
    float angleY = time * 1.2f, angleX = time * 0.7f;
    float cosY = cosf(angleY), sinY = sinf(angleY), cosX = cosf(angleX), sinX = sinf(angleX);
    float m00 = cosY, m01 = sinY * sinX, m02 = sinY * cosX;
    float m10 = 0, m11 = cosX, m12 = -sinX;
    float m20 = -sinY, m21 = cosY * sinX, m22 = cosY * cosX;
    float tx = 0.15f, ty = 0.15f, tz = 0.2f;
    D3D12_RAYTRACING_INSTANCE_DESC* instances = (D3D12_RAYTRACING_INSTANCE_DESC*)s_instanceMapped;
    instances[1].Transform[0][0] = m00; instances[1].Transform[0][1] = m10; instances[1].Transform[0][2] = m20; instances[1].Transform[0][3] = tx;
    instances[1].Transform[1][0] = m01; instances[1].Transform[1][1] = m11; instances[1].Transform[1][2] = m21; instances[1].Transform[1][3] = ty;
    instances[1].Transform[2][0] = m02; instances[1].Transform[2][1] = m12; instances[1].Transform[2][2] = m22; instances[1].Transform[2][3] = tz;
}

static void RebuildTLAS10() {
    if (!s_cmdList || !s_tlas || !s_instanceBuffer) return;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 2;
    tlasInputs.InstanceDescs = s_instanceBuffer->GetGPUVirtualAddress();
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
    tlasBuildDesc.Inputs = tlasInputs;
    tlasBuildDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    tlasBuildDesc.SourceAccelerationStructureData = s_tlas->GetGPUVirtualAddress();
    tlasBuildDesc.DestAccelerationStructureData = s_tlas->GetGPUVirtualAddress();
    tlasBuildDesc.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    s_cmdList->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);
    D3D12_RESOURCE_BARRIER uavBarrier = {};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = s_tlas;
    s_cmdList->ResourceBarrier(1, &uavBarrier);
}

// ============== FEATURE DEFINE BUILDING ==============
// Build array of defines based on current feature flags
static int BuildDXR10Defines(const DXR10Features& f, const wchar_t** defines) {
    int count = 0;
    if (f.spotlight)        defines[count++] = L"FEATURE_SPOTLIGHT";
    if (f.softShadows)      defines[count++] = L"FEATURE_SOFT_SHADOWS";
    if (f.ambientOcclusion) defines[count++] = L"FEATURE_AO";
    if (f.globalIllum)      defines[count++] = L"FEATURE_GI";
    if (f.reflections)      defines[count++] = L"FEATURE_REFLECTIONS";
    if (f.glassRefraction)  defines[count++] = L"FEATURE_GLASS";
    return count;
}

// Recompile shaders with new feature flags
static bool RecompileDXR10Shaders(const DXR10Features& features) {
    if (!s_device) return false;

    Log("[DXR10] Recompiling shaders with features: %s%s%s%s%s%s\n",
        features.spotlight ? "Spot " : "",
        features.softShadows ? "SoftShadow " : "",
        features.ambientOcclusion ? "AO " : "",
        features.globalIllum ? "GI " : "",
        features.reflections ? "Reflect " : "",
        features.glassRefraction ? "Glass " : "");

    // Build defines list
    const wchar_t* featureDefines[10];
    int defineCount = BuildDXR10Defines(features, featureDefines);

    // Build args array: -T lib_6_3 -O3 -D FEATURE_X -D FEATURE_Y ...
    std::vector<const wchar_t*> args;
    args.push_back(L"-T");
    args.push_back(L"lib_6_3");
    args.push_back(L"-O3");
    for (int i = 0; i < defineCount; i++) {
        args.push_back(L"-D");
        args.push_back(featureDefines[i]);
    }

    // Combine shader parts
    std::string shaderCode = std::string(g_dxr10ShaderPart1) + g_dxr10ShaderPart2 + g_dxr10ShaderPart3;

    // Load DXC
    typedef HRESULT(WINAPI* DxcCreateInstanceProc)(REFCLSID, REFIID, LPVOID*);
    HMODULE dxcMod = LoadLibraryW(L"dxcompiler.dll");
    if (!dxcMod) { Log("[DXR10] Cannot load dxcompiler.dll\n"); return false; }
    auto DxcCreate = (DxcCreateInstanceProc)GetProcAddress(dxcMod, "DxcCreateInstance");

    IDxcUtils* utils = nullptr; IDxcCompiler3* compiler = nullptr;
    DxcCreate(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
    DxcCreate(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

    IDxcBlobEncoding* srcBlob = nullptr;
    utils->CreateBlob(shaderCode.c_str(), (UINT)shaderCode.size(), CP_UTF8, &srcBlob);
    DxcBuffer srcBuf = { srcBlob->GetBufferPointer(), srcBlob->GetBufferSize(), CP_UTF8 };

    IDxcResult* result = nullptr;
    compiler->Compile(&srcBuf, args.data(), (UINT)args.size(), nullptr, IID_PPV_ARGS(&result));

    HRESULT status; result->GetStatus(&status);
    if (FAILED(status)) {
        IDxcBlobUtf8* err = nullptr;
        result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&err), nullptr);
        if (err) { Log("[DXR10] Shader error: %s\n", err->GetStringPointer()); err->Release(); }
        srcBlob->Release(); result->Release(); compiler->Release(); utils->Release(); FreeLibrary(dxcMod);
        return false;
    }

    IDxcBlob* shaderBlob = nullptr;
    result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr);
    Log("[DXR10] Shader compiled: %zu bytes\n", shaderBlob->GetBufferSize());

    // Release old PSO
    if (s_rtPSOProps) { s_rtPSOProps->Release(); s_rtPSOProps = nullptr; }
    if (s_rtPSO) { s_rtPSO->Release(); s_rtPSO = nullptr; }

    // Recreate state object with new shader
    D3D12_STATE_SUBOBJECT subobjects[10] = {};
    int subIdx = 0;

    D3D12_DXIL_LIBRARY_DESC libDesc = {};
    libDesc.DXILLibrary.pShaderBytecode = shaderBlob->GetBufferPointer();
    libDesc.DXILLibrary.BytecodeLength = shaderBlob->GetBufferSize();
    subobjects[subIdx].Type = D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY;
    subobjects[subIdx++].pDesc = &libDesc;

    D3D12_HIT_GROUP_DESC hitGroupDesc = {};
    hitGroupDesc.HitGroupExport = L"HitGroup";
    hitGroupDesc.ClosestHitShaderImport = L"ClosestHit";
    hitGroupDesc.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
    subobjects[subIdx].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    subobjects[subIdx++].pDesc = &hitGroupDesc;

    D3D12_HIT_GROUP_DESC shadowHitGroupDesc = {};
    shadowHitGroupDesc.HitGroupExport = L"ShadowHitGroup";
    shadowHitGroupDesc.ClosestHitShaderImport = L"ShadowHit";
    shadowHitGroupDesc.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
    subobjects[subIdx].Type = D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP;
    subobjects[subIdx++].pDesc = &shadowHitGroupDesc;

    D3D12_RAYTRACING_SHADER_CONFIG shaderConfig = {};
    shaderConfig.MaxPayloadSizeInBytes = sizeof(float) * 10 + sizeof(UINT) * 4;
    shaderConfig.MaxAttributeSizeInBytes = sizeof(float) * 2;
    subobjects[subIdx].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG;
    subobjects[subIdx++].pDesc = &shaderConfig;

    subobjects[subIdx].Type = D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE;
    subobjects[subIdx++].pDesc = &s_globalRootSig;

    D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig = {};
    pipelineConfig.MaxTraceRecursionDepth = 1;  // Some GPUs only support depth 1
    subobjects[subIdx].Type = D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG;
    subobjects[subIdx++].pDesc = &pipelineConfig;

    D3D12_STATE_OBJECT_DESC stateDesc = {};
    stateDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    stateDesc.NumSubobjects = subIdx;
    stateDesc.pSubobjects = subobjects;

    HRESULT hr = s_device->CreateStateObject(&stateDesc, IID_PPV_ARGS(&s_rtPSO));
    shaderBlob->Release(); srcBlob->Release(); result->Release(); compiler->Release(); utils->Release(); FreeLibrary(dxcMod);

    if (FAILED(hr)) {
        Log("[DXR10] CreateStateObject failed: 0x%08X\n", hr);
        return false;
    }

    s_rtPSO->QueryInterface(IID_PPV_ARGS(&s_rtPSOProps));

    // Update shader tables with new identifiers
    D3D12_HEAP_PROPERTIES uploadHeap = { D3D12_HEAP_TYPE_UPLOAD };
    UINT shaderIdSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    void* mapped;

    s_rayGenTable->Map(0, nullptr, &mapped);
    memcpy(mapped, s_rtPSOProps->GetShaderIdentifier(L"RayGen"), shaderIdSize);
    s_rayGenTable->Unmap(0, nullptr);

    s_missTable->Map(0, nullptr, &mapped);
    memcpy(mapped, s_rtPSOProps->GetShaderIdentifier(L"Miss"), shaderIdSize);
    memcpy((BYTE*)mapped + s_missRecordSize, s_rtPSOProps->GetShaderIdentifier(L"ShadowMiss"), shaderIdSize);
    s_missTable->Unmap(0, nullptr);

    s_hitGroupTable->Map(0, nullptr, &mapped);
    memcpy(mapped, s_rtPSOProps->GetShaderIdentifier(L"HitGroup"), shaderIdSize);
    memcpy((BYTE*)mapped + s_hitGroupRecordSize, s_rtPSOProps->GetShaderIdentifier(L"ShadowHitGroup"), shaderIdSize);
    s_hitGroupTable->Unmap(0, nullptr);

    s_compiledFeatures = features;
    Log("[DXR10] Shaders recompiled successfully\n");
    return true;
}

// ============== TEXT DRAWING ==============
static void DrawText10(const char* text, float x, float y, float r, float g, float b, float a, float scale) {
    const float charW = 8.0f * scale, charH = 8.0f * scale;
    float startX = x;
    while (*text && s_textVertCount < 5994) {
        char c = *text++;
        if (c == '\n') { y += charH + 2; x = startX; continue; }
        if (c < 32 || c > 127) c = '?';
        int ci = c - 32, row = ci / 16, col = ci % 16;
        float u0 = col / 16.0f, v0 = row / 6.0f, u1 = (col + 1) / 16.0f, v1 = (row + 1) / 6.0f;
        float x0 = x * 2.0f / W - 1.0f, y0 = 1.0f - y * 2.0f / H;
        float x1 = (x + charW) * 2.0f / W - 1.0f, y1 = 1.0f - (y + charH) * 2.0f / H;
        TextVert* v = &s_textVerts[s_textVertCount];
        v[0] = {x0, y0, u0, v0, r, g, b, a}; v[1] = {x1, y0, u1, v0, r, g, b, a}; v[2] = {x0, y1, u0, v1, r, g, b, a};
        v[3] = {x1, y0, u1, v0, r, g, b, a}; v[4] = {x1, y1, u1, v1, r, g, b, a}; v[5] = {x0, y1, u0, v1, r, g, b, a};
        s_textVertCount += 6; x += charW;
    }
}

// Text shader code
static const char* g_textShader10 = R"(
Texture2D fontTex : register(t0);
SamplerState samp : register(s0);
struct VSIn { float2 pos : POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };
PSIn TextVS(VSIn i) { PSIn o; o.pos = float4(i.pos, 0, 1); o.uv = i.uv; o.col = i.col; return o; }
float4 TextPS(PSIn i) : SV_TARGET { return float4(i.col.rgb, i.col.a * fontTex.Sample(samp, i.uv).r); }
)";

// ============== INITIALIZATION ==============
bool InitD3D12DXR10(HWND hwnd) {
    Log("[DXR10] Initializing D3D12 + DXR 1.0...\n");
    HRESULT hr;

    // Note: g_dxr10Features is already set by ShowDxr10SettingsDialog() in main.cpp
    // Do NOT call SetDefaults() here - it would overwrite user's menu selections

    // Create DXGI factory
    IDXGIFactory6* factory = nullptr;
    hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) { Log("[DXR10] CreateDXGIFactory2 failed\n"); return false; }

    // Find DXR-capable adapter
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC1 desc; adapter->GetDesc1(&desc);
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) { adapter->Release(); continue; }
        ID3D12Device5* testDev = nullptr;
        if (SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&testDev)))) {
            s_device = testDev;
            s_gpuName = desc.Description;
            Log("[DXR10] Using GPU: %ls\n", desc.Description);
            break;
        }
        adapter->Release(); adapter = nullptr;
    }
    if (!s_device) { Log("[DXR10] No D3D12 capable GPU found\n"); factory->Release(); return false; }

    // Command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {}; queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    s_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&s_cmdQueue));

    // Swap chain with tearing support
    DXGI_SWAP_CHAIN_DESC1 swapDesc = {};
    swapDesc.Width = W; swapDesc.Height = H; swapDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapDesc.SampleDesc.Count = 1; swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapDesc.BufferCount = 3; swapDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    IDXGISwapChain1* swapChain1 = nullptr;
    factory->CreateSwapChainForHwnd(s_cmdQueue, hwnd, &swapDesc, nullptr, nullptr, &swapChain1);
    factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
    swapChain1->QueryInterface(IID_PPV_ARGS(&s_swapChain));
    swapChain1->Release(); factory->Release(); if (adapter) adapter->Release();
    s_frameIndex = s_swapChain->GetCurrentBackBufferIndex();

    // RTV heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {}; rtvHeapDesc.NumDescriptors = 3; rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    s_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&s_rtvHeap));
    s_rtvDescSize = s_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = s_rtvHeap->GetCPUDescriptorHandleForHeapStart();
    for (UINT i = 0; i < 3; i++) {
        s_swapChain->GetBuffer(i, IID_PPV_ARGS(&s_renderTargets[i]));
        s_device->CreateRenderTargetView(s_renderTargets[i], nullptr, rtvHandle);
        rtvHandle.ptr += s_rtvDescSize;
    }

    // Command allocators and fence
    for (UINT i = 0; i < 3; i++) s_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&s_cmdAlloc[i]));
    s_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&s_fence));
    s_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    // Command list
    ID3D12GraphicsCommandList* baseCmdList = nullptr;
    s_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, s_cmdAlloc[0], nullptr, IID_PPV_ARGS(&baseCmdList));
    baseCmdList->QueryInterface(IID_PPV_ARGS(&s_cmdList));
    baseCmdList->Release();

    D3D12_HEAP_PROPERTIES uploadHeap = { D3D12_HEAP_TYPE_UPLOAD };
    D3D12_HEAP_PROPERTIES defaultHeap = { D3D12_HEAP_TYPE_DEFAULT };
    D3D12_RESOURCE_DESC bufDesc = {}; bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufDesc.Height = 1; bufDesc.DepthOrArraySize = 1; bufDesc.MipLevels = 1; bufDesc.SampleDesc.Count = 1;
    bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    void* mapped;

    // ============== GEOMETRY ==============
    std::vector<DXR10Vert> vertsStatic, vertsCube;
    std::vector<UINT> indsStatic, indsCube;
    BuildCornellBox10(vertsStatic, indsStatic);
    BuildDynamicCube10(vertsCube, indsCube);
    s_vertexCountStatic = (UINT)vertsStatic.size(); s_indexCountStatic = (UINT)indsStatic.size();
    s_vertexCountCube = (UINT)vertsCube.size(); s_indexCountCube = (UINT)indsCube.size();
    Log("[DXR10] Static: %u verts, %u inds | Cube: %u verts, %u inds\n", s_vertexCountStatic, s_indexCountStatic, s_vertexCountCube, s_indexCountCube);

    // Upload geometry
    UINT vbSizeStatic = s_vertexCountStatic * sizeof(DXR10Vert), ibSizeStatic = s_indexCountStatic * sizeof(UINT);
    UINT vbSizeCube = s_vertexCountCube * sizeof(DXR10Vert), ibSizeCube = s_indexCountCube * sizeof(UINT);
    bufDesc.Width = vbSizeStatic; s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_vertexBufferStatic));
    s_vertexBufferStatic->Map(0, nullptr, &mapped); memcpy(mapped, vertsStatic.data(), vbSizeStatic); s_vertexBufferStatic->Unmap(0, nullptr);
    bufDesc.Width = ibSizeStatic; s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_indexBufferStatic));
    s_indexBufferStatic->Map(0, nullptr, &mapped); memcpy(mapped, indsStatic.data(), ibSizeStatic); s_indexBufferStatic->Unmap(0, nullptr);
    bufDesc.Width = vbSizeCube; s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_vertexBufferCube));
    s_vertexBufferCube->Map(0, nullptr, &mapped); memcpy(mapped, vertsCube.data(), vbSizeCube); s_vertexBufferCube->Unmap(0, nullptr);
    bufDesc.Width = ibSizeCube; s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_indexBufferCube));
    s_indexBufferCube->Map(0, nullptr, &mapped); memcpy(mapped, indsCube.data(), ibSizeCube); s_indexBufferCube->Unmap(0, nullptr);

    // Constant buffer
    bufDesc.Width = 256; s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_constantBuffer));
    s_constantBuffer->Map(0, nullptr, &s_cbMapped);

    // ============== BUILD ACCELERATION STRUCTURES ==============
    D3D12_RESOURCE_DESC asDesc = bufDesc; asDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    // BLAS Static
    D3D12_RAYTRACING_GEOMETRY_DESC geomDescStatic = {};
    geomDescStatic.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geomDescStatic.Triangles.VertexBuffer.StartAddress = s_vertexBufferStatic->GetGPUVirtualAddress();
    geomDescStatic.Triangles.VertexBuffer.StrideInBytes = sizeof(DXR10Vert);
    geomDescStatic.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geomDescStatic.Triangles.VertexCount = s_vertexCountStatic;
    geomDescStatic.Triangles.IndexBuffer = s_indexBufferStatic->GetGPUVirtualAddress();
    geomDescStatic.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
    geomDescStatic.Triangles.IndexCount = s_indexCountStatic;
    geomDescStatic.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputsStatic = {};
    blasInputsStatic.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputsStatic.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputsStatic.NumDescs = 1; blasInputsStatic.pGeometryDescs = &geomDescStatic;
    blasInputsStatic.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildStatic = {};
    s_device->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputsStatic, &blasPrebuildStatic);
    asDesc.Width = blasPrebuildStatic.ResultDataMaxSizeInBytes;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_blasStatic));

    // BLAS Cube
    D3D12_RAYTRACING_GEOMETRY_DESC geomDescCube = geomDescStatic;
    geomDescCube.Triangles.VertexBuffer.StartAddress = s_vertexBufferCube->GetGPUVirtualAddress();
    geomDescCube.Triangles.VertexCount = s_vertexCountCube;
    geomDescCube.Triangles.IndexBuffer = s_indexBufferCube->GetGPUVirtualAddress();
    geomDescCube.Triangles.IndexCount = s_indexCountCube;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputsCube = blasInputsStatic;
    blasInputsCube.pGeometryDescs = &geomDescCube;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildCube = {};
    s_device->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputsCube, &blasPrebuildCube);
    asDesc.Width = blasPrebuildCube.ResultDataMaxSizeInBytes;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_blasCube));

    // Scratch buffer
    UINT64 scratchSize = max(blasPrebuildStatic.ScratchDataSizeInBytes, blasPrebuildCube.ScratchDataSizeInBytes);
    scratchSize = max(scratchSize, (UINT64)65536) * 2;
    asDesc.Width = scratchSize;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&s_scratchBuffer));

    // Build BLASes
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuildStatic = {};
    blasBuildStatic.Inputs = blasInputsStatic;
    blasBuildStatic.DestAccelerationStructureData = s_blasStatic->GetGPUVirtualAddress();
    blasBuildStatic.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    s_cmdList->BuildRaytracingAccelerationStructure(&blasBuildStatic, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier = {}; uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = s_blasStatic; s_cmdList->ResourceBarrier(1, &uavBarrier);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC blasBuildCube = {};
    blasBuildCube.Inputs = blasInputsCube;
    blasBuildCube.DestAccelerationStructureData = s_blasCube->GetGPUVirtualAddress();
    blasBuildCube.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    s_cmdList->BuildRaytracingAccelerationStructure(&blasBuildCube, 0, nullptr);
    uavBarrier.UAV.pResource = s_blasCube; s_cmdList->ResourceBarrier(1, &uavBarrier);

    // TLAS instances
    D3D12_RAYTRACING_INSTANCE_DESC instances[2] = {};
    instances[0].Transform[0][0] = instances[0].Transform[1][1] = instances[0].Transform[2][2] = 1.0f;
    instances[0].InstanceMask = 0xFF;
    instances[0].AccelerationStructure = s_blasStatic->GetGPUVirtualAddress();
    instances[1].Transform[0][0] = instances[1].Transform[1][1] = instances[1].Transform[2][2] = 1.0f;
    instances[1].Transform[0][3] = 0.15f; instances[1].Transform[1][3] = 0.15f; instances[1].Transform[2][3] = 0.2f;
    instances[1].InstanceMask = 0xFF;
    instances[1].AccelerationStructure = s_blasCube->GetGPUVirtualAddress();

    bufDesc.Width = sizeof(instances); bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_instanceBuffer));
    s_instanceBuffer->Map(0, nullptr, &s_instanceMapped);
    memcpy(s_instanceMapped, instances, sizeof(instances));

    // Build TLAS
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs = {};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 2;
    tlasInputs.InstanceDescs = s_instanceBuffer->GetGPUVirtualAddress();
    tlasInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPrebuild = {};
    s_device->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasPrebuild);
    asDesc.Width = tlasPrebuild.ResultDataMaxSizeInBytes;
    asDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &asDesc, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nullptr, IID_PPV_ARGS(&s_tlas));

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC tlasBuildDesc = {};
    tlasBuildDesc.Inputs = tlasInputs;
    tlasBuildDesc.DestAccelerationStructureData = s_tlas->GetGPUVirtualAddress();
    tlasBuildDesc.ScratchAccelerationStructureData = s_scratchBuffer->GetGPUVirtualAddress();
    s_cmdList->BuildRaytracingAccelerationStructure(&tlasBuildDesc, 0, nullptr);
    uavBarrier.UAV.pResource = s_tlas; s_cmdList->ResourceBarrier(1, &uavBarrier);

    s_cmdList->Close();
    ID3D12CommandList* lists[] = { s_cmdList };
    s_cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGpu10();
    s_cmdAlloc[0]->Reset();
    s_cmdList->Reset(s_cmdAlloc[0], nullptr);

    Log("[DXR10] Acceleration structures built\n");

    // ============== OUTPUT UAV ==============
    D3D12_RESOURCE_DESC uavDesc = {};
    uavDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    uavDesc.Width = W; uavDesc.Height = H;
    uavDesc.DepthOrArraySize = 1; uavDesc.MipLevels = 1;
    uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavDesc.SampleDesc.Count = 1;
    uavDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &uavDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&s_outputUAV));

    // ============== SRV/UAV HEAP ==============
    // Only need UAV output and TLAS - no vertex/index buffers needed (use primitive ranges)
    D3D12_DESCRIPTOR_HEAP_DESC srvUavHeapDesc = {};
    srvUavHeapDesc.NumDescriptors = 2;  // UAV output, TLAS
    srvUavHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvUavHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    s_device->CreateDescriptorHeap(&srvUavHeapDesc, IID_PPV_ARGS(&s_srvUavHeap));
    UINT descSize = s_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE handle = s_srvUavHeap->GetCPUDescriptorHandleForHeapStart();

    // u0: Output UAV
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavViewDesc = {};
    uavViewDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    uavViewDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    s_device->CreateUnorderedAccessView(s_outputUAV, nullptr, &uavViewDesc, handle);

    // t0: TLAS
    handle.ptr += descSize;
    D3D12_SHADER_RESOURCE_VIEW_DESC tlasSrvDesc = {};
    tlasSrvDesc.Format = DXGI_FORMAT_UNKNOWN;
    tlasSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    tlasSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    tlasSrvDesc.RaytracingAccelerationStructure.Location = s_tlas->GetGPUVirtualAddress();
    s_device->CreateShaderResourceView(nullptr, &tlasSrvDesc, handle);

    // ============== GLOBAL ROOT SIGNATURE ==============
    D3D12_DESCRIPTOR_RANGE uavRange = {}; uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV; uavRange.NumDescriptors = 1;
    D3D12_DESCRIPTOR_RANGE srvRange = {}; srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; srvRange.NumDescriptors = 1;  // Only TLAS
    D3D12_ROOT_PARAMETER rootParams[3] = {};
    rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[0].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[0].DescriptorTable.pDescriptorRanges = &uavRange;
    rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParams[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParams[1].DescriptorTable.pDescriptorRanges = &srvRange;
    rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rootParams[2].Descriptor.ShaderRegister = 0;

    D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
    rsDesc.NumParameters = 3; rsDesc.pParameters = rootParams;
    ID3DBlob* rsBlob = nullptr, *rsErr = nullptr;
    D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr);
    if (rsErr) { Log("[DXR10] Root sig error: %s\n", (char*)rsErr->GetBufferPointer()); rsErr->Release(); }
    s_device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&s_globalRootSig));
    rsBlob->Release();

    // ============== COMPILE RT SHADERS (with feature flags) ==============
    // First, allocate shader table buffers (RecompileDXR10Shaders will fill them)
    UINT shaderIdSize = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
    s_rayGenRecordSize = (shaderIdSize + 255) & ~255;
    s_missRecordSize = (shaderIdSize + 255) & ~255;
    s_hitGroupRecordSize = (shaderIdSize + 255) & ~255;

    bufDesc.Width = s_rayGenRecordSize; bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_rayGenTable));

    bufDesc.Width = s_missRecordSize * 2;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_missTable));

    bufDesc.Width = s_hitGroupRecordSize * 2;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_hitGroupTable));

    // Initial compilation with current features
    if (!RecompileDXR10Shaders(g_dxr10Features)) {
        Log("[DXR10] Initial shader compilation failed\n");
        return false;
    }

    // ============== TEXT RENDERING ==============
    D3D12_STATIC_SAMPLER_DESC sampler = {};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_DESCRIPTOR_RANGE texRange = {}; texRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; texRange.NumDescriptors = 1;
    D3D12_ROOT_PARAMETER textParam = {};
    textParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    textParam.DescriptorTable.NumDescriptorRanges = 1;
    textParam.DescriptorTable.pDescriptorRanges = &texRange;
    textParam.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC textRsDesc = {};
    textRsDesc.NumParameters = 1; textRsDesc.pParameters = &textParam;
    textRsDesc.NumStaticSamplers = 1; textRsDesc.pStaticSamplers = &sampler;
    textRsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

    D3D12SerializeRootSignature(&textRsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rsBlob, &rsErr);
    s_device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&s_textRootSig));
    rsBlob->Release();

    ID3DBlob* textVs = nullptr, *textPs = nullptr;
    D3DCompile(g_textShader10, strlen(g_textShader10), nullptr, nullptr, nullptr, "TextVS", "vs_5_0", 0, 0, &textVs, nullptr);
    D3DCompile(g_textShader10, strlen(g_textShader10), nullptr, nullptr, nullptr, "TextPS", "ps_5_0", 0, 0, &textPs, nullptr);

    D3D12_INPUT_ELEMENT_DESC textLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC textPsoDesc = {};
    textPsoDesc.InputLayout = { textLayout, _countof(textLayout) };
    textPsoDesc.pRootSignature = s_textRootSig;
    textPsoDesc.VS = { textVs->GetBufferPointer(), textVs->GetBufferSize() };
    textPsoDesc.PS = { textPs->GetBufferPointer(), textPs->GetBufferSize() };
    textPsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    textPsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    textPsoDesc.BlendState.RenderTarget[0].BlendEnable = TRUE;
    textPsoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
    textPsoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    textPsoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    textPsoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
    textPsoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    textPsoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    textPsoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    textPsoDesc.SampleMask = UINT_MAX;
    textPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    textPsoDesc.NumRenderTargets = 1;
    textPsoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    textPsoDesc.SampleDesc.Count = 1;
    s_device->CreateGraphicsPipelineState(&textPsoDesc, IID_PPV_ARGS(&s_textPso));
    textVs->Release(); textPs->Release();

    // Font texture
    const int FONT_COLS = 16, FONT_ROWS = 6, TEX_W = FONT_COLS * 8, TEX_H = FONT_ROWS * 8;
    unsigned char texData[TEX_W * TEX_H] = {};
    for (int c = 0; c < 96; c++) {
        int col = c % FONT_COLS, row = c / FONT_COLS;
        for (int y = 0; y < 8; y++) {
            unsigned char bits = g_font8x8[c][y];
            for (int x = 0; x < 8; x++) texData[(row * 8 + y) * TEX_W + col * 8 + x] = (bits & (0x80 >> x)) ? 255 : 0;
        }
    }

    D3D12_RESOURCE_DESC texDesc = {}; texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = TEX_W; texDesc.Height = TEX_H; texDesc.DepthOrArraySize = 1; texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R8_UNORM; texDesc.SampleDesc.Count = 1;
    s_device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&s_fontTexture));

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint; UINT64 uploadSize = 0;
    s_device->GetCopyableFootprints(&texDesc, 0, 1, 0, &footprint, nullptr, nullptr, &uploadSize);
    bufDesc.Width = uploadSize; bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* uploadBuf = nullptr;
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf));
    uploadBuf->Map(0, nullptr, &mapped);
    BYTE* destRow = (BYTE*)mapped + footprint.Offset;
    for (UINT r = 0; r < TEX_H; r++) memcpy(destRow + r * footprint.Footprint.RowPitch, texData + r * TEX_W, TEX_W);
    uploadBuf->Unmap(0, nullptr);

    s_cmdList->Reset(s_cmdAlloc[0], nullptr);
    D3D12_TEXTURE_COPY_LOCATION srcLoc = {}, dstLoc = {};
    srcLoc.pResource = uploadBuf; srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; srcLoc.PlacedFootprint = footprint;
    dstLoc.pResource = s_fontTexture; dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    s_cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

    D3D12_RESOURCE_BARRIER texBarrier = {}; texBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    texBarrier.Transition.pResource = s_fontTexture;
    texBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    texBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    texBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_cmdList->ResourceBarrier(1, &texBarrier);
    s_cmdList->Close();
    s_cmdQueue->ExecuteCommandLists(1, lists);
    WaitForGpu10();
    uploadBuf->Release();

    D3D12_DESCRIPTOR_HEAP_DESC textSrvHeapDesc = {}; textSrvHeapDesc.NumDescriptors = 1;
    textSrvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; textSrvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    s_device->CreateDescriptorHeap(&textSrvHeapDesc, IID_PPV_ARGS(&s_textSrvHeap));
    D3D12_SHADER_RESOURCE_VIEW_DESC texSrvDesc = {}; texSrvDesc.Format = DXGI_FORMAT_R8_UNORM;
    texSrvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    texSrvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    texSrvDesc.Texture2D.MipLevels = 1;
    s_device->CreateShaderResourceView(s_fontTexture, &texSrvDesc, s_textSrvHeap->GetCPUDescriptorHandleForHeapStart());

    bufDesc.Width = 6000 * sizeof(TextVert);
    s_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&s_textVB));
    s_textVB->Map(0, nullptr, &s_textVBMapped);
    s_textVBView.BufferLocation = s_textVB->GetGPUVirtualAddress();
    s_textVBView.SizeInBytes = 6000 * sizeof(TextVert);
    s_textVBView.StrideInBytes = sizeof(TextVert);

    s_cmdAlloc[0]->Reset();
    s_cmdList->Reset(s_cmdAlloc[0], nullptr);
    s_cmdList->Close();

    Log("[DXR10] Initialization complete\n");
    return true;
}

// ============== RENDER ==============
void RenderD3D12DXR10() {
    // Check if features changed and recompile if needed
    if (g_dxr10Features != s_compiledFeatures) {
        WaitForGpu10();  // Wait for GPU before recompiling
        if (!RecompileDXR10Shaders(g_dxr10Features)) {
            Log("[DXR10] WARNING: Shader recompilation failed, reverting features\n");
            g_dxr10Features = s_compiledFeatures;  // Revert to working features
        }
        s_cachedFps = -1;  // Force text rebuild to show new features
    }

    s_cmdAlloc[s_frameIndex]->Reset();
    s_cmdList->Reset(s_cmdAlloc[s_frameIndex], nullptr);

    // Time
    static LARGE_INTEGER startTime = {}, perfFreq = {};
    static UINT frameCount = 0;
    if (startTime.QuadPart == 0) { QueryPerformanceFrequency(&perfFreq); QueryPerformanceCounter(&startTime); }
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    float time = (float)(now.QuadPart - startTime.QuadPart) / perfFreq.QuadPart;
    frameCount++;

    // Update constant buffer with feature parameters
    DXR10CB cb = {};
    cb.time = time;
    cb.lightPos = { 0, 0.92f, 0 };
    cb.lightRadius = g_dxr10Features.lightRadius;
    cb.frameCount = frameCount;
    cb.shadowSamples = g_dxr10Features.shadowSamples;
    cb.aoSamples = g_dxr10Features.aoSamples;
    cb.aoRadius = g_dxr10Features.aoRadius;
    memcpy(s_cbMapped, &cb, sizeof(cb));

    // Update cube transform and rebuild TLAS
    UpdateCubeTransform10(time);
    RebuildTLAS10();

    // Set pipeline state and root signature
    s_cmdList->SetComputeRootSignature(s_globalRootSig);
    ID3D12DescriptorHeap* heaps[] = { s_srvUavHeap };
    s_cmdList->SetDescriptorHeaps(1, heaps);

    UINT descSize = s_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_GPU_DESCRIPTOR_HANDLE gpuHandle = s_srvUavHeap->GetGPUDescriptorHandleForHeapStart();
    s_cmdList->SetComputeRootDescriptorTable(0, gpuHandle);  // UAV
    gpuHandle.ptr += descSize;
    s_cmdList->SetComputeRootDescriptorTable(1, gpuHandle);  // SRVs
    s_cmdList->SetComputeRootConstantBufferView(2, s_constantBuffer->GetGPUVirtualAddress());

    // Dispatch rays
    D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
    dispatchDesc.RayGenerationShaderRecord.StartAddress = s_rayGenTable->GetGPUVirtualAddress();
    dispatchDesc.RayGenerationShaderRecord.SizeInBytes = s_rayGenRecordSize;
    dispatchDesc.MissShaderTable.StartAddress = s_missTable->GetGPUVirtualAddress();
    dispatchDesc.MissShaderTable.SizeInBytes = s_missRecordSize * 2;
    dispatchDesc.MissShaderTable.StrideInBytes = s_missRecordSize;
    dispatchDesc.HitGroupTable.StartAddress = s_hitGroupTable->GetGPUVirtualAddress();
    dispatchDesc.HitGroupTable.SizeInBytes = s_hitGroupRecordSize * 2;
    dispatchDesc.HitGroupTable.StrideInBytes = s_hitGroupRecordSize;
    dispatchDesc.Width = W;
    dispatchDesc.Height = H;
    dispatchDesc.Depth = 1;

    s_cmdList->SetPipelineState1(s_rtPSO);
    s_cmdList->DispatchRays(&dispatchDesc);

    // Copy output to backbuffer
    D3D12_RESOURCE_BARRIER barriers[2] = {};
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = s_outputUAV;
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = s_renderTargets[s_frameIndex];
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    s_cmdList->ResourceBarrier(2, barriers);

    s_cmdList->CopyResource(s_renderTargets[s_frameIndex], s_outputUAV);

    // Transition for text rendering
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
    s_cmdList->ResourceBarrier(2, barriers);

    // Text rendering
    if (fps != s_cachedFps) {
        s_cachedFps = fps;
        s_textVertCount = 0;
        char gpuNameA[128] = {}; size_t converted = 0;
        wcstombs_s(&converted, gpuNameA, 128, s_gpuName.c_str(), 127);
        char buf[256]; float y = 10.0f;
        sprintf_s(buf, "API: Direct3D 12 + DXR 1.0 (TraceRay)");
        DrawText10(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawText10(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;
        sprintf_s(buf, "GPU: %s", gpuNameA);
        DrawText10(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawText10(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;
        sprintf_s(buf, "FPS: %d", fps);
        DrawText10(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawText10(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;
        sprintf_s(buf, "Triangles: %u", (s_indexCountStatic + s_indexCountCube) / 3);
        DrawText10(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawText10(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;
        sprintf_s(buf, "Resolution: %dx%d", W, H);
        DrawText10(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawText10(buf, 10, y, 1, 1, 1, 1, 1.5f); y += 15;

        // Show active features
        y += 5;  // Small gap
        sprintf_s(buf, "Features: %s%s%s%s%s%s",
            g_dxr10Features.spotlight ? "Spot " : "",
            g_dxr10Features.softShadows ? "Shadow " : "",
            g_dxr10Features.ambientOcclusion ? "AO " : "",
            g_dxr10Features.globalIllum ? "GI " : "",
            g_dxr10Features.reflections ? "Refl " : "",
            g_dxr10Features.glassRefraction ? "Glass" : "");
        DrawText10(buf, 11, y+1, 0, 0, 0, 1, 1.5f); DrawText10(buf, 10, y, 0.7f, 1.0f, 0.7f, 1, 1.5f);
    }

    if (s_textVertCount > 0) {
        memcpy(s_textVBMapped, s_textVerts, s_textVertCount * sizeof(TextVert));
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = s_rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rtvHandle.ptr += s_frameIndex * s_rtvDescSize;
        s_cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        D3D12_VIEWPORT vp = { 0, 0, (float)W, (float)H, 0, 1 };
        D3D12_RECT scissor = { 0, 0, (LONG)W, (LONG)H };
        s_cmdList->RSSetViewports(1, &vp);
        s_cmdList->RSSetScissorRects(1, &scissor);
        s_cmdList->SetGraphicsRootSignature(s_textRootSig);
        ID3D12DescriptorHeap* textHeaps[] = { s_textSrvHeap };
        s_cmdList->SetDescriptorHeaps(1, textHeaps);
        s_cmdList->SetGraphicsRootDescriptorTable(0, s_textSrvHeap->GetGPUDescriptorHandleForHeapStart());
        s_cmdList->SetPipelineState(s_textPso);
        s_cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        s_cmdList->IASetVertexBuffers(0, 1, &s_textVBView);
        s_cmdList->DrawInstanced(s_textVertCount, 1, 0, 0);
    }

    // Present
    barriers[0].Transition.pResource = s_renderTargets[s_frameIndex];
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    s_cmdList->ResourceBarrier(1, barriers);

    s_cmdList->Close();
    ID3D12CommandList* lists[] = { s_cmdList };
    s_cmdQueue->ExecuteCommandLists(1, lists);
    s_swapChain->Present(0, DXGI_PRESENT_ALLOW_TEARING);
    MoveToNextFrame10();
}

// ============== CLEANUP ==============
void CleanupD3D12DXR10() {
    WaitForGpu10();
    #define SAFE_RELEASE(x) if(x) { x->Release(); x = nullptr; }
    SAFE_RELEASE(s_rayGenTable); SAFE_RELEASE(s_missTable); SAFE_RELEASE(s_hitGroupTable);
    SAFE_RELEASE(s_rtPSOProps); SAFE_RELEASE(s_rtPSO); SAFE_RELEASE(s_globalRootSig);
    SAFE_RELEASE(s_outputUAV); SAFE_RELEASE(s_srvUavHeap);
    SAFE_RELEASE(s_blasStatic); SAFE_RELEASE(s_blasCube); SAFE_RELEASE(s_tlas);
    SAFE_RELEASE(s_scratchBuffer); SAFE_RELEASE(s_instanceBuffer);
    SAFE_RELEASE(s_vertexBufferStatic); SAFE_RELEASE(s_indexBufferStatic);
    SAFE_RELEASE(s_vertexBufferCube); SAFE_RELEASE(s_indexBufferCube);
    SAFE_RELEASE(s_constantBuffer);
    SAFE_RELEASE(s_textRootSig); SAFE_RELEASE(s_textPso); SAFE_RELEASE(s_textSrvHeap);
    SAFE_RELEASE(s_fontTexture); SAFE_RELEASE(s_textVB);
    SAFE_RELEASE(s_fence); if (s_fenceEvent) { CloseHandle(s_fenceEvent); s_fenceEvent = nullptr; }
    for (UINT i = 0; i < 3; i++) { SAFE_RELEASE(s_cmdAlloc[i]); SAFE_RELEASE(s_renderTargets[i]); }
    SAFE_RELEASE(s_cmdList); SAFE_RELEASE(s_rtvHeap); SAFE_RELEASE(s_swapChain);
    SAFE_RELEASE(s_cmdQueue); SAFE_RELEASE(s_device);
    #undef SAFE_RELEASE
    Log("[DXR10] Cleanup complete\n");
}
