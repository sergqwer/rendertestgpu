#pragma once
// ============== D3D12 DXR RAY TRACING SHADERS ==============
// Cornell Box scene with materials: diffuse, mirror, glass, emissive
// Shader Model 6.5 with inline ray tracing (RayQuery) support

static const char* g_rtShaderCode = R"HLSL(
// ============== CONSTANT BUFFER ==============
cbuffer CB : register(b0) {
    float Time;
    // Feature flags (0 or 1)
    uint rtShadows;
    uint rtSoftShadows;
    uint rtReflections;
    uint rtAO;
    uint rtGI;
    // Parameters
    uint softShadowSamples;
    float shadowSoftness;
    float reflectionStrength;
    float roughness;
    uint aoSamples;
    float aoRadius;
    float aoStrength;
    uint giBounces;
    float giStrength;
    float _pad;
};

// ============== MATERIAL TYPES ==============
#define MAT_DIFFUSE  0
#define MAT_MIRROR   1
#define MAT_GLASS    2
#define MAT_EMISSIVE 3
#define MAT_GLOSSY   4

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
#define OBJ_CUBE_BEHIND 9

// ============== RESOURCES ==============
RaytracingAccelerationStructure Scene : register(t0);

struct VertexData {
    float3 pos;
    float3 norm;
    uint objectID;
    uint materialType;
};
StructuredBuffer<VertexData> VertexBuffer : register(t1);
StructuredBuffer<uint> IndexBuffer : register(t2);

// ============== CONSTANTS ==============
// Camera at z=-4 looking toward +Z (same as base D3D12 renderer)
static const float3 CameraPos = float3(0, 0, -4.0f);
// View/Proj matrices - same as working d3d11_shaders.h
static const matrix View = { 1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,4,1 };
static const matrix Proj = { 1.81066f,0,0,0, 0,2.41421f,0,0, 0,0,1.001f,1, 0,0,-0.1001f,0 };

// Light properties (area light on ceiling)
static const float3 LightPos = float3(0, 0.99f, 0);
static const float3 LightColor = float3(1.0f, 0.95f, 0.8f);
static const float LightIntensity = 2.5f;
static const float LightRadius = 0.3f;  // Area light size

// Object colors
static const float3 Colors[10] = {
    float3(0.9f, 0.9f, 0.9f),   // 0: Floor - white
    float3(0.9f, 0.9f, 0.9f),   // 1: Ceiling - white
    float3(0.9f, 0.9f, 0.9f),   // 2: Back wall - white
    float3(0.85f, 0.1f, 0.1f),  // 3: Left wall - RED
    float3(0.1f, 0.85f, 0.1f),  // 4: Right wall - GREEN
    float3(15.0f, 14.0f, 12.0f),// 5: Light - emissive (bright!)
    float3(0.9f, 0.6f, 0.2f),   // 6: Central cube - orange
    float3(0.95f, 0.95f, 0.95f),// 7: Mirror - neutral
    float3(0.95f, 0.98f, 1.0f), // 8: Glass - slight blue tint
    float3(0.2f, 0.5f, 0.9f)    // 9: Cube behind glass - blue
};

// Cube colors (8 rotating cubes)
static const float3 CubeColors[8] = {
    float3(1.0f, 0.15f, 0.1f),  // 0: Bright Red
    float3(0.1f, 0.9f, 0.2f),   // 1: Bright Green
    float3(0.1f, 0.4f, 1.0f),   // 2: Bright Blue
    float3(1.0f, 0.95f, 0.1f),  // 3: Bright Yellow
    float3(1.0f, 0.95f, 0.1f),  // 4: Bright Yellow
    float3(0.1f, 0.4f, 1.0f),   // 5: Bright Blue
    float3(0.1f, 0.9f, 0.2f),   // 6: Bright Green
    float3(1.0f, 0.15f, 0.1f)   // 7: Bright Red
};

// Get color for object (handles cube colors separately)
float3 GetObjectColor(uint objID, uint cubeIdx) {
    if (objID == OBJ_CUBE) {
        return CubeColors[min(cubeIdx, 7u)];
    }
    return Colors[min(objID, 9u)];
}

// ============== STRUCTURES ==============
struct VSIn {
    float3 pos : POSITION;
    float3 norm : NORMAL;
    uint objectID : OBJECTID;
    uint materialType : MATERIALTYPE;
};

struct PSIn {
    float4 pos : SV_POSITION;
    float3 worldPos : WORLDPOS;
    float3 worldNorm : NORMAL;
    float3 color : COLOR;
    uint objectID : OBJECTID;
    uint materialType : MATERIALTYPE;
};

// ============== UTILITY FUNCTIONS ==============
float3x3 RotY(float a) { float c=cos(a),s=sin(a); return float3x3(c,0,s, 0,1,0, -s,0,c); }
float3x3 RotX(float a) { float c=cos(a),s=sin(a); return float3x3(1,0,0, 0,c,-s, 0,s,c); }

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
    return float(seed) / 4294967295.0f;
}

float3 RandomHemisphere(float3 normal, inout uint seed) {
    float u1 = Random(seed);
    float u2 = Random(seed);
    float r = sqrt(1.0f - u1 * u1);
    float phi = 6.28318530718f * u2;
    float3 dir = float3(cos(phi) * r, sin(phi) * r, u1);
    float3 up = abs(normal.y) < 0.999f ? float3(0,1,0) : float3(1,0,0);
    float3 tangent = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return normalize(tangent * dir.x + bitangent * dir.y + normal * dir.z);
}

float3 RandomInDisk(inout uint seed) {
    float u1 = Random(seed);
    float u2 = Random(seed);
    float r = sqrt(u1);
    float theta = 6.28318530718f * u2;
    return float3(r * cos(theta), 0, r * sin(theta));
}

// ============== RAY TRACING FUNCTIONS ==============

// Get material info from hit
void GetHitInfo(uint primIdx, float2 bary, out float3 hitPos, out float3 hitNorm,
                out float3 hitColor, out uint hitMat, out uint cubeIdx) {
    uint i0 = IndexBuffer[primIdx * 3 + 0];
    uint i1 = IndexBuffer[primIdx * 3 + 1];
    uint i2 = IndexBuffer[primIdx * 3 + 2];

    VertexData v0 = VertexBuffer[i0];
    VertexData v1 = VertexBuffer[i1];
    VertexData v2 = VertexBuffer[i2];

    float w = 1 - bary.x - bary.y;
    hitPos = v0.pos * w + v1.pos * bary.x + v2.pos * bary.y;
    hitNorm = normalize(v0.norm * w + v1.norm * bary.x + v2.norm * bary.y);
    hitMat = v0.materialType;

    // Calculate cube index for rotating cubes (each cube has 12 triangles)
    cubeIdx = (v0.objectID == OBJ_CUBE) ? (primIdx / 12) : 0;
    hitColor = GetObjectColor(v0.objectID, cubeIdx);
}

// Shadow ray with optional soft shadows (area light)
float TraceShadow(float3 origin, float3 lightPos, float lightRadius, uint samples, inout uint seed) {
    float shadow = 0;
    uint actualSamples = (rtSoftShadows != 0 && samples > 1) ? samples : 1;

    for (uint i = 0; i < actualSamples; i++) {
        float3 targetPos = lightPos;
        if (rtSoftShadows != 0 && lightRadius > 0) {
            float3 jitter = RandomInDisk(seed) * lightRadius;
            targetPos += jitter;
        }

        float3 dir = targetPos - origin;
        float dist = length(dir);
        dir /= dist;

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> q;
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = dir;
        ray.TMin = 0.001f;
        ray.TMax = dist - 0.001f;

        q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_NOTHING) {
            shadow += 1.0f;
        } else {
            // Check if we hit glass - glass is semi-transparent for shadows
            uint hitPrim = q.CommittedPrimitiveIndex();
            uint i0 = IndexBuffer[hitPrim * 3];
            if (VertexBuffer[i0].materialType == MAT_GLASS) {
                shadow += 0.7f;  // Glass transmits some light
            }
        }
    }
    return shadow / float(actualSamples);
}

// Trace reflection ray
float3 TraceReflection(float3 origin, float3 dir, uint depth, inout uint seed) {
    if (depth > 2) return float3(0,0,0);

    RayQuery<RAY_FLAG_NONE> q;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.001f;
    ray.TMax = 100.0f;

    q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float3 hitPos, hitNorm, hitColor;
        uint hitMat, cubeIdx;
        GetHitInfo(q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(),
                   hitPos, hitNorm, hitColor, hitMat, cubeIdx);

        // Emissive surfaces
        if (hitMat == MAT_EMISSIVE) {
            return hitColor;
        }

        // Calculate lighting at hit point
        float3 toLight = LightPos - hitPos;
        float lightDist = length(toLight);
        toLight /= lightDist;
        float NdotL = max(dot(hitNorm, toLight), 0);
        float shadow = TraceShadow(hitPos + hitNorm * 0.001f, LightPos, LightRadius,
                                   rtSoftShadows ? softShadowSamples : 1, seed);
        float attenuation = 1.0f / (1.0f + lightDist * lightDist * 0.1f);

        return hitColor * (NdotL * shadow * attenuation * LightIntensity * 0.3f + 0.1f);
    }

    return float3(0.02f, 0.02f, 0.03f);  // Dark ambient
}

// Trace refraction ray (glass)
float3 TraceRefraction(float3 origin, float3 dir, float3 normal, inout uint seed) {
    float eta = 1.0f / 1.5f;  // Air to glass
    float3 refracted = refract(dir, normal, eta);
    if (length(refracted) < 0.001f) {
        // Total internal reflection
        return TraceReflection(origin, reflect(dir, normal), 1, seed);
    }

    RayQuery<RAY_FLAG_NONE> q;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = refracted;
    ray.TMin = 0.001f;
    ray.TMax = 100.0f;

    q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        float3 hitPos, hitNorm, hitColor;
        uint hitMat, cubeIdx;
        GetHitInfo(q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(),
                   hitPos, hitNorm, hitColor, hitMat, cubeIdx);

        if (hitMat == MAT_EMISSIVE) return hitColor;

        // Simple lighting
        float3 toLight = LightPos - hitPos;
        float NdotL = max(dot(hitNorm, normalize(toLight)), 0);
        return hitColor * (NdotL * 0.5f + 0.2f);
    }

    return float3(0.02f, 0.02f, 0.03f);
}

// Ambient occlusion
float TraceAO(float3 origin, float3 normal, float radius, uint samples, inout uint seed) {
    float occlusion = 0;
    for (uint i = 0; i < samples; i++) {
        float3 sampleDir = RandomHemisphere(normal, seed);

        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = sampleDir;
        ray.TMin = 0.001f;
        ray.TMax = radius;

        q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float t = q.CommittedRayT() / radius;
            occlusion += 1.0f - t;
        }
    }
    return 1.0f - (occlusion / float(samples));
}

// Global illumination
float3 TraceGI(float3 origin, float3 normal, float3 albedo, uint bounces, inout uint seed) {
    float3 indirect = float3(0,0,0);
    float3 throughput = float3(1,1,1);
    float3 rayOrigin = origin;
    float3 rayDir = RandomHemisphere(normal, seed);

    for (uint b = 0; b < bounces; b++) {
        RayQuery<RAY_FLAG_NONE> q;
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin = 0.001f;
        ray.TMax = 100.0f;

        q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            float3 hitPos, hitNorm, hitColor;
            uint hitMat, cubeIdx;
            GetHitInfo(q.CommittedPrimitiveIndex(), q.CommittedTriangleBarycentrics(),
                       hitPos, hitNorm, hitColor, hitMat, cubeIdx);

            if (hitMat == MAT_EMISSIVE) {
                indirect += throughput * hitColor * 0.1f;
                break;
            }

            // Direct light at hit point
            float3 toLight = LightPos - hitPos;
            float NdotL = max(dot(hitNorm, normalize(toLight)), 0);
            float shadow = TraceShadow(hitPos + hitNorm * 0.001f, LightPos, 0, 1, seed);
            indirect += throughput * hitColor * NdotL * shadow * 0.5f;

            throughput *= hitColor;
            rayOrigin = hitPos + hitNorm * 0.001f;
            rayDir = RandomHemisphere(hitNorm, seed);
        } else {
            break;
        }
    }

    return indirect * albedo;
}

// ============== VERTEX SHADER ==============
PSIn RTVS(VSIn i) {
    PSIn o;

    // ULTRA-SIMPLE TEST: No rotation, just pass through with View/Proj
    // Cornell Box is centered at origin, camera at z=-4 looking at +Z
    float3 worldPos = i.pos;
    float3 worldNorm = i.norm;

    o.worldPos = worldPos;
    o.worldNorm = worldNorm;
    o.pos = mul(mul(float4(worldPos, 1), View), Proj);

    // DEBUG: Color based on position to see if geometry is correct
    // X: red (-1 to 1), Y: green (-1 to 1), Z: blue (-1 to 1)
    o.color = float3(
        (worldPos.x + 1.0f) * 0.5f,  // 0 to 1 (red)
        (worldPos.y + 1.0f) * 0.5f,  // 0 to 1 (green)
        (worldPos.z + 1.0f) * 0.5f   // 0 to 1 (blue)
    );

    o.objectID = i.objectID;
    o.materialType = i.materialType;

    return o;
}

// ============== PIXEL SHADER ==============
float4 RTPS(PSIn i) : SV_TARGET {
    // ULTRA-SIMPLE TEST: Just output the position-based color
    // This bypasses all RT features to test basic rasterization
    return float4(i.color, 1.0f);
}
)HLSL";
