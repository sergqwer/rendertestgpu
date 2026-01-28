#pragma once
// ============== VULKAN RAY TRACING SHADERS ==============
// GLSL source code for ray tracing shaders
// Compiled to SPIR-V at runtime using shaderc or pre-compiled

#include <cstdint>

// ============== GLSL RAY GENERATION SHADER ==============
static const char* g_rtRayGenGLSL = R"GLSL(
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadEXT vec3 hitValue;
layout(location = 1) rayPayloadEXT float hitT;
layout(location = 2) rayPayloadEXT vec3 hitNormal;
layout(location = 3) rayPayloadEXT vec3 hitPos;
layout(location = 4) rayPayloadEXT uint objectID;
layout(location = 5) rayPayloadEXT uint materialType;
layout(location = 6) rayPayloadEXT bool didHit;
layout(location = 7) rayPayloadEXT uint cubeIdx;

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, rgba8) uniform image2D outputImage;
layout(set = 0, binding = 2) uniform SceneCB {
    float time;
    float lightPosX;
    float lightPosY;
    float lightPosZ;
    float lightRadius;
    uint frameCount;
    int shadowSamples;
    int aoSamples;
    float aoRadius;
    uint features;  // Bit flags for enabled features
} scene;

// Helper to get lightPos as vec3
#define LIGHT_POS vec3(LIGHT_POSX, LIGHT_POSY, LIGHT_POSZ)

// Feature flags (match VulkanRTFeatures)
#define FEATURE_SPOTLIGHT       (1u << 0)
#define FEATURE_SOFT_SHADOWS    (1u << 1)
#define FEATURE_AO              (1u << 2)
#define FEATURE_GI              (1u << 3)
#define FEATURE_REFLECTIONS     (1u << 4)
#define FEATURE_GLASS           (1u << 5)

// Hardcoded camera (same as DXR 1.0)
const vec3 CameraPos = vec3(0.0, 0.0, -2.2);

// Object IDs
#define OBJ_FLOOR      0u
#define OBJ_CEILING    1u
#define OBJ_BACK_WALL  2u
#define OBJ_LEFT_WALL  3u
#define OBJ_RIGHT_WALL 4u
#define OBJ_LIGHT      5u
#define OBJ_CUBE       6u
#define OBJ_MIRROR     7u
#define OBJ_GLASS      8u
#define OBJ_SMALL_CUBE 9u
#define OBJ_FRONT_WALL 10u

// Material types
#define MAT_DIFFUSE  0u
#define MAT_MIRROR   1u
#define MAT_GLASS    2u
#define MAT_EMISSIVE 3u

// Scene colors (match DXR 1.0)
const vec3 Colors[11] = vec3[](
    vec3(0.7, 0.7, 0.7),    // Floor
    vec3(0.9, 0.9, 0.9),    // Ceiling
    vec3(0.7, 0.7, 0.7),    // Back wall
    vec3(0.75, 0.15, 0.15), // Left wall (RED)
    vec3(0.15, 0.75, 0.15), // Right wall (GREEN)
    vec3(15.0, 14.0, 12.0), // Light
    vec3(0.9, 0.6, 0.2),    // Cube (fallback)
    vec3(0.95, 0.95, 0.95), // Mirror
    vec3(0.9, 0.95, 1.0),   // Glass
    vec3(0.9, 0.15, 0.1),   // Small cube
    vec3(0.5, 0.15, 0.7)    // Front wall (purple)
);

// Cube colors
const vec3 CubeColors[8] = vec3[](
    vec3(1.0, 0.15, 0.1),  // Red
    vec3(0.1, 0.9, 0.2),   // Green
    vec3(0.1, 0.4, 1.0),   // Blue
    vec3(1.0, 0.95, 0.1),  // Yellow
    vec3(1.0, 0.95, 0.1),  // Yellow
    vec3(0.1, 0.4, 1.0),   // Blue
    vec3(0.1, 0.9, 0.2),   // Green
    vec3(1.0, 0.15, 0.1)   // Red
);

// Spotlight parameters
const vec3 SpotlightDir = normalize(vec3(0.0, -1.0, 0.15));
const float SpotInnerCos = 0.85;
const float SpotOuterCos = 0.5;

// Wang hash RNG
uint WangHash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

float Random(inout uint seed) {
    seed = WangHash(seed);
    return float(seed) / 4294967295.0;
}

vec3 RandomInDisk(inout uint seed) {
    float r = sqrt(Random(seed));
    float theta = 6.28318530718 * Random(seed);
    return vec3(r * cos(theta), 0.0, r * sin(theta));
}

vec3 RandomInHemisphere(vec3 normal, inout uint seed) {
    float u1 = Random(seed);
    float u2 = Random(seed);
    float r = sqrt(u1);
    float theta = 6.28318530718 * u2;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - u1);
    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

float SpotlightAttenuation(vec3 lightToPoint) {
    vec3 L = normalize(lightToPoint);
    float cosAngle = dot(L, SpotlightDir);
    return clamp((cosAngle - SpotOuterCos) / (SpotInnerCos - SpotOuterCos), 0.0, 1.0);
}

vec3 GetObjectColor(uint objID, uint cubeIdx) {
    if (objID == OBJ_CUBE) {
        return CubeColors[min(cubeIdx, 7u)];
    }
    return Colors[min(objID, 10u)];
}

void main() {
    ivec2 launchIndex = ivec2(gl_LaunchIDEXT.xy);
    ivec2 launchDim = ivec2(gl_LaunchSizeEXT.xy);

    // Generate ray from camera
    vec2 uv = (vec2(launchIndex) + 0.5) / vec2(launchDim);
    vec2 ndc = uv * 2.0 - 1.0;
    ndc.y = -ndc.y;

    float aspectRatio = float(launchDim.x) / float(launchDim.y);
    float tanHalfFovY = 1.0 / 1.73;
    float tanHalfFovX = tanHalfFovY * aspectRatio;

    vec3 rayDir = normalize(vec3(ndc.x * tanHalfFovX, ndc.y * tanHalfFovY, 1.0));
    vec3 rayOrigin = CameraPos;

    // Initialize payload
    hitValue = vec3(0.0);
    hitT = -1.0;
    didHit = false;
    objectID = 0u;
    materialType = 0u;
    cubeIdx = 0u;

    // Trace primary ray
    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, 0, 1, 0, rayOrigin, 0.001, rayDir, 1000.0, 0);

    vec3 finalColor = vec3(0.05, 0.05, 0.08);  // Background

    if (didHit) {
        vec3 baseColor = GetObjectColor(objectID, cubeIdx);

        // Emissive light
        if (objectID == OBJ_LIGHT) {
            finalColor = vec3(1.0, 0.98, 0.9);
        }
        // Mirror reflection
        else if ((scene.features & FEATURE_REFLECTIONS) != 0u && materialType == MAT_MIRROR) {
            vec3 reflectDir = reflect(rayDir, hitNormal);
            hitValue = vec3(0.0);
            didHit = false;
            traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, 0, 1, 0, hitPos + hitNormal * 0.002, 0.001, reflectDir, 100.0, 0);

            if (didHit) {
                vec3 reflColor = GetObjectColor(objectID, 0u);
                if (objectID == OBJ_LIGHT) {
                    reflColor = vec3(1.0, 0.98, 0.9);
                } else {
                    vec3 toLight = normalize(LIGHT_POS - hitPos);
                    float NdotL = max(dot(hitNormal, toLight), 0.0);
                    if ((scene.features & FEATURE_SPOTLIGHT) != 0u) {
                        float reflSpot = SpotlightAttenuation(hitPos - LIGHT_POS);
                        reflColor *= (0.15 + NdotL * reflSpot * 0.85);
                    } else {
                        reflColor *= (0.25 + NdotL * 0.75);
                    }
                }
                finalColor = mix(baseColor * 0.1, reflColor, 0.9);
            } else {
                finalColor = baseColor * 0.3;
            }
        }
        // Glass transparency
        else if ((scene.features & FEATURE_GLASS) != 0u && materialType == MAT_GLASS) {
            vec3 throughOrigin = hitPos + rayDir * 0.01;
            hitValue = vec3(0.0);
            didHit = false;
            traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, 0, 1, 0, throughOrigin, 0.001, rayDir, 100.0, 0);

            vec3 behindColor = vec3(0.05, 0.05, 0.08);
            if (didHit) {
                behindColor = GetObjectColor(objectID, 0u);
                if (objectID != OBJ_LIGHT) {
                    vec3 toLight = normalize(LIGHT_POS - hitPos);
                    float NdotL = max(dot(hitNormal, toLight), 0.0);
                    if ((scene.features & FEATURE_SPOTLIGHT) != 0u) {
                        float glassSpot = SpotlightAttenuation(hitPos - LIGHT_POS);
                        behindColor *= (0.2 + NdotL * glassSpot * 0.8);
                    } else {
                        behindColor *= (0.3 + NdotL * 0.7);
                    }
                }
            }
            float fresnel = pow(1.0 - abs(dot(-rayDir, hitNormal)), 3.0);
            vec3 glassTint = vec3(0.95, 0.97, 1.0);
            finalColor = behindColor * glassTint * (1.0 - fresnel * 0.3);
        }
        // Diffuse surfaces
        else {
            uint seed = uint(launchIndex.x) + uint(launchIndex.y) * 1920u + scene.frameCount * 1920u * 1080u;

            vec3 toLight = normalize(LIGHT_POS - hitPos);
            float NdotL = max(dot(hitNormal, toLight), 0.0);
            float lightDist = length(LIGHT_POS - hitPos);

            // Shadow ray - trace towards light to check occlusion
            float shadow = 1.0;
            vec3 shadowOrigin = hitPos + hitNormal * 0.002;  // Bias to avoid self-intersection

            // Simple shadow - single ray to light center
            didHit = false;
            traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                        0xFF, 0, 1, 0, shadowOrigin, 0.001, toLight, lightDist - 0.01, 0);
            if (didHit) {
                shadow = 0.15;  // In shadow - use ambient only
            }

            // Soft shadows (multiple samples)
            if ((scene.features & FEATURE_SOFT_SHADOWS) != 0u && scene.shadowSamples > 1) {
                float softShadow = 0.0;
                for (int i = 0; i < scene.shadowSamples; i++) {
                    vec3 jitter = RandomInDisk(seed) * scene.lightRadius;
                    vec3 lightSamplePos = LIGHT_POS + jitter;
                    vec3 toLightSample = normalize(lightSamplePos - hitPos);
                    float sampleDist = length(lightSamplePos - hitPos);

                    didHit = false;
                    traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                                0xFF, 0, 1, 0, shadowOrigin, 0.001, toLightSample, sampleDist - 0.01, 0);
                    if (!didHit) {
                        softShadow += 1.0;
                    }
                }
                shadow = max(0.15, softShadow / float(scene.shadowSamples));
            }

            // Ambient occlusion
            float ao = 1.0;
            if ((scene.features & FEATURE_AO) != 0u && scene.aoSamples > 0) {
                float occlusion = 0.0;
                for (int i = 0; i < scene.aoSamples; i++) {
                    vec3 aoDir = RandomInHemisphere(hitNormal, seed);
                    didHit = false;
                    traceRayEXT(topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                                0xFF, 0, 1, 0, shadowOrigin, 0.001, aoDir, scene.aoRadius, 0);
                    if (didHit) {
                        occlusion += 1.0;
                    }
                }
                ao = 1.0 - (occlusion / float(scene.aoSamples)) * 0.7;
            }

            // Final lighting
            float distAtten = 2.5 / (1.0 + lightDist * lightDist * 0.08);
            float totalAtten = distAtten;

            if ((scene.features & FEATURE_SPOTLIGHT) != 0u) {
                float spotAtten = SpotlightAttenuation(hitPos - LIGHT_POS);
                totalAtten *= spotAtten;
            }

            vec3 ambient = baseColor * 0.15 * ao;
            vec3 diffuse = baseColor * NdotL * shadow * totalAtten * ao;
            finalColor = ambient + diffuse;

            // Tone mapping
            finalColor = finalColor / (finalColor + 1.0);
        }
    }

    // Gamma correction
    finalColor = pow(max(finalColor, vec3(0.0)), vec3(1.0 / 2.2));

    imageStore(outputImage, launchIndex, vec4(finalColor, 1.0));
}
)GLSL";

// ============== GLSL CLOSEST HIT SHADER ==============
static const char* g_rtClosestHitGLSL = R"GLSL(
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadInEXT float hitT;
layout(location = 2) rayPayloadInEXT vec3 hitNormal;
layout(location = 3) rayPayloadInEXT vec3 hitPos;
layout(location = 4) rayPayloadInEXT uint objectID;
layout(location = 5) rayPayloadInEXT uint materialType;
layout(location = 6) rayPayloadInEXT bool didHit;
layout(location = 7) rayPayloadInEXT uint cubeIdx;

hitAttributeEXT vec2 attribs;

layout(set = 0, binding = 2) uniform SceneCB {
    float time;
    float lightPosX;
    float lightPosY;
    float lightPosZ;
    float lightRadius;
    uint frameCount;
    int shadowSamples;
    int aoSamples;
    float aoRadius;
    uint features;
} scene;

// Static geometry primitive ranges (same as DXR 1.0)
// Order: floor(2), ceiling(2), back_wall(2), left_wall(2), right_wall(2),
//        light(2), mirror(2), small_cube(12), glass(4), front_wall(2)

#define MAT_DIFFUSE  0u
#define MAT_MIRROR   1u
#define MAT_GLASS    2u
#define MAT_EMISSIVE 3u

mat3 RotateY(float angle) {
    float c = cos(angle), s = sin(angle);
    return mat3(c, 0, s, 0, 1, 0, -s, 0, c);
}

mat3 RotateX(float angle) {
    float c = cos(angle), s = sin(angle);
    return mat3(1, 0, 0, 0, c, -s, 0, s, c);
}

void GetStaticObjectInfo(uint primID, out uint objID, out uint matType, out vec3 normal) {
    matType = MAT_DIFFUSE;
    if (primID < 2u) { objID = 0u; normal = vec3(0, 1, 0); }
    else if (primID < 4u) { objID = 1u; normal = vec3(0, -1, 0); }
    else if (primID < 6u) { objID = 2u; normal = vec3(0, 0, -1); }
    else if (primID < 8u) { objID = 3u; normal = vec3(1, 0, 0); }
    else if (primID < 10u) { objID = 4u; normal = vec3(-1, 0, 0); }
    else if (primID < 12u) { objID = 5u; normal = vec3(0, -1, 0); matType = MAT_EMISSIVE; }
    else if (primID < 14u) { objID = 7u; normal = normalize(vec3(0.707, 0, -0.707)); matType = MAT_MIRROR; }
    else if (primID < 26u) {
        objID = 9u;
        uint faceIdx = (primID - 14u) / 2u;
        if (faceIdx == 0u) normal = vec3(0, 0, 1);
        else if (faceIdx == 1u) normal = vec3(0, 0, -1);
        else if (faceIdx == 2u) normal = vec3(1, 0, 0);
        else if (faceIdx == 3u) normal = vec3(-1, 0, 0);
        else if (faceIdx == 4u) normal = vec3(0, 1, 0);
        else normal = vec3(0, -1, 0);
    }
    else if (primID < 30u) { objID = 8u; normal = vec3(0, 0, -1); matType = MAT_GLASS; }
    else { objID = 10u; normal = vec3(0, 0, 1); }
}

vec3 GetCubeFaceNormal(uint primID) {
    uint localPrim = primID % 12u;
    uint faceIdx = localPrim / 2u;
    vec3 localNormal;
    if (faceIdx == 0u) localNormal = vec3(0, 0, 1);
    else if (faceIdx == 1u) localNormal = vec3(0, 0, -1);
    else if (faceIdx == 2u) localNormal = vec3(1, 0, 0);
    else if (faceIdx == 3u) localNormal = vec3(-1, 0, 0);
    else if (faceIdx == 4u) localNormal = vec3(0, 1, 0);
    else localNormal = vec3(0, -1, 0);

    float angleY = scene.time * 1.2;
    float angleX = scene.time * 0.7;
    mat3 rot = RotateY(angleY) * RotateX(angleX);
    return normalize(rot * localNormal);
}

void main() {
    uint primIdx = gl_PrimitiveID;
    uint instIdx = gl_InstanceID;

    hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    hitT = gl_HitTEXT;
    didHit = true;

    if (instIdx == 0u) {
        // Static geometry
        GetStaticObjectInfo(primIdx, objectID, materialType, hitNormal);
        cubeIdx = 0u;
    } else {
        // Dynamic cubes (instance 1)
        // 8 cubes, each with 12 triangles (6 faces * 2 triangles)
        cubeIdx = primIdx / 12u;
        objectID = 6u;  // OBJ_CUBE
        materialType = MAT_DIFFUSE;
        hitNormal = GetCubeFaceNormal(primIdx);
    }
}
)GLSL";

// ============== GLSL MISS SHADER ==============
static const char* g_rtMissGLSL = R"GLSL(
#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitValue;
layout(location = 1) rayPayloadInEXT float hitT;
layout(location = 2) rayPayloadInEXT vec3 hitNormal;
layout(location = 3) rayPayloadInEXT vec3 hitPos;
layout(location = 4) rayPayloadInEXT uint objectID;
layout(location = 5) rayPayloadInEXT uint materialType;
layout(location = 6) rayPayloadInEXT bool didHit;
layout(location = 7) rayPayloadInEXT uint cubeIdx;

void main() {
    didHit = false;
    cubeIdx = 0u;
    hitValue = vec3(0.05, 0.05, 0.08);
}
)GLSL";

// ============== PRE-COMPILED SPIR-V SHADERS ==============
// Include the pre-compiled SPIR-V bytecode from vulkan_rt_spirv.h
// These were compiled using glslc from Vulkan SDK
#include "vulkan_rt_spirv.h"
