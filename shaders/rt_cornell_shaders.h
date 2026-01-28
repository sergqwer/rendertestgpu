#pragma once
// ============== D3D12 DXR CORNELL BOX SHADERS ==============
// Complete ray-traced Cornell Box with shadows, reflections, glass
//
// SHADER MODES:
// - USE_RAYQUERY: SM 6.5 with inline ray tracing (requires real DXR GPU)
// - Without USE_RAYQUERY: SM 6.0 compatible mode (no ray tracing features)
//
// COMPILE-TIME FEATURES (pass -D FEATURE_XXX to enable):
// - FEATURE_SHADOWS: Ray-traced shadows (requires USE_RAYQUERY)
// - FEATURE_SOFT_SHADOWS: Multiple shadow samples (requires FEATURE_SHADOWS)
// - FEATURE_RT_LIGHTING: Spotlight lighting model (no RayQuery needed)
// - FEATURE_AO: Ray-traced ambient occlusion (requires USE_RAYQUERY)
// - FEATURE_GI: Global illumination (requires USE_RAYQUERY)
// - FEATURE_REFLECTIONS: Mirror and glass reflections (requires USE_RAYQUERY)
// - FEATURE_TEMPORAL_DENOISE: Temporal denoising (no RayQuery needed)

static const char* g_rtCornellShaderCode = R"HLSL(

// ============== CONSTANT BUFFER ==============
// Only contains parameters, not enable flags (those are compile-time now)
cbuffer SceneCB : register(b0) {
    float Time;
    float ShadowSoftness;
    int ShadowSamples;
    int DebugMode;      // 0=normal, 1=objID, 2=normals, 3=reflectDir, 4=shadows, 5=UV, 6=depth

    float ReflectionStrength;
    float AORadius;
    float AOStrength;
    int AOSamples;

    int GIBounces;
    float GIStrength;
    float DenoiseBlendFactor;
    int _Padding;
};

// ============== DEBUG MODE CONSTANTS ==============
#define DEBUG_NONE       0
#define DEBUG_OBJECT_ID  1
#define DEBUG_NORMALS    2
#define DEBUG_REFLECT    3
#define DEBUG_SHADOWS    4
#define DEBUG_UV         5
#define DEBUG_DEPTH      6

// ============== ACCELERATION STRUCTURE (only for RayQuery mode) ==============
#ifdef USE_RAYQUERY
RaytracingAccelerationStructure Scene : register(t0);
#endif

// ============== HISTORY BUFFER FOR TEMPORAL DENOISING ==============
#ifdef FEATURE_TEMPORAL_DENOISE
Texture2D<float4> HistoryBuffer : register(t1);
#endif

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

// ============== SCENE COLORS ==============
static const float3 Colors[11] = {
    float3(0.7, 0.7, 0.7),    // 0: Floor - grey
    float3(0.9, 0.9, 0.9),    // 1: Ceiling - white
    float3(0.7, 0.7, 0.7),    // 2: Back wall - grey (like floor)
    float3(0.75, 0.15, 0.15), // 3: Left wall - RED
    float3(0.15, 0.75, 0.15), // 4: Right wall - GREEN
    float3(15.0, 14.0, 12.0), // 5: Light - bright emissive
    float3(0.9, 0.6, 0.2),    // 6: Cube - orange (fallback)
    float3(0.95, 0.95, 0.95), // 7: Mirror - neutral
    float3(0.9, 0.95, 1.0),   // 8: Glass - slight blue tint
    float3(0.9, 0.15, 0.1),   // 9: Small cube behind glass - RED
    float3(0.5, 0.15, 0.7)    // 10: Front wall - PURPLE
};

// ============== CUBE COLORS (8 cubes, same as D3D12 base renderer) ==============
static const float3 CubeColors[8] = {
    float3(0.95, 0.2, 0.15),  // 0: Red
    float3(0.2, 0.7, 0.3),    // 1: Green
    float3(0.15, 0.5, 0.95),  // 2: Blue
    float3(1.0, 0.85, 0.0),   // 3: Yellow
    float3(1.0, 0.85, 0.0),   // 4: Yellow
    float3(0.15, 0.5, 0.95),  // 5: Blue
    float3(0.2, 0.7, 0.3),    // 6: Green
    float3(0.95, 0.2, 0.15)   // 7: Red
};

// ============== CAMERA & MATRICES ==============
static const float3 CameraPos = float3(0, 0, -2.2);
static const float3 LightPos = float3(0, 0.92, 0);
static const float3 LightColor = float3(1.0, 0.95, 0.9);
static const float LightIntensity = 3.5;
static const float LightRadius = 0.15;

// ============== SPOTLIGHT PARAMETERS ==============
#ifdef FEATURE_RT_LIGHTING
static const float3 SpotlightDir = normalize(float3(0, -1, 0.15));
static const float SpotInnerCos = 0.85;
static const float SpotOuterCos = 0.5;
#endif

// View matrix
static const matrix ViewMat = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 2.2, 1
};

// Projection matrix
static const matrix ProjMat = {
    1.3, 0, 0, 0,
    0, 1.73, 0, 0,
    0, 0, 1.001, 1,
    0, 0, -0.1, 0
};

// ============== VERTEX STRUCTURES ==============
struct VSInput {
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    uint ObjectID : OBJECTID;
    uint MaterialType : MATERIALTYPE;
};

struct PSInput {
    float4 Position : SV_POSITION;
    float3 WorldPos : WORLDPOS;
    float3 Normal : NORMAL;
    nointerpolation uint ObjectID : OBJECTID;
    nointerpolation uint MaterialType : MATERIALTYPE;
};

// ============== UTILITY FUNCTIONS ==============
float3x3 RotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3x3(c, 0, s, 0, 1, 0, -s, 0, c);
}

float3x3 RotateX(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3x3(1, 0, 0, 0, c, -s, 0, s, c);
}

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

#ifdef FEATURE_RT_LIGHTING
float SpotlightAttenuation(float3 lightToPoint) {
    float3 L = normalize(lightToPoint);
    float cosAngle = dot(L, SpotlightDir);
    return saturate((cosAngle - SpotOuterCos) / (SpotInnerCos - SpotOuterCos));
}
#endif

float3 RandomInDisk(inout uint seed) {
    float r = sqrt(Random(seed));
    float theta = 6.28318530718 * Random(seed);
    return float3(r * cos(theta), 0, r * sin(theta));
}

// Generate random direction in hemisphere around normal (cosine-weighted)
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

// ============== RAY TRACING FUNCTIONS ==============

#ifdef FEATURE_SHADOWS
// Trace shadow ray - returns visibility (0 = blocked, 1 = visible)
float TraceShadow(float3 origin, float3 lightPos, inout uint seed) {
    float3 toLight = lightPos - origin;
    float lightDist = length(toLight);
    float3 lightDir = toLight / lightDist;

#ifdef FEATURE_SOFT_SHADOWS
    // Jitter light position for soft shadows
    float3 jitter = RandomInDisk(seed) * LightRadius;
    float3 targetPos = lightPos + jitter;
    toLight = targetPos - origin;
    lightDist = length(toLight);
    lightDir = toLight / lightDist;
#endif

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> query;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = lightDir;
    ray.TMin = 0.001;
    ray.TMax = lightDist - 0.001;

    query.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();

    return (query.CommittedStatus() == COMMITTED_NOTHING) ? 1.0 : 0.0;
}
#endif

// ============== PRIMITIVE/OBJECT HELPERS ==============
float3 GetColorFromPrimitive(uint primID, out uint objID) {
    if (primID < 2) objID = OBJ_FLOOR;
    else if (primID < 4) objID = OBJ_CEILING;
    else if (primID < 6) objID = OBJ_BACK_WALL;
    else if (primID < 8) objID = OBJ_LEFT_WALL;
    else if (primID < 10) objID = OBJ_RIGHT_WALL;
    else if (primID < 12) objID = OBJ_LIGHT;
    else if (primID < 14) objID = OBJ_MIRROR;
    else if (primID < 26) objID = OBJ_SMALL_CUBE;
    else if (primID < 30) objID = OBJ_GLASS;
    else objID = OBJ_FRONT_WALL;
    return Colors[min(objID, 10u)];
}

float3 GetNormalForObject(uint objID) {
    if (objID == OBJ_FLOOR) return float3(0, 1, 0);
    if (objID == OBJ_CEILING) return float3(0, -1, 0);
    if (objID == OBJ_BACK_WALL) return float3(0, 0, -1);
    if (objID == OBJ_LEFT_WALL) return float3(1, 0, 0);
    if (objID == OBJ_RIGHT_WALL) return float3(-1, 0, 0);
    if (objID == OBJ_LIGHT) return float3(0, -1, 0);
    if (objID == OBJ_MIRROR) return normalize(float3(0.707, 0, -0.707));
    if (objID == OBJ_FRONT_WALL) return float3(0, 0, 1);
    return float3(0, 0, -1);
}

float3 GetCubeFaceNormal(uint primID) {
    uint faceIndex = (primID % 12) / 2;
    if (faceIndex == 0) return float3(0, 0, 1);
    if (faceIndex == 1) return float3(0, 0, -1);
    if (faceIndex == 2) return float3(1, 0, 0);
    if (faceIndex == 3) return float3(-1, 0, 0);
    if (faceIndex == 4) return float3(0, 1, 0);
    return float3(0, -1, 0);
}

float3 TransformCubeNormal(float3 localNormal) {
    float angleY = Time * 1.2;
    float angleX = Time * 0.7;
    float3x3 rot = mul(RotateY(angleY), RotateX(angleX));
    return normalize(mul(localNormal, rot));
}

)HLSL"
R"HLSL(

// ============== AMBIENT OCCLUSION ==============
#ifdef FEATURE_AO
float TraceAORay(float3 origin, float3 direction, float maxDist) {
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> query;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.002;
    ray.TMax = maxDist;

    query.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();

    if (query.CommittedStatus() == COMMITTED_NOTHING) return 1.0;
    return query.CommittedRayT() / maxDist;
}

float CalculateAO(float3 worldPos, float3 normal, float radius, int numSamples, inout uint seed) {
    float ao = 0.0;
    for (int i = 0; i < numSamples; i++) {
        float3 sampleDir = RandomInHemisphere(normal, seed);
        ao += TraceAORay(worldPos + normal * 0.002, sampleDir, radius);
    }
    return ao / float(numSamples);
}
#endif

// ============== GLOBAL ILLUMINATION ==============
#ifdef FEATURE_GI
float3 TraceGIRayIterative(float3 origin, float3 direction, int maxBounces, inout uint seed) {
    float3 accumulated = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    float3 rayOrigin = origin;
    float3 rayDir = direction;

    for (int bounce = 0; bounce < maxBounces; bounce++) {
        RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin = 0.002;
        ray.TMax = 50.0;

        query.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
        query.Proceed();

        if (query.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
            accumulated += throughput * float3(0.02, 0.02, 0.03);
            break;
        }

        uint primID = query.CommittedPrimitiveIndex();
        uint instID = query.CommittedInstanceID();
        float t = query.CommittedRayT();
        float3 hitPos = rayOrigin + rayDir * t;

        uint objID;
        float3 hitColor;
        float3 hitNormal;
        if (instID == 1) {
            objID = OBJ_CUBE;
            uint cubeIndex = primID / 12;
            hitColor = CubeColors[min(cubeIndex, 7u)];
            hitNormal = TransformCubeNormal(GetCubeFaceNormal(primID));
        } else {
            hitColor = GetColorFromPrimitive(primID, objID);
            hitNormal = GetNormalForObject(objID);
        }

        if (objID == OBJ_LIGHT) {
            accumulated += throughput * float3(1.0, 0.95, 0.85) * 2.0;
            break;
        }

        float3 toLight = LightPos - hitPos;
        float lightDistSq = dot(toLight, toLight);
        float3 lightDir = toLight / sqrt(lightDistSq);
        float NdotL = max(dot(hitNormal, lightDir), 0.0);

        // Simple shadow check for GI
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQuery;
        RayDesc shadowRay;
        shadowRay.Origin = hitPos + hitNormal * 0.002;
        shadowRay.Direction = lightDir;
        shadowRay.TMin = 0.002;
        shadowRay.TMax = sqrt(lightDistSq);
        shadowQuery.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, shadowRay);
        shadowQuery.Proceed();
        float shadow = (shadowQuery.CommittedStatus() == COMMITTED_NOTHING) ? 1.0 : 0.0;

        float atten = 1.5 / (1.0 + lightDistSq * 0.1);
        accumulated += throughput * hitColor * NdotL * shadow * atten;
        throughput *= hitColor * 0.5;

        float p = max(throughput.r, max(throughput.g, throughput.b));
        if (p < 0.1) break;

        rayOrigin = hitPos + hitNormal * 0.002;
        rayDir = RandomInHemisphere(hitNormal, seed);
    }
    return accumulated;
}

float3 CalculateGI(float3 worldPos, float3 normal, int numBounces, inout uint seed) {
    float3 giDir = RandomInHemisphere(normal, seed);
    return TraceGIRayIterative(worldPos + normal * 0.002, giDir, numBounces, seed);
}
#endif

// ============== REFLECTIONS (for mirror and glass) ==============
#ifdef FEATURE_REFLECTIONS
float3 TraceRayColor(float3 origin, float3 direction) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.002;
    ray.TMax = 50.0;

    query.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
    query.Proceed();

    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        uint primID = query.CommittedPrimitiveIndex();
        uint instID = query.CommittedInstanceID();
        float t = query.CommittedRayT();

        uint objID;
        float3 baseColor;
        float3 normal;

        if (instID == 1) {
            objID = OBJ_CUBE;
            uint cubeIndex = primID / 12;
            baseColor = CubeColors[min(cubeIndex, 7u)];
            normal = TransformCubeNormal(GetCubeFaceNormal(primID));
        } else {
            baseColor = GetColorFromPrimitive(primID, objID);
            normal = GetNormalForObject(objID);
        }

        if (objID == OBJ_LIGHT) return float3(1.0, 0.95, 0.85);

        float3 hitPos = origin + direction * t;
        float3 toLight = LightPos - hitPos;
        float lightDistSq = dot(toLight, toLight);
        float3 lightDir = toLight / sqrt(lightDistSq);
        float NdotL = max(dot(normal, lightDir), 0.0);

#ifdef FEATURE_RT_LIGHTING
        float distAtten = LightIntensity / (1.0 + lightDistSq * 0.1);
        float spotAtten = SpotlightAttenuation(hitPos - LightPos);
        float atten = distAtten * max(spotAtten, 0.5);
        float3 ambient = baseColor * 0.35;
        float3 diffuse = baseColor * NdotL * atten * 0.65;
#else
        float atten = 1.5 / (1.0 + lightDistSq * 0.1);
        float3 ambient = baseColor * 0.3;
        float3 diffuse = baseColor * NdotL * atten * 0.7;
#endif
        return ambient + diffuse;
    }
    return float3(0.03, 0.03, 0.04);
}
#endif

)HLSL"
R"HLSL(

// ============== VERTEX SHADER ==============
PSInput VSMain(VSInput input) {
    PSInput output;
    float3 worldPos = input.Position;
    float3 worldNorm = input.Normal;

    if (input.ObjectID == OBJ_CUBE) {
        float angleY = Time * 1.2;
        float angleX = Time * 0.7;
        float3x3 rot = mul(RotateY(angleY), RotateX(angleX));
        float3 cubeCenter = float3(0.15, 0.15, 0.2);
        worldPos = mul(worldPos, rot) + cubeCenter;
        worldNorm = mul(worldNorm, rot);
    }

    output.WorldPos = worldPos;
    output.Normal = normalize(worldNorm);
    output.ObjectID = input.ObjectID;
    output.MaterialType = input.MaterialType;

    float4 viewPos = mul(float4(worldPos, 1.0), ViewMat);
    output.Position = mul(viewPos, ProjMat);
    return output;
}

// ============== HELPER FUNCTIONS ==============
uint GetMaterialFromObject(uint objID) {
    if (objID == OBJ_LIGHT) return MAT_EMISSIVE;
    if (objID == OBJ_MIRROR) return MAT_MIRROR;
    if (objID == OBJ_GLASS) return MAT_GLASS;
    return MAT_DIFFUSE;
}

float3 DebugObjectIDColor(uint objID) {
    if (objID == 0) return float3(1, 1, 1);
    if (objID == 1) return float3(0.8, 0.8, 0.8);
    if (objID == 2) return float3(0.5, 0.5, 0.5);
    if (objID == 3) return float3(1, 0, 0);
    if (objID == 4) return float3(0, 1, 0);
    if (objID == 5) return float3(1, 1, 0);
    if (objID == 6) return float3(1, 0.5, 0);
    if (objID == 7) return float3(1, 0, 1);
    if (objID == 8) return float3(0, 1, 1);
    if (objID == 9) return float3(0.5, 0, 0);
    return float3(0, 0, 1);
}

// ============== PIXEL SHADER ==============
float4 PSMain(PSInput input) : SV_TARGET {
    float3 worldPos = input.WorldPos;
    float3 normal = normalize(input.Normal);
    uint objID = input.ObjectID;
    uint matType = GetMaterialFromObject(objID);

    float3 viewDir = normalize(CameraPos - worldPos);
    float depth = length(worldPos - CameraPos) / 10.0;

    // ============== DEBUG VISUALIZATION MODES ==============
    if (DebugMode == DEBUG_OBJECT_ID) return float4(DebugObjectIDColor(objID), 1.0);
    if (DebugMode == DEBUG_NORMALS) return float4(normal * 0.5 + 0.5, 1.0);
    if (DebugMode == DEBUG_REFLECT) {
        float3 reflectDir = reflect(-viewDir, normal);
        return float4(reflectDir * 0.5 + 0.5, 1.0);
    }
#ifdef FEATURE_SHADOWS
    if (DebugMode == DEBUG_SHADOWS) {
        uint2 pixelCoord = uint2(input.Position.xy);
        uint seed = pixelCoord.x * 1973 + pixelCoord.y * 9277;
        float vis = TraceShadow(worldPos + normal * 0.002, LightPos, seed);
        return float4(vis, vis, vis, 1.0);
    }
#endif
    if (DebugMode == DEBUG_UV) return float4(frac(worldPos), 1.0);
    if (DebugMode == DEBUG_DEPTH) {
        float d = saturate(1.0 - depth);
        return float4(d, d, d, 1.0);
    }

    // Determine base color
    float3 baseColor;
    if (objID == OBJ_CUBE) {
        baseColor = CubeColors[min(input.MaterialType, 7u)];
    } else {
        baseColor = Colors[min(objID, 10u)];
    }

    // Emissive (light source)
    if (objID == OBJ_LIGHT) return float4(baseColor, 1.0);

    // Random seed for stochastic sampling
    uint2 pixelCoord = uint2(input.Position.xy);
    uint seed = pixelCoord.x * 1973 + pixelCoord.y * 9277 + uint(Time * 1000) * 26699;

    // Calculate lighting
    float3 toLight = LightPos - worldPos;
    float lightDist = length(toLight);
    float3 lightDir = toLight / lightDist;
    float NdotL = max(dot(normal, lightDir), 0.0);

    // ============== SHADOWS ==============
    float shadow = 1.0;
#ifdef FEATURE_SHADOWS
    if (NdotL > 0.0) {
        shadow = 0.0;
#ifdef FEATURE_SOFT_SHADOWS
        uint numSamples = (uint)max(ShadowSamples, 1);
#else
        uint numSamples = 1;
#endif
        for (uint i = 0; i < numSamples; i++) {
            shadow += TraceShadow(worldPos + normal * 0.002, LightPos, seed);
        }
        shadow /= float(numSamples);
    }
#endif

    // ============== LIGHTING ==============
    float3 ambient, diffuse;
#ifdef FEATURE_RT_LIGHTING
    float distAtten = 1.0 / (1.0 + lightDist * lightDist * 0.08);
    float spotAtten = SpotlightAttenuation(worldPos - LightPos);
    float atten = distAtten * spotAtten;
    ambient = baseColor * 0.08;
    diffuse = baseColor * NdotL * shadow * atten * LightColor * LightIntensity;
#else
    float atten = 2.0 / (1.0 + lightDist * lightDist * 0.1);
    ambient = baseColor * 0.2;
    diffuse = baseColor * NdotL * shadow * atten * LightColor;
#endif

    float3 finalColor = ambient + diffuse;

    // ============== AMBIENT OCCLUSION ==============
#ifdef FEATURE_AO
    float ao = CalculateAO(worldPos, normal, AORadius, AOSamples, seed);
    finalColor *= lerp(1.0 - AOStrength, 1.0, ao);
#endif

    // ============== GLOBAL ILLUMINATION ==============
#ifdef FEATURE_GI
    float3 gi = CalculateGI(worldPos, normal, GIBounces, seed);
    finalColor += gi * GIStrength * baseColor;
#endif

    // ============== MIRROR REFLECTION ==============
#ifdef FEATURE_REFLECTIONS
    if (objID == OBJ_MIRROR) {
        float3 mirrorNormal = GetNormalForObject(OBJ_MIRROR);
        float3 reflectDir = reflect(-viewDir, mirrorNormal);
        float3 rayOrigin = worldPos + mirrorNormal * 0.01;
        float3 reflectedColor = TraceRayColor(rayOrigin, reflectDir);
        return float4(reflectedColor, 1);
    }

    // ============== GLASS TRANSPARENCY ==============
    if (objID == OBJ_GLASS) {
        float3 vDir = normalize(worldPos - CameraPos);
        float fresnel = pow(1.0 - abs(dot(-vDir, normal)), 3.0);
        float3 behindColor = TraceRayColor(worldPos + vDir * 0.01, vDir);
        float3 glassTint = float3(0.95, 0.97, 1.0);
        float3 reflectDir = reflect(vDir, normal);
        float3 reflectColor = TraceRayColor(worldPos + normal * 0.002, reflectDir);
        finalColor = lerp(behindColor * glassTint, reflectColor, fresnel * 0.3);
        finalColor += float3(0.1, 0.12, 0.15) * fresnel;
    }
#else
    // Fallback: Mirror and glass render as simple surfaces without RT
    if (objID == OBJ_MIRROR || objID == OBJ_GLASS) {
        // Just use diffuse lighting computed above
    }
#endif

    // Tone mapping
    finalColor = finalColor / (finalColor + 1.0);
    // Gamma correction
    finalColor = pow(finalColor, 1.0 / 2.2);

    // ============== TEMPORAL DENOISING ==============
#ifdef FEATURE_TEMPORAL_DENOISE
    int2 pixelPos = int2(input.Position.xy);
    float4 historyColor = HistoryBuffer.Load(int3(pixelPos, 0));
    finalColor = lerp(finalColor, historyColor.rgb, DenoiseBlendFactor);
#endif

    return float4(finalColor, 1.0);
}

)HLSL";

// ============== TEXT SHADER (unchanged) ==============
static const char* g_textShaderCode = R"HLSL(
Texture2D fontTex : register(t0);
SamplerState samp : register(s0);

struct VSIn { float2 pos : POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };
struct PSIn { float4 pos : SV_POSITION; float2 uv : TEXCOORD; float4 col : COLOR; };

PSIn TextVS(VSIn i) {
    PSIn o;
    o.pos = float4(i.pos, 0, 1);
    o.uv = i.uv;
    o.col = i.col;
    return o;
}

float4 TextPS(PSIn i) : SV_TARGET {
    float a = fontTex.Sample(samp, i.uv).r;
    return float4(i.col.rgb, i.col.a * a);
}
)HLSL";
