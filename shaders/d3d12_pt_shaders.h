#pragma once
// ============== D3D12 PATH TRACING COMPUTE SHADER ==============
// Cornell Box scene with path tracing (matching RT renderer)
// Shader Model 6.5 with RayQuery for inline ray tracing

static const char* g_ptShaderCode = R"HLSL(
// Constant buffer with camera and timing data
cbuffer PathTraceCB : register(b0)
{
    float4x4 InvView;
    float4x4 InvProj;
    float Time;
    uint FrameCount;
    uint Width;
    uint Height;
};

// Vertex structure matching CPU side (pos, normal, objectID, materialType)
struct Vertex
{
    float3 pos;
    float3 normal;
    uint objectID;
    uint materialType;
};

// Material types
#define MAT_DIFFUSE  0
#define MAT_MIRROR   1
#define MAT_GLASS    2
#define MAT_EMISSIVE 3

// Object IDs (matching RT shader)
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

// Resources - must match root signature!
RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<Vertex> Vertices : register(t1);
StructuredBuffer<uint> Indices : register(t2);
RWTexture2D<float4> Output : register(u0);

// Cornell Box colors (matching RT shader)
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

// Cube colors (8 cubes, same as D3D12 base renderer)
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

// Light properties (adjusted for larger room s=2.0)
static const float3 LightPos = float3(0, 1.92, 0);
static const float3 LightColor = float3(1.0, 0.95, 0.9);
static const float LightIntensity = 6.0;  // Increased for larger room
static const float LightRadius = 0.3;     // Larger light source
static const float PI = 3.14159265359;

// Spotlight parameters (matching RT shader)
static const float3 SpotlightDir = normalize(float3(0, -1, 0.15));
static const float SpotInnerCos = 0.85;  // ~32 degrees - full intensity
static const float SpotOuterCos = 0.5;   // ~60 degrees - falloff edge

// Wang hash for random number generation
uint WangHash(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

float RandomFloat(inout uint seed)
{
    seed = WangHash(seed);
    return float(seed) / 4294967296.0;
}

// Rotation matrices (for cube normal transform)
float3x3 RotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3x3(
        c,  0, s,
        0,  1, 0,
        -s, 0, c
    );
}

float3x3 RotateX(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return float3x3(
        1,  0,  0,
        0,  c, -s,
        0,  s,  c
    );
}

// Get cube face normal in object space from primitive ID
float3 GetCubeFaceNormal(uint primID) {
    uint triangleInCube = primID % 12;
    uint faceIndex = triangleInCube / 2;
    // Face order matches AddBox: Front(+Z), Back(-Z), Right(+X), Left(-X), Top(+Y), Bottom(-Y)
    if (faceIndex == 0) return float3(0, 0, 1);
    if (faceIndex == 1) return float3(0, 0, -1);
    if (faceIndex == 2) return float3(1, 0, 0);
    if (faceIndex == 3) return float3(-1, 0, 0);
    if (faceIndex == 4) return float3(0, 1, 0);
    return float3(0, -1, 0);
}

// Transform cube normal from object space to world space
float3 TransformCubeNormal(float3 localNormal) {
    float angleY = Time * 1.2;
    float angleX = Time * 0.7;
    float3x3 rotY = RotateY(angleY);
    float3x3 rotX = RotateX(angleX);
    float3x3 rot = mul(rotY, rotX);
    return normalize(mul(localNormal, rot));
}

// Get color from primitive ID (static geometry only)
// Order: Floor(2), Ceiling(2), Back(2), Left(2), Right(2), Light(2), Mirror(2), SmallCube(12), Glass(4)
float3 GetColorFromPrimitive(uint primID, out uint objID) {
    if (primID < 2) objID = OBJ_FLOOR;
    else if (primID < 4) objID = OBJ_CEILING;
    else if (primID < 6) objID = OBJ_BACK_WALL;
    else if (primID < 8) objID = OBJ_LEFT_WALL;
    else if (primID < 10) objID = OBJ_RIGHT_WALL;
    else if (primID < 12) objID = OBJ_LIGHT;
    else if (primID < 14) objID = OBJ_MIRROR;
    else if (primID < 26) objID = OBJ_SMALL_CUBE;
    else objID = OBJ_GLASS;  // primID 26-29 (no front wall)
    return Colors[min(objID, 10u)];
}

// Get material type from object ID
uint GetMaterialFromObject(uint objID) {
    if (objID == OBJ_LIGHT) return MAT_EMISSIVE;
    if (objID == OBJ_MIRROR) return MAT_MIRROR;
    if (objID == OBJ_GLASS) return MAT_GLASS;
    return MAT_DIFFUSE;
}

// Get approximate normal for object
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

// Spotlight cone attenuation
float SpotlightAttenuation(float3 lightToPoint) {
    float3 L = normalize(lightToPoint);
    float cosAngle = dot(L, SpotlightDir);
    return saturate((cosAngle - SpotOuterCos) / (SpotInnerCos - SpotOuterCos));
}

// Cosine-weighted hemisphere sampling
float3 CosineSampleHemisphere(float2 u, float3 N)
{
    float r = sqrt(u.x);
    float theta = 2.0 * PI * u.y;
    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(max(0.0, 1.0 - u.x));

    float3 up = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(up, N));
    float3 bitangent = cross(N, tangent);

    return normalize(tangent * x + bitangent * y + N * z);
}

// Random point on disk (for area light sampling)
float3 RandomInDisk(inout uint seed)
{
    float u1 = RandomFloat(seed);
    float u2 = RandomFloat(seed);
    float r = sqrt(u1);
    float theta = 2.0 * PI * u2;
    return float3(r * cos(theta), 0, r * sin(theta));
}

// Fresnel-Schlick approximation
float FresnelSchlick(float cosTheta, float F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

[numthreads(8, 8, 1)]
void PathTraceCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (pixel.x >= Width || pixel.y >= Height)
        return;

    uint seed = WangHash(pixel.x + pixel.y * Width + FrameCount * Width * Height);

    // Jittered pixel for AA
    float2 jitter = float2(RandomFloat(seed), RandomFloat(seed));
    float2 uv = (float2(pixel) + jitter) / float2(Width, Height);
    uv = uv * 2.0 - 1.0;
    uv.y = -uv.y;

    // Generate camera ray in WORLD space
    float4 clipPos = float4(uv, 0.0, 1.0);
    float4 viewPos = mul(clipPos, InvProj);
    viewPos /= viewPos.w;

    float3 rayOrigin = mul(float4(0, 0, 0, 1), InvView).xyz;
    float3 rayDir = normalize(mul(float4(viewPos.xyz, 0), InvView).xyz);

    // Path tracing
    float3 radiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);

    for (int bounce = 0; bounce < 4; bounce++)
    {
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin = 0.001;
        ray.TMax = 100.0;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            float t = q.CommittedRayT();
            uint primID = q.CommittedPrimitiveIndex();
            uint instID = q.CommittedInstanceID();
            float2 bary = q.CommittedTriangleBarycentrics();
            float3 hitPos = rayOrigin + rayDir * t;

            // Determine object type, color, normal based on instance
            uint objID;
            float3 albedo;
            float3 hitNormal;
            uint matType;

            if (instID == 1) {
                // Instance 1 = cube BLAS (8 rotating cubes)
                objID = OBJ_CUBE;
                uint cubeIndex = primID / 12;  // Each cube has 12 triangles
                albedo = CubeColors[min(cubeIndex, 7u)];
                float3 localNormal = GetCubeFaceNormal(primID);
                hitNormal = TransformCubeNormal(localNormal);
                matType = MAT_DIFFUSE;
            } else {
                // Instance 0 = static geometry
                albedo = GetColorFromPrimitive(primID, objID);
                matType = GetMaterialFromObject(objID);

                // For static geometry, interpolate normals from vertex data
                uint i0 = Indices[primID * 3 + 0];
                uint i1 = Indices[primID * 3 + 1];
                uint i2 = Indices[primID * 3 + 2];
                float w = 1.0 - bary.x - bary.y;
                hitNormal = normalize(
                    Vertices[i0].normal * w +
                    Vertices[i1].normal * bary.x +
                    Vertices[i2].normal * bary.y
                );
            }

            // Emissive material (light source)
            if (matType == MAT_EMISSIVE) {
                radiance += throughput * float3(1.0, 0.95, 0.85) * 2.0;
                break;
            }

            // Mirror material
            if (matType == MAT_MIRROR) {
                rayDir = reflect(rayDir, hitNormal);
                rayOrigin = hitPos + hitNormal * 0.001;
                throughput *= 0.95;  // Slight absorption
                continue;
            }

            // Glass material
            if (matType == MAT_GLASS) {
                float eta = 1.5;  // Glass IOR
                float cosi = -dot(rayDir, hitNormal);
                float3 n = hitNormal;

                if (cosi < 0) {
                    cosi = -cosi;
                    n = -n;
                    eta = 1.0 / eta;
                }

                float fresnel = FresnelSchlick(cosi, 0.04);

                if (RandomFloat(seed) < fresnel) {
                    rayDir = reflect(rayDir, n);
                    rayOrigin = hitPos + n * 0.001;
                } else {
                    float k = 1.0 - (1.0 / eta) * (1.0 / eta) * (1.0 - cosi * cosi);
                    if (k < 0) {
                        rayDir = reflect(rayDir, n);
                        rayOrigin = hitPos + n * 0.001;
                    } else {
                        rayDir = (1.0 / eta) * rayDir + ((1.0 / eta) * cosi - sqrt(k)) * n;
                        rayOrigin = hitPos - n * 0.001;
                    }
                }
                throughput *= albedo;
                continue;
            }

            // Diffuse material - sample light directly with spotlight
            float3 directLight = float3(0, 0, 0);

            // Sample area light
            float3 lightSample = LightPos + RandomInDisk(seed) * LightRadius;
            float3 toLight = lightSample - hitPos;
            float lightDist = length(toLight);
            toLight /= lightDist;

            float NdotL = max(dot(hitNormal, toLight), 0);
            if (NdotL > 0) {
                // Shadow ray
                RayDesc shadowRay;
                shadowRay.Origin = hitPos + hitNormal * 0.001;
                shadowRay.Direction = toLight;
                shadowRay.TMin = 0.001;
                shadowRay.TMax = lightDist - 0.01;

                RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQ;
                shadowQ.TraceRayInline(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
                shadowQ.Proceed();

                if (shadowQ.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                    // Not in shadow - apply spotlight attenuation
                    float distAtten = 1.0 / (1.0 + lightDist * lightDist * 0.08);
                    float spotAtten = SpotlightAttenuation(hitPos - LightPos);
                    float atten = distAtten * spotAtten * LightIntensity;
                    directLight = albedo * LightColor * NdotL * atten / PI;
                }
            }

            radiance += throughput * directLight;

            // Add ambient for areas outside spotlight
            radiance += throughput * albedo * 0.02;

            // Russian roulette after first bounce
            if (bounce > 0) {
                float p = max(throughput.x, max(throughput.y, throughput.z));
                if (RandomFloat(seed) > p || p < 0.01)
                    break;
                throughput /= p;
            }

            // Indirect bounce - cosine-weighted hemisphere sampling
            float2 u = float2(RandomFloat(seed), RandomFloat(seed));
            rayDir = CosineSampleHemisphere(u, hitNormal);
            rayOrigin = hitPos + hitNormal * 0.001;
            throughput *= albedo;
        }
        else
        {
            // Miss - background color (very dark for closed box)
            radiance += throughput * float3(0.01, 0.01, 0.02);
            break;
        }
    }

    // Tone mapping (simple Reinhard) and gamma
    radiance = radiance / (radiance + 1.0);
    radiance = pow(saturate(radiance), 1.0 / 2.2);

    Output[pixel] = float4(radiance, 1.0);
}
)HLSL";
