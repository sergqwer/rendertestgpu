#pragma once
// ============== D3D12 PATH TRACING FOR DLSS-RR ==============
// Cornell Box scene with G-Buffer outputs for NVIDIA DLSS Ray Reconstruction

static const char* g_ptDlssShaderCode = R"HLSL(
// Constant buffer with camera and timing data
cbuffer PathTraceCB : register(b0)
{
    float4x4 InvView;
    float4x4 InvProj;
    float4x4 PrevViewProj;  // Previous frame's ViewProj for motion vectors
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

// Object IDs (matching RT/PT shaders)
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

// Resources
RaytracingAccelerationStructure Scene : register(t0);
StructuredBuffer<Vertex> Vertices : register(t1);
StructuredBuffer<uint> Indices : register(t2);

// G-Buffer outputs for DLSS-RR
RWTexture2D<float4> OutputColor : register(u0);           // Noisy path traced color (linear HDR)
RWTexture2D<float4> OutputDiffuseAlbedo : register(u1);   // Diffuse albedo
RWTexture2D<float4> OutputSpecularAlbedo : register(u2);  // Specular albedo (F0)
RWTexture2D<float4> OutputNormals : register(u3);         // World-space normals (encoded)
RWTexture2D<float> OutputRoughness : register(u4);        // Roughness
RWTexture2D<float> OutputDepth : register(u5);            // Linear depth
RWTexture2D<float2> OutputMotionVectors : register(u6);   // Screen-space motion vectors

// Cornell Box colors (matching PT shader)
static const float3 Colors[11] = {
    float3(0.7, 0.7, 0.7),    // 0: Floor - grey
    float3(0.9, 0.9, 0.9),    // 1: Ceiling - white
    float3(0.7, 0.7, 0.7),    // 2: Back wall - grey
    float3(0.75, 0.15, 0.15), // 3: Left wall - RED
    float3(0.15, 0.75, 0.15), // 4: Right wall - GREEN
    float3(15.0, 14.0, 12.0), // 5: Light - emissive
    float3(0.9, 0.6, 0.2),    // 6: Cube - orange (fallback)
    float3(0.95, 0.95, 0.95), // 7: Mirror - neutral
    float3(0.9, 0.95, 1.0),   // 8: Glass - slight blue
    float3(0.9, 0.15, 0.1),   // 9: Small cube - RED
    float3(0.5, 0.15, 0.7)    // 10: Front wall - PURPLE
};

// Cube colors (8 cubes)
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

// Roughness per object type (for DLSS G-Buffer)
static const float ObjectRoughness[11] = {
    0.9, 0.9, 0.9, 0.9, 0.9,  // Walls: diffuse
    0.9,                        // Light
    0.5,                        // Cube: medium
    0.02,                       // Mirror: very smooth
    0.02,                       // Glass: very smooth
    0.8,                        // Small cube
    0.9                         // Front wall
};

// Light properties (adjusted for larger room s=2.0)
static const float3 LightPos = float3(0, 1.92, 0);
static const float3 LightColor = float3(1.0, 0.95, 0.9);
static const float LightIntensity = 6.0;  // Increased for larger room
static const float LightRadius = 0.3;     // Larger light source
static const float PI = 3.14159265359;

// Spotlight parameters
static const float3 SpotlightDir = normalize(float3(0, -1, 0.15));
static const float SpotInnerCos = 0.85;
static const float SpotOuterCos = 0.5;

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

// Rotation matrices
float3x3 RotateY(float angle) {
    float c = cos(angle), s = sin(angle);
    return float3x3(c, 0, s, 0, 1, 0, -s, 0, c);
}

float3x3 RotateX(float angle) {
    float c = cos(angle), s = sin(angle);
    return float3x3(1, 0, 0, 0, c, -s, 0, s, c);
}

// Get cube face normal in object space from primitive ID
float3 GetCubeFaceNormal(uint primID) {
    uint triangleInCube = primID % 12;
    uint faceIndex = triangleInCube / 2;
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

// Get color from primitive ID (static geometry)
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

uint GetMaterialFromObject(uint objID) {
    if (objID == OBJ_LIGHT) return MAT_EMISSIVE;
    if (objID == OBJ_MIRROR) return MAT_MIRROR;
    if (objID == OBJ_GLASS) return MAT_GLASS;
    return MAT_DIFFUSE;
}

float SpotlightAttenuation(float3 lightToPoint) {
    float3 L = normalize(lightToPoint);
    float cosAngle = dot(L, SpotlightDir);
    return saturate((cosAngle - SpotOuterCos) / (SpotInnerCos - SpotOuterCos));
}

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

float3 RandomInDisk(inout uint seed)
{
    float u1 = RandomFloat(seed);
    float u2 = RandomFloat(seed);
    float r = sqrt(u1);
    float theta = 2.0 * PI * u2;
    return float3(r * cos(theta), 0, r * sin(theta));
}

float FresnelSchlick(float cosTheta, float F0)
{
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

[numthreads(8, 8, 1)]
void PathTraceDlssCS(uint3 dispatchThreadID : SV_DispatchThreadID)
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

    // Generate camera ray
    float4 clipPos = float4(uv, 0.0, 1.0);
    float4 viewPos = mul(clipPos, InvProj);
    viewPos /= viewPos.w;

    float3 rayOrigin = mul(float4(0, 0, 0, 1), InvView).xyz;
    float3 rayDir = normalize(mul(float4(viewPos.xyz, 0), InvView).xyz);

    // G-Buffer defaults
    float3 firstHitDiffuseAlbedo = float3(0, 0, 0);
    float3 firstHitSpecularAlbedo = float3(0.04, 0.04, 0.04);
    float3 firstHitNormal = float3(0, 0, 1);
    float firstHitRoughness = 1.0;
    float firstHitDepth = 1000.0;
    float2 motionVector = float2(0, 0);
    bool hitSurface = false;
    float3 firstHitWorldPos = float3(0, 0, 0);

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

            uint objID;
            float3 albedo;
            float3 hitNormal;
            uint matType;
            float roughness;

            if (instID == 1) {
                // Instance 1 = cube BLAS (8 rotating cubes)
                objID = OBJ_CUBE;
                uint cubeIndex = primID / 12;
                albedo = CubeColors[min(cubeIndex, 7u)];
                float3 localNormal = GetCubeFaceNormal(primID);
                hitNormal = TransformCubeNormal(localNormal);
                matType = MAT_DIFFUSE;
                roughness = 0.5;
            } else {
                // Instance 0 = static geometry
                albedo = GetColorFromPrimitive(primID, objID);
                matType = GetMaterialFromObject(objID);
                roughness = ObjectRoughness[min(objID, 10u)];

                // Interpolate normals from vertex data
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

            // Store G-Buffer for first hit only
            if (bounce == 0)
            {
                hitSurface = true;
                firstHitWorldPos = hitPos;
                firstHitNormal = hitNormal;
                firstHitDepth = length(hitPos - rayOrigin);
                firstHitRoughness = roughness;

                // Diffuse and specular albedo depend on material
                if (matType == MAT_MIRROR) {
                    firstHitDiffuseAlbedo = float3(0, 0, 0);
                    firstHitSpecularAlbedo = float3(0.95, 0.95, 0.95);
                } else if (matType == MAT_GLASS) {
                    firstHitDiffuseAlbedo = float3(0, 0, 0);
                    firstHitSpecularAlbedo = float3(0.04, 0.04, 0.04);
                } else {
                    firstHitDiffuseAlbedo = albedo;
                    firstHitSpecularAlbedo = float3(0.04, 0.04, 0.04);
                }

                // Compute motion vector (static scene - no motion for walls, motion for cube)
                float3 prevWorldPos = hitPos;  // Default: static
                float3 cubePos = float3(0, 0, 0.5);  // Cube center (matching UpdateCubeTransformPT)
                if (objID == OBJ_CUBE && instID == 1) {
                    // For rotating cube, compute previous frame position
                    float prevTime = Time - (1.0 / 60.0);
                    float3x3 invRot = transpose(mul(RotateY(Time * 1.2), RotateX(Time * 0.7)));
                    float3 objPos = mul(hitPos - cubePos, invRot);  // To object space
                    float3x3 prevRot = mul(RotateY(prevTime * 1.2), RotateX(prevTime * 0.7));
                    prevWorldPos = mul(objPos, prevRot) + cubePos;
                }
                float4 prevClip = mul(float4(prevWorldPos, 1.0), PrevViewProj);
                prevClip.xyz /= prevClip.w;
                float2 prevUV = prevClip.xy * 0.5 + 0.5;
                prevUV.y = 1.0 - prevUV.y;
                float2 currUV = (float2(pixel) + 0.5) / float2(Width, Height);
                motionVector = (currUV - prevUV) * float2(Width, Height);
            }

            // Emissive material
            if (matType == MAT_EMISSIVE) {
                radiance += throughput * float3(1.0, 0.95, 0.85) * 2.0;
                break;
            }

            // Mirror reflection
            if (matType == MAT_MIRROR) {
                rayDir = reflect(rayDir, hitNormal);
                rayOrigin = hitPos + hitNormal * 0.001;
                throughput *= 0.95;
                continue;
            }

            // Glass refraction
            if (matType == MAT_GLASS) {
                float eta = 1.5;
                float cosi = -dot(rayDir, hitNormal);
                float3 n = hitNormal;
                if (cosi < 0) { cosi = -cosi; n = -n; eta = 1.0 / eta; }
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

            // Diffuse - direct lighting with spotlight
            float3 directLight = float3(0, 0, 0);
            float3 lightSample = LightPos + RandomInDisk(seed) * LightRadius;
            float3 toLight = lightSample - hitPos;
            float lightDist = length(toLight);
            toLight /= lightDist;
            float NdotL = max(dot(hitNormal, toLight), 0);

            if (NdotL > 0) {
                RayDesc shadowRay;
                shadowRay.Origin = hitPos + hitNormal * 0.001;
                shadowRay.Direction = toLight;
                shadowRay.TMin = 0.001;
                shadowRay.TMax = lightDist - 0.01;

                RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> shadowQ;
                shadowQ.TraceRayInline(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, shadowRay);
                shadowQ.Proceed();

                if (shadowQ.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
                    float distAtten = 1.0 / (1.0 + lightDist * lightDist * 0.08);
                    float spotAtten = SpotlightAttenuation(hitPos - LightPos);
                    float atten = distAtten * spotAtten * LightIntensity;
                    directLight = albedo * LightColor * NdotL * atten / PI;
                }
            }

            radiance += throughput * directLight;
            radiance += throughput * albedo * 0.02;  // Ambient

            // Russian roulette
            if (bounce > 0) {
                float p = max(throughput.x, max(throughput.y, throughput.z));
                if (RandomFloat(seed) > p || p < 0.01) break;
                throughput /= p;
            }

            // Indirect bounce
            float2 u = float2(RandomFloat(seed), RandomFloat(seed));
            rayDir = CosineSampleHemisphere(u, hitNormal);
            rayOrigin = hitPos + hitNormal * 0.001;
            throughput *= albedo;
        }
        else
        {
            // Miss - dark background (closed box)
            radiance += throughput * float3(0.01, 0.01, 0.02);
            break;
        }
    }

    // Output G-Buffer (linear HDR, no tone mapping - DLSS handles that)
    OutputColor[pixel] = float4(radiance, 1.0);
    OutputDiffuseAlbedo[pixel] = float4(firstHitDiffuseAlbedo, 1.0);
    OutputSpecularAlbedo[pixel] = float4(firstHitSpecularAlbedo, 1.0);
    OutputNormals[pixel] = float4(firstHitNormal * 0.5 + 0.5, 1.0);  // Encode to [0,1]
    OutputRoughness[pixel] = firstHitRoughness;
    OutputDepth[pixel] = firstHitDepth;
    OutputMotionVectors[pixel] = motionVector;
}
)HLSL";
