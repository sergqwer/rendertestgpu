#pragma once
// ============== D3D12 DENOISING SHADER ==============
// Edge-aware bilateral filter using À-Trous wavelet transform

static const char* g_ptDenoiseShaderCode = R"HLSL(
Texture2D<float4> InputTexture : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

cbuffer DenoiseCB : register(b0)
{
    uint Width;
    uint Height;
    uint StepSize;      // 1, 2, 4, 8 for à-trous iterations
    float ColorSigma;   // Color similarity weight
};

// Gaussian kernel 5x5 (precomputed weights)
static const float kernel[3] = { 1.0f, 2.0f/3.0f, 1.0f/6.0f };

float Luminance(float3 c)
{
    return dot(c, float3(0.299f, 0.587f, 0.114f));
}

[numthreads(8, 8, 1)]
void DenoiseCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int2 pixel = int2(dispatchThreadID.xy);
    if (pixel.x >= (int)Width || pixel.y >= (int)Height)
        return;

    float4 centerColor = InputTexture[pixel];
    float centerLum = Luminance(centerColor.rgb);

    float3 colorSum = float3(0, 0, 0);
    float weightSum = 0.0f;

    // 5x5 bilateral filter with à-trous spacing
    int step = (int)StepSize;

    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            int2 samplePos = pixel + int2(dx, dy) * step;

            // Clamp to image bounds
            samplePos = clamp(samplePos, int2(0, 0), int2(Width - 1, Height - 1));

            float4 sampleColor = InputTexture[samplePos];
            float sampleLum = Luminance(sampleColor.rgb);

            // Spatial weight (Gaussian)
            float spatialDist = abs(dx) + abs(dy);
            int ki = min(abs(dx), 2);
            int kj = min(abs(dy), 2);
            float spatialWeight = kernel[ki] * kernel[kj];

            // Color/range weight (edge-stopping)
            float colorDiff = abs(centerLum - sampleLum);
            float colorWeight = exp(-colorDiff * colorDiff / (ColorSigma * ColorSigma + 0.0001f));

            // Also use color difference for RGB
            float3 rgbDiff = centerColor.rgb - sampleColor.rgb;
            float rgbDist = dot(rgbDiff, rgbDiff);
            colorWeight *= exp(-rgbDist / (ColorSigma * ColorSigma * 3.0f + 0.0001f));

            float weight = spatialWeight * colorWeight;
            colorSum += sampleColor.rgb * weight;
            weightSum += weight;
        }
    }

    float3 result = colorSum / max(weightSum, 0.0001f);
    OutputTexture[pixel] = float4(result, 1.0f);
}
)HLSL";
