// ============== VULKAN RENDERER IMPLEMENTATION ==============
// Extracted from main.cpp - Vulkan rendering backend

#define VK_USE_PLATFORM_WIN32_KHR
#include "vulkan.h"
#include "../common.h"
#include "vulkan_shaders.h"
#include "renderer_vulkan.h"

#pragma comment(lib, "vulkan-1.lib")

// External declarations for shared globals
extern int fps;
extern std::wstring gpuName;
extern LARGE_INTEGER g_startTime;
extern LARGE_INTEGER g_perfFreq;
extern const unsigned char g_font8x8[96][8];
extern const UINT W;
extern const UINT H;

// ============== VULKAN GLOBALS ==============
static VkInstance g_vkInstance = VK_NULL_HANDLE;
static VkPhysicalDevice g_vkPhysicalDevice = VK_NULL_HANDLE;
static VkDevice g_vkDevice = VK_NULL_HANDLE;
static VkQueue g_vkGraphicsQueue = VK_NULL_HANDLE;
static VkQueue g_vkPresentQueue = VK_NULL_HANDLE;
static VkSurfaceKHR g_vkSurface = VK_NULL_HANDLE;
static VkSwapchainKHR g_vkSwapchain = VK_NULL_HANDLE;
static std::vector<VkImage> g_vkSwapchainImages;
static std::vector<VkImageView> g_vkSwapchainImageViews;
static VkFormat g_vkSwapchainFormat;
static VkExtent2D g_vkSwapchainExtent;
static VkRenderPass g_vkRenderPass = VK_NULL_HANDLE;
static VkPipelineLayout g_vkPipelineLayout = VK_NULL_HANDLE;
static VkPipeline g_vkPipeline = VK_NULL_HANDLE;
static std::vector<VkFramebuffer> g_vkFramebuffers;
static VkCommandPool g_vkCommandPool = VK_NULL_HANDLE;
static std::vector<VkCommandBuffer> g_vkCommandBuffers;
static VkSemaphore g_vkImageAvailableSemaphore = VK_NULL_HANDLE;
static VkSemaphore g_vkRenderFinishedSemaphore = VK_NULL_HANDLE;
static VkFence g_vkInFlightFence = VK_NULL_HANDLE;
static VkBuffer g_vkVertexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory g_vkVertexBufferMemory = VK_NULL_HANDLE;
static VkBuffer g_vkIndexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory g_vkIndexBufferMemory = VK_NULL_HANDLE;
static VkImage g_vkDepthImage = VK_NULL_HANDLE;
static VkDeviceMemory g_vkDepthImageMemory = VK_NULL_HANDLE;
static VkImageView g_vkDepthImageView = VK_NULL_HANDLE;
static uint32_t g_vkGraphicsFamily = UINT32_MAX;
static uint32_t g_vkPresentFamily = UINT32_MAX;
static uint32_t g_vkIndexCount = 0;
static int g_vkTriangleCount = 0;
static std::string g_vkGpuName;

// Text rendering resources
static VkImage g_vkFontImage = VK_NULL_HANDLE;
static VkDeviceMemory g_vkFontImageMemory = VK_NULL_HANDLE;
static VkImageView g_vkFontImageView = VK_NULL_HANDLE;
static VkSampler g_vkFontSampler = VK_NULL_HANDLE;
static VkDescriptorSetLayout g_vkTextDescSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool g_vkTextDescPool = VK_NULL_HANDLE;
static VkDescriptorSet g_vkTextDescSet = VK_NULL_HANDLE;
static VkPipelineLayout g_vkTextPipelineLayout = VK_NULL_HANDLE;
static VkPipeline g_vkTextPipeline = VK_NULL_HANDLE;
static VkBuffer g_vkTextVertexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory g_vkTextVertexBufferMemory = VK_NULL_HANDLE;
static void* g_vkTextVertexBufferMapped = nullptr;  // Persistently mapped for CPU updates
static const int g_vkMaxTextChars = 256;

// ============== VULKAN VERTEX STRUCTURE ==============
struct VkVert {
    float px, py, pz;
    float nx, ny, nz;
    float r, g, b;  // Per-vertex color
};

// Push constants for MVP matrix and time
struct VkPushConstants {
    float mvp[16];
    float lightDir[4];
    float time;
    float padding[3];
};

// Text vertex structure (position, UV, color)
struct VkTextVert {
    float x, y;     // Screen position in NDC [-1, 1]
    float u, v;     // Texture coordinates
    float r, g, b, a;  // Color
};

// ============== HELPER FUNCTIONS ==============

static uint32_t VkFindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(g_vkPhysicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

static VkShaderModule VkCreateShaderModule(const uint32_t* code, size_t codeSize) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(g_vkDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

static bool VkCreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                           VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(g_vkDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(g_vkDevice, buffer, &memReqs);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = VkFindMemoryType(memReqs.memoryTypeBits, properties);

    if (vkAllocateMemory(g_vkDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        return false;
    }

    vkBindBufferMemory(g_vkDevice, buffer, bufferMemory, 0);
    return true;
}

// ============== GEOMETRY GENERATION ==============

// Generate rounded face for Vulkan (same as D3D11/OpenGL)
void GenRoundedFaceVk(float size, int seg, float offX, float offY, float offZ, int faceIdx,
    float edgeRadius[4], float r, float g, float b, std::vector<VkVert>& verts, std::vector<uint32_t>& inds)
{
    uint32_t base = (uint32_t)verts.size();
    float h = size / 2;

    float faceN[6][3] = {{0,0,1},{0,0,-1},{1,0,0},{-1,0,0},{0,1,0},{0,-1,0}};
    float faceU[6][3] = {{-1,0,0},{1,0,0},{0,0,1},{0,0,-1},{1,0,0},{1,0,0}};
    float faceV[6][3] = {{0,1,0},{0,1,0},{0,1,0},{0,1,0},{0,0,1},{0,0,-1}};

    float fnx = faceN[faceIdx][0], fny = faceN[faceIdx][1], fnz = faceN[faceIdx][2];
    float fux = faceU[faceIdx][0], fuy = faceU[faceIdx][1], fuz = faceU[faceIdx][2];
    float fvx = faceV[faceIdx][0], fvy = faceV[faceIdx][1], fvz = faceV[faceIdx][2];

    for (int j = 0; j <= seg; j++) {
        for (int i = 0; i <= seg; i++) {
            float u = (float)i / seg * 2 - 1;
            float vv = (float)j / seg * 2 - 1;

            float px = u * h, py = vv * h;
            float pz = h;
            float nx = 0, ny = 0, nz = 1;

            float rU_raw = (u > 0) ? edgeRadius[0] : edgeRadius[1];
            float rV_raw = (vv > 0) ? edgeRadius[2] : edgeRadius[3];
            float rU = fabsf(rU_raw), rV = fabsf(rV_raw);
            bool outerU = (rU_raw > 0), outerV = (rV_raw > 0);

            if (rU > 0 || rV > 0) {
                float innerU = h - rU, innerV = h - rV;
                float dx = (rU > 0) ? fmaxf(0, fabsf(px) - innerU) : 0;
                float dy = (rV > 0) ? fmaxf(0, fabsf(py) - innerV) : 0;

                if (dx > 0 || dy > 0) {
                    bool isCorner = (dx > 0 && dy > 0);
                    bool sphericalCorner = isCorner && (outerU || outerV);

                    if (sphericalCorner) {
                        float rad = fmaxf(rU, rV);
                        float dist = sqrtf(dx*dx + dy*dy);
                        if (dist > rad) { dx = dx * rad / dist; dy = dy * rad / dist; }
                        float curveZ = sqrtf(fmaxf(0, rad*rad - dx*dx - dy*dy));
                        pz = (h - rad) + curveZ;
                        px = (u > 0 ? 1 : -1) * (innerU + dx);
                        py = (vv > 0 ? 1 : -1) * (innerV + dy);
                        nx = (u > 0 ? 1 : -1) * dx / rad;
                        ny = (vv > 0 ? 1 : -1) * dy / rad;
                        nz = curveZ / rad;
                    } else if (isCorner) {
                        if (dx >= dy) {
                            float curveZ = sqrtf(fmaxf(0, rU*rU - dx*dx));
                            pz = (h - rU) + curveZ;
                            px = (u > 0 ? 1 : -1) * (innerU + dx);
                            nx = (u > 0 ? 1 : -1) * dx / rU;
                            nz = curveZ / rU;
                        } else {
                            float curveZ = sqrtf(fmaxf(0, rV*rV - dy*dy));
                            pz = (h - rV) + curveZ;
                            py = (vv > 0 ? 1 : -1) * (innerV + dy);
                            ny = (vv > 0 ? 1 : -1) * dy / rV;
                            nz = curveZ / rV;
                        }
                    } else {
                        float rad = (dx > 0) ? rU : rV;
                        float d = (dx > 0) ? dx : dy;
                        float curveZ = sqrtf(fmaxf(0, rad*rad - d*d));
                        pz = (h - rad) + curveZ;
                        if (dx > 0) { px = (u > 0 ? 1 : -1) * (innerU + dx); nx = (u > 0 ? 1 : -1) * dx / rad; }
                        else { py = (vv > 0 ? 1 : -1) * (innerV + dy); ny = (vv > 0 ? 1 : -1) * dy / rad; }
                        nz = curveZ / rad;
                    }
                }
            }

            VkVert vert;
            vert.px = offX + px*fux + py*fvx + pz*fnx;
            vert.py = offY + px*fuy + py*fvy + pz*fny;
            vert.pz = offZ + px*fuz + py*fvz + pz*fnz;

            float nnx = nx*fux + ny*fvx + nz*fnx;
            float nny = nx*fuy + ny*fvy + nz*fny;
            float nnz = nx*fuz + ny*fvz + nz*fnz;
            float len = sqrtf(nnx*nnx + nny*nny + nnz*nnz);
            if (len < 0.001f) len = 1;
            vert.nx = nnx/len;
            vert.ny = nny/len;
            vert.nz = nnz/len;
            vert.r = r; vert.g = g; vert.b = b;
            verts.push_back(vert);
        }
    }

    for (int j = 0; j < seg; j++) {
        for (int i = 0; i < seg; i++) {
            uint32_t idx = base + j * (seg + 1) + i;
            inds.push_back(idx); inds.push_back(idx + seg + 1); inds.push_back(idx + 1);
            inds.push_back(idx + 1); inds.push_back(idx + seg + 1); inds.push_back(idx + seg + 2);
        }
    }
}

void BuildCubeGeometryVk(int cubeID, float r, float g, float b, std::vector<VkVert>& verts, std::vector<uint32_t>& inds)
{
    float cubeSize = 0.95f;
    float outerR = 0.12f, innerR = -0.12f;
    float half = cubeSize / 2;
    int seg = 20;

    int coords[8][3] = {
        {-1, +1, +1}, {+1, +1, +1}, {-1, -1, +1}, {+1, -1, +1},
        {-1, +1, -1}, {+1, +1, -1}, {-1, -1, -1}, {+1, -1, -1},
    };

    int cx = coords[cubeID][0], cy = coords[cubeID][1], cz = coords[cubeID][2];
    float posX = cx * half, posY = cy * half, posZ = cz * half;

    bool renderFace[6] = {(cz > 0), (cz < 0), (cx > 0), (cx < 0), (cy > 0), (cy < 0)};

    for (int f = 0; f < 6; f++) {
        if (!renderFace[f]) continue;

        float er[4];
        switch (f) {
            case 0: er[0] = (cx < 0) ? outerR : innerR; er[1] = (cx > 0) ? outerR : innerR;
                    er[2] = (cy > 0) ? outerR : innerR; er[3] = (cy < 0) ? outerR : innerR; break;
            case 1: er[0] = (cx > 0) ? outerR : innerR; er[1] = (cx < 0) ? outerR : innerR;
                    er[2] = (cy > 0) ? outerR : innerR; er[3] = (cy < 0) ? outerR : innerR; break;
            case 2: er[0] = (cz > 0) ? outerR : innerR; er[1] = (cz < 0) ? outerR : innerR;
                    er[2] = (cy > 0) ? outerR : innerR; er[3] = (cy < 0) ? outerR : innerR; break;
            case 3: er[0] = (cz < 0) ? outerR : innerR; er[1] = (cz > 0) ? outerR : innerR;
                    er[2] = (cy > 0) ? outerR : innerR; er[3] = (cy < 0) ? outerR : innerR; break;
            case 4: er[0] = (cx > 0) ? outerR : innerR; er[1] = (cx < 0) ? outerR : innerR;
                    er[2] = (cz > 0) ? outerR : innerR; er[3] = (cz < 0) ? outerR : innerR; break;
            case 5: er[0] = (cx > 0) ? outerR : innerR; er[1] = (cx < 0) ? outerR : innerR;
                    er[2] = (cz < 0) ? outerR : innerR; er[3] = (cz > 0) ? outerR : innerR; break;
        }
        GenRoundedFaceVk(cubeSize, seg, posX, posY, posZ, f, er, r, g, b, verts, inds);
    }
}

// ============== VULKAN TEXT RENDERING ==============

// Initialize Vulkan text rendering with minimal CPU-GPU interaction
bool InitVulkanText() {
    Log("[INFO] Initializing Vulkan text rendering...\n");

    // Create font texture (128x64, 16 chars x 6 rows of 8x8 characters = 96 chars)
    const int FONT_TEX_W = 128;  // 16 chars * 8 pixels
    const int FONT_TEX_H = 48;   // 6 rows * 8 pixels

    // Build font texture data (RGBA) - same bit order as D3D11
    std::vector<uint8_t> fontData(FONT_TEX_W * FONT_TEX_H * 4, 0);
    for (int charIdx = 0; charIdx < 96; charIdx++) {
        int col = charIdx % 16;
        int row = charIdx / 16;
        for (int y = 0; y < 8; y++) {
            unsigned char bits = g_font8x8[charIdx][y];
            for (int x = 0; x < 8; x++) {
                int px = col * 8 + x;
                int py = row * 8 + y;
                int idx = (py * FONT_TEX_W + px) * 4;
                uint8_t val = (bits & (0x80 >> x)) ? 255 : 0;
                // DEBUG: Use alpha in red channel, set alpha to 255 for all pixels
                // This makes glyphs appear as colored rectangles
                fontData[idx + 0] = val;   // R = glyph (white on black)
                fontData[idx + 1] = val;   // G = glyph
                fontData[idx + 2] = val;   // B = glyph
                fontData[idx + 3] = 255;   // A = fully opaque (DEBUG)
            }
        }
    }

    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    VkDeviceSize imageSize = FONT_TEX_W * FONT_TEX_H * 4;

    if (!VkCreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        stagingBuffer, stagingMemory)) {
        Log("[ERROR] Failed to create staging buffer for font\n");
        return false;
    }

    void* data;
    vkMapMemory(g_vkDevice, stagingMemory, 0, imageSize, 0, &data);
    memcpy(data, fontData.data(), imageSize);
    vkUnmapMemory(g_vkDevice, stagingMemory);

    // Create font image
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = FONT_TEX_W;
    imageInfo.extent.height = FONT_TEX_H;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(g_vkDevice, &imageInfo, nullptr, &g_vkFontImage) != VK_SUCCESS) {
        Log("[ERROR] Failed to create font image\n");
        vkDestroyBuffer(g_vkDevice, stagingBuffer, nullptr);
        vkFreeMemory(g_vkDevice, stagingMemory, nullptr);
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(g_vkDevice, g_vkFontImage, &memReqs);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = VkFindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(g_vkDevice, &allocInfo, nullptr, &g_vkFontImageMemory) != VK_SUCCESS) {
        Log("[ERROR] Failed to allocate font image memory\n");
        vkDestroyImage(g_vkDevice, g_vkFontImage, nullptr);
        vkDestroyBuffer(g_vkDevice, stagingBuffer, nullptr);
        vkFreeMemory(g_vkDevice, stagingMemory, nullptr);
        return false;
    }
    vkBindImageMemory(g_vkDevice, g_vkFontImage, g_vkFontImageMemory, 0);

    // Copy staging buffer to image using a one-time command buffer
    VkCommandBuffer cmdBuf;
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandPool = g_vkCommandPool;
    cmdAllocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_vkDevice, &cmdAllocInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // Transition image to transfer destination
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = g_vkFontImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy buffer to image
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {FONT_TEX_W, FONT_TEX_H, 1};

    vkCmdCopyBufferToImage(cmdBuf, stagingBuffer, g_vkFontImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to shader read
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    vkQueueSubmit(g_vkGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(g_vkGraphicsQueue);
    vkFreeCommandBuffers(g_vkDevice, g_vkCommandPool, 1, &cmdBuf);

    // Cleanup staging buffer
    vkDestroyBuffer(g_vkDevice, stagingBuffer, nullptr);
    vkFreeMemory(g_vkDevice, stagingMemory, nullptr);

    // Create image view
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = g_vkFontImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_vkFontImageView) != VK_SUCCESS) {
        Log("[ERROR] Failed to create font image view\n");
        return false;
    }

    // Create sampler (nearest-neighbor for crisp pixel font)
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(g_vkDevice, &samplerInfo, nullptr, &g_vkFontSampler) != VK_SUCCESS) {
        Log("[ERROR] Failed to create font sampler\n");
        return false;
    }

    // Create descriptor set layout with font texture binding
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    if (vkCreateDescriptorSetLayout(g_vkDevice, &layoutInfo, nullptr, &g_vkTextDescSetLayout) != VK_SUCCESS) {
        Log("[ERROR] Failed to create text descriptor set layout\n");
        return false;
    }

    // Create descriptor pool
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(g_vkDevice, &poolInfo, nullptr, &g_vkTextDescPool) != VK_SUCCESS) {
        Log("[ERROR] Failed to create text descriptor pool\n");
        return false;
    }

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo descAllocInfo = {};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = g_vkTextDescPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &g_vkTextDescSetLayout;

    if (vkAllocateDescriptorSets(g_vkDevice, &descAllocInfo, &g_vkTextDescSet) != VK_SUCCESS) {
        Log("[ERROR] Failed to allocate text descriptor set\n");
        return false;
    }

    // Update descriptor set with font texture
    VkDescriptorImageInfo imageInfoDesc = {};
    imageInfoDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfoDesc.imageView = g_vkFontImageView;
    imageInfoDesc.sampler = g_vkFontSampler;

    VkWriteDescriptorSet descWrite = {};
    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descWrite.dstSet = g_vkTextDescSet;
    descWrite.dstBinding = 0;
    descWrite.dstArrayElement = 0;
    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descWrite.descriptorCount = 1;
    descWrite.pImageInfo = &imageInfoDesc;

    vkUpdateDescriptorSets(g_vkDevice, 1, &descWrite, 0, nullptr);

    // Create text pipeline layout with descriptor set
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &g_vkTextDescSetLayout;

    if (vkCreatePipelineLayout(g_vkDevice, &pipelineLayoutInfo, nullptr, &g_vkTextPipelineLayout) != VK_SUCCESS) {
        Log("[ERROR] Failed to create text pipeline layout\n");
        return false;
    }

    // Create text shaders
    VkShaderModule textVertShader = VkCreateShaderModule(g_vkTextVertShaderCode, sizeof(g_vkTextVertShaderCode));
    VkShaderModule textFragShader = VkCreateShaderModule(g_vkTextFragShaderCode, sizeof(g_vkTextFragShaderCode));

    if (!textVertShader || !textFragShader) {
        Log("[ERROR] Failed to create text shader modules\n");
        if (textVertShader) vkDestroyShaderModule(g_vkDevice, textVertShader, nullptr);
        if (textFragShader) vkDestroyShaderModule(g_vkDevice, textFragShader, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStages[2] = {};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = textVertShader;
    shaderStages[0].pName = "main";
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = textFragShader;
    shaderStages[1].pName = "main";

    // Vertex input: pos(vec2), uv(vec2), color(vec4)
    VkVertexInputBindingDescription bindingDesc = {};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(VkTextVert);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[3] = {};
    attrDescs[0].binding = 0;
    attrDescs[0].location = 0;
    attrDescs[0].format = VK_FORMAT_R32G32_SFLOAT;  // pos
    attrDescs[0].offset = offsetof(VkTextVert, x);
    attrDescs[1].binding = 0;
    attrDescs[1].location = 1;
    attrDescs[1].format = VK_FORMAT_R32G32_SFLOAT;  // uv
    attrDescs[1].offset = offsetof(VkTextVert, u);
    attrDescs[2].binding = 0;
    attrDescs[2].location = 2;
    attrDescs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;  // color
    attrDescs[2].offset = offsetof(VkTextVert, r);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 3;
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)g_vkSwapchainExtent.width;
    viewport.height = (float)g_vkSwapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = g_vkSwapchainExtent;

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;  // No culling for 2D text
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;  // No depth test for 2D overlay
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    // Alpha blending for text
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    // NO dynamic state - use static viewport/scissor like 3D pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;  // Static viewport/scissor
    pipelineInfo.layout = g_vkTextPipelineLayout;
    pipelineInfo.renderPass = g_vkRenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(g_vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &g_vkTextPipeline) != VK_SUCCESS) {
        Log("[ERROR] Failed to create text pipeline\n");
        vkDestroyShaderModule(g_vkDevice, textVertShader, nullptr);
        vkDestroyShaderModule(g_vkDevice, textFragShader, nullptr);
        return false;
    }

    vkDestroyShaderModule(g_vkDevice, textVertShader, nullptr);
    vkDestroyShaderModule(g_vkDevice, textFragShader, nullptr);

    // Create persistently mapped text vertex buffer (6 verts per char * max chars)
    VkDeviceSize textBufferSize = sizeof(VkTextVert) * 6 * g_vkMaxTextChars;
    if (!VkCreateBuffer(textBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        g_vkTextVertexBuffer, g_vkTextVertexBufferMemory)) {
        Log("[ERROR] Failed to create text vertex buffer\n");
        return false;
    }

    // Persistently map for minimal CPU-GPU sync
    VkResult mapResult = vkMapMemory(g_vkDevice, g_vkTextVertexBufferMemory, 0, textBufferSize, 0, &g_vkTextVertexBufferMapped);
    if (mapResult != VK_SUCCESS || !g_vkTextVertexBufferMapped) {
        Log("[ERROR] Failed to map text vertex buffer memory: %d\n", mapResult);
        return false;
    }

    Log("[INFO] Vulkan text rendering initialized successfully\n");
    Log("[INFO] Text pipeline: %p, Text VB mapped: %p\n", (void*)g_vkTextPipeline, g_vkTextVertexBufferMapped);
    return true;
}

// Helper to draw text using Vulkan (generates vertices into mapped buffer, returns vertex count)
static int VkBuildTextVertices(const char* text, float x, float y, float r, float g, float b, float a, float scale) {
    const int FONT_COLS = 16;
    const float CHAR_W = 8.0f * scale;
    const float CHAR_H = 8.0f * scale;
    const float TEX_W = 128.0f, TEX_H = 48.0f;

    // Convert pixel coordinates to NDC [-1, 1]
    float ndcScaleX = 2.0f / (float)g_vkSwapchainExtent.width;
    float ndcScaleY = 2.0f / (float)g_vkSwapchainExtent.height;

    VkTextVert* verts = (VkTextVert*)g_vkTextVertexBufferMapped;
    int vertCount = 0;
    float cx = x, cy = y;

    for (const char* p = text; *p && vertCount < g_vkMaxTextChars * 6 - 6; p++) {
        if (*p == '\n') { cx = x; cy += CHAR_H * 1.4f; continue; }
        if (*p < 32 || *p > 127) continue;

        int idx = *p - 32;
        int col = idx % FONT_COLS, row = idx / FONT_COLS;
        float u0 = col * 8.0f / TEX_W, v0 = row * 8.0f / TEX_H;
        float u1 = u0 + 8.0f / TEX_W, v1 = v0 + 8.0f / TEX_H;

        // Convert pixel coords to NDC (Y is flipped in Vulkan, so we go from top)
        float x0 = cx * ndcScaleX - 1.0f;
        float y0 = cy * ndcScaleY - 1.0f;
        float x1 = (cx + CHAR_W) * ndcScaleX - 1.0f;
        float y1 = (cy + CHAR_H) * ndcScaleY - 1.0f;

        // Two triangles (6 vertices)
        verts[vertCount++] = {x0, y0, u0, v0, r, g, b, a};
        verts[vertCount++] = {x1, y0, u1, v0, r, g, b, a};
        verts[vertCount++] = {x0, y1, u0, v1, r, g, b, a};
        verts[vertCount++] = {x1, y0, u1, v0, r, g, b, a};
        verts[vertCount++] = {x1, y1, u1, v1, r, g, b, a};
        verts[vertCount++] = {x0, y1, u0, v1, r, g, b, a};

        cx += CHAR_W;
    }

    return vertCount;
}

// ============== MAIN VULKAN FUNCTIONS ==============

static bool g_vkFirstFrame = true;
// g_vkTextInitialized is defined in main.cpp (declared extern in renderer_vulkan.h)

bool InitVulkan(HWND hwnd)
{
    Log("[INFO] Initializing Vulkan...\n");

    // Create instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RenderTestGPU";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    const char* extensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME
    };

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = 2;
    createInfo.ppEnabledExtensionNames = extensions;

    if (vkCreateInstance(&createInfo, nullptr, &g_vkInstance) != VK_SUCCESS) {
        Log("[ERROR] Failed to create Vulkan instance\n");
        return false;
    }
    Log("[INFO] Vulkan instance created\n");

    // Create surface
    VkWin32SurfaceCreateInfoKHR surfaceInfo = {};
    surfaceInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfaceInfo.hwnd = hwnd;
    surfaceInfo.hinstance = GetModuleHandle(nullptr);

    if (vkCreateWin32SurfaceKHR(g_vkInstance, &surfaceInfo, nullptr, &g_vkSurface) != VK_SUCCESS) {
        Log("[ERROR] Failed to create Vulkan surface\n");
        return false;
    }
    Log("[INFO] Vulkan surface created\n");

    // Pick physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(g_vkInstance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        Log("[ERROR] No Vulkan-capable GPU found\n");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(g_vkInstance, &deviceCount, devices.data());

    // Select first suitable device
    for (auto device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        uint32_t graphicsFamily = UINT32_MAX, presentFamily = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphicsFamily = i;
            }
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, g_vkSurface, &presentSupport);
            if (presentSupport) {
                presentFamily = i;
            }
        }

        if (graphicsFamily != UINT32_MAX && presentFamily != UINT32_MAX) {
            g_vkPhysicalDevice = device;
            g_vkGraphicsFamily = graphicsFamily;
            g_vkPresentFamily = presentFamily;
            g_vkGpuName = props.deviceName;
            Log("[INFO] Selected GPU: %s\n", props.deviceName);
            break;
        }
    }

    if (g_vkPhysicalDevice == VK_NULL_HANDLE) {
        Log("[ERROR] No suitable GPU found\n");
        return false;
    }

    // Create logical device
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = g_vkGraphicsFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);

    if (g_vkPresentFamily != g_vkGraphicsFamily) {
        queueCreateInfo.queueFamilyIndex = g_vkPresentFamily;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkPhysicalDeviceFeatures deviceFeatures = {};

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
    deviceCreateInfo.enabledExtensionCount = 1;
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;

    if (vkCreateDevice(g_vkPhysicalDevice, &deviceCreateInfo, nullptr, &g_vkDevice) != VK_SUCCESS) {
        Log("[ERROR] Failed to create logical device\n");
        return false;
    }

    vkGetDeviceQueue(g_vkDevice, g_vkGraphicsFamily, 0, &g_vkGraphicsQueue);
    vkGetDeviceQueue(g_vkDevice, g_vkPresentFamily, 0, &g_vkPresentQueue);
    Log("[INFO] Vulkan device created\n");

    // Create swapchain
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vkPhysicalDevice, g_vkSurface, &capabilities);

    g_vkSwapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;
    g_vkSwapchainExtent = { W, H };

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    // Select best present mode for no VSync (MAILBOX > IMMEDIATE > FIFO)
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_vkPhysicalDevice, g_vkSurface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_vkPhysicalDevice, g_vkSurface, &presentModeCount, presentModes.data());

    VkPresentModeKHR selectedPresentMode = VK_PRESENT_MODE_FIFO_KHR;  // Always available fallback
    for (auto mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            selectedPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;  // Best option - triple buffering, no VSync, no tearing
        }
        if (mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            selectedPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;  // Good - no VSync
        }
    }
    Log("[INFO] Present mode: %s\n",
        selectedPresentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX (no VSync)" :
        selectedPresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "IMMEDIATE (no VSync)" : "FIFO (VSync)");

    VkSwapchainCreateInfoKHR swapchainInfo = {};
    swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainInfo.surface = g_vkSurface;
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = g_vkSwapchainFormat;
    swapchainInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchainInfo.imageExtent = g_vkSwapchainExtent;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchainInfo.preTransform = capabilities.currentTransform;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainInfo.presentMode = selectedPresentMode;
    swapchainInfo.clipped = VK_TRUE;

    if (g_vkGraphicsFamily != g_vkPresentFamily) {
        uint32_t queueFamilyIndices[] = { g_vkGraphicsFamily, g_vkPresentFamily };
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainInfo.queueFamilyIndexCount = 2;
        swapchainInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    if (vkCreateSwapchainKHR(g_vkDevice, &swapchainInfo, nullptr, &g_vkSwapchain) != VK_SUCCESS) {
        Log("[ERROR] Failed to create swapchain\n");
        return false;
    }

    vkGetSwapchainImagesKHR(g_vkDevice, g_vkSwapchain, &imageCount, nullptr);
    g_vkSwapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(g_vkDevice, g_vkSwapchain, &imageCount, g_vkSwapchainImages.data());
    Log("[INFO] Swapchain created with %u images\n", imageCount);

    // Create image views
    g_vkSwapchainImageViews.resize(g_vkSwapchainImages.size());
    for (size_t i = 0; i < g_vkSwapchainImages.size(); i++) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = g_vkSwapchainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = g_vkSwapchainFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        if (vkCreateImageView(g_vkDevice, &viewInfo, nullptr, &g_vkSwapchainImageViews[i]) != VK_SUCCESS) {
            Log("[ERROR] Failed to create image view %zu\n", i);
            return false;
        }
    }

    // Create depth buffer
    VkImageCreateInfo depthImageInfo = {};
    depthImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depthImageInfo.imageType = VK_IMAGE_TYPE_2D;
    depthImageInfo.extent.width = W;
    depthImageInfo.extent.height = H;
    depthImageInfo.extent.depth = 1;
    depthImageInfo.mipLevels = 1;
    depthImageInfo.arrayLayers = 1;
    depthImageInfo.format = VK_FORMAT_D32_SFLOAT;
    depthImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthImageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(g_vkDevice, &depthImageInfo, nullptr, &g_vkDepthImage) != VK_SUCCESS) {
        Log("[ERROR] Failed to create depth image\n");
        return false;
    }

    VkMemoryRequirements depthMemReqs;
    vkGetImageMemoryRequirements(g_vkDevice, g_vkDepthImage, &depthMemReqs);

    VkMemoryAllocateInfo depthAllocInfo = {};
    depthAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    depthAllocInfo.allocationSize = depthMemReqs.size;
    depthAllocInfo.memoryTypeIndex = VkFindMemoryType(depthMemReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(g_vkDevice, &depthAllocInfo, nullptr, &g_vkDepthImageMemory) != VK_SUCCESS) {
        Log("[ERROR] Failed to allocate depth image memory\n");
        return false;
    }
    vkBindImageMemory(g_vkDevice, g_vkDepthImage, g_vkDepthImageMemory, 0);

    VkImageViewCreateInfo depthViewInfo = {};
    depthViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthViewInfo.image = g_vkDepthImage;
    depthViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthViewInfo.format = VK_FORMAT_D32_SFLOAT;
    depthViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthViewInfo.subresourceRange.baseMipLevel = 0;
    depthViewInfo.subresourceRange.levelCount = 1;
    depthViewInfo.subresourceRange.baseArrayLayer = 0;
    depthViewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(g_vkDevice, &depthViewInfo, nullptr, &g_vkDepthImageView) != VK_SUCCESS) {
        Log("[ERROR] Failed to create depth image view\n");
        return false;
    }
    Log("[INFO] Depth buffer created\n");

    // Create render pass
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = g_vkSwapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef = { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = { colorAttachment, depthAttachment };

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 2;
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(g_vkDevice, &renderPassInfo, nullptr, &g_vkRenderPass) != VK_SUCCESS) {
        Log("[ERROR] Failed to create render pass\n");
        return false;
    }
    Log("[INFO] Render pass created\n");

    // Create pipeline
    VkShaderModule vertModule = VkCreateShaderModule(g_vkVertShaderCode, sizeof(g_vkVertShaderCode));
    VkShaderModule fragModule = VkCreateShaderModule(g_vkFragShaderCode, sizeof(g_vkFragShaderCode));

    if (!vertModule || !fragModule) {
        Log("[ERROR] Failed to create shader modules\n");
        return false;
    }

    VkPipelineShaderStageCreateInfo shaderStages[2] = {};
    shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shaderStages[0].module = vertModule;
    shaderStages[0].pName = "main";
    shaderStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderStages[1].module = fragModule;
    shaderStages[1].pName = "main";

    VkVertexInputBindingDescription bindingDesc = {};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(VkVert);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[3] = {};
    attrDescs[0].binding = 0;
    attrDescs[0].location = 0;
    attrDescs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[0].offset = offsetof(VkVert, px);
    attrDescs[1].binding = 0;
    attrDescs[1].location = 1;
    attrDescs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[1].offset = offsetof(VkVert, nx);
    attrDescs[2].binding = 0;
    attrDescs[2].location = 2;
    attrDescs[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    attrDescs[2].offset = offsetof(VkVert, r);

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDesc;
    vertexInputInfo.vertexAttributeDescriptionCount = 3;
    vertexInputInfo.pVertexAttributeDescriptions = attrDescs;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport = { 0, 0, (float)W, (float)H, 0, 1 };
    VkRect2D scissor = { {0, 0}, {W, H} };

    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    // Counter-clockwise is standard front face for right-handed coordinates
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(VkPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(g_vkDevice, &pipelineLayoutInfo, nullptr, &g_vkPipelineLayout) != VK_SUCCESS) {
        Log("[ERROR] Failed to create pipeline layout\n");
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = g_vkPipelineLayout;
    pipelineInfo.renderPass = g_vkRenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(g_vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &g_vkPipeline) != VK_SUCCESS) {
        Log("[ERROR] Failed to create graphics pipeline\n");
        return false;
    }

    vkDestroyShaderModule(g_vkDevice, vertModule, nullptr);
    vkDestroyShaderModule(g_vkDevice, fragModule, nullptr);
    Log("[INFO] Pipeline created\n");

    // Create framebuffers
    g_vkFramebuffers.resize(g_vkSwapchainImageViews.size());
    for (size_t i = 0; i < g_vkSwapchainImageViews.size(); i++) {
        VkImageView attachments[] = { g_vkSwapchainImageViews[i], g_vkDepthImageView };

        VkFramebufferCreateInfo fbInfo = {};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = g_vkRenderPass;
        fbInfo.attachmentCount = 2;
        fbInfo.pAttachments = attachments;
        fbInfo.width = W;
        fbInfo.height = H;
        fbInfo.layers = 1;

        if (vkCreateFramebuffer(g_vkDevice, &fbInfo, nullptr, &g_vkFramebuffers[i]) != VK_SUCCESS) {
            Log("[ERROR] Failed to create framebuffer %zu\n", i);
            return false;
        }
    }
    Log("[INFO] Framebuffers created\n");

    // Create command pool and buffers
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = g_vkGraphicsFamily;

    if (vkCreateCommandPool(g_vkDevice, &poolInfo, nullptr, &g_vkCommandPool) != VK_SUCCESS) {
        Log("[ERROR] Failed to create command pool\n");
        return false;
    }

    g_vkCommandBuffers.resize(g_vkFramebuffers.size());
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = g_vkCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)g_vkCommandBuffers.size();

    if (vkAllocateCommandBuffers(g_vkDevice, &allocInfo, g_vkCommandBuffers.data()) != VK_SUCCESS) {
        Log("[ERROR] Failed to allocate command buffers\n");
        return false;
    }
    Log("[INFO] Command buffers created\n");

    // Create sync objects
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateSemaphore(g_vkDevice, &semaphoreInfo, nullptr, &g_vkImageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(g_vkDevice, &semaphoreInfo, nullptr, &g_vkRenderFinishedSemaphore) != VK_SUCCESS ||
        vkCreateFence(g_vkDevice, &fenceInfo, nullptr, &g_vkInFlightFence) != VK_SUCCESS) {
        Log("[ERROR] Failed to create sync objects\n");
        return false;
    }
    Log("[INFO] Sync objects created\n");

    // Create vertex and index buffers
    Log("[INFO] Building Vulkan cube geometry...\n");
    std::vector<VkVert> vertices;
    std::vector<uint32_t> indices;

    const float colors[8][3] = {
        {0.95f, 0.20f, 0.15f}, {0.20f, 0.70f, 0.30f}, {0.15f, 0.50f, 0.95f}, {1.00f, 0.85f, 0.00f},
        {1.00f, 0.85f, 0.00f}, {0.15f, 0.50f, 0.95f}, {0.20f, 0.70f, 0.30f}, {0.95f, 0.20f, 0.15f}
    };

    for (int c = 0; c < 8; c++) {
        BuildCubeGeometryVk(c, colors[c][0], colors[c][1], colors[c][2], vertices, indices);
    }

    g_vkIndexCount = (uint32_t)indices.size();
    g_vkTriangleCount = g_vkIndexCount / 3;
    Log("[INFO] Vulkan geometry: %d vertices, %d indices (%d triangles)\n",
        (int)vertices.size(), (int)indices.size(), g_vkTriangleCount);

    VkDeviceSize vertexBufferSize = sizeof(VkVert) * vertices.size();
    VkDeviceSize indexBufferSize = sizeof(uint32_t) * indices.size();

    if (!VkCreateBuffer(vertexBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        g_vkVertexBuffer, g_vkVertexBufferMemory)) {
        Log("[ERROR] Failed to create vertex buffer\n");
        return false;
    }

    void* data;
    vkMapMemory(g_vkDevice, g_vkVertexBufferMemory, 0, vertexBufferSize, 0, &data);
    memcpy(data, vertices.data(), vertexBufferSize);
    vkUnmapMemory(g_vkDevice, g_vkVertexBufferMemory);

    if (!VkCreateBuffer(indexBufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        g_vkIndexBuffer, g_vkIndexBufferMemory)) {
        Log("[ERROR] Failed to create index buffer\n");
        return false;
    }

    vkMapMemory(g_vkDevice, g_vkIndexBufferMemory, 0, indexBufferSize, 0, &data);
    memcpy(data, indices.data(), indexBufferSize);
    vkUnmapMemory(g_vkDevice, g_vkIndexBufferMemory);

    Log("[INFO] Vulkan buffers created\n");
    Log("[INFO] Vulkan initialization complete\n");
    return true;
}

void RenderVulkan()
{
    VkResult result;

    result = vkWaitForFences(g_vkDevice, 1, &g_vkInFlightFence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS && g_vkFirstFrame) Log("[VK ERROR] vkWaitForFences: %d\n", result);

    result = vkResetFences(g_vkDevice, 1, &g_vkInFlightFence);
    if (result != VK_SUCCESS && g_vkFirstFrame) Log("[VK ERROR] vkResetFences: %d\n", result);

    uint32_t imageIndex;
    result = vkAcquireNextImageKHR(g_vkDevice, g_vkSwapchain, UINT64_MAX, g_vkImageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
    if (result != VK_SUCCESS && g_vkFirstFrame) Log("[VK ERROR] vkAcquireNextImageKHR: %d\n", result);

    // Get time
    LARGE_INTEGER nowTime;
    QueryPerformanceCounter(&nowTime);
    float t = (float)(nowTime.QuadPart - g_startTime.QuadPart) / g_perfFreq.QuadPart;

    // Build matrices in COLUMN-MAJOR format for GLSL (mat4 * vec4)
    // In column-major, each column is stored consecutively
    // mat[col][row] -> array[col * 4 + row]

    float aspect = (float)W / (float)H;
    float fov = 45.0f * 3.14159265f / 180.0f;
    float nearZ = 0.1f, farZ = 100.0f;
    float tanHalfFov = tanf(fov / 2.0f);
    float f = 1.0f / tanHalfFov;

    // Vulkan perspective matrix (column-major, Y-flip, Z [0,1])
    // Column 0: (f/aspect, 0, 0, 0)
    // Column 1: (0, -f, 0, 0)  -- negative for Y-flip
    // Column 2: (0, 0, far/(near-far), -1)
    // Column 3: (0, 0, near*far/(near-far), 0)
    float proj[16] = {
        f / aspect, 0.0f, 0.0f, 0.0f,           // column 0
        0.0f, -f, 0.0f, 0.0f,                   // column 1 (Y-flip)
        0.0f, 0.0f, farZ / (nearZ - farZ), -1.0f,  // column 2
        0.0f, 0.0f, (nearZ * farZ) / (nearZ - farZ), 0.0f  // column 3
    };

    // View matrix: camera at (0, 0, 4) looking at origin
    // Translation (0, 0, -4) in column 3
    float view[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,   // column 0
        0.0f, 1.0f, 0.0f, 0.0f,   // column 1
        0.0f, 0.0f, 1.0f, 0.0f,   // column 2
        0.0f, 0.0f, -4.0f, 1.0f   // column 3 (translation)
    };

    // Rotation matrices (column-major)
    float cy = cosf(t * 1.2f), sy = sinf(t * 1.2f);
    float cx = cosf(t * 0.7f), sx = sinf(t * 0.7f);

    // Rotation around Y axis (column-major)
    float rotY[16] = {
        cy, 0.0f, -sy, 0.0f,    // column 0
        0.0f, 1.0f, 0.0f, 0.0f, // column 1
        sy, 0.0f, cy, 0.0f,     // column 2
        0.0f, 0.0f, 0.0f, 1.0f  // column 3
    };

    // Rotation around X axis (column-major)
    float rotX[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,   // column 0
        0.0f, cx, sx, 0.0f,       // column 1
        0.0f, -sx, cx, 0.0f,      // column 2
        0.0f, 0.0f, 0.0f, 1.0f    // column 3
    };

    // Column-major matrix multiply: result = a * b
    // result[col][row] = sum(a[k][row] * b[col][k])
    auto matMulColMajor = [](float* out, const float* a, const float* b) {
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += a[k * 4 + row] * b[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
    };

    // MVP = Projection * View * RotX * RotY (right-to-left application)
    float rot[16], viewRot[16], mvp[16];
    matMulColMajor(rot, rotX, rotY);        // rot = rotX * rotY
    matMulColMajor(viewRot, view, rot);     // viewRot = view * rot
    matMulColMajor(mvp, proj, viewRot);     // mvp = proj * viewRot

    // Push constants (already in column-major, no transpose needed)
    VkPushConstants pc;
    memcpy(pc.mvp, mvp, sizeof(mvp));

    // Light direction in world space
    float lx = 0.2f, ly = 1.0f, lz = 0.3f;
    float llen = sqrtf(lx*lx + ly*ly + lz*lz);
    lx /= llen; ly /= llen; lz /= llen;

    // Transform light to object space (inverse rotation = transpose for orthogonal matrix)
    // rot is column-major, so transpose means: new_row = old_col
    // lightObj = rot^T * lightWorld
    float lightObjX = rot[0]*lx + rot[1]*ly + rot[2]*lz;   // row 0 of rot^T
    float lightObjY = rot[4]*lx + rot[5]*ly + rot[6]*lz;   // row 1 of rot^T
    float lightObjZ = rot[8]*lx + rot[9]*ly + rot[10]*lz;  // row 2 of rot^T

    pc.lightDir[0] = lightObjX; pc.lightDir[1] = lightObjY; pc.lightDir[2] = lightObjZ; pc.lightDir[3] = 0;
    pc.time = t;

    // Record command buffer
    VkCommandBuffer cmd = g_vkCommandBuffers[imageIndex];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkClearValue clearValues[2] = {};
    clearValues[0].color = {{0.5f, 0.5f, 0.5f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo = {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = g_vkRenderPass;
    renderPassBeginInfo.framebuffer = g_vkFramebuffers[imageIndex];
    renderPassBeginInfo.renderArea.extent = g_vkSwapchainExtent;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vkPipeline);

    VkBuffer vertexBuffers[] = { g_vkVertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(cmd, g_vkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdPushConstants(cmd, g_vkPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vkCmdDrawIndexed(cmd, g_vkIndexCount, 1, 0, 0, 0);

    // Render text overlay if initialized
    if (g_vkTextInitialized && g_vkTextPipeline && g_vkTextVertexBufferMapped) {
        // Calculate FPS
        static LARGE_INTEGER lastFpsTime = {0};
        static int frameCount = 0;
        static float fps = 0.0f;
        LARGE_INTEGER currentTime;
        QueryPerformanceCounter(&currentTime);
        frameCount++;
        double elapsed = (double)(currentTime.QuadPart - lastFpsTime.QuadPart) / g_perfFreq.QuadPart;
        if (elapsed >= 0.5) {
            fps = (float)(frameCount / elapsed);
            frameCount = 0;
            lastFpsTime = currentTime;
        }

        // Build text string (same format as D3D11/D3D12)
        char textBuf[512];
        snprintf(textBuf, sizeof(textBuf), "API: Vulkan\nGPU: %s\nFPS: %.0f\nTriangles: %d\nResolution: %ux%u",
                 g_vkGpuName.c_str(), fps, g_vkTriangleCount,
                 g_vkSwapchainExtent.width, g_vkSwapchainExtent.height);

        // Build vertices (shadow first, then text)
        int totalVerts = 0;
        VkTextVert* verts = (VkTextVert*)g_vkTextVertexBufferMapped;

        // Text styling (same as D3D11: scale 1.5, shadow offset)
        float scale = 1.5f;
        float shadowOff = 2.0f;
        const int FONT_COLS = 16;
        const float CHAR_W = 8.0f * scale;
        const float CHAR_H = 8.0f * scale;
        const float TEX_W = 128.0f, TEX_H = 48.0f;
        float ndcScaleX = 2.0f / (float)g_vkSwapchainExtent.width;
        float ndcScaleY = 2.0f / (float)g_vkSwapchainExtent.height;

        // Helper lambda to add text vertices
        auto addText = [&](const char* text, float startX, float startY, float r, float g, float b, float a) {
            float cx = startX, cy = startY;
            for (const char* p = text; *p && totalVerts < g_vkMaxTextChars * 6 - 6; p++) {
                if (*p == '\n') { cx = startX; cy += CHAR_H * 1.4f; continue; }
                if (*p < 32 || *p > 127) continue;

                int idx = *p - 32;
                int col = idx % FONT_COLS, row = idx / FONT_COLS;
                float u0 = col * 8.0f / TEX_W, v0 = row * 8.0f / TEX_H;
                float u1 = u0 + 8.0f / TEX_W, v1 = v0 + 8.0f / TEX_H;

                float x0 = cx * ndcScaleX - 1.0f;
                float y0 = cy * ndcScaleY - 1.0f;
                float x1 = (cx + CHAR_W) * ndcScaleX - 1.0f;
                float y1 = (cy + CHAR_H) * ndcScaleY - 1.0f;

                verts[totalVerts++] = {x0, y0, u0, v0, r, g, b, a};
                verts[totalVerts++] = {x1, y0, u1, v0, r, g, b, a};
                verts[totalVerts++] = {x0, y1, u0, v1, r, g, b, a};
                verts[totalVerts++] = {x1, y0, u1, v0, r, g, b, a};
                verts[totalVerts++] = {x1, y1, u1, v1, r, g, b, a};
                verts[totalVerts++] = {x0, y1, u0, v1, r, g, b, a};
                cx += CHAR_W;
            }
        };

        // Add shadow and main text (top-left corner like D3D11)
        float textX = 10.0f;
        float textY = 10.0f;
        addText(textBuf, textX + shadowOff, textY + shadowOff, 0.0f, 0.0f, 0.0f, 0.7f);  // Shadow (black, semi-transparent)
        addText(textBuf, textX, textY, 1.0f, 1.0f, 1.0f, 1.0f);  // White text

        if (totalVerts > 0) {
            // Render text in SAME render pass as 3D content (after 3D, before ending pass)
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vkTextPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_vkTextPipelineLayout,
                                    0, 1, &g_vkTextDescSet, 0, nullptr);

            VkBuffer textVBs[] = { g_vkTextVertexBuffer };
            VkDeviceSize textOffsets[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, textVBs, textOffsets);
            vkCmdDraw(cmd, totalVerts, 1, 0, 0);
        }
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);

    // Submit
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkSemaphore waitSemaphores[] = { g_vkImageAvailableSemaphore };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VkSemaphore signalSemaphores[] = { g_vkRenderFinishedSemaphore };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    result = vkQueueSubmit(g_vkGraphicsQueue, 1, &submitInfo, g_vkInFlightFence);
    if (result != VK_SUCCESS && g_vkFirstFrame) Log("[VK ERROR] vkQueueSubmit: %d\n", result);

    // Present
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &g_vkSwapchain;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(g_vkPresentQueue, &presentInfo);
    if (result != VK_SUCCESS && g_vkFirstFrame) Log("[VK ERROR] vkQueuePresentKHR: %d\n", result);

    if (g_vkFirstFrame) {
        Log("[VK DEBUG] First frame rendered. imageIndex=%u, indexCount=%u\n", imageIndex, g_vkIndexCount);
        g_vkFirstFrame = false;
    }

}

void CleanupVulkan()
{
    Log("[INFO] Cleaning up Vulkan...\n");

    if (g_vkDevice) vkDeviceWaitIdle(g_vkDevice);

    // Cleanup text rendering resources
    if (g_vkTextVertexBufferMapped) {
        vkUnmapMemory(g_vkDevice, g_vkTextVertexBufferMemory);
        g_vkTextVertexBufferMapped = nullptr;
    }
    if (g_vkTextVertexBuffer) { vkDestroyBuffer(g_vkDevice, g_vkTextVertexBuffer, nullptr); g_vkTextVertexBuffer = VK_NULL_HANDLE; }
    if (g_vkTextVertexBufferMemory) { vkFreeMemory(g_vkDevice, g_vkTextVertexBufferMemory, nullptr); g_vkTextVertexBufferMemory = VK_NULL_HANDLE; }
    if (g_vkTextPipeline) { vkDestroyPipeline(g_vkDevice, g_vkTextPipeline, nullptr); g_vkTextPipeline = VK_NULL_HANDLE; }
    if (g_vkTextPipelineLayout) { vkDestroyPipelineLayout(g_vkDevice, g_vkTextPipelineLayout, nullptr); g_vkTextPipelineLayout = VK_NULL_HANDLE; }
    if (g_vkTextDescPool) { vkDestroyDescriptorPool(g_vkDevice, g_vkTextDescPool, nullptr); g_vkTextDescPool = VK_NULL_HANDLE; }
    if (g_vkTextDescSetLayout) { vkDestroyDescriptorSetLayout(g_vkDevice, g_vkTextDescSetLayout, nullptr); g_vkTextDescSetLayout = VK_NULL_HANDLE; }
    if (g_vkFontSampler) { vkDestroySampler(g_vkDevice, g_vkFontSampler, nullptr); g_vkFontSampler = VK_NULL_HANDLE; }
    if (g_vkFontImageView) { vkDestroyImageView(g_vkDevice, g_vkFontImageView, nullptr); g_vkFontImageView = VK_NULL_HANDLE; }
    if (g_vkFontImage) { vkDestroyImage(g_vkDevice, g_vkFontImage, nullptr); g_vkFontImage = VK_NULL_HANDLE; }
    if (g_vkFontImageMemory) { vkFreeMemory(g_vkDevice, g_vkFontImageMemory, nullptr); g_vkFontImageMemory = VK_NULL_HANDLE; }
    g_vkTextInitialized = false;

    if (g_vkIndexBuffer) vkDestroyBuffer(g_vkDevice, g_vkIndexBuffer, nullptr);
    if (g_vkIndexBufferMemory) vkFreeMemory(g_vkDevice, g_vkIndexBufferMemory, nullptr);
    if (g_vkVertexBuffer) vkDestroyBuffer(g_vkDevice, g_vkVertexBuffer, nullptr);
    if (g_vkVertexBufferMemory) vkFreeMemory(g_vkDevice, g_vkVertexBufferMemory, nullptr);

    if (g_vkInFlightFence) vkDestroyFence(g_vkDevice, g_vkInFlightFence, nullptr);
    if (g_vkRenderFinishedSemaphore) vkDestroySemaphore(g_vkDevice, g_vkRenderFinishedSemaphore, nullptr);
    if (g_vkImageAvailableSemaphore) vkDestroySemaphore(g_vkDevice, g_vkImageAvailableSemaphore, nullptr);

    if (g_vkCommandPool) vkDestroyCommandPool(g_vkDevice, g_vkCommandPool, nullptr);

    for (auto fb : g_vkFramebuffers) if (fb) vkDestroyFramebuffer(g_vkDevice, fb, nullptr);
    g_vkFramebuffers.clear();

    if (g_vkPipeline) vkDestroyPipeline(g_vkDevice, g_vkPipeline, nullptr);
    if (g_vkPipelineLayout) vkDestroyPipelineLayout(g_vkDevice, g_vkPipelineLayout, nullptr);
    if (g_vkRenderPass) vkDestroyRenderPass(g_vkDevice, g_vkRenderPass, nullptr);

    if (g_vkDepthImageView) vkDestroyImageView(g_vkDevice, g_vkDepthImageView, nullptr);
    if (g_vkDepthImage) vkDestroyImage(g_vkDevice, g_vkDepthImage, nullptr);
    if (g_vkDepthImageMemory) vkFreeMemory(g_vkDevice, g_vkDepthImageMemory, nullptr);

    for (auto iv : g_vkSwapchainImageViews) if (iv) vkDestroyImageView(g_vkDevice, iv, nullptr);
    g_vkSwapchainImageViews.clear();

    if (g_vkSwapchain) vkDestroySwapchainKHR(g_vkDevice, g_vkSwapchain, nullptr);
    if (g_vkDevice) vkDestroyDevice(g_vkDevice, nullptr);
    if (g_vkSurface) vkDestroySurfaceKHR(g_vkInstance, g_vkSurface, nullptr);
    if (g_vkInstance) vkDestroyInstance(g_vkInstance, nullptr);

    g_vkInstance = VK_NULL_HANDLE;
    g_vkDevice = VK_NULL_HANDLE;
    g_vkSwapchain = VK_NULL_HANDLE;

    Log("[INFO] Vulkan cleanup complete\n");
}
