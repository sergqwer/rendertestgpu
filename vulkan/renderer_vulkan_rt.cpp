// ============== VULKAN RAY TRACING RENDERER ==============
// Uses VK_KHR_ray_tracing_pipeline extension (equivalent to DXR 1.0)
// Cornell Box scene with configurable features - same as DXR 1.0
// Requires: VK_KHR_acceleration_structure, VK_KHR_ray_tracing_pipeline,
//           VK_KHR_buffer_device_address, VK_KHR_deferred_host_operations

#define VK_USE_PLATFORM_WIN32_KHR
#include "vulkan.h"
#include "../common.h"
#include "renderer_vulkan_rt.h"
#include "vulkan_rt_shaders.h"
#include "vulkan_shaders.h"  // For text rendering shaders

#pragma comment(lib, "vulkan-1.lib")

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <set>

// Note: shaderc can be used for runtime GLSL->SPIR-V compilation
// but requires Vulkan SDK with shaderc_combined.lib
// For now, we use pre-compiled SPIR-V or show informational message

// External declarations
extern int fps;
extern std::wstring gpuName;
extern LARGE_INTEGER g_startTime;
extern LARGE_INTEGER g_perfFreq;
extern const unsigned char g_font8x8[96][8];
extern const UINT W;
extern const UINT H;

// Global feature flags (defined in common.h, instantiated here)
VulkanRTFeatures g_vulkanRTFeatures = {};

// ============== VULKAN RT FUNCTION POINTERS ==============
static PFN_vkGetBufferDeviceAddressKHR pvkGetBufferDeviceAddressKHR = nullptr;
static PFN_vkCreateAccelerationStructureKHR pvkCreateAccelerationStructureKHR = nullptr;
static PFN_vkDestroyAccelerationStructureKHR pvkDestroyAccelerationStructureKHR = nullptr;
static PFN_vkGetAccelerationStructureBuildSizesKHR pvkGetAccelerationStructureBuildSizesKHR = nullptr;
static PFN_vkCmdBuildAccelerationStructuresKHR pvkCmdBuildAccelerationStructuresKHR = nullptr;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pvkGetAccelerationStructureDeviceAddressKHR = nullptr;
static PFN_vkCreateRayTracingPipelinesKHR pvkCreateRayTracingPipelinesKHR = nullptr;
static PFN_vkGetRayTracingShaderGroupHandlesKHR pvkGetRayTracingShaderGroupHandlesKHR = nullptr;
static PFN_vkCmdTraceRaysKHR pvkCmdTraceRaysKHR = nullptr;

// ============== CONSTANTS ==============
static const uint32_t FRAME_COUNT = 2;

// ============== VULKAN RT GLOBALS ==============
static VkInstance s_instance = VK_NULL_HANDLE;
static VkPhysicalDevice s_physicalDevice = VK_NULL_HANDLE;
static VkDevice s_device = VK_NULL_HANDLE;
static VkQueue s_graphicsQueue = VK_NULL_HANDLE;
static VkQueue s_presentQueue = VK_NULL_HANDLE;
static VkSurfaceKHR s_surface = VK_NULL_HANDLE;
static VkSwapchainKHR s_swapchain = VK_NULL_HANDLE;
static std::vector<VkImage> s_swapchainImages;
static std::vector<VkImageView> s_swapchainImageViews;
static VkFormat s_swapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;
static VkExtent2D s_swapchainExtent = {};
static VkCommandPool s_commandPool = VK_NULL_HANDLE;
static std::vector<VkCommandBuffer> s_commandBuffers;
static VkSemaphore s_imageAvailableSemaphore = VK_NULL_HANDLE;
static VkSemaphore s_renderFinishedSemaphore = VK_NULL_HANDLE;
static VkFence s_inFlightFence = VK_NULL_HANDLE;
static uint32_t s_graphicsFamily = UINT32_MAX;
static uint32_t s_presentFamily = UINT32_MAX;
static std::string s_gpuName;

// Ray tracing properties
static VkPhysicalDeviceRayTracingPipelinePropertiesKHR s_rtProperties = {};
static VkPhysicalDeviceAccelerationStructureFeaturesKHR s_asFeatures = {};

// Acceleration structures
static VkAccelerationStructureKHR s_blasStatic = VK_NULL_HANDLE;
static VkAccelerationStructureKHR s_blasCubes = VK_NULL_HANDLE;
static VkAccelerationStructureKHR s_tlas = VK_NULL_HANDLE;
static VkBuffer s_blasStaticBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_blasStaticMemory = VK_NULL_HANDLE;
static VkBuffer s_blasCubesBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_blasCubesMemory = VK_NULL_HANDLE;
static VkBuffer s_tlasBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_tlasMemory = VK_NULL_HANDLE;
static VkBuffer s_instanceBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_instanceMemory = VK_NULL_HANDLE;
static void* s_instanceMapped = nullptr;
static VkBuffer s_tlasScratchBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_tlasScratchMemory = VK_NULL_HANDLE;
static VkDeviceSize s_tlasScratchSize = 0;

// Geometry buffers
static VkBuffer s_staticVertexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_staticVertexMemory = VK_NULL_HANDLE;
static VkBuffer s_staticIndexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_staticIndexMemory = VK_NULL_HANDLE;
static VkBuffer s_cubesVertexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_cubesVertexMemory = VK_NULL_HANDLE;
static VkBuffer s_cubesIndexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_cubesIndexMemory = VK_NULL_HANDLE;
static uint32_t s_staticVertexCount = 0;
static uint32_t s_staticIndexCount = 0;
static uint32_t s_cubesVertexCount = 0;
static uint32_t s_cubesIndexCount = 0;

// Ray tracing pipeline
static VkPipeline s_rtPipeline = VK_NULL_HANDLE;
static VkPipelineLayout s_rtPipelineLayout = VK_NULL_HANDLE;
static VkDescriptorSetLayout s_rtDescSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool s_rtDescPool = VK_NULL_HANDLE;
static VkDescriptorSet s_rtDescSet = VK_NULL_HANDLE;

// Shader binding table
static VkBuffer s_sbtBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_sbtMemory = VK_NULL_HANDLE;
static VkStridedDeviceAddressRegionKHR s_raygenRegion = {};
static VkStridedDeviceAddressRegionKHR s_missRegion = {};
static VkStridedDeviceAddressRegionKHR s_hitRegion = {};
static VkStridedDeviceAddressRegionKHR s_callableRegion = {};

// Output image (storage image for ray tracing output)
static VkImage s_outputImage = VK_NULL_HANDLE;
static VkDeviceMemory s_outputMemory = VK_NULL_HANDLE;
static VkImageView s_outputImageView = VK_NULL_HANDLE;

// Uniform buffer
static VkBuffer s_uniformBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_uniformMemory = VK_NULL_HANDLE;
static void* s_uniformMapped = nullptr;

// Text rendering
static VkImage s_fontImage = VK_NULL_HANDLE;
static VkDeviceMemory s_fontMemory = VK_NULL_HANDLE;
static VkImageView s_fontImageView = VK_NULL_HANDLE;
static VkSampler s_fontSampler = VK_NULL_HANDLE;
static VkDescriptorSetLayout s_textDescSetLayout = VK_NULL_HANDLE;
static VkDescriptorPool s_textDescPool = VK_NULL_HANDLE;
static VkDescriptorSet s_textDescSet = VK_NULL_HANDLE;
static VkPipelineLayout s_textPipelineLayout = VK_NULL_HANDLE;
static VkPipeline s_textPipeline = VK_NULL_HANDLE;
static VkRenderPass s_textRenderPass = VK_NULL_HANDLE;
static std::vector<VkFramebuffer> s_framebuffers;
static VkBuffer s_textVertexBuffer = VK_NULL_HANDLE;
static VkDeviceMemory s_textVertexMemory = VK_NULL_HANDLE;
static void* s_textVertexMapped = nullptr;
static TextVert s_textVerts[6000];
static uint32_t s_textVertCount = 0;
static int s_cachedFps = -1;

// Frame tracking
static uint32_t s_frameCount = 0;

// Feature tracking for recompilation
static VulkanRTFeatures s_compiledFeatures = {};

// ============== VERTEX STRUCTURE (matches DXR 1.0) ==============
#pragma pack(push, 1)
struct VkRTVertex {
    float px, py, pz;     // Position
    float nx, ny, nz;     // Normal
    float r, g, b;        // Color
    uint32_t materialType; // 0=diffuse, 1=mirror, 2=glass, 3=emissive
};
#pragma pack(pop)
static_assert(sizeof(VkRTVertex) == 40, "VkRTVertex must be 40 bytes");

// Uniform buffer structure (matches DXR 1.0)
struct VkRTUniforms {
    float time;           // Animation time
    float lightPos[3];    // Light position (x, y, z)
    float lightRadius;    // For soft shadows
    uint32_t frameCount;  // Frame number for RNG
    int32_t shadowSamples;// Number of shadow samples
    int32_t aoSamples;    // Number of AO samples
    float aoRadius;       // AO sample radius
    uint32_t features;    // Feature flags
};

// Float3 helper
struct float3 { float x, y, z; };

// ============== HELPER FUNCTIONS ==============

static uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(s_physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    Log("[VkRT] ERROR: Failed to find suitable memory type\n");
    return UINT32_MAX;
}

static bool CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                         VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(s_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create buffer\n");
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(s_device, buffer, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlagsInfo = {};
    allocFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        allocFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    }

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) ? &allocFlagsInfo : nullptr;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memReqs.memoryTypeBits, properties);

    if (allocInfo.memoryTypeIndex == UINT32_MAX) {
        vkDestroyBuffer(s_device, buffer, nullptr);
        return false;
    }

    if (vkAllocateMemory(s_device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to allocate buffer memory\n");
        vkDestroyBuffer(s_device, buffer, nullptr);
        return false;
    }

    vkBindBufferMemory(s_device, buffer, memory, 0);
    return true;
}

static VkDeviceAddress GetBufferDeviceAddress(VkBuffer buffer) {
    VkBufferDeviceAddressInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return pvkGetBufferDeviceAddressKHR(s_device, &info);
}

static VkCommandBuffer BeginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = s_commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(s_device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

static void EndSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(s_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(s_graphicsQueue);

    vkFreeCommandBuffers(s_device, s_commandPool, 1, &commandBuffer);
}

static void CopyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer = BeginSingleTimeCommands();
    VkBufferCopy copyRegion = {};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    EndSingleTimeCommands(commandBuffer);
}

// ============== TEXT RENDERING ==============
static void DrawTextVkRT(const char* text, float x, float y, float r, float g, float b, float a, float scale) {
    const float charW = 8.0f * scale;
    const float charH = 8.0f * scale;
    const float texCharW = 8.0f / 128.0f;
    const float texCharH = 8.0f / 48.0f;

    while (*text && s_textVertCount < 5994) {
        char c = *text++;
        if (c < 32 || c > 127) c = '?';
        int idx = c - 32;
        int tx = idx % 16;
        int ty = idx / 16;

        float u0 = tx * texCharW;
        float v0 = ty * texCharH;
        float u1 = u0 + texCharW;
        float v1 = v0 + texCharH;

        float x0 = (x / W) * 2.0f - 1.0f;
        float y0 = (y / H) * 2.0f - 1.0f;
        float x1 = ((x + charW) / W) * 2.0f - 1.0f;
        float y1 = ((y + charH) / H) * 2.0f - 1.0f;

        s_textVerts[s_textVertCount++] = {x0, y0, u0, v0, r, g, b, a};
        s_textVerts[s_textVertCount++] = {x1, y0, u1, v0, r, g, b, a};
        s_textVerts[s_textVertCount++] = {x0, y1, u0, v1, r, g, b, a};
        s_textVerts[s_textVertCount++] = {x1, y0, u1, v0, r, g, b, a};
        s_textVerts[s_textVertCount++] = {x1, y1, u1, v1, r, g, b, a};
        s_textVerts[s_textVertCount++] = {x0, y1, u0, v1, r, g, b, a};

        x += charW;
    }
}

// ============== TEXT INITIALIZATION ==============

static VkShaderModule CreateShaderModule(const uint32_t* code, size_t codeSize) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = codeSize;
    createInfo.pCode = code;

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(s_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return shaderModule;
}

static bool CreateTextRenderPass() {
    // Create a render pass that loads existing swapchain content
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = s_swapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // Load existing content
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dependency.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(s_device, &renderPassInfo, nullptr, &s_textRenderPass) != VK_SUCCESS) {
        Log("[VkRT] Failed to create text render pass\n");
        return false;
    }
    return true;
}

static bool CreateTextFramebuffers() {
    s_framebuffers.resize(s_swapchainImageViews.size());

    for (size_t i = 0; i < s_swapchainImageViews.size(); i++) {
        VkImageView attachments[] = { s_swapchainImageViews[i] };

        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = s_textRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = s_swapchainExtent.width;
        framebufferInfo.height = s_swapchainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(s_device, &framebufferInfo, nullptr, &s_framebuffers[i]) != VK_SUCCESS) {
            Log("[VkRT] Failed to create framebuffer %zu\n", i);
            return false;
        }
    }
    return true;
}

static bool InitTextResources() {
    Log("[VkRT] Initializing text rendering...\n");

    // Create render pass and framebuffers first
    if (!CreateTextRenderPass()) return false;
    if (!CreateTextFramebuffers()) return false;

    // ========== Create Font Texture ==========
    const int FONT_TEX_W = 128;
    const int FONT_TEX_H = 48;

    // Build font texture data (RGBA)
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
                fontData[idx + 0] = val;
                fontData[idx + 1] = val;
                fontData[idx + 2] = val;
                fontData[idx + 3] = 255;
            }
        }
    }

    // Create staging buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    VkDeviceSize imageSize = FONT_TEX_W * FONT_TEX_H * 4;

    if (!CreateBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      stagingBuffer, stagingMemory)) {
        Log("[VkRT] Failed to create font staging buffer\n");
        return false;
    }

    void* data;
    vkMapMemory(s_device, stagingMemory, 0, imageSize, 0, &data);
    memcpy(data, fontData.data(), imageSize);
    vkUnmapMemory(s_device, stagingMemory);

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

    if (vkCreateImage(s_device, &imageInfo, nullptr, &s_fontImage) != VK_SUCCESS) {
        Log("[VkRT] Failed to create font image\n");
        vkDestroyBuffer(s_device, stagingBuffer, nullptr);
        vkFreeMemory(s_device, stagingMemory, nullptr);
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(s_device, s_fontImage, &memReqs);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(s_device, &allocInfo, nullptr, &s_fontMemory) != VK_SUCCESS) {
        Log("[VkRT] Failed to allocate font memory\n");
        vkDestroyImage(s_device, s_fontImage, nullptr);
        vkDestroyBuffer(s_device, stagingBuffer, nullptr);
        vkFreeMemory(s_device, stagingMemory, nullptr);
        return false;
    }
    vkBindImageMemory(s_device, s_fontImage, s_fontMemory, 0);

    // Copy staging buffer to image
    VkCommandBuffer cmdBuf;
    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandPool = s_commandPool;
    cmdAllocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(s_device, &cmdAllocInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    // Transition to transfer dst
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = s_fontImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {(uint32_t)FONT_TEX_W, (uint32_t)FONT_TEX_H, 1};

    vkCmdCopyBufferToImage(cmdBuf, stagingBuffer, s_fontImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

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
    vkQueueSubmit(s_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(s_graphicsQueue);
    vkFreeCommandBuffers(s_device, s_commandPool, 1, &cmdBuf);

    // Cleanup staging
    vkDestroyBuffer(s_device, stagingBuffer, nullptr);
    vkFreeMemory(s_device, stagingMemory, nullptr);

    // ========== Create Image View ==========
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = s_fontImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(s_device, &viewInfo, nullptr, &s_fontImageView) != VK_SUCCESS) {
        Log("[VkRT] Failed to create font image view\n");
        return false;
    }

    // ========== Create Sampler ==========
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

    if (vkCreateSampler(s_device, &samplerInfo, nullptr, &s_fontSampler) != VK_SUCCESS) {
        Log("[VkRT] Failed to create font sampler\n");
        return false;
    }

    // ========== Create Descriptor Set Layout ==========
    VkDescriptorSetLayoutBinding binding = {};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;

    if (vkCreateDescriptorSetLayout(s_device, &layoutInfo, nullptr, &s_textDescSetLayout) != VK_SUCCESS) {
        Log("[VkRT] Failed to create text descriptor set layout\n");
        return false;
    }

    // ========== Create Descriptor Pool ==========
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(s_device, &poolInfo, nullptr, &s_textDescPool) != VK_SUCCESS) {
        Log("[VkRT] Failed to create text descriptor pool\n");
        return false;
    }

    // ========== Allocate Descriptor Set ==========
    VkDescriptorSetAllocateInfo descAllocInfo = {};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = s_textDescPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &s_textDescSetLayout;

    if (vkAllocateDescriptorSets(s_device, &descAllocInfo, &s_textDescSet) != VK_SUCCESS) {
        Log("[VkRT] Failed to allocate text descriptor set\n");
        return false;
    }

    // Update descriptor set
    VkDescriptorImageInfo imageInfoDesc = {};
    imageInfoDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imageInfoDesc.imageView = s_fontImageView;
    imageInfoDesc.sampler = s_fontSampler;

    VkWriteDescriptorSet descWrite = {};
    descWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descWrite.dstSet = s_textDescSet;
    descWrite.dstBinding = 0;
    descWrite.dstArrayElement = 0;
    descWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descWrite.descriptorCount = 1;
    descWrite.pImageInfo = &imageInfoDesc;

    vkUpdateDescriptorSets(s_device, 1, &descWrite, 0, nullptr);

    // ========== Create Pipeline Layout ==========
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &s_textDescSetLayout;

    if (vkCreatePipelineLayout(s_device, &pipelineLayoutInfo, nullptr, &s_textPipelineLayout) != VK_SUCCESS) {
        Log("[VkRT] Failed to create text pipeline layout\n");
        return false;
    }

    // ========== Create Shaders ==========
    VkShaderModule textVertShader = CreateShaderModule(g_vkTextVertShaderCode, sizeof(g_vkTextVertShaderCode));
    VkShaderModule textFragShader = CreateShaderModule(g_vkTextFragShaderCode, sizeof(g_vkTextFragShaderCode));

    if (!textVertShader || !textFragShader) {
        Log("[VkRT] Failed to create text shader modules\n");
        if (textVertShader) vkDestroyShaderModule(s_device, textVertShader, nullptr);
        if (textFragShader) vkDestroyShaderModule(s_device, textFragShader, nullptr);
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

    // ========== Vertex Input ==========
    VkVertexInputBindingDescription bindingDesc = {};
    bindingDesc.binding = 0;
    bindingDesc.stride = sizeof(TextVert);
    bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attrDescs[3] = {};
    attrDescs[0].binding = 0;
    attrDescs[0].location = 0;
    attrDescs[0].format = VK_FORMAT_R32G32_SFLOAT;
    attrDescs[0].offset = offsetof(TextVert, x);
    attrDescs[1].binding = 0;
    attrDescs[1].location = 1;
    attrDescs[1].format = VK_FORMAT_R32G32_SFLOAT;
    attrDescs[1].offset = offsetof(TextVert, u);
    attrDescs[2].binding = 0;
    attrDescs[2].location = 2;
    attrDescs[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrDescs[2].offset = offsetof(TextVert, r);

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
    viewport.width = (float)s_swapchainExtent.width;
    viewport.height = (float)s_swapchainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = s_swapchainExtent;

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
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    // Alpha blending
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

    // ========== Create Pipeline ==========
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
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = s_textPipelineLayout;
    pipelineInfo.renderPass = s_textRenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(s_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &s_textPipeline) != VK_SUCCESS) {
        Log("[VkRT] Failed to create text pipeline\n");
        vkDestroyShaderModule(s_device, textVertShader, nullptr);
        vkDestroyShaderModule(s_device, textFragShader, nullptr);
        return false;
    }

    vkDestroyShaderModule(s_device, textVertShader, nullptr);
    vkDestroyShaderModule(s_device, textFragShader, nullptr);

    // ========== Create Text Vertex Buffer ==========
    VkDeviceSize textBufferSize = sizeof(TextVert) * 6000;
    if (!CreateBuffer(textBufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      s_textVertexBuffer, s_textVertexMemory)) {
        Log("[VkRT] Failed to create text vertex buffer\n");
        return false;
    }

    if (vkMapMemory(s_device, s_textVertexMemory, 0, textBufferSize, 0, &s_textVertexMapped) != VK_SUCCESS) {
        Log("[VkRT] Failed to map text vertex buffer\n");
        return false;
    }

    Log("[VkRT] Text rendering initialized successfully\n");
    return true;
}

// ============== GEOMETRY GENERATION (matches DXR 1.0 exactly) ==============

static void GenerateCornellBox(std::vector<VkRTVertex>& verts, std::vector<uint32_t>& indices) {
    // Materials: 0=diffuse, 1=mirror, 2=glass, 3=emissive

    // Floor (white/grey diffuse)
    uint32_t baseIdx = (uint32_t)verts.size();
    verts.push_back({-1, -1, -1, 0, 1, 0, 0.7f, 0.7f, 0.7f, 0});
    verts.push_back({ 1, -1, -1, 0, 1, 0, 0.7f, 0.7f, 0.7f, 0});
    verts.push_back({ 1, -1,  1, 0, 1, 0, 0.7f, 0.7f, 0.7f, 0});
    verts.push_back({-1, -1,  1, 0, 1, 0, 0.7f, 0.7f, 0.7f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Ceiling (white diffuse)
    baseIdx = (uint32_t)verts.size();
    verts.push_back({-1, 1, -1, 0, -1, 0, 0.9f, 0.9f, 0.9f, 0});
    verts.push_back({-1, 1,  1, 0, -1, 0, 0.9f, 0.9f, 0.9f, 0});
    verts.push_back({ 1, 1,  1, 0, -1, 0, 0.9f, 0.9f, 0.9f, 0});
    verts.push_back({ 1, 1, -1, 0, -1, 0, 0.9f, 0.9f, 0.9f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Back wall (grey diffuse)
    baseIdx = (uint32_t)verts.size();
    verts.push_back({-1, -1, 1, 0, 0, -1, 0.7f, 0.7f, 0.7f, 0});
    verts.push_back({ 1, -1, 1, 0, 0, -1, 0.7f, 0.7f, 0.7f, 0});
    verts.push_back({ 1,  1, 1, 0, 0, -1, 0.7f, 0.7f, 0.7f, 0});
    verts.push_back({-1,  1, 1, 0, 0, -1, 0.7f, 0.7f, 0.7f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Left wall (RED diffuse)
    baseIdx = (uint32_t)verts.size();
    verts.push_back({-1, -1, -1, 1, 0, 0, 0.75f, 0.15f, 0.15f, 0});
    verts.push_back({-1, -1,  1, 1, 0, 0, 0.75f, 0.15f, 0.15f, 0});
    verts.push_back({-1,  1,  1, 1, 0, 0, 0.75f, 0.15f, 0.15f, 0});
    verts.push_back({-1,  1, -1, 1, 0, 0, 0.75f, 0.15f, 0.15f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Right wall (GREEN diffuse)
    baseIdx = (uint32_t)verts.size();
    verts.push_back({1, -1, -1, -1, 0, 0, 0.15f, 0.75f, 0.15f, 0});
    verts.push_back({1,  1, -1, -1, 0, 0, 0.15f, 0.75f, 0.15f, 0});
    verts.push_back({1,  1,  1, -1, 0, 0, 0.15f, 0.75f, 0.15f, 0});
    verts.push_back({1, -1,  1, -1, 0, 0, 0.15f, 0.75f, 0.15f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Light panel on ceiling (emissive white)
    baseIdx = (uint32_t)verts.size();
    float ls = 0.3f;
    verts.push_back({-ls, 0.99f, -ls, 0, -1, 0, 15.0f, 14.0f, 12.0f, 3});
    verts.push_back({-ls, 0.99f,  ls, 0, -1, 0, 15.0f, 14.0f, 12.0f, 3});
    verts.push_back({ ls, 0.99f,  ls, 0, -1, 0, 15.0f, 14.0f, 12.0f, 3});
    verts.push_back({ ls, 0.99f, -ls, 0, -1, 0, 15.0f, 14.0f, 12.0f, 3});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Mirror panel (angled, back-left) - matches DXR 1.0 exactly
    baseIdx = (uint32_t)verts.size();
    const float mh = 0.5f, mw = 0.4f, mcx = -0.6f, mcy = 0.0f, mcz = 0.6f;
    const float c45 = 0.707f;  // cos(45 degrees)
    float3 mnorm = {c45, 0, -c45};
    // Mirror quad facing rotating cubes (45 degree angle)
    verts.push_back({mcx - c45*mw, mcy - mh, mcz - c45*mw, mnorm.x, mnorm.y, mnorm.z, 0.95f, 0.95f, 0.95f, 1});
    verts.push_back({mcx + c45*mw, mcy - mh, mcz + c45*mw, mnorm.x, mnorm.y, mnorm.z, 0.95f, 0.95f, 0.95f, 1});
    verts.push_back({mcx + c45*mw, mcy + mh, mcz + c45*mw, mnorm.x, mnorm.y, mnorm.z, 0.95f, 0.95f, 0.95f, 1});
    verts.push_back({mcx - c45*mw, mcy + mh, mcz - c45*mw, mnorm.x, mnorm.y, mnorm.z, 0.95f, 0.95f, 0.95f, 1});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Small red cube (static, near left wall - matches DXR)
    baseIdx = (uint32_t)verts.size();
    float scx = -0.5f, scy = -0.85f, scz = 0.3f, scs = 0.13f;
    // Front
    verts.push_back({scx-scs, scy-scs, scz-scs, 0, 0, -1, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy-scs, scz-scs, 0, 0, -1, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy+scs, scz-scs, 0, 0, -1, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx-scs, scy+scs, scz-scs, 0, 0, -1, 0.9f, 0.15f, 0.1f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);
    // Back
    baseIdx = (uint32_t)verts.size();
    verts.push_back({scx+scs, scy-scs, scz+scs, 0, 0, 1, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx-scs, scy-scs, scz+scs, 0, 0, 1, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx-scs, scy+scs, scz+scs, 0, 0, 1, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy+scs, scz+scs, 0, 0, 1, 0.9f, 0.15f, 0.1f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);
    // Right
    baseIdx = (uint32_t)verts.size();
    verts.push_back({scx+scs, scy-scs, scz-scs, 1, 0, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy-scs, scz+scs, 1, 0, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy+scs, scz+scs, 1, 0, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy+scs, scz-scs, 1, 0, 0, 0.9f, 0.15f, 0.1f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);
    // Left
    baseIdx = (uint32_t)verts.size();
    verts.push_back({scx-scs, scy-scs, scz+scs, -1, 0, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx-scs, scy-scs, scz-scs, -1, 0, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx-scs, scy+scs, scz-scs, -1, 0, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx-scs, scy+scs, scz+scs, -1, 0, 0, 0.9f, 0.15f, 0.1f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);
    // Top
    baseIdx = (uint32_t)verts.size();
    verts.push_back({scx-scs, scy+scs, scz-scs, 0, 1, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy+scs, scz-scs, 0, 1, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy+scs, scz+scs, 0, 1, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx-scs, scy+scs, scz+scs, 0, 1, 0, 0.9f, 0.15f, 0.1f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);
    // Bottom
    baseIdx = (uint32_t)verts.size();
    verts.push_back({scx-scs, scy-scs, scz+scs, 0, -1, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy-scs, scz+scs, 0, -1, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx+scs, scy-scs, scz-scs, 0, -1, 0, 0.9f, 0.15f, 0.1f, 0});
    verts.push_back({scx-scs, scy-scs, scz-scs, 0, -1, 0, 0.9f, 0.15f, 0.1f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Glass panel (directly in front of small red cube - matches DXR)
    // Glass is just in front of the cube at scz - 0.18
    baseIdx = (uint32_t)verts.size();
    float glassZ = scz - 0.18f;  // Just in front of cube
    float glassY = scy - 0.02f;  // Slightly below cube bottom
    float glassH = 0.35f;        // Covers the cube height
    float glassW = 0.18f;        // Covers the cube width
    // Front face (towards camera, normal -Z)
    verts.push_back({scx - glassW, glassY,          glassZ, 0, 0, -1, 0.9f, 0.95f, 1.0f, 2});
    verts.push_back({scx + glassW, glassY,          glassZ, 0, 0, -1, 0.9f, 0.95f, 1.0f, 2});
    verts.push_back({scx + glassW, glassY + glassH, glassZ, 0, 0, -1, 0.9f, 0.95f, 1.0f, 2});
    verts.push_back({scx - glassW, glassY + glassH, glassZ, 0, 0, -1, 0.9f, 0.95f, 1.0f, 2});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);
    // Back face (towards cube, normal +Z)
    baseIdx = (uint32_t)verts.size();
    verts.push_back({scx + glassW, glassY,          glassZ, 0, 0, 1, 0.9f, 0.95f, 1.0f, 2});
    verts.push_back({scx - glassW, glassY,          glassZ, 0, 0, 1, 0.9f, 0.95f, 1.0f, 2});
    verts.push_back({scx - glassW, glassY + glassH, glassZ, 0, 0, 1, 0.9f, 0.95f, 1.0f, 2});
    verts.push_back({scx + glassW, glassY + glassH, glassZ, 0, 0, 1, 0.9f, 0.95f, 1.0f, 2});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);

    // Front wall (purple) - behind camera at z=-3.0 to block outside light
    // Camera is at z=-2.2, so wall at z=-3.0 is behind camera
    baseIdx = (uint32_t)verts.size();
    float fwz = -3.0f;
    float fws = 2.0f;  // Large wall to cover everything
    verts.push_back({-fws, -fws, fwz, 0, 0, 1, 0.5f, 0.15f, 0.7f, 0});
    verts.push_back({ fws, -fws, fwz, 0, 0, 1, 0.5f, 0.15f, 0.7f, 0});
    verts.push_back({ fws,  fws, fwz, 0, 0, 1, 0.5f, 0.15f, 0.7f, 0});
    verts.push_back({-fws,  fws, fwz, 0, 0, 1, 0.5f, 0.15f, 0.7f, 0});
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+1); indices.push_back(baseIdx+2);
    indices.push_back(baseIdx+0); indices.push_back(baseIdx+2); indices.push_back(baseIdx+3);
}

static void GenerateRotatingCubes(std::vector<VkRTVertex>& verts, std::vector<uint32_t>& indices) {
    // 8 small rotating cubes in 2x2x2 arrangement (matches DXR)
    // Cubes touch each other (spacing = smallSize)
    const float smallSize = 0.11f;  // Half-size of each small cube
    const float spacing = smallSize; // Cubes touch exactly

    // Colors matching DXR shader's CubeColors array
    float colors[8][3] = {
        {1.0f, 0.15f, 0.1f},  // Red
        {0.1f, 0.9f, 0.2f},   // Green
        {0.1f, 0.4f, 1.0f},   // Blue
        {1.0f, 0.95f, 0.1f},  // Yellow
        {1.0f, 0.95f, 0.1f},  // Yellow
        {0.1f, 0.4f, 1.0f},   // Blue
        {0.1f, 0.9f, 0.2f},   // Green
        {1.0f, 0.15f, 0.1f}   // Red
    };

    // Coordinates matching DXR BuildDynamicCube
    int coords[8][3] = {
        {-1, +1, +1}, {+1, +1, +1}, {-1, -1, +1}, {+1, -1, +1},
        {-1, +1, -1}, {+1, +1, -1}, {-1, -1, -1}, {+1, -1, -1},
    };

    float cubeSize = smallSize;

    for (int c = 0; c < 8; c++) {
        float cx = coords[c][0] * spacing;
        float cy = coords[c][1] * spacing;
        float cz = coords[c][2] * spacing;
        float cr = colors[c][0];
        float cg = colors[c][1];
        float cb = colors[c][2];
        float s = cubeSize;

        uint32_t baseIdx = (uint32_t)verts.size();

        // 6 faces x 4 vertices = 24 vertices per cube
        // Front (Z-)
        verts.push_back({cx-s, cy-s, cz-s, 0, 0, -1, cr, cg, cb, 0});
        verts.push_back({cx+s, cy-s, cz-s, 0, 0, -1, cr, cg, cb, 0});
        verts.push_back({cx+s, cy+s, cz-s, 0, 0, -1, cr, cg, cb, 0});
        verts.push_back({cx-s, cy+s, cz-s, 0, 0, -1, cr, cg, cb, 0});
        // Back (Z+)
        verts.push_back({cx+s, cy-s, cz+s, 0, 0, 1, cr, cg, cb, 0});
        verts.push_back({cx-s, cy-s, cz+s, 0, 0, 1, cr, cg, cb, 0});
        verts.push_back({cx-s, cy+s, cz+s, 0, 0, 1, cr, cg, cb, 0});
        verts.push_back({cx+s, cy+s, cz+s, 0, 0, 1, cr, cg, cb, 0});
        // Right (X+)
        verts.push_back({cx+s, cy-s, cz-s, 1, 0, 0, cr, cg, cb, 0});
        verts.push_back({cx+s, cy-s, cz+s, 1, 0, 0, cr, cg, cb, 0});
        verts.push_back({cx+s, cy+s, cz+s, 1, 0, 0, cr, cg, cb, 0});
        verts.push_back({cx+s, cy+s, cz-s, 1, 0, 0, cr, cg, cb, 0});
        // Left (X-)
        verts.push_back({cx-s, cy-s, cz+s, -1, 0, 0, cr, cg, cb, 0});
        verts.push_back({cx-s, cy-s, cz-s, -1, 0, 0, cr, cg, cb, 0});
        verts.push_back({cx-s, cy+s, cz-s, -1, 0, 0, cr, cg, cb, 0});
        verts.push_back({cx-s, cy+s, cz+s, -1, 0, 0, cr, cg, cb, 0});
        // Top (Y+)
        verts.push_back({cx-s, cy+s, cz-s, 0, 1, 0, cr, cg, cb, 0});
        verts.push_back({cx+s, cy+s, cz-s, 0, 1, 0, cr, cg, cb, 0});
        verts.push_back({cx+s, cy+s, cz+s, 0, 1, 0, cr, cg, cb, 0});
        verts.push_back({cx-s, cy+s, cz+s, 0, 1, 0, cr, cg, cb, 0});
        // Bottom (Y-)
        verts.push_back({cx-s, cy-s, cz+s, 0, -1, 0, cr, cg, cb, 0});
        verts.push_back({cx+s, cy-s, cz+s, 0, -1, 0, cr, cg, cb, 0});
        verts.push_back({cx+s, cy-s, cz-s, 0, -1, 0, cr, cg, cb, 0});
        verts.push_back({cx-s, cy-s, cz-s, 0, -1, 0, cr, cg, cb, 0});

        // Indices for 6 faces (2 triangles each)
        for (int f = 0; f < 6; f++) {
            uint32_t fi = baseIdx + f * 4;
            indices.push_back(fi+0); indices.push_back(fi+1); indices.push_back(fi+2);
            indices.push_back(fi+0); indices.push_back(fi+2); indices.push_back(fi+3);
        }
    }
}

// ============== LOAD RT EXTENSION FUNCTIONS ==============
static bool LoadRTExtensions() {
    pvkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)
        vkGetDeviceProcAddr(s_device, "vkGetBufferDeviceAddressKHR");
    pvkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)
        vkGetDeviceProcAddr(s_device, "vkCreateAccelerationStructureKHR");
    pvkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)
        vkGetDeviceProcAddr(s_device, "vkDestroyAccelerationStructureKHR");
    pvkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)
        vkGetDeviceProcAddr(s_device, "vkGetAccelerationStructureBuildSizesKHR");
    pvkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)
        vkGetDeviceProcAddr(s_device, "vkCmdBuildAccelerationStructuresKHR");
    pvkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
        vkGetDeviceProcAddr(s_device, "vkGetAccelerationStructureDeviceAddressKHR");
    pvkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)
        vkGetDeviceProcAddr(s_device, "vkCreateRayTracingPipelinesKHR");
    pvkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)
        vkGetDeviceProcAddr(s_device, "vkGetRayTracingShaderGroupHandlesKHR");
    pvkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)
        vkGetDeviceProcAddr(s_device, "vkCmdTraceRaysKHR");

    if (!pvkGetBufferDeviceAddressKHR || !pvkCreateAccelerationStructureKHR ||
        !pvkDestroyAccelerationStructureKHR || !pvkGetAccelerationStructureBuildSizesKHR ||
        !pvkCmdBuildAccelerationStructuresKHR || !pvkGetAccelerationStructureDeviceAddressKHR ||
        !pvkCreateRayTracingPipelinesKHR || !pvkGetRayTracingShaderGroupHandlesKHR ||
        !pvkCmdTraceRaysKHR) {
        Log("[VkRT] ERROR: Failed to load ray tracing extension functions\n");
        return false;
    }

    Log("[VkRT] Ray tracing extension functions loaded successfully\n");
    return true;
}

// ============== CREATE GEOMETRY BUFFERS ==============
static bool CreateGeometryBuffers() {
    Log("[VkRT] Creating geometry buffers...\n");

    // Generate static geometry (Cornell Box)
    std::vector<VkRTVertex> staticVerts;
    std::vector<uint32_t> staticInds;
    GenerateCornellBox(staticVerts, staticInds);
    s_staticVertexCount = (uint32_t)staticVerts.size();
    s_staticIndexCount = (uint32_t)staticInds.size();

    // Generate rotating cubes
    std::vector<VkRTVertex> cubeVerts;
    std::vector<uint32_t> cubeInds;
    GenerateRotatingCubes(cubeVerts, cubeInds);
    s_cubesVertexCount = (uint32_t)cubeVerts.size();
    s_cubesIndexCount = (uint32_t)cubeInds.size();

    // Create staging buffers and copy to device local buffers
    VkDeviceSize staticVBSize = staticVerts.size() * sizeof(VkRTVertex);
    VkDeviceSize staticIBSize = staticInds.size() * sizeof(uint32_t);
    VkDeviceSize cubesVBSize = cubeVerts.size() * sizeof(VkRTVertex);
    VkDeviceSize cubesIBSize = cubeInds.size() * sizeof(uint32_t);

    // Static vertex buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    CreateBuffer(staticVBSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingMemory);
    void* data;
    vkMapMemory(s_device, stagingMemory, 0, staticVBSize, 0, &data);
    memcpy(data, staticVerts.data(), staticVBSize);
    vkUnmapMemory(s_device, stagingMemory);

    CreateBuffer(staticVBSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 s_staticVertexBuffer, s_staticVertexMemory);
    CopyBuffer(stagingBuffer, s_staticVertexBuffer, staticVBSize);
    vkDestroyBuffer(s_device, stagingBuffer, nullptr);
    vkFreeMemory(s_device, stagingMemory, nullptr);

    // Static index buffer
    CreateBuffer(staticIBSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingMemory);
    vkMapMemory(s_device, stagingMemory, 0, staticIBSize, 0, &data);
    memcpy(data, staticInds.data(), staticIBSize);
    vkUnmapMemory(s_device, stagingMemory);

    CreateBuffer(staticIBSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 s_staticIndexBuffer, s_staticIndexMemory);
    CopyBuffer(stagingBuffer, s_staticIndexBuffer, staticIBSize);
    vkDestroyBuffer(s_device, stagingBuffer, nullptr);
    vkFreeMemory(s_device, stagingMemory, nullptr);

    // Cubes vertex buffer
    CreateBuffer(cubesVBSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingMemory);
    vkMapMemory(s_device, stagingMemory, 0, cubesVBSize, 0, &data);
    memcpy(data, cubeVerts.data(), cubesVBSize);
    vkUnmapMemory(s_device, stagingMemory);

    CreateBuffer(cubesVBSize,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 s_cubesVertexBuffer, s_cubesVertexMemory);
    CopyBuffer(stagingBuffer, s_cubesVertexBuffer, cubesVBSize);
    vkDestroyBuffer(s_device, stagingBuffer, nullptr);
    vkFreeMemory(s_device, stagingMemory, nullptr);

    // Cubes index buffer
    CreateBuffer(cubesIBSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingMemory);
    vkMapMemory(s_device, stagingMemory, 0, cubesIBSize, 0, &data);
    memcpy(data, cubeInds.data(), cubesIBSize);
    vkUnmapMemory(s_device, stagingMemory);

    CreateBuffer(cubesIBSize,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 s_cubesIndexBuffer, s_cubesIndexMemory);
    CopyBuffer(stagingBuffer, s_cubesIndexBuffer, cubesIBSize);
    vkDestroyBuffer(s_device, stagingBuffer, nullptr);
    vkFreeMemory(s_device, stagingMemory, nullptr);

    Log("[VkRT] Geometry buffers created: Static %u verts/%u inds, Cubes %u verts/%u inds\n",
        s_staticVertexCount, s_staticIndexCount, s_cubesVertexCount, s_cubesIndexCount);
    return true;
}

// ============== CREATE BLAS ==============
static bool CreateBLAS(VkBuffer vertexBuffer, VkBuffer indexBuffer,
                       uint32_t vertexCount, uint32_t indexCount,
                       VkAccelerationStructureKHR& blas, VkBuffer& blasBuffer, VkDeviceMemory& blasMemory) {
    // Geometry description
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.vertexData.deviceAddress = GetBufferDeviceAddress(vertexBuffer);
    geometry.geometry.triangles.vertexStride = sizeof(VkRTVertex);
    geometry.geometry.triangles.maxVertex = vertexCount - 1;
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.indexData.deviceAddress = GetBufferDeviceAddress(indexBuffer);

    // Build info
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t primitiveCount = indexCount / 3;

    // Get size requirements
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pvkGetAccelerationStructureBuildSizesKHR(s_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                              &buildInfo, &primitiveCount, &sizeInfo);

    // Create buffer for BLAS
    CreateBuffer(sizeInfo.accelerationStructureSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, blasBuffer, blasMemory);

    // Create acceleration structure
    VkAccelerationStructureCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = blasBuffer;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    if (pvkCreateAccelerationStructureKHR(s_device, &createInfo, nullptr, &blas) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create BLAS\n");
        return false;
    }

    // Create scratch buffer
    VkBuffer scratchBuffer;
    VkDeviceMemory scratchMemory;
    CreateBuffer(sizeInfo.buildScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchMemory);

    // Build BLAS
    buildInfo.dstAccelerationStructure = blas;
    buildInfo.scratchData.deviceAddress = GetBufferDeviceAddress(scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = BeginSingleTimeCommands();
    pvkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    EndSingleTimeCommands(cmd);

    // Cleanup scratch
    vkDestroyBuffer(s_device, scratchBuffer, nullptr);
    vkFreeMemory(s_device, scratchMemory, nullptr);

    return true;
}

// ============== CREATE TLAS ==============
static bool CreateTLAS() {
    Log("[VkRT] Creating TLAS...\n");

    // Get BLAS device addresses
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;

    addressInfo.accelerationStructure = s_blasStatic;
    VkDeviceAddress blasStaticAddr = pvkGetAccelerationStructureDeviceAddressKHR(s_device, &addressInfo);

    addressInfo.accelerationStructure = s_blasCubes;
    VkDeviceAddress blasCubesAddr = pvkGetAccelerationStructureDeviceAddressKHR(s_device, &addressInfo);

    // Create instance data (2 instances: static geometry + rotating cubes)
    VkAccelerationStructureInstanceKHR instances[2] = {};

    // Instance 0: Static geometry (identity transform)
    instances[0].transform.matrix[0][0] = 1.0f;
    instances[0].transform.matrix[1][1] = 1.0f;
    instances[0].transform.matrix[2][2] = 1.0f;
    instances[0].instanceCustomIndex = 0;
    instances[0].mask = 0xFF;
    instances[0].instanceShaderBindingTableRecordOffset = 0;
    instances[0].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instances[0].accelerationStructureReference = blasStaticAddr;

    // Instance 1: Rotating cubes (will be updated each frame)
    instances[1].transform.matrix[0][0] = 1.0f;
    instances[1].transform.matrix[1][1] = 1.0f;
    instances[1].transform.matrix[2][2] = 1.0f;
    instances[1].instanceCustomIndex = 1;
    instances[1].mask = 0xFF;
    instances[1].instanceShaderBindingTableRecordOffset = 0;
    instances[1].flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instances[1].accelerationStructureReference = blasCubesAddr;

    // Create instance buffer
    VkDeviceSize instanceBufferSize = sizeof(instances);
    CreateBuffer(instanceBufferSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 s_instanceBuffer, s_instanceMemory);

    vkMapMemory(s_device, s_instanceMemory, 0, instanceBufferSize, 0, &s_instanceMapped);
    memcpy(s_instanceMapped, instances, instanceBufferSize);

    // Build TLAS
    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = GetBufferDeviceAddress(s_instanceBuffer);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t instanceCount = 2;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pvkGetAccelerationStructureBuildSizesKHR(s_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                              &buildInfo, &instanceCount, &sizeInfo);

    CreateBuffer(sizeInfo.accelerationStructureSize,
                 VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, s_tlasBuffer, s_tlasMemory);

    VkAccelerationStructureCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = s_tlasBuffer;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    if (pvkCreateAccelerationStructureKHR(s_device, &createInfo, nullptr, &s_tlas) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create TLAS\n");
        return false;
    }

    // Create persistent scratch buffer for TLAS updates
    s_tlasScratchSize = sizeInfo.buildScratchSize;
    CreateBuffer(s_tlasScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, s_tlasScratchBuffer, s_tlasScratchMemory);

    buildInfo.dstAccelerationStructure = s_tlas;
    buildInfo.scratchData.deviceAddress = GetBufferDeviceAddress(s_tlasScratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBuffer cmd = BeginSingleTimeCommands();
    pvkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);
    EndSingleTimeCommands(cmd);

    // Don't destroy scratch buffer - we need it for updates!

    Log("[VkRT] TLAS created with %u instances\n", instanceCount);
    return true;
}

// ============== UPDATE CUBE TRANSFORM ==============
static void UpdateCubeTransform(float time) {
    if (!s_instanceMapped) return;

    // Rotation angles (same as DXR: Y*1.2, X*0.7)
    float angleY = time * 1.2f;
    float angleX = time * 0.7f;

    float cosY = cosf(angleY), sinY = sinf(angleY);
    float cosX = cosf(angleX), sinX = sinf(angleX);

    // Combined rotation: RotY * RotX
    float m00 = cosY;
    float m01 = sinY * sinX;
    float m02 = sinY * cosX;
    float m10 = 0;
    float m11 = cosX;
    float m12 = -sinX;
    float m20 = -sinY;
    float m21 = cosY * sinX;
    float m22 = cosY * cosX;

    // Cube position (same as DXR)
    float tx = 0.15f;
    float ty = 0.15f;
    float tz = 0.2f;

    // Update instance 1 transform (cube)
    VkAccelerationStructureInstanceKHR* instances = (VkAccelerationStructureInstanceKHR*)s_instanceMapped;

    // Transform is 3x4 row-major (transpose of rotation + translation)
    instances[1].transform.matrix[0][0] = m00; instances[1].transform.matrix[0][1] = m10; instances[1].transform.matrix[0][2] = m20; instances[1].transform.matrix[0][3] = tx;
    instances[1].transform.matrix[1][0] = m01; instances[1].transform.matrix[1][1] = m11; instances[1].transform.matrix[1][2] = m21; instances[1].transform.matrix[1][3] = ty;
    instances[1].transform.matrix[2][0] = m02; instances[1].transform.matrix[2][1] = m12; instances[1].transform.matrix[2][2] = m22; instances[1].transform.matrix[2][3] = tz;
}

// ============== REBUILD TLAS ==============
static void RebuildTLAS(VkCommandBuffer cmd) {
    if (!s_tlas || !s_instanceBuffer || !s_tlasScratchBuffer) return;

    VkAccelerationStructureGeometryKHR geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = GetBufferDeviceAddress(s_instanceBuffer);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = s_tlas;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.scratchData.deviceAddress = GetBufferDeviceAddress(s_tlasScratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo = {};
    rangeInfo.primitiveCount = 2;  // 2 instances
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    pvkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfo);

    // Memory barrier to ensure TLAS is built before ray tracing
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ============== CREATE OUTPUT IMAGE ==============
static bool CreateOutputImage() {
    Log("[VkRT] Creating output image...\n");

    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;  // Shader swizzles to BGRA
    imageInfo.extent = {s_swapchainExtent.width, s_swapchainExtent.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (vkCreateImage(s_device, &imageInfo, nullptr, &s_outputImage) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create output image\n");
        return false;
    }

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(s_device, s_outputImage, &memReqs);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(s_device, &allocInfo, nullptr, &s_outputMemory) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to allocate output image memory\n");
        return false;
    }

    vkBindImageMemory(s_device, s_outputImage, s_outputMemory, 0);

    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = s_outputImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;  // Shader swizzles to BGRA
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(s_device, &viewInfo, nullptr, &s_outputImageView) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create output image view\n");
        return false;
    }

    // Transition to general layout
    VkCommandBuffer cmd = BeginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = s_outputImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    EndSingleTimeCommands(cmd);

    Log("[VkRT] Output image created (%ux%u)\n", s_swapchainExtent.width, s_swapchainExtent.height);
    return true;
}

// ============== CREATE UNIFORM BUFFER ==============
static bool CreateUniformBuffer() {
    VkDeviceSize bufferSize = sizeof(VkRTUniforms);

    CreateBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 s_uniformBuffer, s_uniformMemory);

    vkMapMemory(s_device, s_uniformMemory, 0, bufferSize, 0, &s_uniformMapped);

    // Initialize with default values
    VkRTUniforms* uniforms = (VkRTUniforms*)s_uniformMapped;
    uniforms->time = 0.0f;
    uniforms->lightPos[0] = 0.0f;
    uniforms->lightPos[1] = 0.92f;
    uniforms->lightPos[2] = 0.0f;
    uniforms->lightRadius = g_vulkanRTFeatures.lightRadius;
    uniforms->frameCount = 0;
    uniforms->shadowSamples = g_vulkanRTFeatures.shadowSamples;
    uniforms->aoSamples = g_vulkanRTFeatures.aoSamples;
    uniforms->aoRadius = g_vulkanRTFeatures.aoRadius;

    // Build feature flags
    uniforms->features = 0;
    if (g_vulkanRTFeatures.spotlight) uniforms->features |= (1u << 0);
    if (g_vulkanRTFeatures.softShadows) uniforms->features |= (1u << 1);
    if (g_vulkanRTFeatures.ambientOcclusion) uniforms->features |= (1u << 2);
    if (g_vulkanRTFeatures.globalIllum) uniforms->features |= (1u << 3);
    if (g_vulkanRTFeatures.reflections) uniforms->features |= (1u << 4);
    if (g_vulkanRTFeatures.glassRefraction) uniforms->features |= (1u << 5);

    Log("[VkRT] Uniform buffer created\n");
    return true;
}

// ============== CREATE DESCRIPTOR SET LAYOUT ==============
static bool CreateDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding bindings[3] = {};

    // Binding 0: Acceleration structure
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    // Binding 1: Output image
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Binding 2: Uniform buffer
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(s_device, &layoutInfo, nullptr, &s_rtDescSetLayout) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create descriptor set layout\n");
        return false;
    }

    Log("[VkRT] Descriptor set layout created\n");
    return true;
}

// ============== CREATE DESCRIPTOR POOL AND SET ==============
static bool CreateDescriptorSet() {
    // Create pool
    VkDescriptorPoolSize poolSizes[3] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = 1;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[2].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(s_device, &poolInfo, nullptr, &s_rtDescPool) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create descriptor pool\n");
        return false;
    }

    // Allocate set
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = s_rtDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &s_rtDescSetLayout;

    if (vkAllocateDescriptorSets(s_device, &allocInfo, &s_rtDescSet) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to allocate descriptor set\n");
        return false;
    }

    // Update descriptors
    VkWriteDescriptorSetAccelerationStructureKHR asWrite = {};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &s_tlas;

    VkDescriptorImageInfo imageInfo = {};
    imageInfo.imageView = s_outputImageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.buffer = s_uniformBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = sizeof(VkRTUniforms);

    VkWriteDescriptorSet writes[3] = {};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asWrite;
    writes[0].dstSet = s_rtDescSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = s_rtDescSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &imageInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = s_rtDescSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(s_device, 3, writes, 0, nullptr);

    Log("[VkRT] Descriptor set created and updated\n");
    return true;
}

// ============== CREATE RT PIPELINE ==============
static bool CreateRTPipeline() {
    Log("[VkRT] Creating ray tracing pipeline...\n");

    // Create shader modules from pre-compiled SPIR-V
    VkShaderModuleCreateInfo raygenModuleInfo = {};
    raygenModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    raygenModuleInfo.codeSize = g_rtRayGenSPIRV_size;
    raygenModuleInfo.pCode = g_rtRayGenSPIRV;

    VkShaderModule raygenModule;
    if (vkCreateShaderModule(s_device, &raygenModuleInfo, nullptr, &raygenModule) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create raygen shader module\n");
        return false;
    }

    VkShaderModuleCreateInfo chitModuleInfo = {};
    chitModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    chitModuleInfo.codeSize = g_rtClosestHitSPIRV_size;
    chitModuleInfo.pCode = g_rtClosestHitSPIRV;

    VkShaderModule chitModule;
    if (vkCreateShaderModule(s_device, &chitModuleInfo, nullptr, &chitModule) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create closesthit shader module\n");
        vkDestroyShaderModule(s_device, raygenModule, nullptr);
        return false;
    }

    VkShaderModuleCreateInfo missModuleInfo = {};
    missModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    missModuleInfo.codeSize = g_rtMissSPIRV_size;
    missModuleInfo.pCode = g_rtMissSPIRV;

    VkShaderModule missModule;
    if (vkCreateShaderModule(s_device, &missModuleInfo, nullptr, &missModule) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create miss shader module\n");
        vkDestroyShaderModule(s_device, raygenModule, nullptr);
        vkDestroyShaderModule(s_device, chitModule, nullptr);
        return false;
    }

    // Shader stages
    VkPipelineShaderStageCreateInfo stages[3] = {};

    // Raygen
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = raygenModule;
    stages[0].pName = "main";

    // Miss
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = missModule;
    stages[1].pName = "main";

    // Closest hit
    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[2].module = chitModule;
    stages[2].pName = "main";

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR groups[3] = {};

    // Raygen group
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Miss group
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Hit group
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[2].generalShader = VK_SHADER_UNUSED_KHR;
    groups[2].closestHitShader = 2;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Create pipeline layout
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &s_rtDescSetLayout;

    if (vkCreatePipelineLayout(s_device, &layoutInfo, nullptr, &s_rtPipelineLayout) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create pipeline layout\n");
        vkDestroyShaderModule(s_device, raygenModule, nullptr);
        vkDestroyShaderModule(s_device, chitModule, nullptr);
        vkDestroyShaderModule(s_device, missModule, nullptr);
        return false;
    }

    // Create ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipelineInfo.stageCount = 3;
    pipelineInfo.pStages = stages;
    pipelineInfo.groupCount = 3;
    pipelineInfo.pGroups = groups;
    pipelineInfo.maxPipelineRayRecursionDepth = 2;
    pipelineInfo.layout = s_rtPipelineLayout;

    if (pvkCreateRayTracingPipelinesKHR(s_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &s_rtPipeline) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create ray tracing pipeline\n");
        vkDestroyPipelineLayout(s_device, s_rtPipelineLayout, nullptr);
        s_rtPipelineLayout = VK_NULL_HANDLE;
        vkDestroyShaderModule(s_device, raygenModule, nullptr);
        vkDestroyShaderModule(s_device, chitModule, nullptr);
        vkDestroyShaderModule(s_device, missModule, nullptr);
        return false;
    }

    // Cleanup shader modules (no longer needed after pipeline creation)
    vkDestroyShaderModule(s_device, raygenModule, nullptr);
    vkDestroyShaderModule(s_device, chitModule, nullptr);
    vkDestroyShaderModule(s_device, missModule, nullptr);

    Log("[VkRT] Ray tracing pipeline created\n");
    return true;
}

// ============== CREATE SHADER BINDING TABLE ==============
static bool CreateShaderBindingTable() {
    Log("[VkRT] Creating shader binding table...\n");

    const uint32_t handleSize = s_rtProperties.shaderGroupHandleSize;
    const uint32_t handleAlignment = s_rtProperties.shaderGroupHandleAlignment;
    const uint32_t baseAlignment = s_rtProperties.shaderGroupBaseAlignment;

    // Align handle size to handle alignment
    const uint32_t handleSizeAligned = (handleSize + handleAlignment - 1) & ~(handleAlignment - 1);

    // Calculate sizes for each region (1 shader per region)
    const uint32_t raygenSize = (1 * handleSizeAligned + baseAlignment - 1) & ~(baseAlignment - 1);
    const uint32_t missSize = (1 * handleSizeAligned + baseAlignment - 1) & ~(baseAlignment - 1);
    const uint32_t hitSize = (1 * handleSizeAligned + baseAlignment - 1) & ~(baseAlignment - 1);

    const VkDeviceSize sbtSize = raygenSize + missSize + hitSize;

    // Get shader group handles
    const uint32_t groupCount = 3;
    std::vector<uint8_t> handles(groupCount * handleSize);
    if (pvkGetRayTracingShaderGroupHandlesKHR(s_device, s_rtPipeline, 0, groupCount, handles.size(), handles.data()) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to get shader group handles\n");
        return false;
    }

    // Create SBT buffer
    if (!CreateBuffer(sbtSize,
                      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      s_sbtBuffer, s_sbtMemory)) {
        Log("[VkRT] ERROR: Failed to create SBT buffer\n");
        return false;
    }

    // Map and fill SBT
    void* sbtData;
    vkMapMemory(s_device, s_sbtMemory, 0, sbtSize, 0, &sbtData);

    uint8_t* pSbt = (uint8_t*)sbtData;
    VkDeviceAddress sbtAddress = GetBufferDeviceAddress(s_sbtBuffer);

    // Raygen region
    memcpy(pSbt, handles.data() + 0 * handleSize, handleSize);
    s_raygenRegion.deviceAddress = sbtAddress;
    s_raygenRegion.stride = raygenSize;
    s_raygenRegion.size = raygenSize;

    // Miss region
    memcpy(pSbt + raygenSize, handles.data() + 1 * handleSize, handleSize);
    s_missRegion.deviceAddress = sbtAddress + raygenSize;
    s_missRegion.stride = handleSizeAligned;
    s_missRegion.size = missSize;

    // Hit region
    memcpy(pSbt + raygenSize + missSize, handles.data() + 2 * handleSize, handleSize);
    s_hitRegion.deviceAddress = sbtAddress + raygenSize + missSize;
    s_hitRegion.stride = handleSizeAligned;
    s_hitRegion.size = hitSize;

    // Callable region (empty)
    s_callableRegion = {};

    vkUnmapMemory(s_device, s_sbtMemory);

    Log("[VkRT] Shader binding table created (raygen: %llu, miss: %llu, hit: %llu)\n",
        s_raygenRegion.deviceAddress, s_missRegion.deviceAddress, s_hitRegion.deviceAddress);
    return true;
}

// ============== INITIALIZATION ==============
bool InitVulkanRT(HWND hwnd) {
    Log("[VkRT] Initializing Vulkan Ray Tracing renderer...\n");

    // Note: g_vulkanRTFeatures is already set by ShowVulkanRTSettingsDialog() in main.cpp

    // ========== Step 1: Create Vulkan Instance ==========
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RenderTestGPU - Vulkan RT";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Custom";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    const char* instanceExtensions[] = {
        VK_KHR_SURFACE_EXTENSION_NAME,
        VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };

    VkInstanceCreateInfo instanceInfo = {};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = 3;
    instanceInfo.ppEnabledExtensionNames = instanceExtensions;

    if (vkCreateInstance(&instanceInfo, nullptr, &s_instance) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create Vulkan instance\n");
        return false;
    }
    Log("[VkRT] Vulkan instance created\n");

    // ========== Step 2: Create Surface ==========
    VkWin32SurfaceCreateInfoKHR surfaceInfo = {};
    surfaceInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfaceInfo.hinstance = GetModuleHandle(nullptr);
    surfaceInfo.hwnd = hwnd;

    if (vkCreateWin32SurfaceKHR(s_instance, &surfaceInfo, nullptr, &s_surface) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create window surface\n");
        return false;
    }
    Log("[VkRT] Window surface created\n");

    // ========== Step 3: Select Physical Device ==========
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(s_instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        Log("[VkRT] ERROR: No Vulkan physical devices found\n");
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(s_instance, &deviceCount, devices.data());

    // Select first discrete GPU, or first available
    s_physicalDevice = devices[0];
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            s_physicalDevice = device;
            s_gpuName = props.deviceName;
            Log("[VkRT] Selected GPU: %s\n", s_gpuName.c_str());
            break;
        }
        if (s_gpuName.empty()) {
            s_gpuName = props.deviceName;
        }
    }
    if (s_gpuName.empty()) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(s_physicalDevice, &props);
        s_gpuName = props.deviceName;
        Log("[VkRT] Selected GPU: %s\n", s_gpuName.c_str());
    }

    if (s_physicalDevice == VK_NULL_HANDLE) {
        Log("[VkRT] ERROR: No GPU found\n");
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 4: Find Queue Families ==========
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(s_physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(s_physicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            s_graphicsFamily = i;
        }
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(s_physicalDevice, i, s_surface, &presentSupport);
        if (presentSupport) {
            s_presentFamily = i;
        }
        if (s_graphicsFamily != UINT32_MAX && s_presentFamily != UINT32_MAX) break;
    }

    if (s_graphicsFamily == UINT32_MAX || s_presentFamily == UINT32_MAX) {
        Log("[VkRT] ERROR: Could not find suitable queue families\n");
        return false;
    }

    // ========== Step 5: Create Logical Device with RT Extensions ==========
    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {s_graphicsFamily, s_presentFamily};

    for (uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Check available extensions first
    uint32_t availExtCount = 0;
    vkEnumerateDeviceExtensionProperties(s_physicalDevice, nullptr, &availExtCount, nullptr);
    std::vector<VkExtensionProperties> availExts(availExtCount);
    vkEnumerateDeviceExtensionProperties(s_physicalDevice, nullptr, &availExtCount, availExts.data());

    Log("[VkRT] Available device extensions (%u):\n", availExtCount);
    auto hasExt = [&](const char* name) {
        for (const auto& e : availExts) {
            if (strcmp(e.extensionName, name) == 0) return true;
        }
        return false;
    };

    // Required extensions
    const char* requiredExts[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        VK_KHR_SPIRV_1_4_EXTENSION_NAME,
        VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME
    };

    // Log which are available/missing
    std::vector<const char*> deviceExtensions;
    for (const char* ext : requiredExts) {
        bool avail = hasExt(ext);
        Log("[VkRT]   %s: %s\n", ext, avail ? "YES" : "NO");
        if (avail) deviceExtensions.push_back(ext);
    }

    // Check for any RT-related extensions
    Log("[VkRT] All RT-related extensions on this GPU:\n");
    for (const auto& e : availExts) {
        if (strstr(e.extensionName, "ray") || strstr(e.extensionName, "accel") ||
            strstr(e.extensionName, "RT") || strstr(e.extensionName, "trace")) {
            Log("[VkRT]   %s\n", e.extensionName);
        }
    }

    // Enable required features
    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures = {};
    bufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures = {};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.pNext = &bufferDeviceAddressFeatures;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelStructFeatures = {};
    accelStructFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    accelStructFeatures.pNext = &rtPipelineFeatures;
    accelStructFeatures.accelerationStructure = VK_TRUE;

    VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.pNext = &accelStructFeatures;

    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &deviceFeatures2;
    deviceInfo.queueCreateInfoCount = (uint32_t)queueCreateInfos.size();
    deviceInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();

    Log("[VkRT] Enabling %u extensions\n", (uint32_t)deviceExtensions.size());

    if (vkCreateDevice(s_physicalDevice, &deviceInfo, nullptr, &s_device) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create logical device\n");
        return false;
    }
    Log("[VkRT] Logical device created with RT extensions\n");

    vkGetDeviceQueue(s_device, s_graphicsFamily, 0, &s_graphicsQueue);
    vkGetDeviceQueue(s_device, s_presentFamily, 0, &s_presentQueue);

    // Load RT extension functions
    if (!LoadRTExtensions()) {
        Log("[VkRT] ERROR: Failed to load ray tracing extension functions\n");
        return false;
    }

    // Get RT properties
    s_rtProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 deviceProps2 = {};
    deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProps2.pNext = &s_rtProperties;
    vkGetPhysicalDeviceProperties2(s_physicalDevice, &deviceProps2);
    Log("[VkRT] RT Properties - Shader Group Handle Size: %u, Max Recursion: %u\n",
        s_rtProperties.shaderGroupHandleSize, s_rtProperties.maxRayRecursionDepth);

    // ========== Step 6: Create Swapchain ==========
    VkSurfaceCapabilitiesKHR surfaceCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(s_physicalDevice, s_surface, &surfaceCaps);

    s_swapchainExtent = {W, H};
    if (surfaceCaps.currentExtent.width != UINT32_MAX) {
        s_swapchainExtent = surfaceCaps.currentExtent;
    }

    uint32_t imageCount = surfaceCaps.minImageCount + 1;
    if (surfaceCaps.maxImageCount > 0 && imageCount > surfaceCaps.maxImageCount) {
        imageCount = surfaceCaps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchainInfo = {};
    swapchainInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainInfo.surface = s_surface;
    swapchainInfo.minImageCount = imageCount;
    swapchainInfo.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    swapchainInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchainInfo.imageExtent = s_swapchainExtent;
    swapchainInfo.imageArrayLayers = 1;
    swapchainInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    uint32_t queueFamilyIndices[] = {s_graphicsFamily, s_presentFamily};
    if (s_graphicsFamily != s_presentFamily) {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainInfo.queueFamilyIndexCount = 2;
        swapchainInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        swapchainInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    swapchainInfo.preTransform = surfaceCaps.currentTransform;
    swapchainInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainInfo.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;  // No vsync
    swapchainInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(s_device, &swapchainInfo, nullptr, &s_swapchain) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create swapchain\n");
        return false;
    }
    Log("[VkRT] Swapchain created (%ux%u)\n", s_swapchainExtent.width, s_swapchainExtent.height);

    // Get swapchain images
    vkGetSwapchainImagesKHR(s_device, s_swapchain, &imageCount, nullptr);
    s_swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(s_device, s_swapchain, &imageCount, s_swapchainImages.data());
    s_swapchainFormat = VK_FORMAT_B8G8R8A8_UNORM;

    // Create image views
    s_swapchainImageViews.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = s_swapchainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = s_swapchainFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(s_device, &viewInfo, nullptr, &s_swapchainImageViews[i]);
    }

    // ========== Step 7: Create Command Pool ==========
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = s_graphicsFamily;

    if (vkCreateCommandPool(s_device, &poolInfo, nullptr, &s_commandPool) != VK_SUCCESS) {
        Log("[VkRT] ERROR: Failed to create command pool\n");
        return false;
    }

    // Create command buffers
    s_commandBuffers.resize(imageCount);
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = s_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = imageCount;
    vkAllocateCommandBuffers(s_device, &allocInfo, s_commandBuffers.data());

    // ========== Step 8: Create Sync Objects ==========
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    vkCreateSemaphore(s_device, &semaphoreInfo, nullptr, &s_imageAvailableSemaphore);
    vkCreateSemaphore(s_device, &semaphoreInfo, nullptr, &s_renderFinishedSemaphore);
    vkCreateFence(s_device, &fenceInfo, nullptr, &s_inFlightFence);

    // Set GPU name for display
    gpuName = std::wstring(s_gpuName.begin(), s_gpuName.end());

    Log("[VkRT] Base Vulkan RT initialization complete!\n");
    Log("[VkRT] GPU: %s\n", s_gpuName.c_str());

    // ========== Step 9: Create Geometry Buffers ==========
    if (!CreateGeometryBuffers()) {
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 10: Create BLAS ==========
    Log("[VkRT] Creating BLAS for static geometry...\n");
    if (!CreateBLAS(s_staticVertexBuffer, s_staticIndexBuffer, s_staticVertexCount, s_staticIndexCount,
                    s_blasStatic, s_blasStaticBuffer, s_blasStaticMemory)) {
        CleanupVulkanRT();
        return false;
    }
    Log("[VkRT] Static BLAS created\n");

    Log("[VkRT] Creating BLAS for cubes...\n");
    if (!CreateBLAS(s_cubesVertexBuffer, s_cubesIndexBuffer, s_cubesVertexCount, s_cubesIndexCount,
                    s_blasCubes, s_blasCubesBuffer, s_blasCubesMemory)) {
        CleanupVulkanRT();
        return false;
    }
    Log("[VkRT] Cubes BLAS created\n");

    // ========== Step 11: Create TLAS ==========
    if (!CreateTLAS()) {
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 12: Create Output Image ==========
    if (!CreateOutputImage()) {
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 13: Create Uniform Buffer ==========
    if (!CreateUniformBuffer()) {
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 14: Create Descriptor Set Layout ==========
    if (!CreateDescriptorSetLayout()) {
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 15: Create Descriptor Set ==========
    if (!CreateDescriptorSet()) {
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 16: Create RT Pipeline ==========
    if (!CreateRTPipeline()) {
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 17: Create Shader Binding Table ==========
    if (!CreateShaderBindingTable()) {
        CleanupVulkanRT();
        return false;
    }

    // ========== Step 18: Initialize Text Rendering ==========
    if (!InitTextResources()) {
        Log("[VkRT] Warning: Text rendering initialization failed (non-fatal)\n");
        // Don't fail init - text is optional
    }

    Log("[VkRT] ===== Vulkan RT fully initialized! =====\n");
    return true;
}

void RenderVulkanRT() {
    // Debug: Log RT state once at first frame
    static bool s_firstFrame = true;
    if (s_firstFrame) {
        s_firstFrame = false;
        Log("[VkRT] === First Render Frame Debug ===\n");
        Log("[VkRT] RT Pipeline: %p\n", (void*)s_rtPipeline);
        Log("[VkRT] RT PipelineLayout: %p\n", (void*)s_rtPipelineLayout);
        Log("[VkRT] RT DescSet: %p\n", (void*)s_rtDescSet);
        Log("[VkRT] TLAS: %p\n", (void*)s_tlas);
        Log("[VkRT] Output Image: %p, View: %p\n", (void*)s_outputImage, (void*)s_outputImageView);
        Log("[VkRT] Uniform Buffer: %p (mapped: %p)\n", (void*)s_uniformBuffer, s_uniformMapped);
        Log("[VkRT] Swapchain Extent: %ux%u\n", s_swapchainExtent.width, s_swapchainExtent.height);
        Log("[VkRT] SBT Raygen: addr=%llu stride=%llu size=%llu\n",
            s_raygenRegion.deviceAddress, s_raygenRegion.stride, s_raygenRegion.size);
        Log("[VkRT] SBT Miss: addr=%llu stride=%llu size=%llu\n",
            s_missRegion.deviceAddress, s_missRegion.stride, s_missRegion.size);
        Log("[VkRT] SBT Hit: addr=%llu stride=%llu size=%llu\n",
            s_hitRegion.deviceAddress, s_hitRegion.stride, s_hitRegion.size);
        Log("[VkRT] Static geometry: %u verts, %u indices\n", s_staticVertexCount, s_staticIndexCount);
        Log("[VkRT] Cubes geometry: %u verts, %u indices\n", s_cubesVertexCount, s_cubesIndexCount);
    }

    // Wait for previous frame
    vkWaitForFences(s_device, 1, &s_inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(s_device, 1, &s_inFlightFence);

    // Acquire next image
    uint32_t imageIndex;
    vkAcquireNextImageKHR(s_device, s_swapchain, UINT64_MAX, s_imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    // Update uniform buffer with current time and feature flags
    LARGE_INTEGER currentTime;
    QueryPerformanceCounter(&currentTime);
    float elapsedTime = (float)(currentTime.QuadPart - g_startTime.QuadPart) / (float)g_perfFreq.QuadPart;

    VkRTUniforms* uniforms = (VkRTUniforms*)s_uniformMapped;
    uniforms->time = elapsedTime;
    uniforms->lightPos[0] = 0.0f;
    uniforms->lightPos[1] = 0.92f;
    uniforms->lightPos[2] = 0.0f;
    uniforms->lightRadius = g_vulkanRTFeatures.lightRadius;
    uniforms->frameCount = s_frameCount;
    uniforms->shadowSamples = g_vulkanRTFeatures.shadowSamples;
    uniforms->aoSamples = g_vulkanRTFeatures.aoSamples;
    uniforms->aoRadius = g_vulkanRTFeatures.aoRadius;

    // Build feature flags
    uniforms->features = 0;
    if (g_vulkanRTFeatures.spotlight) uniforms->features |= (1u << 0);
    if (g_vulkanRTFeatures.softShadows) uniforms->features |= (1u << 1);
    if (g_vulkanRTFeatures.ambientOcclusion) uniforms->features |= (1u << 2);
    if (g_vulkanRTFeatures.globalIllum) uniforms->features |= (1u << 3);
    if (g_vulkanRTFeatures.reflections) uniforms->features |= (1u << 4);
    if (g_vulkanRTFeatures.glassRefraction) uniforms->features |= (1u << 5);

    // Update cube transform for animation
    UpdateCubeTransform(elapsedTime);

    // Reset and begin command buffer
    VkCommandBuffer cmd = s_commandBuffers[imageIndex];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Rebuild TLAS with updated cube transform
    RebuildTLAS(cmd);

    // Bind ray tracing pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, s_rtPipeline);

    // Bind descriptor set
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, s_rtPipelineLayout,
                            0, 1, &s_rtDescSet, 0, nullptr);

    // Dispatch rays
    pvkCmdTraceRaysKHR(cmd, &s_raygenRegion, &s_missRegion, &s_hitRegion, &s_callableRegion,
                       s_swapchainExtent.width, s_swapchainExtent.height, 1);

    // Transition output image for copy
    VkImageMemoryBarrier outputBarrier = {};
    outputBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    outputBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputBarrier.image = s_outputImage;
    outputBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    outputBarrier.subresourceRange.baseMipLevel = 0;
    outputBarrier.subresourceRange.levelCount = 1;
    outputBarrier.subresourceRange.baseArrayLayer = 0;
    outputBarrier.subresourceRange.layerCount = 1;
    outputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    // Transition swapchain image for copy
    VkImageMemoryBarrier swapBarrier = {};
    swapBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    swapBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    swapBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    swapBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    swapBarrier.image = s_swapchainImages[imageIndex];
    swapBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    swapBarrier.subresourceRange.baseMipLevel = 0;
    swapBarrier.subresourceRange.levelCount = 1;
    swapBarrier.subresourceRange.baseArrayLayer = 0;
    swapBarrier.subresourceRange.layerCount = 1;
    swapBarrier.srcAccessMask = 0;
    swapBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkImageMemoryBarrier barriers[2] = {outputBarrier, swapBarrier};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 2, barriers);

    // Copy output image to swapchain
    VkImageCopy copyRegion = {};
    copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.srcSubresource.layerCount = 1;
    copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.dstSubresource.layerCount = 1;
    copyRegion.extent = {s_swapchainExtent.width, s_swapchainExtent.height, 1};

    vkCmdCopyImage(cmd, s_outputImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   s_swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   1, &copyRegion);

    // Transition output image back to general for next frame
    outputBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    outputBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                         0, 0, nullptr, 0, nullptr, 1, &outputBarrier);

    // ========== TEXT RENDERING ==========
    if (s_textPipeline && s_textVertexMapped && !s_framebuffers.empty()) {
        // Calculate FPS
        static LARGE_INTEGER lastFpsTime = {0};
        static int fpsFrameCount = 0;
        static float displayFps = 0.0f;
        LARGE_INTEGER fpsTime;
        QueryPerformanceCounter(&fpsTime);
        fpsFrameCount++;
        double fpsElapsed = (double)(fpsTime.QuadPart - lastFpsTime.QuadPart) / g_perfFreq.QuadPart;
        if (fpsElapsed >= 0.5) {
            displayFps = (float)(fpsFrameCount / fpsElapsed);
            fpsFrameCount = 0;
            lastFpsTime = fpsTime;
        }

        // Build text string
        char textBuf[512];
        snprintf(textBuf, sizeof(textBuf), "API: Vulkan RT (VK_KHR_ray_tracing_pipeline)\nGPU: %s\nFPS: %.0f\nTriangles: %d\nResolution: %ux%u",
                 s_gpuName.c_str(), displayFps, 200,  // Approximate triangle count for RT
                 s_swapchainExtent.width, s_swapchainExtent.height);

        // Build vertices
        s_textVertCount = 0;
        float scale = 1.5f;
        float shadowOff = 2.0f;
        float charW = 8.0f * scale;
        float charH = 8.0f * scale;
        float textX = 10.0f, textY = 10.0f;

        // Lambda to add text
        auto addTextVerts = [&](const char* text, float startX, float startY, float cr, float cg, float cb, float ca) {
            float cx = startX, cy = startY;
            for (const char* p = text; *p && s_textVertCount < 5994; p++) {
                if (*p == '\n') { cx = startX; cy += charH * 1.4f; continue; }
                if (*p < 32 || *p > 127) continue;
                int idx = *p - 32;
                int col = idx % 16, row = idx / 16;
                float u0 = col * 8.0f / 128.0f, v0 = row * 8.0f / 48.0f;
                float u1 = u0 + 8.0f / 128.0f, v1 = v0 + 8.0f / 48.0f;
                float x0 = cx / s_swapchainExtent.width * 2.0f - 1.0f;
                float y0 = cy / s_swapchainExtent.height * 2.0f - 1.0f;
                float x1 = (cx + charW) / s_swapchainExtent.width * 2.0f - 1.0f;
                float y1 = (cy + charH) / s_swapchainExtent.height * 2.0f - 1.0f;

                s_textVerts[s_textVertCount++] = {x0, y0, u0, v0, cr, cg, cb, ca};
                s_textVerts[s_textVertCount++] = {x1, y0, u1, v0, cr, cg, cb, ca};
                s_textVerts[s_textVertCount++] = {x0, y1, u0, v1, cr, cg, cb, ca};
                s_textVerts[s_textVertCount++] = {x1, y0, u1, v0, cr, cg, cb, ca};
                s_textVerts[s_textVertCount++] = {x1, y1, u1, v1, cr, cg, cb, ca};
                s_textVerts[s_textVertCount++] = {x0, y1, u0, v1, cr, cg, cb, ca};
                cx += charW;
            }
        };

        // Shadow + main text
        addTextVerts(textBuf, textX + shadowOff, textY + shadowOff, 0.0f, 0.0f, 0.0f, 0.7f);
        addTextVerts(textBuf, textX, textY, 1.0f, 1.0f, 1.0f, 1.0f);

        if (s_textVertCount > 0) {
            // Copy to GPU
            memcpy(s_textVertexMapped, s_textVerts, s_textVertCount * sizeof(TextVert));

            // Begin render pass (swapchain is in TRANSFER_DST_OPTIMAL, pass will transition to PRESENT_SRC_KHR)
            VkRenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = s_textRenderPass;
            renderPassInfo.framebuffer = s_framebuffers[imageIndex];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = s_swapchainExtent;

            vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_textPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_textPipelineLayout,
                                    0, 1, &s_textDescSet, 0, nullptr);

            VkBuffer textVBs[] = { s_textVertexBuffer };
            VkDeviceSize textOffsets[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, textVBs, textOffsets);
            vkCmdDraw(cmd, s_textVertCount, 1, 0, 0);

            vkCmdEndRenderPass(cmd);
        } else {
            // No text - still need to transition swapchain to present
            swapBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            swapBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            swapBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            swapBarrier.dstAccessMask = 0;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &swapBarrier);
        }
    } else {
        // No text resources - transition swapchain for present
        swapBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        swapBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        swapBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        swapBarrier.dstAccessMask = 0;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &swapBarrier);
    }

    vkEndCommandBuffer(cmd);

    // Submit
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &s_imageAvailableSemaphore;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &s_renderFinishedSemaphore;

    vkQueueSubmit(s_graphicsQueue, 1, &submitInfo, s_inFlightFence);

    // Present
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &s_renderFinishedSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &s_swapchain;
    presentInfo.pImageIndices = &imageIndex;

    vkQueuePresentKHR(s_presentQueue, &presentInfo);

    s_frameCount++;
}

void CleanupVulkanRT() {
    Log("[VkRT] Cleanup\n");

    if (s_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(s_device);
    }

    // Release all resources in reverse order of creation
    #define SAFE_DESTROY_BUFFER(buf, mem) \
        if (buf != VK_NULL_HANDLE) { vkDestroyBuffer(s_device, buf, nullptr); buf = VK_NULL_HANDLE; } \
        if (mem != VK_NULL_HANDLE) { vkFreeMemory(s_device, mem, nullptr); mem = VK_NULL_HANDLE; }

    #define SAFE_DESTROY_IMAGE(img, mem, view) \
        if (view != VK_NULL_HANDLE) { vkDestroyImageView(s_device, view, nullptr); view = VK_NULL_HANDLE; } \
        if (img != VK_NULL_HANDLE) { vkDestroyImage(s_device, img, nullptr); img = VK_NULL_HANDLE; } \
        if (mem != VK_NULL_HANDLE) { vkFreeMemory(s_device, mem, nullptr); mem = VK_NULL_HANDLE; }

    // Text resources
    SAFE_DESTROY_BUFFER(s_textVertexBuffer, s_textVertexMemory);
    if (s_fontSampler) { vkDestroySampler(s_device, s_fontSampler, nullptr); s_fontSampler = VK_NULL_HANDLE; }
    SAFE_DESTROY_IMAGE(s_fontImage, s_fontMemory, s_fontImageView);
    if (s_textPipeline) { vkDestroyPipeline(s_device, s_textPipeline, nullptr); s_textPipeline = VK_NULL_HANDLE; }
    if (s_textPipelineLayout) { vkDestroyPipelineLayout(s_device, s_textPipelineLayout, nullptr); s_textPipelineLayout = VK_NULL_HANDLE; }
    if (s_textDescPool) { vkDestroyDescriptorPool(s_device, s_textDescPool, nullptr); s_textDescPool = VK_NULL_HANDLE; }
    if (s_textDescSetLayout) { vkDestroyDescriptorSetLayout(s_device, s_textDescSetLayout, nullptr); s_textDescSetLayout = VK_NULL_HANDLE; }
    if (s_textRenderPass) { vkDestroyRenderPass(s_device, s_textRenderPass, nullptr); s_textRenderPass = VK_NULL_HANDLE; }

    // RT resources
    SAFE_DESTROY_BUFFER(s_sbtBuffer, s_sbtMemory);
    if (s_rtPipeline) { vkDestroyPipeline(s_device, s_rtPipeline, nullptr); s_rtPipeline = VK_NULL_HANDLE; }
    if (s_rtPipelineLayout) { vkDestroyPipelineLayout(s_device, s_rtPipelineLayout, nullptr); s_rtPipelineLayout = VK_NULL_HANDLE; }
    if (s_rtDescPool) { vkDestroyDescriptorPool(s_device, s_rtDescPool, nullptr); s_rtDescPool = VK_NULL_HANDLE; }
    if (s_rtDescSetLayout) { vkDestroyDescriptorSetLayout(s_device, s_rtDescSetLayout, nullptr); s_rtDescSetLayout = VK_NULL_HANDLE; }

    SAFE_DESTROY_IMAGE(s_outputImage, s_outputMemory, s_outputImageView);
    SAFE_DESTROY_BUFFER(s_uniformBuffer, s_uniformMemory);

    // Acceleration structures
    if (s_tlas && pvkDestroyAccelerationStructureKHR) {
        pvkDestroyAccelerationStructureKHR(s_device, s_tlas, nullptr);
        s_tlas = VK_NULL_HANDLE;
    }
    SAFE_DESTROY_BUFFER(s_tlasBuffer, s_tlasMemory);
    SAFE_DESTROY_BUFFER(s_instanceBuffer, s_instanceMemory);
    SAFE_DESTROY_BUFFER(s_tlasScratchBuffer, s_tlasScratchMemory);

    if (s_blasCubes && pvkDestroyAccelerationStructureKHR) {
        pvkDestroyAccelerationStructureKHR(s_device, s_blasCubes, nullptr);
        s_blasCubes = VK_NULL_HANDLE;
    }
    SAFE_DESTROY_BUFFER(s_blasCubesBuffer, s_blasCubesMemory);

    if (s_blasStatic && pvkDestroyAccelerationStructureKHR) {
        pvkDestroyAccelerationStructureKHR(s_device, s_blasStatic, nullptr);
        s_blasStatic = VK_NULL_HANDLE;
    }
    SAFE_DESTROY_BUFFER(s_blasStaticBuffer, s_blasStaticMemory);

    // Geometry buffers
    SAFE_DESTROY_BUFFER(s_cubesVertexBuffer, s_cubesVertexMemory);
    SAFE_DESTROY_BUFFER(s_cubesIndexBuffer, s_cubesIndexMemory);
    SAFE_DESTROY_BUFFER(s_staticVertexBuffer, s_staticVertexMemory);
    SAFE_DESTROY_BUFFER(s_staticIndexBuffer, s_staticIndexMemory);

    // Framebuffers
    for (auto fb : s_framebuffers) {
        if (fb) vkDestroyFramebuffer(s_device, fb, nullptr);
    }
    s_framebuffers.clear();

    // Swapchain
    for (auto iv : s_swapchainImageViews) {
        if (iv) vkDestroyImageView(s_device, iv, nullptr);
    }
    s_swapchainImageViews.clear();
    s_swapchainImages.clear();
    if (s_swapchain) { vkDestroySwapchainKHR(s_device, s_swapchain, nullptr); s_swapchain = VK_NULL_HANDLE; }

    // Sync objects
    if (s_inFlightFence) { vkDestroyFence(s_device, s_inFlightFence, nullptr); s_inFlightFence = VK_NULL_HANDLE; }
    if (s_renderFinishedSemaphore) { vkDestroySemaphore(s_device, s_renderFinishedSemaphore, nullptr); s_renderFinishedSemaphore = VK_NULL_HANDLE; }
    if (s_imageAvailableSemaphore) { vkDestroySemaphore(s_device, s_imageAvailableSemaphore, nullptr); s_imageAvailableSemaphore = VK_NULL_HANDLE; }

    // Command pool
    if (s_commandPool) { vkDestroyCommandPool(s_device, s_commandPool, nullptr); s_commandPool = VK_NULL_HANDLE; }

    // Device and instance
    if (s_device) { vkDestroyDevice(s_device, nullptr); s_device = VK_NULL_HANDLE; }
    if (s_surface) { vkDestroySurfaceKHR(s_instance, s_surface, nullptr); s_surface = VK_NULL_HANDLE; }
    if (s_instance) { vkDestroyInstance(s_instance, nullptr); s_instance = VK_NULL_HANDLE; }

    #undef SAFE_DESTROY_BUFFER
    #undef SAFE_DESTROY_IMAGE

    Log("[VkRT] Cleanup complete\n");
}
