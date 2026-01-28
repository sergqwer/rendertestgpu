// ============== OPENGL RENDERER IMPLEMENTATION ==============
// Extracted from main.cpp

// Windows headers MUST come before GL headers for WINGDIAPI/APIENTRY
#include <Windows.h>

// OpenGL headers
#include <GL/gl.h>
#include <GL/glu.h>
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")

#include "../common.h"
#include <vector>
#include <cstring>

// ============== EXTERN DECLARATIONS ==============
// Shared globals from main module (most are in common.h)
// g_hMainWnd is declared in common.h

// ============== OPENGL GLOBALS ==============
static HDC g_glHDC = nullptr;
static HGLRC g_glRC = nullptr;
static GLuint g_glFontBase = 0;
static GLuint g_glCubeLists[8] = {0};  // Display lists for 8 cubes
static int g_glTriangleCount = 0;

// ============== OPENGL VERTEX STRUCTURE ==============
// OpenGL vertex structure matching D3D11
struct GLVert {
    float px, py, pz;
    float nx, ny, nz;
};

// ============== GEOMETRY GENERATION ==============

// Generate rounded face - port of D3D11's GenRoundedFace
void GenRoundedFaceGL(float size, int seg, float offX, float offY, float offZ, int faceIdx,
    float edgeRadius[4], std::vector<GLVert>& verts, std::vector<unsigned int>& inds)
{
    unsigned int base = (unsigned int)verts.size();
    float h = size / 2;

    // Face normals, U and V directions for each face
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
                        float r = fmaxf(rU, rV);
                        float dist = sqrtf(dx*dx + dy*dy);
                        if (dist > r) { dx = dx * r / dist; dy = dy * r / dist; }
                        float curveZ = sqrtf(fmaxf(0, r*r - dx*dx - dy*dy));
                        pz = (h - r) + curveZ;
                        px = (u > 0 ? 1 : -1) * (innerU + dx);
                        py = (vv > 0 ? 1 : -1) * (innerV + dy);
                        nx = (u > 0 ? 1 : -1) * dx / r;
                        ny = (vv > 0 ? 1 : -1) * dy / r;
                        nz = curveZ / r;
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
                        float r = (dx > 0) ? rU : rV;
                        float d = (dx > 0) ? dx : dy;
                        float curveZ = sqrtf(fmaxf(0, r*r - d*d));
                        pz = (h - r) + curveZ;
                        if (dx > 0) { px = (u > 0 ? 1 : -1) * (innerU + dx); nx = (u > 0 ? 1 : -1) * dx / r; }
                        else { py = (vv > 0 ? 1 : -1) * (innerV + dy); ny = (vv > 0 ? 1 : -1) * dy / r; }
                        nz = curveZ / r;
                    }
                }
            }

            GLVert vert;
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
            verts.push_back(vert);
        }
    }

    for (int j = 0; j < seg; j++) {
        for (int i = 0; i < seg; i++) {
            unsigned int idx = base + j * (seg + 1) + i;
            inds.push_back(idx); inds.push_back(idx + seg + 1); inds.push_back(idx + 1);
            inds.push_back(idx + 1); inds.push_back(idx + seg + 1); inds.push_back(idx + seg + 2);
        }
    }
}

// Build all geometry for one cube (matching D3D11's BuildAllGeometry for single cube)
void BuildCubeGeometryGL(int cubeID, std::vector<GLVert>& verts, std::vector<unsigned int>& inds)
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
        GenRoundedFaceGL(cubeSize, seg, posX, posY, posZ, f, er, verts, inds);
    }
}

// ============== ERROR CHECKING ==============

// Helper to check and log OpenGL errors
static bool CheckGLError(const char* operation)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        const char* errStr = "Unknown";
        switch (err) {
            case GL_INVALID_ENUM: errStr = "GL_INVALID_ENUM"; break;
            case GL_INVALID_VALUE: errStr = "GL_INVALID_VALUE"; break;
            case GL_INVALID_OPERATION: errStr = "GL_INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW: errStr = "GL_STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW: errStr = "GL_STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY: errStr = "GL_OUT_OF_MEMORY"; break;
        }
        Log("[ERROR] OpenGL error after %s: %s (0x%X)\n", operation, errStr, err);
        return false;
    }
    return true;
}

// ============== INITIALIZATION ==============

bool InitOpenGL(HWND hwnd)
{
    Log("[INFO] Initializing OpenGL...\n");

    g_glHDC = GetDC(hwnd);
    if (!g_glHDC) {
        Log("[ERROR] Failed to get device context\n");
        return false;
    }

    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int pixelFormat = ChoosePixelFormat(g_glHDC, &pfd);
    if (!pixelFormat || !SetPixelFormat(g_glHDC, pixelFormat, &pfd)) {
        Log("[ERROR] Failed to set pixel format\n");
        ReleaseDC(hwnd, g_glHDC);
        g_glHDC = nullptr;
        return false;
    }

    g_glRC = wglCreateContext(g_glHDC);
    if (!g_glRC || !wglMakeCurrent(g_glHDC, g_glRC)) {
        Log("[ERROR] Failed to create/activate OpenGL context\n");
        if (g_glRC) wglDeleteContext(g_glRC);
        ReleaseDC(hwnd, g_glHDC);
        g_glRC = nullptr;
        g_glHDC = nullptr;
        return false;
    }

    const char* vendor = (const char*)glGetString(GL_VENDOR);
    const char* renderer = (const char*)glGetString(GL_RENDERER);
    const char* version = (const char*)glGetString(GL_VERSION);
    Log("[INFO] OpenGL Vendor: %s\n", vendor ? vendor : "Unknown");
    Log("[INFO] OpenGL Renderer: %s\n", renderer ? renderer : "Unknown");
    Log("[INFO] OpenGL Version: %s\n", version ? version : "Unknown");

    // Clear any pending errors
    while (glGetError() != GL_NO_ERROR) {}

    // Setup OpenGL state
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);  // Counter-clockwise due to face transform handedness

    glDisable(GL_LIGHTING);
    glShadeModel(GL_SMOOTH);

    if (!CheckGLError("basic state setup")) {
        Log("[WARN] OpenGL state setup had errors, continuing...\n");
    }

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)W / (double)H, 0.1, 100.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

    if (!CheckGLError("matrix/clear setup")) {
        Log("[WARN] Matrix setup had errors, continuing...\n");
    }

    // Create bitmap font - may fail on non-Windows OpenGL implementations
    Log("[INFO] Creating bitmap font...\n");
    g_glFontBase = glGenLists(96);
    if (g_glFontBase == 0) {
        Log("[WARN] glGenLists(96) returned 0 - font display lists not available\n");
    } else {
        HFONT font = CreateFontA(14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
            ANSI_CHARSET, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
            FF_DONTCARE | DEFAULT_PITCH, "Consolas");
        if (font) {
            HFONT oldFont = (HFONT)SelectObject(g_glHDC, font);
            BOOL fontResult = wglUseFontBitmaps(g_glHDC, 32, 96, g_glFontBase);
            if (!fontResult) {
                Log("[WARN] wglUseFontBitmaps failed (error %lu) - text overlay disabled\n", GetLastError());
                glDeleteLists(g_glFontBase, 96);
                g_glFontBase = 0;
            }
            SelectObject(g_glHDC, oldFont);
            DeleteObject(font);
        } else {
            Log("[WARN] CreateFontA failed - text overlay disabled\n");
            glDeleteLists(g_glFontBase, 96);
            g_glFontBase = 0;
        }
    }
    CheckGLError("font creation");

    // Build rounded cube geometry for each of the 8 cubes (matching D3D11 exactly)
    Log("[INFO] Building rounded cube geometry...\n");
    g_glTriangleCount = 0;

    for (int c = 0; c < 8; c++) {
        std::vector<GLVert> verts;
        std::vector<unsigned int> inds;
        BuildCubeGeometryGL(c, verts, inds);

        Log("[INFO] Cube %d: %zu vertices, %zu indices\n", c, verts.size(), inds.size());

        g_glCubeLists[c] = glGenLists(1);
        if (g_glCubeLists[c] == 0) {
            Log("[ERROR] glGenLists(1) failed for cube %d\n", c);
            continue;
        }

        glNewList(g_glCubeLists[c], GL_COMPILE);
        glBegin(GL_TRIANGLES);
        for (size_t i = 0; i < inds.size(); i++) {
            GLVert& v = verts[inds[i]];
            glNormal3f(v.nx, v.ny, v.nz);
            glVertex3f(v.px, v.py, v.pz);
        }
        glEnd();
        glEndList();

        if (!CheckGLError("display list creation")) {
            Log("[ERROR] Failed to create display list for cube %d\n", c);
        }

        g_glTriangleCount += (int)inds.size() / 3;
    }

    Log("[INFO] OpenGL geometry: %d triangles total\n", g_glTriangleCount);

    // Final error check
    if (!CheckGLError("initialization complete")) {
        Log("[WARN] OpenGL initialization completed with errors\n");
    }

    Log("[INFO] OpenGL initialization complete\n");
    return true;
}

// ============== TEXT RENDERING ==============

void DrawTextGL(const char* text, float x, float y)
{
    // Skip if font not available (e.g., on Mesa/Zink)
    if (g_glFontBase == 0) return;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, W, H, 0, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);

    // Shadow
    glColor3f(0.0f, 0.0f, 0.0f);
    glRasterPos2f(x + 1.5f, y + 1.5f);
    glPushAttrib(GL_LIST_BIT);
    glListBase(g_glFontBase - 32);
    glCallLists((GLsizei)strlen(text), GL_UNSIGNED_BYTE, text);
    glPopAttrib();

    // Main text
    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2f(x, y);
    glPushAttrib(GL_LIST_BIT);
    glListBase(g_glFontBase - 32);
    glCallLists((GLsizei)strlen(text), GL_UNSIGNED_BYTE, text);
    glPopAttrib();

    glEnable(GL_DEPTH_TEST);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// ============== RENDERING ==============

void RenderOpenGL()
{
    static int frameNum = 0;
    static bool errorLogged = false;
    frameNum++;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Get time for animation
    LARGE_INTEGER nowTime;
    QueryPerformanceCounter(&nowTime);
    float t = (float)(nowTime.QuadPart - g_startTime.QuadPart) / g_perfFreq.QuadPart;

    // D3D11 colors for 8 cubes (matching exactly)
    const float colors[8][3] = {
        {0.95f, 0.20f, 0.15f},  // 0: Red
        {0.20f, 0.70f, 0.30f},  // 1: Green
        {0.15f, 0.50f, 0.95f},  // 2: Blue
        {1.00f, 0.85f, 0.00f},  // 3: Yellow
        {1.00f, 0.85f, 0.00f},  // 4: Yellow
        {0.15f, 0.50f, 0.95f},  // 5: Blue
        {0.20f, 0.70f, 0.30f},  // 6: Green
        {0.95f, 0.20f, 0.15f}   // 7: Red
    };

    // Setup OpenGL fixed-function lighting to match D3D11 shader:
    // D3D11 formula: color = baseColor * (diffuse * 0.65 + 0.35)
    // OpenGL: color = ambient_light * material + diffuse_light * material * dot(N,L)
    // Set ambient_light = 0.35, diffuse_light = 0.65 to match
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    // Light direction (normalized) - same as D3D11: normalize(0.2, 1.0, 0.3)
    // For directional light in OpenGL, w=0
    float lx = 0.2f, ly = 1.0f, lz = 0.3f;
    float llen = sqrtf(lx*lx + ly*ly + lz*lz);
    lx /= llen; ly /= llen; lz /= llen;
    GLfloat lightPos[] = {lx, ly, lz, 0.0f};  // Directional light (w=0)

    // Ambient light = 0.35 (constant term in D3D11 formula)
    GLfloat ambientLight[] = {0.35f, 0.35f, 0.35f, 1.0f};
    // Diffuse light = 0.65 (multiplier for dot(N,L) in D3D11 formula)
    GLfloat diffuseLight[] = {0.65f, 0.65f, 0.65f, 1.0f};

    GLfloat specularLight[] = {0.0f, 0.0f, 0.0f, 1.0f};  // No specular
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
    glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);

    // Set light position in world space (before model rotation)
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    // Setup view matrix: camera at z=4 looking at origin (D3D11 style)
    glTranslatef(0, 0, -4);  // Camera at (0, 0, 4) looking at origin

    // Apply combined rotation: RotY(Time*1.2) * RotX(Time*0.7) - whole scene rotates
    float rotY = t * 1.2f * 180.0f / 3.14159265f;  // Convert to degrees
    float rotX = t * 0.7f * 180.0f / 3.14159265f;
    glRotatef(rotX, 1, 0, 0);
    glRotatef(rotY, 0, 1, 0);

    // Enable color material so glColor affects material properties
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    // Draw 8 cubes using rounded display lists with per-vertex lighting
    // Note: Display lists already contain geometry positioned at +/-half offsets,
    // so no additional translation is needed here
    for (int i = 0; i < 8; i++) {
        if (g_glCubeLists[i] == 0) continue;  // Skip if display list not created

        // Set material color (affects both ambient and diffuse via GL_COLOR_MATERIAL)
        glColor3f(colors[i][0], colors[i][1], colors[i][2]);

        // Call the pre-built rounded cube display list (already positioned)
        glCallList(g_glCubeLists[i]);
    }

    // Check for errors after rendering (only log once)
    if (!errorLogged && !CheckGLError("cube rendering")) {
        Log("[ERROR] OpenGL error during cube rendering at frame %d\n", frameNum);
        errorLogged = true;
    }

    // Disable lighting for text overlay
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);

    // Draw text overlay - use OpenGL's own GPU name
    const char* glRenderer = (const char*)glGetString(GL_RENDERER);
    if (!glRenderer) glRenderer = "Unknown";

    char infoText[512];
    sprintf_s(infoText, "API: OpenGL\nGPU: %s\nFPS: %d\nTriangles: %d\nResolution: %ux%u",
        glRenderer, fps, g_glTriangleCount, W, H);

    char* context = nullptr;
    char* line = strtok_s(infoText, "\n", &context);
    float textY = 20.0f;
    while (line) {
        DrawTextGL(line, 10.0f, textY);
        textY += 16.0f;
        line = strtok_s(nullptr, "\n", &context);
    }

    if (!SwapBuffers(g_glHDC)) {
        if (!errorLogged) {
            Log("[ERROR] SwapBuffers failed at frame %d (error %lu)\n", frameNum, GetLastError());
            errorLogged = true;
        }
    }
}

// ============== CLEANUP ==============

void CleanupOpenGL()
{
    Log("[INFO] Cleaning up OpenGL...\n");

    // Delete all 8 rounded cube display lists
    for (int i = 0; i < 8; i++) {
        if (g_glCubeLists[i]) {
            glDeleteLists(g_glCubeLists[i], 1);
            g_glCubeLists[i] = 0;
        }
    }

    if (g_glFontBase) {
        glDeleteLists(g_glFontBase, 96);
        g_glFontBase = 0;
    }

    if (g_glRC) {
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(g_glRC);
        g_glRC = nullptr;
    }

    if (g_glHDC && g_hMainWnd) {
        ReleaseDC(g_hMainWnd, g_glHDC);
        g_glHDC = nullptr;
    }

    Log("[INFO] OpenGL cleanup complete\n");
}
