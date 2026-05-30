package com.vladih.computer_vision.flutter_vision.utils;

import android.graphics.Bitmap;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES30;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * GPU-accelerated NV21 → RGBA bitmap converter using OpenGL ES 3.0.
 * Uses an offscreen EGL pbuffer context so no window/surface is required.
 */
public class GlYuvConverter implements AutoCloseable {
    private static final String TAG = "GlYuvConverter";

    private final EGLDisplay eglDisplay;
    private final EGLContext eglContext;
    private final EGLSurface eglSurface;

    private final int program;
    private final int quadVbo;
    private final int yTex;
    private final int uvTex;
    private final int fbo;

    private int outputTex  = 0;
    private int lastWidth  = -1;
    private int lastHeight = -1;

    // Per-frame buffers — reallocated only when dimensions change
    private ByteBuffer yBuf   = null;
    private ByteBuffer uvBuf  = null;
    private ByteBuffer pixBuf = null;

    // ------------------------------------------------------------------ shaders

    private static final String VERTEX_SHADER =
        "#version 300 es\n" +
        "in vec2 aPos;\n" +
        "in vec2 aUV;\n" +
        "out vec2 vUV;\n" +
        "void main() {\n" +
        "    gl_Position = vec4(aPos, 0.0, 1.0);\n" +
        "    vUV = aUV;\n" +
        "}\n";

    // NV21 VU plane: first byte = V (Cr), second byte = U (Cb).
    // BT.601 limited range: Y in [16,235], U/V in [16,240].
    private static final String FRAGMENT_SHADER =
        "#version 300 es\n" +
        "precision mediump float;\n" +
        "uniform sampler2D uY;\n" +
        "uniform sampler2D uVU;\n" +
        "in  vec2 vUV;\n" +
        "out vec4 fragColor;\n" +
        "void main() {\n" +
        "    float y = (texture(uY,  vUV).r - 16.0/255.0) * (255.0/219.0);\n" +
        "    vec2  vu = texture(uVU, vUV).rg;\n" +
        "    float v  = (vu.r - 128.0/255.0) * (255.0/224.0);\n" +  // Cr
        "    float u  = (vu.g - 128.0/255.0) * (255.0/224.0);\n" +  // Cb
        "    float r  = clamp(y + 1.596 * v,              0.0, 1.0);\n" +
        "    float g  = clamp(y - 0.391 * u - 0.813 * v,  0.0, 1.0);\n" +
        "    float b  = clamp(y + 2.018 * u,              0.0, 1.0);\n" +
        "    fragColor = vec4(r, g, b, 1.0);\n" +
        "}\n";

    // Full-screen triangle strip.
    // Screen bottom (y = -1) → texcoord y = 0 → first row of NV21 data (image top).
    // This makes glReadPixels (which reads from y=0 up) return rows top-to-bottom,
    // matching Bitmap.copyPixelsFromBuffer's expected order.
    private static final float[] QUAD = {
        -1f, -1f,  0f, 0f,
         1f, -1f,  1f, 0f,
        -1f,  1f,  0f, 1f,
         1f,  1f,  1f, 1f,
    };

    // ------------------------------------------------------------------ init

    public GlYuvConverter() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        int[] ver = new int[2];
        if (!EGL14.eglInitialize(eglDisplay, ver, 0, ver, 1)) {
            throw new RuntimeException("eglInitialize failed: " + EGL14.eglGetError());
        }

        int[] cfgAttribs = {
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE,    EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RED_SIZE,        8,
            EGL14.EGL_GREEN_SIZE,      8,
            EGL14.EGL_BLUE_SIZE,       8,
            EGL14.EGL_ALPHA_SIZE,      8,
            EGL14.EGL_NONE
        };
        EGLConfig[] cfgs = new EGLConfig[1];
        int[] numCfgs = new int[1];
        EGL14.eglChooseConfig(eglDisplay, cfgAttribs, 0, cfgs, 0, 1, numCfgs, 0);
        if (numCfgs[0] == 0) throw new RuntimeException("No suitable EGL config");

        int[] ctxAttribs = { EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE };
        eglContext = EGL14.eglCreateContext(eglDisplay, cfgs[0], EGL14.EGL_NO_CONTEXT, ctxAttribs, 0);
        if (eglContext == EGL14.EGL_NO_CONTEXT) {
            throw new RuntimeException("eglCreateContext failed: " + EGL14.eglGetError());
        }

        int[] pbAttribs = { EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE };
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, cfgs[0], pbAttribs, 0);
        if (eglSurface == EGL14.EGL_NO_SURFACE) {
            throw new RuntimeException("eglCreatePbufferSurface failed: " + EGL14.eglGetError());
        }

        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);

        program = buildProgram(VERTEX_SHADER, FRAGMENT_SHADER);

        // Quad VBO
        FloatBuffer fb = ByteBuffer.allocateDirect(QUAD.length * 4)
            .order(ByteOrder.nativeOrder()).asFloatBuffer();
        fb.put(QUAD).rewind();
        int[] buf = new int[1];
        GLES30.glGenBuffers(1, buf, 0);
        quadVbo = buf[0];
        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, quadVbo);
        GLES30.glBufferData(GLES30.GL_ARRAY_BUFFER, QUAD.length * 4, fb, GLES30.GL_STATIC_DRAW);

        // Input textures (Y and VU)
        int[] tex = new int[2];
        GLES30.glGenTextures(2, tex, 0);
        yTex  = tex[0];
        uvTex = tex[1];
        for (int t : tex) {
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, t);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_LINEAR);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_S,     GLES30.GL_CLAMP_TO_EDGE);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_WRAP_T,     GLES30.GL_CLAMP_TO_EDGE);
        }

        // Framebuffer
        int[] fboArr = new int[1];
        GLES30.glGenFramebuffers(1, fboArr, 0);
        fbo = fboArr[0];

        Log.d(TAG, "GlYuvConverter ready (OpenGL ES 3.0)");
    }

    // ------------------------------------------------------------------ convert

    public Bitmap convert(byte[] nv21, int width, int height) {
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);

        int ySize  = width * height;
        int uvSize = ySize / 2;

        // Reallocate output texture and FBO attachment when dimensions change
        if (width != lastWidth || height != lastHeight) {
            if (outputTex != 0) GLES30.glDeleteTextures(1, new int[]{outputTex}, 0);
            int[] tex = new int[1];
            GLES30.glGenTextures(1, tex, 0);
            outputTex = tex[0];
            GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, outputTex);
            GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RGBA8,
                width, height, 0, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, null);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST);
            GLES30.glTexParameteri(GLES30.GL_TEXTURE_2D, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_NEAREST);
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo);
            GLES30.glFramebufferTexture2D(GLES30.GL_FRAMEBUFFER, GLES30.GL_COLOR_ATTACHMENT0,
                GLES30.GL_TEXTURE_2D, outputTex, 0);

            // Reallocate cached CPU buffers
            yBuf   = ByteBuffer.allocateDirect(ySize).order(ByteOrder.nativeOrder());
            uvBuf  = ByteBuffer.allocateDirect(uvSize).order(ByteOrder.nativeOrder());
            pixBuf = ByteBuffer.allocateDirect(ySize * 4).order(ByteOrder.nativeOrder());

            lastWidth  = width;
            lastHeight = height;
        }

        // Upload Y plane: full resolution, single channel (GL_R8)
        yBuf.clear();
        yBuf.put(nv21, 0, ySize);
        yBuf.rewind();
        GLES30.glActiveTexture(GLES30.GL_TEXTURE0);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, yTex);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_R8,
            width, height, 0, GLES30.GL_RED, GLES30.GL_UNSIGNED_BYTE, yBuf);

        // Upload VU plane: half resolution, two-channel GL_RG8 (V=.r, U=.g)
        uvBuf.clear();
        uvBuf.put(nv21, ySize, uvSize);
        uvBuf.rewind();
        GLES30.glActiveTexture(GLES30.GL_TEXTURE1);
        GLES30.glBindTexture(GLES30.GL_TEXTURE_2D, uvTex);
        GLES30.glTexImage2D(GLES30.GL_TEXTURE_2D, 0, GLES30.GL_RG8,
            width / 2, height / 2, 0, GLES30.GL_RG, GLES30.GL_UNSIGNED_BYTE, uvBuf);

        // Render
        GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, fbo);
        GLES30.glViewport(0, 0, width, height);
        GLES30.glUseProgram(program);
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uY"),  0);
        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "uVU"), 1);

        GLES30.glBindBuffer(GLES30.GL_ARRAY_BUFFER, quadVbo);
        int posLoc = GLES30.glGetAttribLocation(program, "aPos");
        int uvLoc  = GLES30.glGetAttribLocation(program, "aUV");
        GLES30.glEnableVertexAttribArray(posLoc);
        GLES30.glVertexAttribPointer(posLoc, 2, GLES30.GL_FLOAT, false, 16, 0);
        GLES30.glEnableVertexAttribArray(uvLoc);
        GLES30.glVertexAttribPointer(uvLoc,  2, GLES30.GL_FLOAT, false, 16, 8);
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4);

        // Read pixels back to CPU
        pixBuf.clear();
        GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, pixBuf);
        pixBuf.rewind();

        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(pixBuf);
        return bitmap;
    }

    // ------------------------------------------------------------------ cleanup

    @Override
    public void close() {
        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext);
        if (outputTex != 0) GLES30.glDeleteTextures(1,    new int[]{outputTex}, 0);
        GLES30.glDeleteTextures(1,     new int[]{yTex},    0);
        GLES30.glDeleteTextures(1,     new int[]{uvTex},   0);
        GLES30.glDeleteFramebuffers(1, new int[]{fbo},     0);
        GLES30.glDeleteBuffers(1,      new int[]{quadVbo}, 0);
        GLES30.glDeleteProgram(program);
        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT);
        EGL14.eglDestroyContext(eglDisplay, eglContext);
        EGL14.eglDestroySurface(eglDisplay, eglSurface);
        EGL14.eglTerminate(eglDisplay);
        Log.d(TAG, "GlYuvConverter closed");
    }

    // ------------------------------------------------------------------ helpers

    private static int buildProgram(String vertSrc, String fragSrc) {
        int vert = compileShader(GLES30.GL_VERTEX_SHADER,   vertSrc);
        int frag = compileShader(GLES30.GL_FRAGMENT_SHADER, fragSrc);
        int prog = GLES30.glCreateProgram();
        GLES30.glAttachShader(prog, vert);
        GLES30.glAttachShader(prog, frag);
        GLES30.glLinkProgram(prog);
        int[] status = new int[1];
        GLES30.glGetProgramiv(prog, GLES30.GL_LINK_STATUS, status, 0);
        GLES30.glDeleteShader(vert);
        GLES30.glDeleteShader(frag);
        if (status[0] == GLES30.GL_FALSE) {
            String log = GLES30.glGetProgramInfoLog(prog);
            GLES30.glDeleteProgram(prog);
            throw new RuntimeException("Shader link error: " + log);
        }
        return prog;
    }

    private static int compileShader(int type, String src) {
        int shader = GLES30.glCreateShader(type);
        GLES30.glShaderSource(shader, src);
        GLES30.glCompileShader(shader);
        int[] status = new int[1];
        GLES30.glGetShaderiv(shader, GLES30.GL_COMPILE_STATUS, status, 0);
        if (status[0] == GLES30.GL_FALSE) {
            String log = GLES30.glGetShaderInfoLog(shader);
            GLES30.glDeleteShader(shader);
            throw new RuntimeException("Shader compile error: " + log);
        }
        return shader;
    }
}
