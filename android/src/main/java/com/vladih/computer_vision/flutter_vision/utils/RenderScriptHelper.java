package com.vladih.computer_vision.flutter_vision.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

public class RenderScriptHelper implements AutoCloseable {
    private static final String TAG = "RenderScriptHelper";
    private static RenderScriptHelper instance;
    private static final Object lock = new Object();
    private volatile boolean isClosed = false;

    private RenderScriptHelper() {}

    public static RenderScriptHelper getInstance(Context context) {
        if (instance == null || instance.isClosed) {
            synchronized (lock) {
                if (instance == null || instance.isClosed) {
                    instance = new RenderScriptHelper();
                }
            }
        }
        return instance;
    }

    public static Bitmap getBitmapFromNV21(Context context, byte[] nv21, int width, int height) {
        if (context == null) {
            throw new IllegalArgumentException("Context cannot be null");
        }
        if (nv21 == null || nv21.length == 0) {
            throw new IllegalArgumentException("Invalid NV21 data");
        }
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Invalid dimensions: " + width + "x" + height);
        }

        try {
            int[] argb = nv21ToArgb(nv21, width, height);
            Bitmap bitmap = Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888);
            Log.d(TAG, String.format("Successfully converted NV21 to bitmap (%dx%d)", width, height));
            return bitmap;
        } catch (Exception e) {
            Log.e(TAG, "Error creating bitmap from NV21", e);
            throw new RuntimeException("Bitmap creation from NV21 failed: " + e.getMessage());
        }
    }

    // BT.601 limited-range YUV to ARGB conversion.
    // NV21 layout: Y plane (width*height bytes) then interleaved V,U plane.
    private static int[] nv21ToArgb(byte[] nv21, int width, int height) {
        int[] argb = new int[width * height];
        int ySize = width * height;
        int i = 0;
        for (int y = 0; y < height; y++) {
            int uvRow = ySize + (y >> 1) * width;
            for (int x = 0; x < width; x++) {
                int c = (nv21[y * width + x] & 0xFF) - 16;
                int v = (nv21[uvRow + (x & ~1)] & 0xFF) - 128;     // Cr
                int u = (nv21[uvRow + (x & ~1) + 1] & 0xFF) - 128; // Cb
                int yScaled = 298 * c + 128;
                int r = clamp((yScaled + 409 * v) >> 8);
                int g = clamp((yScaled - 100 * u - 208 * v) >> 8);
                int b = clamp((yScaled + 516 * u) >> 8);
                argb[i++] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        return argb;
    }

    private static int clamp(int val) {
        return Math.max(0, Math.min(255, val));
    }

    @Override
    public void close() {
        isClosed = true;
    }

    @Deprecated
    public void cleanup() {
        close();
    }

    public boolean isClosed() {
        return isClosed;
    }

    public String getMemoryInfo() {
        return isClosed ? "RenderScriptHelper: CLOSED" : "RenderScriptHelper: Active (software renderer)";
    }

    public static void resetInstance() {
        synchronized (lock) {
            if (instance != null) {
                instance.close();
                instance = null;
                Log.d(TAG, "RenderScriptHelper instance reset");
            }
        }
    }
}
