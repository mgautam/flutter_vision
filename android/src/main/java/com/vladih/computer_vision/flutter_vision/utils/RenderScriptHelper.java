package com.vladih.computer_vision.flutter_vision.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

public class RenderScriptHelper implements AutoCloseable {
    private static final String TAG = "RenderScriptHelper";
    private static RenderScriptHelper instance;
    private static final Object lock = new Object();
    private volatile boolean isClosed = false;

    private final GlYuvConverter converter;

    private RenderScriptHelper() {
        converter = new GlYuvConverter();
    }

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
            Bitmap bitmap = getInstance(context).converter.convert(nv21, width, height);
            Log.d(TAG, String.format("Successfully converted NV21 to bitmap (%dx%d)", width, height));
            return bitmap;
        } catch (Exception e) {
            Log.e(TAG, "Error creating bitmap from NV21", e);
            throw new RuntimeException("Bitmap creation from NV21 failed: " + e.getMessage());
        }
    }

    @Override
    public void close() {
        if (isClosed) return;
        synchronized (lock) {
            if (isClosed) return;
            try {
                converter.close();
            } catch (Exception e) {
                Log.e(TAG, "Error closing GlYuvConverter", e);
            }
            isClosed = true;
            Log.d(TAG, "RenderScriptHelper closed");
        }
    }

    @Deprecated
    public void cleanup() {
        close();
    }

    public boolean isClosed() {
        return isClosed;
    }

    public String getMemoryInfo() {
        return isClosed ? "RenderScriptHelper: CLOSED" : "RenderScriptHelper: Active (OpenGL ES 3.0)";
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
