package com.example;

public class JniDemo {
    public static native int add(int a, int b);
    public static native long add(long a, long b);

    static {
        System.loadLibrary("jni_demo");
    }
}
