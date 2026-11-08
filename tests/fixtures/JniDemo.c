#include <jni.h>

JNIEXPORT jint JNICALL Java_com_example_JniDemo_add__II(JNIEnv *env, jclass cls, jint a, jint b) {
    return a + b;
}

JNIEXPORT jlong JNICALL Java_com_example_JniDemo_add__JJ(JNIEnv *env, jclass cls, jlong a, jlong b) {
    return a + b;
}

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    return JNI_VERSION_1_8;
}
