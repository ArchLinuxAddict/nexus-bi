# Test fixtures

These fixtures are compiled by the integration tests and the end-to-end demo. They are not executed by the platform; only compiled and statically analyzed.

- `simple.c` — arithmetic functions and a `main`.
- `unsafe.c` — `strcpy`/`system` call sites for evidence-linked security findings.
- `loops.c` — a function with a loop, used to verify explicit unresolved reconstruction.
- `JniDemo.java` / `JniDemo.c` — JNI overloads and `JNI_OnLoad` for Java/JNI reconstruction tests.
