FIXTURES = {
    "arithmetic": r'''
#include <stdint.h>
int add(int a, int b) { return a + b; }
int subtract(int a, int b) { return a - b; }
int multiply(int a, int b) { return a * b; }
unsigned rotate_left(unsigned value, unsigned count) { return (value << count) | (value >> (32 - count)); }
int main(void) { return add(2, 3) + subtract(9, 4) + multiply(3, 7); }
''',
    "loops": r'''
#include <stdint.h>
uint64_t sum_to(unsigned n) { uint64_t total = 0; for (unsigned i = 1; i <= n; i++) total += i; return total; }
uint64_t factorial(unsigned n) { uint64_t value = 1; for (unsigned i = 2; i <= n; i++) value *= i; return value; }
int main(void) { return (int)(sum_to(10) + factorial(5)); }
''',
    "branching": r'''
int classify(int value) { if (value < 0) return -1; if (value == 0) return 0; return 1; }
int maximum(int a, int b) { return a > b ? a : b; }
int main(void) { return classify(-7) + classify(0) + maximum(4, 9); }
''',
    "recursion": r'''
unsigned factorial(unsigned n) { if (n <= 1) return 1; return n * factorial(n - 1); }
unsigned fibonacci(unsigned n) { if (n < 2) return n; return fibonacci(n - 1) + fibonacci(n - 2); }
int main(void) { return (int)(factorial(5) + fibonacci(7)); }
''',
    "pointers": r'''
void swap(int *a, int *b) { int temporary = *a; *a = *b; *b = temporary; }
int sum_array(const int *values, unsigned count) { int total = 0; for (unsigned i = 0; i < count; i++) total += values[i]; return total; }
int main(void) { int a = 3, b = 8; int values[3] = {1, 2, 3}; swap(&a, &b); return sum_array(values, 3) + a + b; }
''',
    "arrays": r'''
void reverse(int *values, unsigned count) { for (unsigned i = 0; i < count / 2; i++) { int t = values[i]; values[i] = values[count - 1 - i]; values[count - 1 - i] = t; } }
int main(void) { int values[4] = {1, 2, 3, 4}; reverse(values, 4); return values[0] + values[3]; }
''',
    "structs": r'''
struct point { int x; int y; };
struct point make_point(int x, int y) { struct point p = {x, y}; return p; }
int distance_squared(struct point p) { return p.x * p.x + p.y * p.y; }
int main(void) { struct point p = make_point(3, 4); return distance_squared(p); }
''',
    "unions": r'''
union value { int integer; unsigned char bytes[4]; };
unsigned low_byte(int input) { union value v; v.integer = input; return v.bytes[0]; }
int main(void) { return (int)low_byte(0x12345678); }
''',
    "function_pointers": r'''
typedef int (*binary_op)(int, int);
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
int apply(binary_op operation, int a, int b) { return operation(a, b); }
int main(void) { return apply(add, 2, 3) + apply(multiply, 4, 5); }
''',
    "memory": r'''
#include <stdlib.h>
int *allocate_values(unsigned count) { int *values = (int *)malloc(count * sizeof(int)); if (!values) return 0; for (unsigned i = 0; i < count; i++) values[i] = (int)i; return values; }
void release_values(int *values) { free(values); }
int main(void) { int *values = allocate_values(8); if (!values) return 1; int result = values[7]; release_values(values); return result; }
''',
    "strings": r'''
#include <stddef.h>
size_t string_length(const char *text) { size_t length = 0; while (text[length]) length++; return length; }
int string_compare(const char *a, const char *b) { while (*a && *a == *b) { a++; b++; } return (unsigned char)*a - (unsigned char)*b; }
int main(void) { return (int)string_length("nexus") + string_compare("a", "b"); }
''',
    "file_operations": r'''
#include <stdio.h>
int write_text(const char *path, const char *text) { FILE *file = fopen(path, "w"); if (!file) return -1; int result = fputs(text, file); fclose(file); return result; }
int main(void) { return write_text("nexus-demo.txt", "nexus"); }
''',
    "sorting": r'''
void insertion_sort(int *values, unsigned count) { for (unsigned i = 1; i < count; i++) { int key = values[i]; unsigned j = i; while (j > 0 && values[j - 1] > key) { values[j] = values[j - 1]; j--; } values[j] = key; } }
int main(void) { int values[5] = {5, 1, 4, 2, 3}; insertion_sort(values, 5); return values[0] + values[4]; }
''',
    "error_handling": r'''
int parse_positive(const char *text, int *output) { if (!text || !output) return -1; int value = 0; for (unsigned i = 0; text[i]; i++) { if (text[i] < '0' || text[i] > '9') return -2; value = value * 10 + (text[i] - '0'); } *output = value; return 0; }
int main(void) { int value = 0; int status = parse_positive("123", &value); return status == 0 ? value : 1; }
''',
    "linked_list": r'''
#include <stdlib.h>
struct node { int value; struct node *next; };
struct node *push_front(struct node *head, int value) { struct node *item = (struct node *)malloc(sizeof(struct node)); if (!item) return head; item->value = value; item->next = head; return item; }
int sum_list(const struct node *head) { int total = 0; while (head) { total += head->value; head = head->next; } return total; }
void free_list(struct node *head) { while (head) { struct node *next = head->next; free(head); head = next; } }
int main(void) { struct node *head = 0; for (int i = 0; i < 4; i++) head = push_front(head, i); int result = sum_list(head); free_list(head); return result; }
''',
    "hash_table": r'''
#include <stdlib.h>
#define TABLE_SIZE 16
struct entry { int key; int value; struct entry *next; };
static struct entry *table[TABLE_SIZE];
static unsigned bucket_for(int key) { unsigned value = (unsigned)key; return value % TABLE_SIZE; }
void table_insert(int key, int value) { unsigned bucket = bucket_for(key); for (struct entry *item = table[bucket]; item; item = item->next) { if (item->key == key) { item->value = value; return; } } struct entry *item = (struct entry *)malloc(sizeof(struct entry)); if (!item) return; item->key = key; item->value = value; item->next = table[bucket]; table[bucket] = item; }
int table_lookup(int key) { for (struct entry *item = table[bucket_for(key)]; item; item = item->next) if (item->key == key) return item->value; return -1; }
int main(void) { table_insert(3, 30); table_insert(19, 190); return table_lookup(3) + table_lookup(19); }
''',
    "state_machine": r'''
enum state { IDLE, RUNNING, DONE };
int step(enum state current, int event) { switch (current) { case IDLE: return event == 1 ? RUNNING : IDLE; case RUNNING: return event == 2 ? DONE : RUNNING; case DONE: return DONE; default: return IDLE; } }
int main(void) { enum state current = IDLE; current = (enum state)step(current, 1); current = (enum state)step(current, 2); return (int)current; }
''',
    "matrix": r'''
void matrix_multiply(const int left[2][2], const int right[2][2], int result[2][2]) { for (int row = 0; row < 2; row++) for (int column = 0; column < 2; column++) { int total = 0; for (int inner = 0; inner < 2; inner++) total += left[row][inner] * right[inner][column]; result[row][column] = total; } }
int main(void) { int left[2][2] = {{1, 2}, {3, 4}}; int right[2][2] = {{5, 6}, {7, 8}}; int result[2][2]; matrix_multiply(left, right, result); return result[0][0] + result[1][1]; }
''',
    "serialization": r'''
#include <stdint.h>
uint32_t pack_le(uint16_t high, uint16_t low) { return ((uint32_t)high << 16) | low; }
uint16_t unpack_high(uint32_t value) { return (uint16_t)(value >> 16); }
uint16_t unpack_low(uint32_t value) { return (uint16_t)value; }
int main(void) { uint32_t packed = pack_le(0x1234, 0x5678); return unpack_high(packed) + unpack_low(packed); }
''',
    "checksum": r'''
#include <stdint.h>
static uint32_t rotate_left(uint32_t value, unsigned count) { return (value << count) | (value >> (32 - count)); }
uint32_t checksum(const unsigned char *data, unsigned length) { uint32_t value = 2166136261u; for (unsigned i = 0; i < length; i++) { value ^= data[i]; value *= 16777619u; value = rotate_left(value, 5); } return value; }
int main(void) { const unsigned char data[] = {1, 2, 3, 4}; return (int)checksum(data, 4); }
''',
}


def fixture_spec(name: str) -> dict:
    features = {
        "arithmetic": ["integer arithmetic", "bit manipulation"],
        "loops": ["bounded loops", "accumulators"],
        "branching": ["conditional branches", "ternary-like control flow"],
        "recursion": ["recursive calls", "call graph cycles"],
        "pointers": ["pointer arguments", "pointer writes", "array access"],
        "arrays": ["array indexing", "in-place mutation"],
        "structs": ["struct layout", "struct return"],
        "unions": ["union interpretation", "byte access"],
        "function_pointers": ["function pointers", "callbacks", "indirect calls"],
        "memory": ["heap allocation", "allocation ownership", "free"],
        "strings": ["string traversal", "string comparison"],
        "file_operations": ["file I/O", "external library calls"],
        "sorting": ["nested loops", "in-place sorting"],
        "error_handling": ["input validation", "error codes", "pointer output"],
        "linked_list": ["linked structures", "heap allocation", "pointer traversal", "free"],
        "hash_table": ["hash buckets", "chained structures", "lookup", "heap allocation"],
        "state_machine": ["switch dispatch", "state transitions", "enum"],
        "matrix": ["nested loops", "2D arrays", "arithmetic"],
        "serialization": ["shifts and masks", "packing", "endianness-like layout"],
        "checksum": ["bit rotation", "byte loop", "hash-like transformation"],
    }
    return {
        "name": name,
        "purpose": f"Controlled C fixture covering {', '.join(features.get(name, []))}",
        "features": features.get(name, []),
        "source": f"{name}.c",
        "expected_behavior": "Functions are deterministic for the documented input domains.",
        "tests_executed": False,
        "notes": "Synthetic fixture authored for NEXUS-BI dataset generation.",
    }
