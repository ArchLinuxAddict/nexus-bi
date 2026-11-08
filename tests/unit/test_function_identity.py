from nexus.data.identity import (
    extract_function_by_lines,
    extract_function_by_name,
    normalize_c_source,
    source_hashes,
)


def test_normalization_ignores_comments_and_whitespace_but_preserves_literals():
    left = "int add(int a, int b) { /* comment */ return a + b; }"
    right = "int add(int a,int b){\n// other comment\nreturn a+b;\n}"
    assert normalize_c_source(left) == normalize_c_source(right)
    assert source_hashes(left)["normalized_source_hash"] == source_hashes(right)["normalized_source_hash"]
    assert source_hashes(left)["raw_source_hash"] != source_hashes(right)["raw_source_hash"]


def test_literals_are_not_normalized_away():
    left = 'const char *x = "alpha";'
    right = 'const char *x = "beta";'
    assert source_hashes(left)["normalized_source_hash"] != source_hashes(right)["normalized_source_hash"]


def test_extract_function_by_name_and_lines():
    source = "int helper(int x) {\n  return x + 1;\n}\n\nint target(int x) {\n  return helper(x);\n}\n"
    assert extract_function_by_name(source, "target") == "int target(int x) {\n  return helper(x);\n}"
    assert extract_function_by_lines(source, 5, 7) == "int target(int x) {\n  return helper(x);\n}"


def test_missing_source_is_explicitly_unavailable():
    hashes = source_hashes("")
    assert hashes["token_count"] == 0
    assert hashes["shingle_hashes"] == ()
    assert extract_function_by_name("", "missing") is None
    assert extract_function_by_lines("int x;", 10, 12) is None
