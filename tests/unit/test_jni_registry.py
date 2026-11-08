from nexus.jni.registry import _stack_table_fields, _valid_signature


def test_valid_signature_accepts_descriptor_semicolons():
    assert _valid_signature("()Ljava/lang/String;")
    assert _valid_signature("(ILjava/lang/Class;)V")
    assert not _valid_signature("()Ljava/lang/String;); code")
    assert not _valid_signature("(I")


def test_stack_table_recovery_matches_libhello_shape():
    source = """
    local_78 = local_40;
    pcStack_70 = local_60;
    local_68 = FUN_00127368;
    """
    fields = _stack_table_fields(
        source,
        "local_78",
        1,
        None,  # type: ignore[arg-type]
        0x100000,
        ["registerNativesForClass", "(ILjava/lang/Class;)V"],
    )
    assert fields == [
        {
            "name": "registerNativesForClass",
            "signature": "(ILjava/lang/Class;)V",
            "address": 0x27368,
        }
    ]


def test_tail_payload_detection_reports_near_tail():
    from nexus.jni.registry import tail_payload_offsets

    data = b"X" * 64 + b"AKP1" + b"Y" * 64
    found = tail_payload_offsets(data, max_tail=len(data))
    assert found[0]["magic"] == "AKP1"
    assert found[0]["offset"] == 64
    assert found[0]["near_tail"] is True


def test_protection_profile_claims_ark_safe_only_with_strong_marker():
    from types import SimpleNamespace

    from nexus.jni.registry import protection_profile

    raw = b"prefix ark-native-payload suffix"
    section = SimpleNamespace(name=".rodata", addr=0, size=len(raw))
    fake = SimpleNamespace(
        sections=[section],
        raw=raw,
        read_va=lambda addr, size: raw[addr:addr + size],
    )
    profile = protection_profile(fake)  # type: ignore[arg-type]
    assert profile["ark_safe"] is True
    assert profile["label"] == "ark-safe-vmp"
    assert profile["runtime_decryption"] is True

    raw = b"prefix /proc/self/maps suffix"
    section = SimpleNamespace(name=".rodata", addr=0, size=len(raw))
    maps_only = SimpleNamespace(
        sections=[section],
        raw=raw,
        read_va=lambda addr, size: raw[addr:addr + size],
    )
    profile = protection_profile(maps_only)  # type: ignore[arg-type]
    assert profile["ark_safe"] is False
