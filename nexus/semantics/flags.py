"""Explicit x86-64 flag semantics.

Pure, deterministic flag-effect definitions used by semantic IR and the
reconstruction emitter.  The former external CPU-oracle harness is not part of
the runtime model and has been removed.
"""

from dataclasses import dataclass

ALL_FLAGS = ("CF", "OF", "SF", "ZF", "AF", "PF")
BITS = {"CF": 0, "PF": 2, "AF": 4, "ZF": 6, "SF": 7, "OF": 11}


@dataclass(frozen=True)
class FlagEffect:
    written: frozenset[str]
    preserved: frozenset[str]
    undefined: frozenset[str]
    reads: frozenset[str]


def flag_effect(mnemonic: str, width: int, count: int = 1) -> FlagEffect:
    all_set = frozenset(ALL_FLAGS)
    if mnemonic in {"add", "sub", "adc", "sbb", "neg", "cmp"}:
        return FlagEffect(all_set, frozenset(), frozenset(), frozenset({"CF"} if mnemonic in {"adc", "sbb"} else set()))
    if mnemonic in {"inc", "dec"}:
        return FlagEffect(frozenset({"OF", "SF", "ZF", "AF", "PF"}), frozenset({"CF"}), frozenset(), frozenset())
    if mnemonic in {"and", "or", "xor", "test"}:
        return FlagEffect(frozenset({"CF", "OF", "SF", "ZF", "PF"}), frozenset(), frozenset({"AF"}), frozenset())
    if mnemonic in {"shl", "shr", "sar"}:
        if count % width == 0:
            return FlagEffect(frozenset(), all_set, frozenset(), frozenset())
        written = {"CF", "SF", "ZF", "PF"}
        undefined = {"AF"}
        if count % width == 1:
            written.add("OF")
        else:
            undefined.add("OF")
        return FlagEffect(frozenset(written), frozenset(), frozenset(undefined), frozenset())
    if mnemonic in {"rol", "ror"}:
        if count % width == 0:
            return FlagEffect(frozenset(), all_set, frozenset(), frozenset())
        if count % width == 1:
            return FlagEffect(frozenset({"CF", "OF"}), frozenset({"SF", "ZF", "AF", "PF"}), frozenset(), frozenset())
        return FlagEffect(frozenset({"CF"}), frozenset({"SF", "ZF", "AF", "PF"}), frozenset({"OF"}), frozenset())
    if mnemonic == "bt":
        return FlagEffect(frozenset({"CF"}), frozenset(), frozenset({"OF", "SF", "ZF", "AF", "PF"}), frozenset())
    if mnemonic in {"bsr", "bsf"}:
        return FlagEffect(frozenset({"ZF"}), frozenset(), frozenset({"CF", "OF", "SF", "AF", "PF"}), frozenset())
    if mnemonic in {"mul", "imul"}:
        return FlagEffect(frozenset({"CF", "OF"}), frozenset(), frozenset({"SF", "ZF", "AF", "PF"}), frozenset())
    if mnemonic in {"div", "idiv"}:
        return FlagEffect(frozenset(), frozenset(), all_set, frozenset())
    if mnemonic == "not":
        return FlagEffect(frozenset(), all_set, frozenset(), frozenset())
    if mnemonic in {"mov", "lea", "movzx", "movsx", "movsxd", "push", "pop", "call", "ret"}:
        return FlagEffect(frozenset(), all_set, frozenset(), frozenset())
    return FlagEffect(frozenset(), frozenset(), all_set, frozenset())


def _parity(value: int) -> int:
    return 1 if (bin(value & 0xFF).count("1") % 2) == 0 else 0


def flag_values(mnemonic: str, left: int, right: int, width: int, carry: int = 0, count: int = 1) -> dict[str, int]:
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    a, b = left & mask, right & mask
    if mnemonic in {"cmp", "sub"}:
        result = (a - b) & mask
        cf = 1 if a < b else 0
    elif mnemonic == "sbb":
        result = (a - b - carry) & mask
        cf = 1 if a < (b + carry) else 0
    elif mnemonic == "add":
        result = (a + b) & mask
        cf = 1 if result < a else 0
    elif mnemonic == "adc":
        result = (a + b + carry) & mask
        cf = 1 if (a + b + carry) > mask else 0
    elif mnemonic == "inc":
        result = (a + 1) & mask
        cf = carry
    elif mnemonic == "dec":
        result = (a - 1) & mask
        cf = carry
    elif mnemonic == "neg":
        result = (-a) & mask
        cf = 1 if a != 0 else 0
    elif mnemonic in {"and", "test"}:
        result = (a & b) & mask
        cf = 0
    elif mnemonic == "or":
        result = (a | b) & mask
        cf = 0
    elif mnemonic == "xor":
        result = (a ^ b) & mask
        cf = 0
    elif mnemonic in {"shl", "shr", "sar", "rol", "ror"}:
        n = count % width
        if n == 0:
            return {}
        if mnemonic == "shl":
            result = (a << n) & mask
            cf = (a >> (width - n)) & 1
        elif mnemonic == "shr":
            result = (a >> n) & mask
            cf = (a >> (n - 1)) & 1
        elif mnemonic == "sar":
            signed = a - (1 << width) if a & sign else a
            result = (signed >> n) & mask
            cf = (a >> (n - 1)) & 1
        elif mnemonic == "rol":
            result = ((a << n) | (a >> (width - n))) & mask
            cf = result & 1
        else:
            result = ((a >> n) | (a << (width - n))) & mask
            cf = (result >> (width - 1)) & 1
    else:
        return {}
    sf = 1 if result & sign else 0
    zf = 1 if result == 0 else 0
    pf = _parity(result)
    if mnemonic in {"add", "adc"}:
        of = 1 if ((a ^ result) & (b ^ result) & sign) else 0
        af = 1 if ((a ^ b ^ result) & 0x10) else 0
    elif mnemonic in {"sub", "sbb", "cmp"}:
        of = 1 if ((a ^ b) & (a ^ result) & sign) else 0
        af = 1 if ((a ^ b ^ result) & 0x10) else 0
    elif mnemonic in {"inc", "dec"}:
        if mnemonic == "inc":
            of = 1 if result == sign else 0
        else:
            of = 1 if result == (sign - 1) else 0
        af = 1 if ((a ^ 1 ^ result) & 0x10) else 0
    elif mnemonic == "neg":
        of = 1 if a == sign else 0
        af = 1 if ((0 ^ a ^ result) & 0x10) else 0
    elif mnemonic in {"and", "or", "xor", "test"}:
        of = 0
        af = 0
    elif mnemonic in {"shl", "shr", "sar"}:
        if n == 1:
            if mnemonic == "shl":
                of = (result >> (width - 1)) ^ cf
            elif mnemonic == "shr":
                of = (a >> (width - 1)) & 1
            else:
                of = 0
        else:
            of = 0
        af = 0
    elif mnemonic in {"rol", "ror"}:
        if n == 1:
            if mnemonic == "rol":
                of = ((result >> (width - 1)) & 1) ^ cf
            else:
                of = ((result >> (width - 2)) & 1) ^ ((result >> (width - 1)) & 1)
        else:
            of = 0
        af = 0
    else:
        return {}
    values = {"CF": cf, "OF": of, "SF": sf, "ZF": zf, "AF": af, "PF": pf}
    if mnemonic in {"and", "or", "xor", "test"}:
        values.pop("AF")
    if mnemonic in {"shl", "shr", "sar"}:
        values.pop("AF")
    return values


def _special_values(mnemonic: str, a: int, b: int, width: int) -> dict[str, int]:
    mask = (1 << width) - 1
    if mnemonic == "bt":
        return {"CF": (a >> (b % width)) & 1}
    if mnemonic in {"bsr", "bsf"}:
        return {"ZF": 1 if (b & mask) == 0 else 0}
    if mnemonic in {"mul", "imul"}:
        if mnemonic == "mul":
            product = (a & mask) * (b & mask)
            high = (product >> width) & mask
            return {"CF": 1 if high else 0, "OF": 1 if high else 0}
        signed_a = a - (1 << width) if a & (1 << (width - 1)) else a
        signed_b = b - (1 << width) if b & (1 << (width - 1)) else b
        product = signed_a * signed_b
        low = product & mask
        signed_low = low - (1 << width) if low & (1 << (width - 1)) else low
        high = product >> width
        return {"CF": 1 if high != (0 if signed_low >= 0 else -1) else 0, "OF": 1 if high != (0 if signed_low >= 0 else -1) else 0}
    return {}

