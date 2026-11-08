import os
import struct
from pathlib import Path
from typing import Any

from nexus.core.errors import ElfError
from nexus.core.hashing import sha256_bytes
from nexus.elf.model import (
    DynamicEntry,
    ElfFile,
    ElfHeader,
    Note,
    ProgramHeader,
    Relocation,
    SectionHeader,
    Symbol,
)

ELF_MAGIC = b"\x7fELF"
ELFCLASS32 = 1
ELFCLASS64 = 2
ELFDATA2LSB = 1
ELFDATA2MSB = 2
EV_CURRENT = 1

ET_NAMES = {0: "ET_NONE", 1: "ET_REL", 2: "ET_EXEC", 3: "ET_DYN", 4: "ET_CORE"}
MACHINE_NAMES = {
    0: "none", 3: "x86", 8: "mips", 20: "powerpc", 21: "powerpc64",
    40: "arm", 62: "x86_64", 183: "aarch64", 243: "riscv",
}
PT_NAMES = {
    0: "PT_NULL", 1: "PT_LOAD", 2: "PT_DYNAMIC", 3: "PT_INTERP",
    4: "PT_NOTE", 5: "PT_SHLIB", 6: "PT_PHDR", 7: "PT_TLS",
    0x6474E550: "PT_GNU_EH_FRAME", 0x6474E551: "PT_GNU_STACK",
    0x6474E552: "PT_GNU_RELRO", 0x6474E553: "PT_GNU_PROPERTY",
}
SHT_NAMES = {
    0: "SHT_NULL", 1: "SHT_PROGBITS", 2: "SHT_SYMTAB", 3: "SHT_STRTAB",
    4: "SHT_RELA", 5: "SHT_HASH", 6: "SHT_DYNAMIC", 7: "SHT_NOTE",
    8: "SHT_NOBITS", 9: "SHT_REL", 10: "SHT_SHLIB", 11: "SHT_DYNSYM",
    14: "SHT_INIT_ARRAY", 15: "SHT_FINI_ARRAY", 16: "SHT_PREINIT_ARRAY",
    17: "SHT_GROUP", 18: "SHT_SYMTAB_SHNDX",
    0x6FFFFFF5: "SHT_GNU_ATTRIBUTES", 0x6FFFFFF6: "SHT_GNU_HASH",
    0x6FFFFFF7: "SHT_GNU_LIBLIST", 0x6FFFFFF8: "SHT_CHECKSUM",
    0x6FFFFFFD: "SHT_GNU_VERDEF", 0x6FFFFFFE: "SHT_GNU_VERNEED",
    0x6FFFFFFF: "SHT_GNU_VERSYM",
}
DT_NAMES = {
    0: "DT_NULL", 1: "DT_NEEDED", 2: "DT_PLTRELSZ", 3: "DT_PLTGOT",
    4: "DT_HASH", 5: "DT_STRTAB", 6: "DT_SYMTAB", 7: "DT_RELA",
    8: "DT_RELASZ", 9: "DT_RELAENT", 10: "DT_STRSZ", 11: "DT_SYMENT",
    12: "DT_INIT", 13: "DT_FINI", 14: "DT_SONAME", 15: "DT_RPATH",
    16: "DT_SYMBOLIC", 17: "DT_REL", 18: "DT_RELSZ", 19: "DT_RELENT",
    20: "DT_PLTREL", 21: "DT_DEBUG", 22: "DT_TEXTREL", 23: "DT_JMPREL",
    24: "DT_BIND_NOW", 25: "DT_INIT_ARRAY", 26: "DT_FINI_ARRAY",
    27: "DT_INIT_ARRAYSZ", 28: "DT_FINI_ARRAYSZ", 29: "DT_RUNPATH",
    30: "DT_FLAGS", 32: "DT_PREINIT_ARRAY", 33: "DT_PREINIT_ARRAYSZ",
    34: "DT_SYMTAB_SHNDX", 35: "DT_RELRSZ", 36: "DT_RELR", 37: "DT_RELRENT",
    0x6FFFFEF5: "DT_GNU_HASH", 0x6FFFFFF0: "DT_VERSYM",
    0x6FFFFFFC: "DT_VERDEF", 0x6FFFFFFD: "DT_VERDEFNUM",
    0x6FFFFFFE: "DT_VERNEED", 0x6FFFFFFF: "DT_VERNEEDNUM",
    0x6FFFFFFB: "DT_FLAGS_1",
}
BINDING_NAMES = {0: "LOCAL", 1: "GLOBAL", 2: "WEAK", 10: "GNU_UNIQUE"}
SYMBOL_TYPE_NAMES = {
    0: "NOTYPE", 1: "OBJECT", 2: "FUNC", 3: "SECTION", 4: "FILE",
    5: "COMMON", 6: "TLS", 10: "GNU_IFUNC",
}
VISIBILITY_NAMES = {0: "DEFAULT", 1: "INTERNAL", 2: "HIDDEN", 3: "PROTECTED"}
X86_64_RELOCATIONS = {
    0: "R_X86_64_NONE", 1: "R_X86_64_64", 2: "R_X86_64_PC32",
    3: "R_X86_64_GOT32", 4: "R_X86_64_PLT32", 5: "R_X86_64_COPY",
    6: "R_X86_64_GLOB_DAT", 7: "R_X86_64_JUMP_SLOT", 8: "R_X86_64_RELATIVE",
    9: "R_X86_64_GOTPCREL", 10: "R_X86_64_32", 11: "R_X86_64_32S",
    12: "R_X86_64_16", 13: "R_X86_64_PC16", 14: "R_X86_64_8",
    15: "R_X86_64_PC8", 16: "R_X86_64_DTPMOD64", 17: "R_X86_64_DTPOFF64",
    18: "R_X86_64_TPOFF64", 19: "R_X86_64_TLSGD", 20: "R_X86_64_TLSLD",
    21: "R_X86_64_DTPOFF32", 22: "R_X86_64_GOTTPOFF", 23: "R_X86_64_TPOFF32",
    24: "R_X86_64_PC64", 25: "R_X86_64_GOTOFF64", 26: "R_X86_64_GOTPC32",
    27: "R_X86_64_GOT64", 28: "R_X86_64_GOTPCREL64", 29: "R_X86_64_GOTPC64",
    30: "R_X86_64_GOTPLT64", 31: "R_X86_64_PLTOFF64", 32: "R_X86_64_SIZE32",
    33: "R_X86_64_SIZE64", 34: "R_X86_64_GOTPC32_TLSDESC",
    35: "R_X86_64_TLSDESC_CALL", 36: "R_X86_64_TLSDESC", 37: "R_X86_64_IRELATIVE",
    38: "R_X86_64_RELATIVE64", 41: "R_X86_64_GOTPCRELX", 42: "R_X86_64_REX_GOTPCRELX",
}


def _name(mapping: dict[int, str], value: int) -> str:
    return mapping.get(value, f"0x{value:x}")


def _cstring(data: bytes, offset: int, limit: int | None = None) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\x00", offset)
    if end < 0:
        end = len(data)
    if limit is not None:
        end = min(end, offset + limit)
    return data[offset:end].decode("utf-8", errors="replace")


def _check_range(offset: int, size: int, total: int, what: str) -> None:
    if offset < 0 or size < 0 or offset > total or size > total - offset:
        raise ElfError(f"{what} is outside the file", details={"offset": offset, "size": size, "file_size": total})


def _read_section_data(elf: ElfFile, section: SectionHeader) -> bytes:
    if section.sh_type == 8:
        return b""
    if section.offset > len(elf.raw) or section.size > len(elf.raw) - section.offset:
        return b""
    return elf.raw[section.offset:section.offset + section.size]


def _read_string_table(elf: ElfFile, index: int) -> bytes:
    section = elf.section_by_index(index)
    if section is None or section.sh_type != 3:
        return b""
    return _read_section_data(elf, section)


def _parse_notes(data: bytes, file_offset: int, endian: str) -> list[Note]:
    notes = []
    position = 0
    header_fmt = endian + "III"
    while position + 12 <= len(data):
        namesz, descsz, note_type = struct.unpack_from(header_fmt, data, position)
        position += 12
        if namesz > len(data) - position:
            break
        name_bytes = data[position:position + namesz]
        position += namesz
        position = (position + 3) & ~3
        if descsz > len(data) - position:
            break
        desc = data[position:position + descsz]
        position += descsz
        position = (position + 3) & ~3
        name = name_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        notes.append(Note(
            name=name,
            note_type=note_type,
            descriptor=desc,
            offset=file_offset,
            descriptor_hex=desc.hex(),
        ))
        if position <= 0:
            break
    return notes


def _parse_version_tables(elf: ElfFile) -> tuple[dict[int, str], list[dict[str, Any]]]:
    names: dict[int, str] = {}
    details: list[dict[str, Any]] = []
    for section in elf.sections:
        if section.sh_type not in (0x6FFFFFFD, 0x6FFFFFFE):
            continue
        data = _read_section_data(elf, section)
        strings = _read_string_table(elf, section.link)
        if section.sh_type == 0x6FFFFFFE:
            position = 0
            while position + 16 <= len(data):
                version, count, file_name, aux, next_offset = struct.unpack_from(elf.endian + "HHIII", data, position)
                entry = {"kind": "need", "version": version, "file": _cstring(strings, file_name)}
                aux_pos = position + aux
                for _ in range(count):
                    if aux_pos + 16 > len(data):
                        break
                    vna_hash, flags, other, name_off, vna_next = struct.unpack_from(elf.endian + "IHHII", data, aux_pos)
                    name = _cstring(strings, name_off)
                    if other & 0x7FFF:
                        names[other & 0x7FFF] = name
                    entry.setdefault("aux", []).append({"name": name, "hash": vna_hash, "flags": flags, "index": other})
                    if vna_next == 0:
                        break
                    aux_pos += vna_next
                details.append(entry)
                if next_offset == 0:
                    break
                position += next_offset
        else:
            position = 0
            while position + 20 <= len(data):
                version, flags, index, count, vd_hash, aux, next_offset = struct.unpack_from(elf.endian + "HHHHIII", data, position)
                entry = {"kind": "definition", "version": version, "flags": flags, "index": index, "hash": vd_hash}
                aux_pos = position + aux
                for aux_index in range(count):
                    if aux_pos + 8 > len(data):
                        break
                    name_off, vda_next = struct.unpack_from(elf.endian + "II", data, aux_pos)
                    name = _cstring(strings, name_off)
                    if aux_index == 0 and index:
                        names[index] = name
                    entry.setdefault("aux", []).append({"name": name})
                    if vda_next == 0:
                        break
                    aux_pos += vda_next
                details.append(entry)
                if next_offset == 0:
                    break
                position += next_offset
    return names, details


def _parse_symbols(elf: ElfFile, section: SectionHeader, table_name: str, versions: list[int] | None) -> list[Symbol]:
    data = _read_section_data(elf, section)
    strings = _read_string_table(elf, section.link)
    entry_size = section.entsize or (24 if elf.header.elf_class == "ELF64" else 16)
    if entry_size == 0:
        return []
    count = min(len(data) // entry_size, 2_000_000)
    symbols = []
    endian = elf.endian
    for index in range(count):
        offset = index * entry_size
        if elf.header.elf_class == "ELF64":
            st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(endian + "IBBHQQ", data, offset)
        else:
            st_name, st_value, st_size, st_info, st_other, st_shndx = struct.unpack_from(endian + "IIIBBH", data, offset)
        name = _cstring(strings, st_name)
        binding = BINDING_NAMES.get(st_info >> 4, f"BIND_{st_info >> 4}")
        symbol_type = SYMBOL_TYPE_NAMES.get(st_info & 0xF, f"TYPE_{st_info & 0xF}")
        visibility = VISIBILITY_NAMES.get(st_other & 0x3, f"VIS_{st_other & 0x3}")
        version = None
        if versions is not None and index < len(versions):
            version_index = versions[index] & 0x7FFF
            version = elf.version_names.get(version_index) if version_index else None
        if index == 0 and not name:
            continue
        symbols.append(Symbol(
            index=index,
            name=name,
            value=st_value,
            size=st_size,
            info=st_info,
            other=st_other,
            shndx=st_shndx,
            binding=binding,
            symbol_type=symbol_type,
            visibility=visibility,
            table=table_name,
            version=version,
        ))
    return symbols


def _parse_relocations(elf: ElfFile, section: SectionHeader, symbol_tables: dict[int, list[Symbol]], versions: list[int] | None) -> list[Relocation]:
    data = _read_section_data(elf, section)
    entry_size = section.entsize or (24 if section.sh_type == 4 else 16 if elf.header.elf_class == "ELF64" else 12 if section.sh_type == 4 else 8)
    if entry_size == 0:
        return []
    count = min(len(data) // entry_size, 2_000_000)
    symbols = symbol_tables.get(section.link, [])
    relocations = []
    endian = elf.endian
    for index in range(count):
        offset = index * entry_size
        if elf.header.elf_class == "ELF64":
            r_offset, r_info = struct.unpack_from(endian + "QQ", data, offset)
            addend = struct.unpack_from(endian + "q", data, offset + 16)[0] if section.sh_type == 4 else None
            reloc_type = r_info & 0xFFFFFFFF
            symbol_index = r_info >> 32
        else:
            r_offset, r_info = struct.unpack_from(endian + "II", data, offset)
            addend = struct.unpack_from(endian + "i", data, offset + 8)[0] if section.sh_type == 4 else None
            reloc_type = r_info & 0xFF
            symbol_index = r_info >> 8
        symbol_name = symbols[symbol_index].name if 0 <= symbol_index < len(symbols) else None
        type_name = X86_64_RELOCATIONS.get(reloc_type, f"RELOC_{reloc_type}") if elf.header.machine_name == "x86_64" else f"RELOC_{reloc_type}"
        relocations.append(Relocation(
            index=index,
            offset=r_offset,
            info=r_info,
            addend=addend,
            relocation_type=reloc_type,
            type_name=type_name,
            symbol_index=symbol_index,
            symbol_name=symbol_name,
            section_index=section.info,
            table=SHT_NAMES.get(section.sh_type, str(section.sh_type)),
        ))
    return relocations


def parse_elf(path: str | Path) -> ElfFile:
    file_path = Path(path)
    max_bytes = int(os.environ.get("NEXUS_MAX_BINARY_BYTES", str(256 * 1024 * 1024)))
    try:
        file_size = file_path.stat().st_size
    except OSError as exc:
        raise ElfError(f"unable to stat file: {exc}") from exc
    if file_size > max_bytes:
        raise ElfError("binary exceeds the configured maximum size", details={"size": file_size, "max_bytes": max_bytes})
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise ElfError(f"unable to read file: {exc}") from exc
    if len(data) < 16:
        raise ElfError("file is too small to be ELF", details={"size": len(data)})
    if data[:4] != ELF_MAGIC:
        raise ElfError("invalid ELF magic", details={"magic": data[:4].hex()})
    elf_class = data[4]
    data_encoding = data[5]
    if elf_class not in (ELFCLASS32, ELFCLASS64):
        raise ElfError(f"unsupported ELF class: {elf_class}")
    if data_encoding not in (ELFDATA2LSB, ELFDATA2MSB):
        raise ElfError(f"unsupported ELF data encoding: {data_encoding}")
    elf_class_name = "ELF32" if elf_class == ELFCLASS32 else "ELF64"
    endian = "<" if data_encoding == ELFDATA2LSB else ">"
    header_fmt = endian + ("16sHHIQQQIHHHHHH" if elf_class == ELFCLASS64 else "16sHHIIIIIHHHHHH")
    header_size = struct.calcsize(header_fmt)
    if len(data) < header_size:
        raise ElfError("truncated ELF header", details={"size": len(data), "required": header_size})
    values = struct.unpack_from(header_fmt, data, 0)
    (_, e_type, e_machine, e_version, e_entry, e_phoff, e_shoff, e_flags,
     e_ehsize, e_phentsize, e_phnum, e_shentsize, e_shnum, e_shstrndx) = values
    if e_ehsize < header_size:
        raise ElfError("invalid ELF header size", details={"ehsize": e_ehsize, "minimum": header_size})
    expected_phentsize = 56 if elf_class == ELFCLASS64 else 32
    expected_shentsize = 64 if elf_class == ELFCLASS64 else 40
    if e_phnum and e_phentsize < expected_phentsize:
        raise ElfError("invalid program header entry size", details={"phentsize": e_phentsize})
    if e_shnum and e_shentsize < expected_shentsize:
        raise ElfError("invalid section header entry size", details={"shentsize": e_shentsize})

    section0 = None
    if e_shoff and e_shentsize >= expected_shentsize:
        _check_range(e_shoff, e_shentsize, len(data), "section header 0")
        section0 = _unpack_section_header(data, e_shoff, e_shentsize, endian, elf_class, 0, "", 0)
    if e_shnum == 0 and section0 is not None:
        e_shnum = section0.size
    if e_phnum == 0xFFFF and section0 is not None:
        e_phnum = section0.info
    if e_shstrndx == 0xFFFF and section0 is not None:
        e_shstrndx = section0.link
    if e_shnum > 2_000_000 or e_phnum > 2_000_000:
        raise ElfError("unreasonable header count", details={"sections": e_shnum, "segments": e_phnum})
    if e_phnum:
        _check_range(e_phoff, e_phnum * e_phentsize, len(data), "program header table")
    if e_shnum:
        _check_range(e_shoff, e_shnum * e_shentsize, len(data), "section header table")

    header = ElfHeader(
        elf_class=elf_class_name,
        endianness="little" if data_encoding == ELFDATA2LSB else "big",
        osabi=data[7],
        abi_version=data[8],
        e_type=e_type,
        machine=e_machine,
        machine_name=_name(MACHINE_NAMES, e_machine),
        e_version=e_version,
        entry=e_entry,
        phoff=e_phoff,
        shoff=e_shoff,
        flags=e_flags,
        ehsize=e_ehsize,
        phentsize=e_phentsize,
        phnum=e_phnum,
        shentsize=e_shentsize,
        shnum=e_shnum,
        shstrndx=e_shstrndx,
        type_name=_name(ET_NAMES, e_type),
    )
    elf = ElfFile(
        path=str(file_path),
        size=len(data),
        sha256=sha256_bytes(data),
        header=header,
        raw=data,
        endian=endian,
    )

    for index in range(e_phnum):
        offset = e_phoff + index * e_phentsize
        if elf_class == ELFCLASS64:
            p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align = struct.unpack_from(endian + "IIQQQQQQ", data, offset)
        else:
            p_type, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_flags, p_align = struct.unpack_from(endian + "IIIIIIII", data, offset)
        if p_filesz:
            _check_range(p_offset, p_filesz, len(data), f"program header {index} file range")
        elf.program_headers.append(ProgramHeader(
            index=index, p_type=p_type, type_name=_name(PT_NAMES, p_type),
            flags=p_flags, offset=p_offset, vaddr=p_vaddr, paddr=p_paddr,
            filesz=p_filesz, memsz=p_memsz, align=p_align,
        ))

    if e_shstrndx < e_shnum and e_shstrndx >= 0:
        shstr_section = _unpack_section_header(
            data, e_shoff + e_shstrndx * e_shentsize, e_shentsize, endian, elf_class, e_shstrndx, "", 0
        )
        shstr = data[shstr_section.offset:shstr_section.offset + shstr_section.size] if shstr_section.sh_type != 8 else b""
    else:
        shstr = b""

    for index in range(e_shnum):
        offset = e_shoff + index * e_shentsize
        if elf_class == ELFCLASS64:
            sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack_from(endian + "IIQQQQIIQQ", data, offset)
        else:
            sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack_from(endian + "IIIIIIIIII", data, offset)
        name = _cstring(shstr, sh_name, 4096)
        section = SectionHeader(
            index=index, name_offset=sh_name, name=name, sh_type=sh_type,
            type_name=_name(SHT_NAMES, sh_type), flags=sh_flags, addr=sh_addr,
            offset=sh_offset, size=sh_size, link=sh_link, info=sh_info,
            addralign=sh_addralign, entsize=sh_entsize,
        )
        if sh_type != 8 and sh_size:
            _check_range(sh_offset, sh_size, len(data), f"section {index} ({name or 'unnamed'})")
        elf.sections.append(section)

    for segment in elf.program_headers:
        if segment.p_type == 3:
            segment_data = data[segment.offset:segment.offset + segment.filesz]
            elf.metadata["has_interpreter"] = True
            elf.metadata["interpreter"] = segment_data.rstrip(b"\x00").decode("utf-8", errors="replace")
        if segment.p_type == 2:
            elf.metadata["has_dynamic"] = True
        if segment.p_type == 0x6474E551:
            elf.metadata["gnu_stack_executable"] = bool(segment.flags & 1)
        if segment.p_type == 0x6474E552:
            elf.metadata["relro"] = True

    for section in elf.sections:
        if section.sh_type == 7:
            section_data = _read_section_data(elf, section)
            elf.notes.extend(_parse_notes(section_data, section.offset, endian))
    seen_note_offsets = {note.offset for note in elf.notes}
    for segment in elf.program_headers:
        if segment.p_type == 4 and segment.offset not in seen_note_offsets:
            segment_data = data[segment.offset:segment.offset + segment.filesz]
            elf.notes.extend(_parse_notes(segment_data, segment.offset, endian))
            seen_note_offsets.add(segment.offset)
    for note in elf.notes:
        if note.name == "GNU" and note.note_type == 3 and note.descriptor:
            elf.build_id = note.descriptor.hex()
            break

    elf.version_names, version_details = _parse_version_tables(elf)
    if version_details:
        elf.metadata["gnu_versions"] = version_details

    versym_section = next((section for section in elf.sections if section.sh_type == 0x6FFFFFFF), None)
    versym: list[int] | None = None
    if versym_section is not None:
        versym_data = _read_section_data(elf, versym_section)
        versym = [struct.unpack_from(endian + "H", versym_data, i)[0] for i in range(0, len(versym_data) - 1, 2)]
        elf.metadata["symbol_version_count"] = len(versym)

    symbol_tables: dict[int, list[Symbol]] = {}
    for section in elf.sections:
        if section.sh_type not in (2, 11):
            continue
        symbols = _parse_symbols(elf, section, "symtab" if section.sh_type == 2 else "dynsym", versym if section.sh_type == 11 else None)
        symbol_tables[section.index] = symbols
        if section.sh_type == 2:
            elf.symbols.extend(symbols)
        else:
            elf.dynamic_symbols.extend(symbols)

    for section in elf.sections:
        if section.sh_type in (4, 9):
            elf.relocations.extend(_parse_relocations(elf, section, symbol_tables, versym))

    dynamic_section = next((section for section in elf.sections if section.sh_type == 6), None)
    dynstr = _read_string_table(elf, dynamic_section.link) if dynamic_section is not None else b""
    if dynamic_section is not None:
        dynamic_data = _read_section_data(elf, dynamic_section)
        entry_size = dynamic_section.entsize or (16 if elf_class == ELFCLASS64 else 8)
        for index in range(len(dynamic_data) // entry_size):
            offset = index * entry_size
            if elf_class == ELFCLASS64:
                tag, value = struct.unpack_from(endian + "qQ", dynamic_data, offset)
            else:
                tag, value = struct.unpack_from(endian + "iI", dynamic_data, offset)
            if tag == 0:
                break
            tag_name = DT_NAMES.get(tag, f"DT_{tag:x}")
            elf.dynamic_entries.append(DynamicEntry(tag=tag, tag_name=tag_name, value=value))
            if tag == 1:
                elf.dependencies.append({"name": _cstring(dynstr, value), "kind": "needed", "value": value})
            elif tag == 14:
                elf.dependencies.append({"name": _cstring(dynstr, value), "kind": "soname", "value": value})
            elif tag == 15:
                elf.dependencies.append({"name": _cstring(dynstr, value), "kind": "rpath", "value": value})
            elif tag == 29:
                elf.dependencies.append({"name": _cstring(dynstr, value), "kind": "runpath", "value": value})
            elif tag == 24:
                elf.metadata["bind_now"] = True
            elif tag == 30:
                elf.metadata["dt_flags"] = value
            elif tag == 0x6FFFFFFB:
                elf.metadata["dt_flags_1"] = value
                if value & 0x1:
                    elf.metadata["bind_now"] = True
    if elf.metadata.get("dt_flags", 0) & 0x8:
        elf.metadata["bind_now"] = True

    imports = [symbol.name for symbol in elf.dynamic_symbols if symbol.is_import and symbol.name]
    if "__stack_chk_fail" in imports:
        elf.metadata["stack_canary"] = True
    if any(name.startswith("__asan_") for name in imports):
        elf.metadata["sanitizer"] = "asan"
    elf.metadata["has_dynamic"] = bool(elf.metadata.get("has_dynamic") or dynamic_section is not None)
    return elf


def _unpack_section_header(data: bytes, offset: int, entry_size: int, endian: str, elf_class: int, index: int, name: str, shstrndx: int) -> SectionHeader:
    if elf_class == ELFCLASS64:
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack_from(endian + "IIQQQQIIQQ", data, offset)
    else:
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = struct.unpack_from(endian + "IIIIIIIIII", data, offset)
    return SectionHeader(
        index=index, name_offset=sh_name, name=name, sh_type=sh_type,
        type_name=_name(SHT_NAMES, sh_type), flags=sh_flags, addr=sh_addr,
        offset=sh_offset, size=sh_size, link=sh_link, info=sh_info,
        addralign=sh_addralign, entsize=sh_entsize,
    )
