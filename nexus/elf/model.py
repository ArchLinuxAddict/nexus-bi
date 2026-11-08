from dataclasses import dataclass, field
from typing import Any


@dataclass
class ElfHeader:
    elf_class: str
    endianness: str
    osabi: int
    abi_version: int
    e_type: int
    machine: int
    machine_name: str
    e_version: int
    entry: int
    phoff: int
    shoff: int
    flags: int
    ehsize: int
    phentsize: int
    phnum: int
    shentsize: int
    shnum: int
    shstrndx: int
    type_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.elf_class,
            "endianness": self.endianness,
            "osabi": self.osabi,
            "abi_version": self.abi_version,
            "type": self.type_name,
            "e_type": self.e_type,
            "machine": self.machine_name,
            "e_machine": self.machine,
            "version": self.e_version,
            "entry": self.entry,
            "program_header_offset": self.phoff,
            "section_header_offset": self.shoff,
            "flags": self.flags,
            "header_size": self.ehsize,
            "program_header_entry_size": self.phentsize,
            "program_header_count": self.phnum,
            "section_header_entry_size": self.shentsize,
            "section_header_count": self.shnum,
            "section_name_string_table_index": self.shstrndx,
        }


@dataclass
class ProgramHeader:
    index: int
    p_type: int
    type_name: str
    flags: int
    offset: int
    vaddr: int
    paddr: int
    filesz: int
    memsz: int
    align: int

    @property
    def permissions(self) -> list[str]:
        names = []
        if self.flags & 4:
            names.append("read")
        if self.flags & 2:
            names.append("write")
        if self.flags & 1:
            names.append("execute")
        return names

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "type": self.type_name,
            "p_type": self.p_type,
            "flags": self.permissions,
            "offset": self.offset,
            "vaddr": self.vaddr,
            "paddr": self.paddr,
            "filesz": self.filesz,
            "memsz": self.memsz,
            "align": self.align,
        }


@dataclass
class SectionHeader:
    index: int
    name_offset: int
    name: str
    sh_type: int
    type_name: str
    flags: int
    addr: int
    offset: int
    size: int
    link: int
    info: int
    addralign: int
    entsize: int

    @property
    def flag_names(self) -> list[str]:
        mapping = [
            (1, "write"), (2, "alloc"), (4, "exec"), (0x10, "merge"),
            (0x20, "strings"), (0x40, "info_link"), (0x80, "link_order"),
            (0x100, "os_nonconforming"), (0x200, "group"), (0x400, "tls"),
            (0x800, "compressed"),
        ]
        return [name for bit, name in mapping if self.flags & bit]

    def contains_file_range(self, offset: int, size: int) -> bool:
        return self.sh_type != 8 and offset >= self.offset and offset + size <= self.offset + self.size

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "type": self.type_name,
            "sh_type": self.sh_type,
            "flags": self.flag_names,
            "addr": self.addr,
            "offset": self.offset,
            "size": self.size,
            "link": self.link,
            "info": self.info,
            "addralign": self.addralign,
            "entsize": self.entsize,
        }


@dataclass
class Symbol:
    index: int
    name: str
    value: int
    size: int
    info: int
    other: int
    shndx: int
    binding: str
    symbol_type: str
    visibility: str
    table: str
    version: str | None = None

    @property
    def is_function(self) -> bool:
        return self.symbol_type == "FUNC" and self.shndx != 0

    @property
    def is_import(self) -> bool:
        return self.shndx == 0 and self.binding in {"GLOBAL", "WEAK"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "value": self.value,
            "size": self.size,
            "binding": self.binding,
            "type": self.symbol_type,
            "visibility": self.visibility,
            "section_index": self.shndx,
            "table": self.table,
            "version": self.version,
        }


@dataclass
class Relocation:
    index: int
    offset: int
    info: int
    addend: int | None
    relocation_type: int
    type_name: str
    symbol_index: int
    symbol_name: str | None
    section_index: int
    table: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "offset": self.offset,
            "type": self.type_name,
            "relocation_type": self.relocation_type,
            "symbol_index": self.symbol_index,
            "symbol_name": self.symbol_name,
            "addend": self.addend,
            "section_index": self.section_index,
            "table": self.table,
        }


@dataclass
class Note:
    name: str
    note_type: int
    descriptor: bytes
    offset: int
    descriptor_hex: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.note_type,
            "offset": self.offset,
            "descriptor_hex": self.descriptor_hex,
        }


@dataclass
class DynamicEntry:
    tag: int
    tag_name: str
    value: int

    def to_dict(self) -> dict[str, Any]:
        return {"tag": self.tag_name, "tag_value": self.tag, "value": self.value}


@dataclass
class ElfFile:
    path: str
    size: int
    sha256: str
    header: ElfHeader
    program_headers: list[ProgramHeader] = field(default_factory=list)
    sections: list[SectionHeader] = field(default_factory=list)
    symbols: list[Symbol] = field(default_factory=list)
    dynamic_symbols: list[Symbol] = field(default_factory=list)
    relocations: list[Relocation] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    dynamic_entries: list[DynamicEntry] = field(default_factory=list)
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    build_id: str | None = None
    version_names: dict[int, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: bytes = field(default=b"", repr=False)
    endian: str = "<"

    def section_by_name(self, name: str) -> SectionHeader | None:
        for section in self.sections:
            if section.name == name:
                return section
        return None

    def section_by_index(self, index: int) -> SectionHeader | None:
        if 0 <= index < len(self.sections):
            return self.sections[index]
        return None

    def load_segments(self) -> list[ProgramHeader]:
        return [segment for segment in self.program_headers if segment.p_type == 1]

    def va_to_offset(self, va: int) -> int | None:
        for segment in self.load_segments():
            if segment.vaddr <= va < segment.vaddr + segment.filesz:
                return segment.offset + (va - segment.vaddr)
        return None

    def offset_to_va(self, offset: int) -> int | None:
        for segment in self.load_segments():
            if segment.offset <= offset < segment.offset + segment.filesz:
                return segment.vaddr + (offset - segment.offset)
        return None

    def read_va(self, va: int, size: int) -> bytes | None:
        offset = self.va_to_offset(va)
        if offset is None or offset + size > len(self.raw):
            return None
        return self.raw[offset:offset + size]

    def read_offset(self, offset: int, size: int) -> bytes | None:
        if offset < 0 or size < 0 or offset + size > len(self.raw):
            return None
        return self.raw[offset:offset + size]

    def function_symbols(self) -> list[Symbol]:
        seen = set()
        result = []
        for symbol in self.symbols + self.dynamic_symbols:
            if not (symbol.is_function and symbol.size > 0):
                continue
            if symbol.value in seen:
                continue
            seen.add(symbol.value)
            result.append(symbol)
        return result

    def named_symbols(self) -> list[Symbol]:
        return [symbol for symbol in self.symbols + self.dynamic_symbols if symbol.name]

    @property
    def is_pie(self) -> bool:
        return self.header.e_type == 3 and self.metadata.get("has_interpreter", False)

    @property
    def is_shared_object(self) -> bool:
        return self.header.e_type == 3 and not self.is_pie

    @property
    def is_static(self) -> bool:
        return not self.metadata.get("has_dynamic", False) and not self.metadata.get("has_interpreter", False)

    @property
    def is_stripped(self) -> bool:
        return not any(section.sh_type == 2 for section in self.sections)

    def inspect_dict(self, symbol_limit: int = 200, relocation_limit: int = 200) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "header": self.header.to_dict(),
            "segments": [segment.to_dict() for segment in self.program_headers],
            "sections": [section.to_dict() for section in self.sections],
            "symbols": [symbol.to_dict() for symbol in self.symbols[:symbol_limit]],
            "dynamic_symbols": [symbol.to_dict() for symbol in self.dynamic_symbols[:symbol_limit]],
            "relocations": [relocation.to_dict() for relocation in self.relocations[:relocation_limit]],
            "notes": [note.to_dict() for note in self.notes],
            "dynamic_entries": [entry.to_dict() for entry in self.dynamic_entries],
            "dependencies": self.dependencies,
            "build_id": self.build_id,
            "version_names": self.version_names,
            "stripped": self.is_stripped,
            "pie": self.is_pie,
            "shared_object": self.is_shared_object,
            "static": self.is_static,
            "metadata": self.metadata,
            "counts": {
                "program_headers": len(self.program_headers),
                "sections": len(self.sections),
                "symbols": len(self.symbols),
                "dynamic_symbols": len(self.dynamic_symbols),
                "relocations": len(self.relocations),
                "notes": len(self.notes),
            },
        }
