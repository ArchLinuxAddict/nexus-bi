import re
from pathlib import Path
from typing import Any

from nexus.acquisition.models import LicenseEvidence
from nexus.core.hashing import sha256_file

ALLOWED_SPDX = {
    "0BSD",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSD-3-Clause-Clear",
    "CC0-1.0",
    "ISC",
    "MIT",
    "Unlicense",
    "WTFPL",
    "Zlib",
}
COPYLEFT_SPDX = {
    "AGPL-3.0",
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "GPL-2.0",
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "LGPL-2.1",
    "LGPL-2.1-only",
    "LGPL-2.1-or-later",
    "LGPL-3.0",
    "LGPL-3.0-only",
    "LGPL-3.0-or-later",
    "MPL-2.0",
    "EPL-2.0",
    "CDDL-1.0",
    "SSPL-1.0",
}
RESTRICTED_SPDX = {"BUSL-1.1", "Commons-Clause", "Elastic-2.0", "Proprietary"}
LICENSE_FILE_NAMES = {
    "license",
    "license.txt",
    "license.md",
    "license.rst",
    "licence",
    "licence.txt",
    "licence.md",
    "copying",
    "copying.txt",
    "copying.md",
    "copyright",
    "unlicense",
    "unlicense.txt",
    "license-mit",
    "license-bsd",
    "license.apache-2.0",
}
MAX_LICENSE_BYTES = 2 * 1024 * 1024
EXCLUDED_PARTS = {"test", "tests", "testing", "examples", "example", "third_party", "third-party", "vendor", "externals", "crates", "contrib"}


def _normalize(text: str) -> str:
    value = text.lower().replace("\u2019", "'").replace("\u2018", "'").replace("\u00a0", " ")
    value = value.replace("*", " ")
    value = re.sub(r"[\t\r\n]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _license_files(root: Path, ignore_paths: list[str] | None = None) -> list[Path]:
    found: list[Path] = []
    ignored = tuple(ignore_paths or ())
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(str(relative).replace("\\", "/").startswith(prefix) for prefix in ignored):
            continue
        if any(part.lower() in EXCLUDED_PARTS for part in relative.parts[:-1]):
            continue
        if len(relative.parts) > 3:
            continue
        name = path.name.lower()
        if name in LICENSE_FILE_NAMES or name.startswith("license.") or name.startswith("licence.") or name.startswith("copying."):
            if path.stat().st_size <= MAX_LICENSE_BYTES:
                found.append(path)
    found.sort(key=lambda item: (len(item.relative_to(root).parts), len(item.name), str(item)))
    return found


def _spdx_from_text(text: str) -> tuple[list[str], str | None]:
    values = []
    expression = None
    for match in re.finditer(r"spdx-license-identifier\s*:\s*([^\n*]+)", text, re.IGNORECASE):
        raw = match.group(1).strip()
        if expression is None:
            expression = raw
        for part in re.split(r"\s+(?:OR|AND)\s+|[()]", raw):
            token = part.strip()
            if token and token not in {"OR", "AND"}:
                values.append(token)
    return values, expression


def _match_license(text: str) -> tuple[str | None, str, float, str | None]:
    value = _normalize(text)
    spdx_tokens, spdx_expression = _spdx_from_text(text)
    if spdx_tokens:
        unique = sorted(set(spdx_tokens))
        if len(unique) == 1:
            return unique[0], "spdx_identifier", 0.99, None
        return None, "spdx_identifier", 0.9, spdx_expression
    if "gnu affero general public license" in value:
        version = "3.0" if "version 3" in value else "unknown"
        return f"AGPL-{version}", "text_signature", 0.95, None
    if "gnu lesser general public license" in value:
        version = "3.0" if "version 3" in value else "2.1" if "version 2.1" in value else "unknown"
        return f"LGPL-{version}", "text_signature", 0.95, None
    if "gnu general public license" in value:
        version = "3.0" if "version 3" in value else "2.0" if "version 2" in value else "unknown"
        return f"GPL-{version}", "text_signature", 0.95, None
    if "mozilla public license" in value and "2.0" in value:
        return "MPL-2.0", "text_signature", 0.95, None
    if "business source license" in value:
        return "BUSL-1.1", "text_signature", 0.9, None
    if "server side public license" in value:
        return "SSPL-1.0", "text_signature", 0.9, None
    if "do what the fuck you want to public license" in value:
        return "WTFPL", "text_signature", 0.95, None
    if "apache license" in value and "version 2.0" in value and ("apache.org/licenses" in value or "licensed under the apache license" in value):
        return "Apache-2.0", "text_signature", 0.98, None
    if "this is free and unencumbered software released into the public domain" in value or "unlicense.org" in value:
        return "Unlicense", "text_signature", 0.98, None
    if "cc0 1.0 universal" in value or ("creative commons" in value and "cc0" in value and "public domain dedication" in value):
        return "CC0-1.0", "text_signature", 0.97, None
    if "permission is hereby granted, free of charge, to any person obtaining a copy" in value and "the above copyright notice and this permission notice shall be included" in value:
        return "MIT", "text_signature", 0.98, None
    if "redistribution and use in source and binary forms" in value and "redistributions of source code must retain" in value and "redistributions in binary form must reproduce" in value:
        if "neither the name" in value or "names of its contributors may not be used to endorse" in value:
            return "BSD-3-Clause", "text_signature", 0.97, None
        return "BSD-2-Clause", "text_signature", 0.97, None
    if "permission to use, copy, modify, and/or distribute this software for any purpose with or without fee" in value or "permission to use, copy, modify, and distribute this software for any purpose with or without fee" in value:
        if "provided that the above copyright notice and this permission notice appear in all copies" in value:
            return "ISC", "text_signature", 0.96, None
        return "0BSD", "text_signature", 0.9, None
    if "this software is provided 'as-is'" in value and "altered source versions must be plainly marked" in value:
        return "Zlib", "text_signature", 0.97, None
    if "public domain" in value:
        return None, "public_domain_language", 0.4, "LicenseRef-PublicDomain"
    return None, "unknown", 0.0, None


def _expression_allowed(expression: str | None) -> bool:
    if not expression:
        return False
    tokens = [token.strip() for token in re.split(r"\s+(?:OR|AND)\s+|[()]", expression) if token.strip() and token.strip() not in {"OR", "AND"}]
    if not tokens:
        return False
    return all(token in ALLOWED_SPDX for token in tokens)


def license_allowed(spdx: str | None, expression: str | None = None) -> bool:
    if expression:
        return _expression_allowed(expression)
    return spdx in ALLOWED_SPDX


def classify_license(evidence: LicenseEvidence, expected: str | None = None) -> LicenseEvidence:
    if not evidence.found:
        evidence.status = "quarantined"
        evidence.reason = "no license file"
        evidence.confidence = 0.0
        return evidence
    if evidence.expression:
        tokens = [token.strip() for token in re.split(r"\s+(?:OR|AND)\s+|[()]", evidence.expression) if token.strip() and token.strip() not in {"OR", "AND"}]
    else:
        tokens = [evidence.spdx] if evidence.spdx else []
    if not tokens:
        evidence.status = "quarantined"
        evidence.reason = evidence.reason or "license not identified"
        return evidence
    blocked = [token for token in tokens if token not in ALLOWED_SPDX]
    if blocked:
        evidence.status = "quarantined"
        evidence.reason = f"license outside permissive allowlist: {', '.join(blocked)}"
        return evidence
    if expected and evidence.spdx and evidence.spdx != expected and not evidence.expression:
        evidence.status = "quarantined"
        evidence.reason = f"declared license {expected} conflicts with detected {evidence.spdx}"
        return evidence
    evidence.status = "accepted"
    evidence.reason = "permissive license with file evidence"
    return evidence


def _select_primary(files: list[Path], root: Path) -> Path | None:
    if not files:
        return None
    root_level = [path for path in files if len(path.relative_to(root).parts) == 1]
    pool = root_level or files
    preference = ["license", "licence", "copying", "unlicense", "copyright"]
    for name in preference:
        for path in pool:
            if path.name.lower() == name or path.name.lower().startswith(f"{name}."):
                return path
    return pool[0]


def detect_license(root: str | Path, expected: str | None = None, ignore_paths: list[str] | None = None) -> LicenseEvidence:
    base = Path(root)
    files = _license_files(base, ignore_paths=ignore_paths)
    primary = _select_primary(files, base)
    if primary is None:
        return LicenseEvidence(found=False, status="quarantined", reason="no license file")
    texts = {}
    for path in files:
        try:
            texts[path] = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    if primary not in texts:
        return LicenseEvidence(found=False, status="quarantined", reason="license file unreadable", license_file=str(primary.relative_to(base)))
    spdx, method, confidence, expression = _match_license(texts[primary])
    evidence = LicenseEvidence(
        found=True,
        status="pending",
        spdx=spdx,
        expression=expression,
        license_file=str(primary.relative_to(base)),
        sha256=sha256_file(primary),
        size=primary.stat().st_size,
        method=method,
        confidence=confidence,
        detected_from=[str(primary.relative_to(base))],
        reason=None,
    )
    secondary = []
    for path, text in texts.items():
        if path == primary:
            continue
        other_spdx, other_method, other_confidence, other_expression = _match_license(text)
        if other_spdx is None and other_expression is None:
            continue
        secondary.append(
            {
                "path": str(path.relative_to(base)),
                "sha256": sha256_file(path),
                "spdx": other_spdx,
                "expression": other_expression,
                "method": other_method,
                "confidence": other_confidence,
            }
        )
    evidence.secondary_files = secondary
    other_licenses = {item["spdx"] or item["expression"] for item in secondary if item["spdx"] or item["expression"]}
    own = {spdx or expression}
    others = {item for item in other_licenses if item not in own}
    if others:
        combined = " OR ".join(sorted(token for token in own | others if token))
        if _expression_allowed(combined):
            evidence.expression = combined
            evidence.spdx = None
            evidence.method = "multi_file_expression"
            evidence.reason = "multiple license files form a permissive expression"
        else:
            evidence.reason = "conflicting license files: " + ", ".join(sorted(str(token) for token in others))
    if evidence.reason and "conflicting" in evidence.reason and evidence.status == "pending":
        evidence.status = "quarantined"
        return evidence
    return classify_license(evidence, expected)


def license_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_spdx: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for record in records:
        license_record = record.get("license") or {}
        spdx = license_record.get("spdx") or license_record.get("expression") or "unknown"
        by_spdx[spdx] = by_spdx.get(spdx, 0) + 1
        status = license_record.get("status") or "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "projects": len(records),
        "accepted": statuses.get("accepted", 0),
        "quarantined": statuses.get("quarantined", 0),
        "by_spdx": dict(sorted(by_spdx.items())),
        "statuses": dict(sorted(statuses.items())),
        "allowlist": sorted(ALLOWED_SPDX),
        "policy": "Only permissive licenses with a license-file hash are accepted. Unknown, public-domain-only language without SPDX, copyleft, source-available and conflicting licenses are quarantined.",
    }
