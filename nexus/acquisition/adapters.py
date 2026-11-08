import json
import lzma
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexus.acquisition.models import ArchiveRecord, ResolvedSource, SourceCandidate
from nexus.core.hashing import sha256_file

DEFAULT_USER_AGENT = "nexus-bi-acquisition/0.1 (+research; contact: local)"
MAX_ARCHIVE_BYTES = 80 * 1024 * 1024
MAX_INDEX_BYTES = 128 * 1024 * 1024


class AdapterError(Exception):
    pass


class AdapterUnavailable(AdapterError):
    pass


class RateLimited(AdapterError):
    pass


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes
    url: str


@dataclass
class HttpClient:
    timeout: int = 60
    user_agent: str = DEFAULT_USER_AGENT
    max_bytes: int = MAX_ARCHIVE_BYTES
    retries: int = 2
    rate_limit_state: dict[str, Any] = field(default_factory=dict)

    def get(self, url: str, headers: dict[str, str] | None = None, max_bytes: int | None = None) -> HttpResponse:
        limit = max_bytes or self.max_bytes
        request_headers = {"User-Agent": self.user_agent, "Accept": "*/*"}
        request_headers.update(headers or {})
        last_error = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(url, headers=request_headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read(limit + 1)
                    if len(payload) > limit:
                        raise AdapterError(f"response exceeds {limit} bytes: {url}")
                    record = HttpResponse(status=response.status, headers={key.lower(): value for key, value in response.headers.items()}, body=payload, url=response.geturl())
                    self.rate_limit_state[url] = {
                        "status": record.status,
                        "remaining": record.headers.get("x-ratelimit-remaining"),
                        "reset": record.headers.get("x-ratelimit-reset"),
                        "observed_at": int(time.time()),
                    }
                    return record
            except urllib.error.HTTPError as exc:
                body = exc.read(4096).decode("utf-8", errors="replace") if exc.fp else ""
                if exc.code in {403, 429} and "rate limit" in body.lower():
                    raise RateLimited(f"provider rate limit for {url}: {body[:300]}") from exc
                last_error = AdapterError(f"HTTP {exc.code} for {url}: {body[:300]}")
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = AdapterError(f"request failed for {url}: {exc}")
            if attempt < self.retries:
                time.sleep(min(2 ** attempt, 5))
        raise last_error or AdapterError(f"request failed for {url}")

    def get_json(self, url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
        response = self.get(url, {**(headers or {}), "Accept": "application/json"}, max_bytes=8 * 1024 * 1024)
        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError(f"invalid JSON from {url}: {exc}") from exc


def _git_ls_remote(repository_url: str, ref: str, timeout: int = 60) -> str | None:
    try:
        result = subprocess.run(
            ["git", "ls-remote", repository_url, ref],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("TMPDIR", "/tmp"), "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{40,64}", parts[0]):
            return parts[0]
    return None


def _download_to(url: str, target: Path, client: HttpClient, expected_sha: str | None = None, expected_size: int | None = None, timeout: int = 60) -> ArchiveRecord:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    resumed = part.exists() and part.stat().st_size > 0
    headers = {}
    if resumed:
        headers["Range"] = f"bytes={part.stat().st_size}-"
    request = urllib.request.Request(url, headers={"User-Agent": client.user_agent, **headers})
    mode = "ab" if resumed else "wb"
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, open(part, mode) as output:
            if resumed and response.status != 206:
                output.close()
                mode = "wb"
                request = urllib.request.Request(url, headers={"User-Agent": client.user_agent})
                with urllib.request.urlopen(request, timeout=timeout) as fresh, open(part, "wb") as reset:
                    _copy_limited(fresh, reset)
                resumed = False
            else:
                _copy_limited(response, output)
    except urllib.error.HTTPError as exc:
        raise AdapterError(f"download failed HTTP {exc.code}: {url}") from exc
    except urllib.error.URLError as exc:
        raise AdapterError(f"download failed: {url}: {exc}") from exc
    if expected_size is not None and part.stat().st_size != expected_size:
        raise AdapterError(f"archive size mismatch for {url}: expected {expected_size}, got {part.stat().st_size}")
    digest = sha256_file(part)
    if expected_sha and digest != expected_sha:
        raise AdapterError(f"archive checksum mismatch for {url}: expected {expected_sha}, got {digest}")
    part.replace(target)
    return ArchiveRecord(url=url, path=str(target), sha256=digest, size=target.stat().st_size, resumed=resumed)


def _copy_limited(source, target, limit: int = MAX_ARCHIVE_BYTES) -> None:
    total = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise AdapterError(f"archive exceeds {limit} bytes")
        target.write(chunk)


class BaseAdapter:
    name = "base"
    fetch_supported = True

    def __init__(self, client: HttpClient | None = None) -> None:
        self.client = client or HttpClient()

    def resolve(self, candidate: SourceCandidate) -> ResolvedSource:
        raise NotImplementedError

    def fetch(self, resolved: ResolvedSource, target: Path) -> ArchiveRecord:
        raise NotImplementedError

    def discover(self, query: str, limit: int = 20) -> list[SourceCandidate]:
        raise AdapterUnavailable(f"{self.name} discovery is not implemented")


class GitHubAdapter(BaseAdapter):
    name = "github"
    api = "https://api.github.com"

    def resolve(self, candidate: SourceCandidate) -> ResolvedSource:
        expected = candidate.commit
        resolved_commit = expected
        warnings = []
        if candidate.ref and candidate.ref != candidate.commit:
            live = _git_ls_remote(f"https://github.com/{candidate.repository}.git", candidate.ref)
            if live:
                if expected and live != expected:
                    warnings.append(f"requested ref {candidate.ref} resolves to {live}, not declared commit {expected}")
                resolved_commit = expected or live
            elif expected is None:
                raise AdapterError(f"unable to resolve {candidate.repository} ref {candidate.ref}")
            else:
                warnings.append("git ls-remote unavailable; using declared commit without ref verification")
        if not resolved_commit:
            raise AdapterError(f"no commit for {candidate.repository}")
        return ResolvedSource(
            provider=self.name,
            repository=candidate.repository,
            canonical_url=candidate.canonical_url or f"https://github.com/{candidate.repository}",
            commit=resolved_commit,
            ref=candidate.ref,
            metadata={"api": self.api, "clone_url": f"https://github.com/{candidate.repository}.git"},
            warnings=warnings,
        )

    def fetch(self, resolved: ResolvedSource, target: Path) -> ArchiveRecord:
        url = f"https://codeload.github.com/{resolved.repository}/tar.gz/{resolved.commit}"
        return _download_to(url, target, self.client, expected_sha=None, timeout=90)

    def discover(self, query: str, limit: int = 20) -> list[SourceCandidate]:
        url = f"{self.api}/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc&per_page={min(limit, 50)}"
        payload = self.client.get_json(url)
        candidates = []
        for item in payload.get("items", [])[:limit]:
            candidates.append(
                SourceCandidate(
                    project_id=item["full_name"].replace("/", "__").lower(),
                    adapter=self.name,
                    repository=item["full_name"],
                    canonical_url=item["html_url"],
                    category="other",
                    description=item.get("description") or "",
                    topics=item.get("topics") or [],
                    expected_license=(item.get("license") or {}).get("spdx_id"),
                    provider_metadata={
                        "stars": item.get("stargazers_count"),
                        "size_kb": item.get("size"),
                        "default_branch": item.get("default_branch"),
                        "pushed_at": item.get("pushed_at"),
                        "archived": item.get("archived"),
                    },
                )
            )
        return candidates


class GitLabAdapter(BaseAdapter):
    name = "gitlab"

    def resolve(self, candidate: SourceCandidate) -> ResolvedSource:
        resolved_commit = candidate.commit
        warnings = []
        if candidate.ref and candidate.ref != candidate.commit:
            live = _git_ls_remote(f"https://gitlab.com/{candidate.repository}.git", candidate.ref)
            if live:
                if candidate.commit and live != candidate.commit:
                    warnings.append(f"requested ref resolves to {live}, not declared commit {candidate.commit}")
                resolved_commit = candidate.commit or live
            elif not candidate.commit:
                raise AdapterError(f"unable to resolve GitLab ref {candidate.ref}")
        if not resolved_commit:
            raise AdapterError(f"no commit for {candidate.repository}")
        return ResolvedSource(
            provider=self.name,
            repository=candidate.repository,
            canonical_url=candidate.canonical_url or f"https://gitlab.com/{candidate.repository}",
            commit=resolved_commit,
            ref=candidate.ref,
            metadata={"clone_url": f"https://gitlab.com/{candidate.repository}.git"},
            warnings=warnings,
        )

    def fetch(self, resolved: ResolvedSource, target: Path) -> ArchiveRecord:
        name = resolved.repository.rstrip("/").split("/")[-1]
        url = f"https://gitlab.com/{resolved.repository}/-/archive/{resolved.commit}/{name}-{resolved.commit}.tar.gz"
        return _download_to(url, target, self.client, timeout=90)

    def discover(self, query: str, limit: int = 20) -> list[SourceCandidate]:
        url = f"https://gitlab.com/api/v4/projects?search={urllib.parse.quote(query)}&per_page={min(limit, 50)}&order_by=star_count&sort=desc&visibility=public"
        payload = self.client.get_json(url)
        if not isinstance(payload, list):
            raise AdapterError("unexpected GitLab discovery response")
        candidates = []
        for item in payload[:limit]:
            candidates.append(
                SourceCandidate(
                    project_id=str(item.get("path_with_namespace", item.get("id"))).replace("/", "__").lower(),
                    adapter=self.name,
                    repository=item.get("path_with_namespace", ""),
                    canonical_url=item.get("web_url", ""),
                    category="other",
                    description=item.get("description") or "",
                    topics=item.get("topics") or [],
                    provider_metadata={"stars": item.get("star_count"), "default_branch": item.get("default_branch"), "last_activity_at": item.get("last_activity_at")},
                )
            )
        return candidates


class SoftwareHeritageAdapter(BaseAdapter):
    name = "softwareheritage"
    api = "https://archive.softwareheritage.org/api/1"
    fetch_supported = False

    def resolve(self, candidate: SourceCandidate) -> ResolvedSource:
        origin_url = candidate.canonical_url or f"https://github.com/{candidate.repository}"
        encoded = urllib.parse.quote(origin_url, safe="")
        try:
            visits = self.client.get_json(f"{self.api}/origin/{encoded}/visits/?limit=20")
        except AdapterError as exc:
            raise AdapterUnavailable(f"Software Heritage origin lookup failed: {exc}") from exc
        entries = visits if isinstance(visits, list) else visits.get("results", [])
        if not entries:
            raise AdapterUnavailable(f"origin not archived by Software Heritage: {origin_url}")
        snapshot = entries[0].get("snapshot")
        revision = candidate.commit
        corroborated = False
        if candidate.commit:
            try:
                revision_payload = self.client.get_json(f"{self.api}/revision/{candidate.commit}/")
                corroborated = isinstance(revision_payload, dict) and bool(revision_payload.get("id"))
                if corroborated:
                    revision = revision_payload.get("id") or candidate.commit
            except AdapterError:
                corroborated = False
        return ResolvedSource(
            provider=self.name,
            repository=candidate.repository,
            canonical_url=origin_url,
            commit=str(revision or candidate.commit or ""),
            ref=candidate.ref,
            metadata={"swh_revision": revision, "swh_snapshot": snapshot, "visits": len(entries), "fetch_supported": False, "revision_corroborated": corroborated},
            warnings=[] if corroborated else ["Software Heritage revision is not corroborated"],
        )

    def fetch(self, resolved: ResolvedSource, target: Path) -> ArchiveRecord:
        raise AdapterUnavailable("Software Heritage vault downloads are not enabled; use the origin archive for acquisition and this adapter for corroboration")


class DebianAdapter(BaseAdapter):
    name = "debian"
    index_url = "https://deb.debian.org/debian/dists/{suite}/main/source/Sources.xz"
    pool = "https://deb.debian.org/debian"

    def __init__(self, client: HttpClient | None = None, cache_dir: str | Path | None = None) -> None:
        super().__init__(client)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(os.environ.get("NEXUS_CACHE", "/tmp/nexus-cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _index(self, suite: str = "stable") -> list[dict[str, Any]]:
        path = self.cache_dir / f"debian-{suite}-Sources.xz"
        if not path.exists():
            response = self.client.get(self.index_url.format(suite=suite), max_bytes=MAX_INDEX_BYTES)
            path.write_bytes(response.body)
        try:
            text = lzma.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        except lzma.LZMAError as exc:
            raise AdapterError(f"unable to decompress Debian Sources index: {exc}") from exc
        entries = []
        for block in text.split("\n\n"):
            if not block.strip():
                continue
            fields: dict[str, Any] = {}
            key = None
            for line in block.splitlines():
                if line.startswith(" ") and key:
                    fields[key] += "\n" + line.strip()
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                fields[key.strip()] = value.strip()
            if "Package" in fields:
                entries.append(fields)
        return entries

    def resolve(self, candidate: SourceCandidate) -> ResolvedSource:
        package = candidate.extra.get("package") or candidate.repository
        suite = candidate.extra.get("suite", "stable")
        wanted = candidate.extra.get("version")
        entries = [entry for entry in self._index(suite) if entry.get("Package") == package]
        if wanted:
            entries = [entry for entry in entries if entry.get("Version") == wanted]
        if not entries:
            raise AdapterUnavailable(f"Debian package not found: {package} {wanted or ''}".strip())
        entry = sorted(entries, key=lambda item: item.get("Version", ""))[-1]
        files = []
        for line in entry.get("Files", "").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                files.append({"md5": parts[0], "size": int(parts[1]), "name": parts[2]})
        checksums = {}
        for line in entry.get("Checksums-Sha256", "").splitlines():
            parts = line.split()
            if len(parts) >= 3:
                checksums[parts[2]] = {"sha256": parts[0], "size": int(parts[1])}
        orig = next((item for item in files if ".orig.tar" in item["name"]), None)
        if orig is None:
            raise AdapterUnavailable(f"Debian source package has no orig tar: {package}")
        orig["sha256"] = checksums.get(orig["name"], {}).get("sha256")
        return ResolvedSource(
            provider=self.name,
            repository=package,
            canonical_url=f"https://packages.debian.org/source/{suite}/{package}",
            commit=entry.get("Version", ""),
            ref=suite,
            metadata={
                "package": package,
                "version": entry.get("Version"),
                "directory": entry.get("Directory"),
                "homepage": entry.get("Homepage"),
                "files": files,
                "orig": orig,
                "debian_tar": next((item for item in files if ".debian.tar" in item["name"]), None),
                "copyright_file": "debian/copyright",
            },
        )

    def fetch(self, resolved: ResolvedSource, target: Path) -> ArchiveRecord:
        orig = resolved.metadata.get("orig") or {}
        directory = resolved.metadata.get("directory")
        if not directory or not orig.get("name"):
            raise AdapterError("Debian resolver did not provide an orig archive")
        url = f"{self.pool}/{directory}/{orig['name']}"
        return _download_to(url, target, self.client, expected_sha=orig.get("sha256"), expected_size=orig.get("size"), timeout=120)


class LocalArchiveAdapter(BaseAdapter):
    name = "local"
    fetch_supported = True

    def resolve(self, candidate: SourceCandidate) -> ResolvedSource:
        archive = candidate.extra.get("archive_path")
        if not archive:
            raise AdapterError("local adapter requires extra.archive_path")
        path = Path(archive)
        if not path.exists():
            raise AdapterError(f"local archive does not exist: {path}")
        return ResolvedSource(
            provider=self.name,
            repository=candidate.repository,
            canonical_url=candidate.canonical_url or f"file://{path.resolve()}",
            commit=candidate.commit or sha256_file(path)[:40],
            ref=candidate.ref,
            metadata={"archive_path": str(path.resolve()), "declared_commit": candidate.commit},
        )

    def fetch(self, resolved: ResolvedSource, target: Path) -> ArchiveRecord:
        source = Path(resolved.metadata["archive_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return ArchiveRecord(url=f"file://{source}", path=str(target), sha256=sha256_file(target), size=target.stat().st_size, media_type="application/gzip")


ADAPTERS = {
    "github": GitHubAdapter,
    "gitlab": GitLabAdapter,
    "softwareheritage": SoftwareHeritageAdapter,
    "debian": DebianAdapter,
    "local": LocalArchiveAdapter,
}


def adapter_for(name: str, client: HttpClient | None = None, cache_dir: str | Path | None = None) -> BaseAdapter:
    key = name.lower()
    if key not in ADAPTERS:
        raise AdapterError(f"unknown adapter: {name}")
    if key == "debian":
        return DebianAdapter(client=client, cache_dir=cache_dir)
    return ADAPTERS[key](client=client)


def discover(adapter: str, query: str, limit: int = 20, client: HttpClient | None = None) -> list[SourceCandidate]:
    return adapter_for(adapter, client=client).discover(query, limit=limit)
