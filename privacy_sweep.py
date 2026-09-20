#!/usr/bin/env python3
"""Privacy sweep: fail if a publishable file leaks personal data.

Checks every tracked file for personal emails, other people's identifiers, home
directory paths, API keys/tokens, private keys, and absolute paths that name a
real user. Exit code 1 when anything is found, so it can gate a release.

Usage:  python3 privacy_sweep.py [--fix-hints]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "runs",
             "build", "dist", ".venv", "node_modules"}
SKIP_SUFFIX = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz",
               ".apk", ".so", ".bin", ".html"}

PATTERNS = [
    ("personal-email", re.compile(r"[A-Za-z0-9._%+-]+@(?:gmail|googlemail|yahoo|"
                                 r"hotmail|outlook|qq|163|126)\.[A-Za-z.]{2,}")),
    ("institutional-email", re.compile(r"@(?:connect\.hku\.hk|ha\.org\.hk|"
                                       r"hku\.hk)")),
    ("home-path", re.compile(r"/home/[A-Za-z0-9._-]+/")),
    ("windows-user-path", re.compile(r"[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9._-]+")),
    ("api-key", re.compile(r"\b(?:sk-[A-Za-z0-9]{12,}|nvapi-[A-Za-z0-9_-]{12,}|"
                           r"ghp_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,}|"
                           r"xox[baprs]-[A-Za-z0-9-]{10,})\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("oauth-client-secret", re.compile(r"GOCSPX-[A-Za-z0-9_-]{10,}")),
    ("phone-number", re.compile(r"(?:\+852|852)[\s-]?\d{4}[\s-]?\d{4}\b")),
    ("hardcoded-token", re.compile(r"(?:refresh_token|access_token|sessionKey|"
                                   r"cookie)\s*[=:]\s*[\"'][A-Za-z0-9._%-]{20,}")),
]
ALLOW = [
    re.compile(r"lesterppo@users\.noreply\.github\.com"),
    re.compile(r"example\.com"),
    re.compile(r"\bHOME\b|Path\.home\(\)|~/\."),
]


def tracked_files(root: Path) -> list:
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files"],
                             capture_output=True, text=True, check=True).stdout
        files = [root / line for line in out.splitlines() if line.strip()]
        if files:
            return files
    except Exception:  # noqa: BLE001
        pass
    files = []
    for p in root.rglob("*"):
        if p.is_dir() or any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--fix-hints", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    findings = []
    for path in tracked_files(root):
        if path.suffix.lower() in SKIP_SUFFIX:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root)
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, rx in PATTERNS:
                m = rx.search(line)
                if not m:
                    continue
                if any(a.search(line) for a in ALLOW):
                    continue
                findings.append((name, str(rel), lineno, m.group(0)[:60]))
    if not findings:
        print(f"privacy sweep clean: {len(tracked_files(root))} files checked in {root}")
        return 0
    print(f"privacy sweep FAILED: {len(findings)} finding(s)")
    for name, rel, lineno, sample in findings:
        print(f"  {name:22s} {rel}:{lineno}  {sample}")
    if args.fix_hints:
        print("\nhints: use get_hermes_home()/Path.home(); replace personal addresses "
              "with placeholders; keep credentials in ~/.hermes/secrets/ or env vars")
    return 1


if __name__ == "__main__":
    sys.exit(main())
