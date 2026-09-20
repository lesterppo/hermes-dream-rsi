"""LLM back-ends: discovery agent + policy-development agent.

Backends (``--agent``):
  ``deepseek[:model]``   DeepSeek API over HTTPS (key: $DEEPSEEK_API_KEY or ~/.dsh/.env)
  ``openai:base|model|keyfile``  any OpenAI-compatible endpoint
  ``dsh``                DeepSeek Harness headless agent (edits files itself)
  ``cmd:<template>``     your own command; {prompt} and {dir} are substituted
  ``mock``               deterministic offline generator (tests, smoke runs)

One-shot back-ends return text; the file protocol in prompts.py turns that text
into files.  Agentic back-ends (dsh) write files directly in the workspace.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

FILE_BLOCK = re.compile(r"<<<FILE:\s*(?P<path>[^>\n]+?)\s*>>>\n(?P<body>.*?)\n?<<<END>>>",
                        re.DOTALL)
FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\n(?P<body>.*?)```", re.DOTALL)


class AgentError(RuntimeError):
    pass


@dataclass
class AgentResult:
    ok: bool
    text: str = ""
    backend: str = ""
    model: str = ""
    files: Dict[str, str] = field(default_factory=dict)
    wrote_files: bool = False       # agent wrote files itself (agentic back-ends)
    error: Optional[str] = None
    latency_s: float = 0.0
    usage: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "backend": self.backend, "model": self.model,
            "files": sorted(self.files), "wrote_files": self.wrote_files,
            "error": self.error, "latency_s": round(self.latency_s, 2),
            "usage": self.usage,
        }


_DSML = "\uFF5C\uFF5CDSML\uFF5C\uFF5C"          # ｜｜DSML｜｜ tool-call markup
DSML_BLOCK = re.compile(re.escape(_DSML) + r".*?(?:>.*?" + re.escape(_DSML)
                        + r"\w+>|$)", re.DOTALL)


def strip_tool_markup(text: str) -> str:
    """Remove tool-call markup a chat model may emit despite having no tools."""
    if not text:
        return ""
    out = DSML_BLOCK.sub("", text)
    if _DSML in out:                      # unbalanced / truncated markup
        out = out.split(_DSML, 1)[0]
    return out


SAFE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,119}$")


def _safe_rel_path(path: str) -> Optional[str]:
    """Accept only a plausible file path from a model's FILE header.

    Absolute paths are tolerated (the caller remaps by basename), but traversal
    segments and anything that is not a file name (prose or math captured by the
    header regex) are rejected instead of becoming a filesystem error.
    """
    candidate = path.strip().strip("`'\"")
    if not candidate:
        return None
    parts = [p for p in candidate.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    rel = "/".join(parts)
    if not SAFE_PATH.match(rel) or rel.endswith("/"):
        return None
    return rel


MD_LINK = re.compile(r"\[([^\]\n]{1,300})\]\((https?://[^)\s]+)\)")
MD_ESCAPE = re.compile(r"\\([_*\[\]()#`~|])")


def delink(body: str) -> str:
    """Undo markdown autolinking that mangles URLs inside generated code.

    Chat back-ends (notably Gemini's web UI) rewrite bare URLs into
    ``[url](url)`` markdown, which corrupts string literals in the artefact even
    though the code is otherwise correct.
    """
    if not body:
        return body
    def _fix(m):
        label, url = m.group(1), m.group(2)
        stripped = label.strip()
        if stripped == url or stripped.rstrip("/") == url.rstrip("/") or \
                stripped in url:
            return url
        return m.group(0)
    out = MD_LINK.sub(_fix, body)
    return MD_ESCAPE.sub(r"\1", out)


def extract_files(text: str, default_name: Optional[str] = None,
                  expected_name: Optional[str] = None) -> Dict[str, str]:
    """Parse the <<<FILE: path>>> protocol; fall back to a single code fence.

    ``expected_name`` pins the file the caller actually needs: models routinely
    emit an absolute path (which would land in a nonsense nested directory), so a
    basename match is remapped onto the expected relative path.  Paths that are not
    plausible file names (prose or math captured by the header regex) are dropped
    rather than turned into a filesystem error.
    """
    text = strip_tool_markup(text)
    out: Dict[str, str] = {}
    for m in FILE_BLOCK.finditer(text or ""):
        rel = _safe_rel_path(m.group("path"))
        if rel:
            out[rel] = delink(m.group("body")).rstrip() + "\n"
    if not out and expected_name:
        fences = FENCE.findall(text or "")
        if fences:
            out[expected_name] = delink(max(fences, key=len)).rstrip() + "\n"
    if not out and default_name:
        fences = FENCE.findall(text or "")
        if fences:
            out[default_name] = delink(max(fences, key=len)).rstrip() + "\n"
    if expected_name:
        norm: Dict[str, str] = {}
        for rel, body in out.items():
            name = Path(rel).name
            if name == expected_name:
                norm[expected_name] = body
            else:
                norm[rel] = body
        out = norm
    return out


# --------------------------------------------------------------------------- #
# back-ends                                                                   #
# --------------------------------------------------------------------------- #
NO_TOOLS_SYSTEM = (
    "You are an offline coding assistant working on files inside your working "
    "directory. You have NO tools, NO shell, NO file access and NO internet: you "
    "cannot run commands or inspect the filesystem. Never emit tool-call or "
    "function-call markup (no DSML, no <invoke>, no JSON tool calls). Answer by "
    "writing files with the required file protocol only."
)


class Backend:
    name = "backend"
    model = ""

    def complete(self, prompt: str, cwd: Optional[str] = None,
                 system: Optional[str] = None) -> AgentResult:
        raise NotImplementedError


def _deepseek_key() -> Optional[str]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()
    env = Path.home() / ".dsh" / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY"):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


class OpenAIChatBackend(Backend):
    """Any OpenAI-compatible /chat/completions endpoint.

    Streams by default: long unary generations are routinely cut by the provider
    (IncompleteRead / connection reset), while the SSE stream survives them, and a
    transient failure is retried with backoff instead of losing the cycle.
    """

    def __init__(self, base_url: str, model: str, api_key: str,
                 name: str = "openai", max_tokens: int = 32768,
                 timeout: int = 600, reasoning_effort: Optional[str] = None,
                 system_prompt: str = NO_TOOLS_SYSTEM, stream: bool = True,
                 retries: int = 3, idle_timeout: int = 240) -> None:
        self.system_prompt = system_prompt
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.name = name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.stream = stream
        self.retries = max(1, int(retries))
        self.idle_timeout = int(idle_timeout)

    def complete(self, prompt: str, cwd: Optional[str] = None,
                 system: Optional[str] = None) -> AgentResult:
        last: AgentResult = AgentResult(False, backend=self.name, model=self.model,
                                        error="no attempt")
        for attempt in range(1, self.retries + 1):
            result = self._attempt(prompt, system)
            if result.ok and result.text.strip():
                result.usage = dict(result.usage or {})
                result.usage["attempt"] = attempt
                return result
            last = result
            if attempt < self.retries:
                time.sleep(min(20.0, 3.0 * attempt))
        return last

    # ---------------------------------------------------------------- request
    def _attempt(self, prompt: str, system: Optional[str]) -> AgentResult:
        sys_text = system or self.system_prompt
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": ([{"role": "system", "content": sys_text}] if sys_text else [])
                        + [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "stream": bool(self.stream),
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Accept": "text/event-stream, application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if self.stream:
                    self._arm_idle_timeout(resp)
                    text, usage = self._read_stream(resp)
                else:
                    data = json.loads(resp.read().decode())
                    choice = (data.get("choices") or [{}])[0]
                    msg = choice.get("message") or {}
                    text = msg.get("content") or msg.get("reasoning_content") or ""
                    usage = dict(data.get("usage") or {})
                    usage["finish_reason"] = choice.get("finish_reason")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="ignore")[:400]
            return AgentResult(False, backend=self.name, model=self.model,
                               error=f"HTTP {exc.code}: {detail}",
                               latency_s=time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            return AgentResult(False, backend=self.name, model=self.model,
                               error=f"{type(exc).__name__}: {exc}",
                               latency_s=time.time() - t0)
        if not text.strip():
            return AgentResult(False, backend=self.name, model=self.model,
                               error="empty completion", latency_s=time.time() - t0,
                               usage=usage)
        # a stream that ended without a finish_reason was cut, not completed
        if usage.get("stream_error") and usage.get("finish_reason") is None:
            return AgentResult(False, text=text, backend=self.name,
                               model=self.model,
                               error=f"stream cut: {usage['stream_error']}",
                               latency_s=time.time() - t0, usage=usage)
        return AgentResult(True, text=text, backend=self.name, model=self.model,
                           latency_s=time.time() - t0, usage=usage)

    def _arm_idle_timeout(self, resp: Any, idle: Optional[int] = None) -> None:
        """Fail a stalled stream instead of hanging on provider silence."""
        seconds = int(idle or self.idle_timeout)
        try:
            sock = resp.fp.raw._sock  # type: ignore[attr-defined]
            sock.settimeout(seconds)
        except Exception:  # noqa: BLE001 — best effort, urlopen timeout still applies
            pass

    def _read_stream(self, resp: Any) -> tuple:
        """Parse an SSE completion stream; partial text survives a broken pipe."""
        chunks: List[str] = []
        usage: Dict[str, Any] = {}
        finish = None
        try:
            for raw in resp:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if data.get("usage"):
                    usage = dict(data["usage"])
                choice = (data.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or choice.get("message") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    chunks.append(piece)
        except Exception as exc:  # noqa: BLE001 — keep whatever arrived
            usage["stream_error"] = f"{type(exc).__name__}: {exc}"
        usage["finish_reason"] = finish
        return "".join(chunks), usage



class DshBackend(Backend):
    """DeepSeek Harness headless agent: writes files in the workspace itself."""

    name = "dsh"

    def __init__(self, profile: str = "headless", timeout: int = 900,
                 model: str = "") -> None:
        self.profile = profile
        self.timeout = timeout
        self.model = model or "dsh-default"

    def complete(self, prompt: str, cwd: Optional[str] = None,
                 system: Optional[str] = None) -> AgentResult:
        import time
        t0 = time.time()
        full = f"{system}\n\n{prompt}" if system else prompt
        try:
            proc = subprocess.run(["dsh", "--profile", self.profile, full],
                                  cwd=cwd, capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return AgentResult(False, backend=self.name, model=self.model,
                               error=f"timeout after {self.timeout}s",
                               latency_s=time.time() - t0)
        if proc.returncode != 0:
            return AgentResult(False, backend=self.name, model=self.model,
                               error=f"rc={proc.returncode}: {proc.stderr[-300:]}",
                               latency_s=time.time() - t0)
        return AgentResult(True, text=proc.stdout, backend=self.name,
                           model=self.model, wrote_files=True,
                           latency_s=time.time() - t0)


class CmdBackend(Backend):
    """Shell template; {prompt} is written to a temp file and passed as {file}."""

    name = "cmd"

    def __init__(self, template: str, timeout: int = 900, model: str = "") -> None:
        self.template = template
        self.timeout = timeout
        self.model = model or "cmd"

    def complete(self, prompt: str, cwd: Optional[str] = None,
                 system: Optional[str] = None) -> AgentResult:
        import tempfile
        import time
        t0 = time.time()
        with tempfile.NamedTemporaryFile("w", suffix=".prompt", delete=False) as fh:
            fh.write(prompt)
            pf = fh.name
        cmd = self.template.format(prompt=shlex.quote(prompt), file=shlex.quote(pf),
                                   dir=shlex.quote(cwd or "."))
        try:
            proc = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True,
                                  text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return AgentResult(False, backend=self.name, model=self.model,
                               error=f"timeout after {self.timeout}s",
                               latency_s=time.time() - t0)
        finally:
            try:
                os.unlink(pf)
            except OSError:
                pass
        if proc.returncode != 0:
            return AgentResult(False, backend=self.name, model=self.model,
                               error=f"rc={proc.returncode}: {proc.stderr[-300:]}",
                               latency_s=time.time() - t0)
        return AgentResult(True, text=proc.stdout, backend=self.name,
                           model=self.model, latency_s=time.time() - t0)


class MockBackend(Backend):
    """Deterministic offline generator used by tests and dry runs.

    Produces task-appropriate artifacts that improve monotonically with the
    attempt index so the whole RSI loop is exercisable without an LLM.  When the
    prompt is a policy-improvement prompt it returns the seeded adaptive
    portfolio policy, so the dreaming stage is exercisable too.
    """

    name = "mock"

    def __init__(self, model: str = "mock", program_name: str = "solution.py",
                 task: str = "") -> None:
        self.model = model
        self.program_name = program_name
        self.task = task

    def complete(self, prompt: str, cwd: Optional[str] = None,
                 system: Optional[str] = None) -> AgentResult:
        if "prefix-only exploration policy" in (prompt or ""):
            return self._policy_response()
        attempt = 0
        branch = 0
        m = re.search(r"branch\s+(\d+)\s*/\s*attempt\s+(\d+)", prompt or "")
        if m:
            branch, attempt = int(m.group(1)), int(m.group(2))
        body = self._artifact(branch, attempt)
        text = (f"<<<FILE: proposal.md>>>\nMock attempt b{branch} a{attempt}.\n<<<END>>>\n"
                f"<<<FILE: {self.program_name}>>>\n{body}\n<<<END>>>\n")
        return AgentResult(True, text=text, backend=self.name, model=self.model,
                           files={})

    def _policy_response(self) -> AgentResult:
        src = (_PKG / "policies" / "portfolio.py").read_text(encoding="utf-8")
        text = (f"<<<FILE: method.py>>>\n{src}\n<<<END>>>\n")
        return AgentResult(True, text=text, backend=self.name, model=self.model,
                           files={})

    def _artifact(self, branch: int, attempt: int) -> str:
        if self.task == "lasso_path":
            return MOCK_LASSO.format(iters=20 + 10 * (branch + attempt))
        return MOCK_CIRCLE.format(n=10, jitter=0.5 - 0.03 * (branch * 3 + attempt))


_PKG = Path(__file__).resolve().parent



MOCK_CIRCLE = '''"""Mock circle-packing candidate."""
N = 10


def pack(n=N):
    pts = []
    for i in range(n):
        x = ((i * 0.61803398875) % 1.0)
        y = ((i * 0.38196601125) % 1.0)
        pts.append((0.05 + ({jitter} * 0.9) * x, 0.05 + ({jitter} * 0.9) * y))
    return pts
'''

MOCK_LASSO = '''"""Mock lasso-path candidate."""
import numpy as np


def lasso_path(X, y, lam_path):
    n, p = X.shape
    beta = np.zeros((p, len(lam_path)))
    for k, lam in enumerate(lam_path):
        b = getattr(lasso_path, "_warm", None)
        b = np.zeros(p) if b is None else b.copy()
        for _ in range({iters}):
            grad = X.T @ (X @ b - y) / n
            b = np.sign(b - grad / 1.0) * np.maximum(np.abs(b - grad) - lam, 0.0)
        lasso_path._warm = b
        beta[:, k] = b
    return beta
'''


# --------------------------------------------------------------------------- #
# factory                                                                     #
# --------------------------------------------------------------------------- #
def _gemini_cli_path() -> Optional[str]:
    for cand in (Path.home() / ".local" / "bin" / "gemini.py",
                 Path.home() / "gemini-cli" / "gemini.py"):
        if cand.exists():
            return str(cand)
    return _which("gemini.py") or _which("gemini")


def make_agent(spec: str = "deepseek", model: str = "", timeout: int = 900,
               program_name: str = "solution.py", task: str = "",
               max_tokens: int = 32768) -> Backend:
    spec = (spec or "deepseek").strip()
    if spec == "mock":
        return MockBackend(model="mock", program_name=program_name, task=task)
    if spec == "dsh":
        return DshBackend(timeout=timeout, model=model)
    if spec.startswith("gemini"):
        # Zero-API-cost discovery agent: the Gemini web CLI (browser-cookie auth).
        # Its stdout is pointer JSON, so the template writes the response to a file
        # in the attempt dir and cats it back for the FILE protocol.
        cli = _gemini_cli_path()
        if not cli:
            raise AgentError("gemini CLI not found (expected ~/.local/bin/gemini.py)")
        mdl = spec.split(":", 1)[1] if ":" in spec else (model or "flash")
        template = (f'python3 {shlex.quote(cli)} -m {shlex.quote(mdl)} '
                    f'-p "$(cat {{file}})" -o {{dir}}/resp.md >/dev/null 2>&1; '
                    f'cat {{dir}}/resp.md')
        return CmdBackend(template, timeout=timeout, model=f"gemini-{mdl}")
    if spec.startswith("cmd:"):
        return CmdBackend(spec[4:], timeout=timeout, model=model)
    if spec.startswith("openai:"):
        payload = spec[len("openai:"):]
        parts = payload.split("|")
        base = parts[0]
        mdl = parts[1] if len(parts) > 1 and parts[1] else (model or "gpt-4o-mini")
        keyfile = parts[2] if len(parts) > 2 else ""
        key = os.environ.get("OPENAI_API_KEY", "")
        if keyfile:
            key = Path(keyfile).expanduser().read_text(encoding="utf-8").strip()
        if not key:
            raise AgentError(f"no api key for {base} (env OPENAI_API_KEY or keyfile)")
        return OpenAIChatBackend(base, mdl, key, name="openai",
                                 timeout=timeout, max_tokens=max_tokens)
    if spec.startswith("deepseek"):
        mdl = spec.split(":", 1)[1] if ":" in spec else (model or "deepseek-v4-flash")
        key = _deepseek_key()
        if not key:
            raise AgentError("DEEPSEEK_API_KEY not found (env or ~/.dsh/.env)")
        return OpenAIChatBackend("https://api.deepseek.com", mdl, key,
                                 name="deepseek", timeout=timeout,
                                 max_tokens=max_tokens)
    raise AgentError(f"unknown agent spec {spec!r}")


def agent_list() -> List[Dict[str, Any]]:
    rows = [{"spec": "deepseek", "ready": bool(_deepseek_key()),
             "note": "DeepSeek API, one-shot text -> files"}]
    rows.append({"spec": "dsh", "ready": bool(_which("dsh")),
                 "note": "DeepSeek Harness headless, agentic file edits"})
    rows.append({"spec": "gemini[:flash|pro|thinking|lite]",
                 "ready": bool(_gemini_cli_path()),
                 "note": "zero-API-cost discovery agent via Gemini web CLI cookies"})
    rows.append({"spec": "mock", "ready": True, "note": "offline deterministic"})
    rows.append({"spec": "openai:<base>|<model>|<keyfile>", "ready": True,
                 "note": "any OpenAI-compatible endpoint"})
    rows.append({"spec": "cmd:<template>", "ready": True,
                 "note": "custom command, {prompt}/{file}/{dir}"})
    return rows


def _which(binary: str) -> Optional[str]:
    from shutil import which
    return which(binary)
