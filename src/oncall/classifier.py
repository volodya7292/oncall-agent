"""Deterministic readonly/mutating/catastrophic classifier — the safety core.

This module is the ONLY thing standing between the model and a mutating tool call
without explicit user approval. It must be:

  * Default-deny: anything it can't prove read-only is mutating.
  * Compositional for Bash: pipes, &&, ||, ; chains of read-only commands are
    one classifier decision, not a cascade.
  * Independent of the LLM: pure-Python rules, no model calls.

Anything new added here MUST come with table-driven tests in
tests/test_classifier.py.
"""

from __future__ import annotations

import re
from typing import Any

import bashlex
import sqlglot
import sqlglot.expressions as exp

from .models import ClassifierVerdict, Verdict


# ---------------------------------------------------------------------------
# Catastrophic detection — STRUCTURAL (against parsed AST), not raw-text.
#
# Raw-text matching false-positives on prose inside heredocs / quoted strings
# (commit messages, embedded Python, comments). We only flag a command as
# catastrophic when its PROGRAM and ARGS, as parsed by bashlex, match the
# dangerous shape — never on text that happens to live inside a string literal.
#
# The single exception is the classic fork bomb, which is structurally
# distinctive enough that a raw-text match is safe.
# ---------------------------------------------------------------------------

_FORK_BOMB_RE = re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")

_CATASTROPHIC_PROGRAMS: frozenset[str] = frozenset({
    "shutdown", "reboot", "halt", "poweroff",
})

_DANGEROUS_PATHS: frozenset[str] = frozenset({
    "/", "/*", "~", "~/", "$HOME", "${HOME}", "/*/*",
})

_DD_DEVICE_RE = re.compile(r"^of=/dev/(sd|nvme|hd|disk|xvd|md)")

_PYTHON_PROGRAM_RE = re.compile(r"^python(\d+(\.\d+)?)?$")
_SCRIPT_LANG_BY_PROGRAM: dict[str, str] = {
    "node": "javascript", "nodejs": "javascript",
    "ruby": "ruby", "perl": "perl",
}
# Fallback heredoc regex for cases bashlex can't parse (e.g. quoted delimiters).
# Captures the body between `<< [-]?['"]?DELIM['"]?` and the line containing
# just DELIM. Used only when bashlex fails.
_HEREDOC_RE = re.compile(
    r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*\n(.*?)\n\1(?:\n|$)",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Read-only program tables
# ---------------------------------------------------------------------------

_ALWAYS_READONLY: frozenset[str] = frozenset({
    "ls", "cat", "head", "tail", "file", "stat", "wc", "grep", "rg", "ack",
    "fd", "tree", "pwd", "whoami", "id", "env", "printenv", "date",
    "uname", "hostname", "jq", "yq", "dig", "nslookup", "ping", "traceroute",
    "df", "du", "free", "top", "ps", "uptime", "which", "type", "echo",
    "printf", "basename", "dirname", "realpath", "true", "false", "test",
    "sort", "uniq", "tr", "cut", "tee",  # tee writes when given args; handled below
    "column", "expand", "fold", "rev", "nl", "tac", "less", "more",
    "cd", "pushd", "popd", "dirs",   # shell builtins, no side effects outside the shell
    "diff", "cmp", "comm",
    "md5sum", "sha1sum", "sha256sum", "shasum", "b3sum", "cksum",
    "od", "xxd", "hexdump",
    "tar", "zip", "unzip",  # tar/zip default to creating archives — actually mutating; remove these
})
# tar/zip CAN mutate; we don't want them in always-readonly. Strip them out.
_ALWAYS_READONLY = _ALWAYS_READONLY - frozenset({"tar", "zip", "unzip"})

# Programs we explicitly handle below; not in _ALWAYS_READONLY because they
# need subcommand inspection.
_GIT_READONLY_SUB: frozenset[str] = frozenset({
    "status", "diff", "log", "show", "blame", "ls-files", "ls-tree", "remote",
    "rev-parse", "describe", "name-rev", "branch", "tag", "config",
    "shortlog", "reflog",
})
_KUBECTL_READONLY_SUB: frozenset[str] = frozenset({
    "get", "describe", "logs", "top", "version", "api-resources",
    "api-versions", "explain", "auth", "cluster-info",
})
_DOCKER_READONLY_SUB: frozenset[str] = frozenset({
    "ps", "inspect", "logs", "images", "version", "info", "history",
    "stats", "port",
})
_REDIS_READONLY_CMDS: frozenset[str] = frozenset({
    "GET", "MGET", "HGET", "HGETALL", "HKEYS", "HVALS", "KEYS", "SCAN",
    "HSCAN", "SSCAN", "ZSCAN", "DBSIZE", "INFO", "TYPE", "EXISTS", "TTL",
    "PTTL", "OBJECT", "MEMORY", "CLIENT", "CONFIG", "PING", "ECHO",
    "RANDOMKEY", "LLEN", "LRANGE", "SMEMBERS", "SCARD", "SISMEMBER",
    "ZRANGE", "ZRANGEBYSCORE", "ZCARD", "ZSCORE", "GETRANGE", "STRLEN",
})

_SAFE_REDIRECT_TARGETS: frozenset[str] = frozenset({
    "/dev/null", "/dev/stderr", "/dev/stdout",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(tool_name: str, tool_input: dict[str, Any]) -> Verdict:
    """Deterministic verdict for one tool call. Default-deny posture."""
    if tool_name == "Bash":
        return _classify_bash(str(tool_input.get("command", "")))
    if tool_name in ("Read", "Glob", "Grep"):
        return _readonly_verdict(tool_name, tool_input)
    if tool_name in ("Edit", "Write", "NotebookEdit", "MultiEdit"):
        target = tool_input.get("file_path", "")
        return Verdict(
            kind=ClassifierVerdict.MUTATING,
            canonical=f"{tool_name}({target})" if target else f"{tool_name}({tool_input!r})",
            blast_radius=f"Writes to file '{target}'." if target else "Modifies file on disk.",
        )
    if tool_name in ("WebFetch", "WebSearch"):
        target = tool_input.get("url") or tool_input.get("query") or ""
        return Verdict(
            kind=ClassifierVerdict.READONLY,
            canonical=f"{tool_name}({target!r})",
            blast_radius=f"Read-only web access via {tool_name}.",
        )
    if tool_name == "mcp__oncall__messenger_inbox":
        return _classify_messenger(tool_input)
    if tool_name == "mcp__oncall__memory":
        return _classify_memory(tool_input)
    if tool_name == "mcp__oncall__ask_user":
        q = str(tool_input.get("question", ""))[:80]
        return Verdict(
            kind=ClassifierVerdict.READONLY,
            canonical=f"ask_user({q!r})",
            blast_radius="Relays a question to the human; no external mutation.",
        )
    # Unknown tool → mutating (default-deny posture).
    return Verdict(
        kind=ClassifierVerdict.MUTATING,
        canonical=f"{tool_name}({tool_input!r})",
        blast_radius=f"Unknown tool '{tool_name}'; default-deny.",
        reason="unknown_tool",
    )


# ---------------------------------------------------------------------------
# Bash classification
# ---------------------------------------------------------------------------

def _classify_bash(command: str) -> Verdict:
    canonical = command.strip()
    if not canonical:
        return Verdict(
            kind=ClassifierVerdict.MUTATING,
            canonical="(empty)",
            blast_radius="Empty bash command.",
            reason="empty",
        )
    # Fork bomb is structurally distinctive enough for a raw-text check.
    if _FORK_BOMB_RE.search(canonical):
        return Verdict(
            kind=ClassifierVerdict.CATASTROPHIC,
            canonical=canonical,
            blast_radius="Fork bomb — would exhaust process table.",
            reason="catastrophic:fork_bomb",
        )
    try:
        trees = bashlex.parse(canonical)
    except Exception as e:
        return Verdict(
            kind=ClassifierVerdict.MUTATING,
            canonical=canonical,
            blast_radius="Bash syntax not safely parseable.",
            reason=f"parse_error:{type(e).__name__}",
        )
    # First pass: scan the AST for catastrophic commands (program + args).
    for tree in trees:
        cat = _scan_catastrophic(tree)
        if cat:
            return Verdict(
                kind=ClassifierVerdict.CATASTROPHIC,
                canonical=canonical,
                blast_radius="Catastrophic command — irreversible system damage.",
                reason=f"catastrophic:{cat}",
            )
    # Extract embedded scripts (Python via -c / heredoc, etc.) for summarization.
    scripts: list[dict[str, str]] = []
    for tree in trees:
        scripts.extend(_extract_embedded_scripts(tree))
    embedded = scripts[0] if scripts else None
    # Second pass: readonly?
    for tree in trees:
        ok, why = _classify_ast(tree)
        if not ok:
            blast = "Includes at least one mutating step."
            if embedded:
                blast = (f"Embedded {embedded['language']} script. "
                         f"Summarize before approving.")
            return Verdict(
                kind=ClassifierVerdict.MUTATING,
                canonical=canonical,
                blast_radius=blast,
                reason=why,
                embedded_code=embedded,
            )
    return Verdict(
        kind=ClassifierVerdict.READONLY,
        canonical=canonical,
        blast_radius="Read-only shell command.",
    )


# ---------------------------------------------------------------------------
# Embedded-script extraction (Python via -c, heredoc, etc.)
# ---------------------------------------------------------------------------

def _extract_embedded_scripts(node) -> list[dict[str, str]]:
    """Walk an AST node, return any embedded scripts found."""
    found: list[dict[str, str]] = []
    kind = node.kind
    if kind == "command":
        s = _command_embedded_script(node)
        if s:
            found.append(s)
    elif kind in ("pipeline", "list"):
        for p in node.parts:
            if p.kind in ("pipe", "operator", "reservedword"):
                continue
            found.extend(_extract_embedded_scripts(p))
    elif kind == "compound":
        for child in getattr(node, "list", []) or []:
            found.extend(_extract_embedded_scripts(child))
    return found


def _command_embedded_script(node) -> dict[str, str] | None:
    """If the command is `python -c '<script>'`, `python << EOF ... EOF`,
    `node -e <script>`, etc., return the extracted script + language."""
    parts = node.parts
    words = [p for p in parts if p.kind == "word"]
    redirects = [p for p in parts if p.kind == "redirect"]
    if not words:
        return None
    program = _word_text(words[0]).lower()
    args = [_word_text(w) for w in words[1:]]

    lang = _script_language(program)
    if lang is None:
        return None

    # `<prog> -c '<script>'` or `<prog> -e '<script>'`
    flag = "-c" if lang == "python" else "-e"
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return {"language": lang, "source": args[i + 1]}
        if a.startswith(f"{flag}=") and len(a) > len(flag) + 1:
            return {"language": lang, "source": a[len(flag) + 1:]}

    # Heredoc: any `<<` redirect on the command. bashlex stores the body
    # in `redirect.heredoc.value`, *including* a trailing delimiter we need
    # to strip.
    for r in redirects:
        op = r.type
        if op not in ("<<", "<<-"):
            continue
        body = _heredoc_body(r)
        if body is not None:
            return {"language": lang, "source": body}

    return None


def _heredoc_body(redirect) -> str | None:
    """Return the body of a heredoc redirect, with the trailing delimiter stripped."""
    delim = _word_text(redirect.output) if redirect.output else ""
    hd = getattr(redirect, "heredoc", None)
    raw = getattr(hd, "value", None) if hd else None
    if not isinstance(raw, str):
        return None
    # bashlex includes a trailing delimiter line; strip it.
    if delim and raw.endswith(delim):
        body = raw[: -len(delim)]
        body = body.rstrip("\n")
    else:
        body = raw
    return body


def _script_language(program: str) -> str | None:
    if _PYTHON_PROGRAM_RE.match(program):
        return "python"
    return _SCRIPT_LANG_BY_PROGRAM.get(program)


# ---------------------------------------------------------------------------
# Structural catastrophic scan
# ---------------------------------------------------------------------------

def _scan_catastrophic(node) -> str | None:
    """Recurse the AST. Return a short reason string if catastrophic, else None."""
    kind = node.kind
    if kind == "command":
        return _command_catastrophic(node)
    if kind == "pipeline":
        # Detect classic `curl ... | sh` / `wget ... | bash`.
        stages = [p for p in node.parts if p.kind != "pipe"]
        if len(stages) >= 2:
            left_prog = _stage_program(stages[0])
            for st in stages[1:]:
                rp = _stage_program(st)
                if rp in ("sh", "bash", "zsh") and left_prog in ("curl", "wget", "fetch"):
                    return f"pipe_to_shell:{left_prog}_to_{rp}"
        for st in stages:
            r = _scan_catastrophic(st)
            if r:
                return r
        return None
    if kind == "list":
        for p in node.parts:
            if p.kind in ("operator", "reservedword"):
                continue
            r = _scan_catastrophic(p)
            if r:
                return r
        return None
    if kind == "compound":
        # Loops, conditionals — scan the body. Use the node's `list` child if present.
        for child in getattr(node, "list", []) or []:
            r = _scan_catastrophic(child)
            if r:
                return r
        return None
    return None


def _stage_program(node) -> str:
    """If `node` is (or wraps) a Command, return its first word; else ''."""
    if node.kind == "command":
        for p in node.parts:
            if p.kind == "word":
                return _word_text(p).lower()
    return ""


def _command_catastrophic(node) -> str | None:
    words = [p for p in node.parts if p.kind == "word"]
    if not words:
        return None
    program = _word_text(words[0]).lower()
    args = [_word_text(w) for w in words[1:]]

    if program in _CATASTROPHIC_PROGRAMS:
        return f"program:{program}"

    if program == "init" and args and args[0] in ("0", "6"):
        return f"init_runlevel:{args[0]}"

    if program == "rm":
        if any(a == "--no-preserve-root" for a in args):
            return "rm_no_preserve_root"
        # rm with any dangerous path as a target
        for a in args:
            if a in _DANGEROUS_PATHS:
                return f"rm_target:{a}"

    if program.startswith("mkfs"):
        return f"program:{program}"

    if program == "dd":
        for a in args:
            if _DD_DEVICE_RE.match(a):
                return f"dd_to_device:{a}"

    if program == "chmod" and "-R" in args and "777" in args:
        for a in args:
            if a in _DANGEROUS_PATHS:
                return f"chmod_777_to:{a}"

    return None


def _classify_ast(node) -> tuple[bool, str | None]:
    """Recurse into a bashlex AST node. Returns (is_readonly, reason_if_not)."""
    kind = node.kind
    if kind == "command":
        return _classify_command(node)
    if kind == "pipeline":
        for p in node.parts:
            if p.kind == "pipe":
                continue
            ok, why = _classify_ast(p)
            if not ok:
                return False, why
        return True, None
    if kind == "list":
        for p in node.parts:
            if p.kind in ("operator", "reservedword"):
                continue
            ok, why = _classify_ast(p)
            if not ok:
                return False, why
        return True, None
    if kind == "compound":
        # if/while/for/case/{ }/( ) — too complex to introspect safely
        return False, "compound_command"
    # function definitions, assignments, etc. — conservative deny
    return False, f"unsupported_node:{kind}"


def _classify_command(node) -> tuple[bool, str | None]:
    parts = node.parts
    words = [p for p in parts if p.kind == "word"]
    redirects = [p for p in parts if p.kind == "redirect"]
    assignments = [p for p in parts if p.kind == "assignment"]

    # A bare assignment (FOO=bar with no command) is read-only-ish — only sets a var
    # in the current shell. But it's so rare in agent-written commands that we keep
    # default-deny if there's no command word.
    if not words:
        return False, "no_command_word"

    # Variable assignments that prefix a command (e.g., `FOO=bar ls`) are OK as long
    # as the command itself is readonly. We accept the assignments without checking
    # — they can only affect the spawned process's env.
    _ = assignments

    # Redirects: any output redirect to a non-/dev/null target is mutating.
    for r in redirects:
        if not _redirect_is_readonly(r):
            return False, "writes_to_file"

    # Walk command-substitutions inside word parts
    for w in words:
        for sub in getattr(w, "parts", []) or []:
            sub_kind = getattr(sub, "kind", "")
            if sub_kind == "commandsubstitution":
                ok, why = _classify_ast(sub.command)
                if not ok:
                    return False, f"cmdsub:{why}"
            elif sub_kind == "processsubstitution":
                return False, "process_substitution"

    program = _word_text(words[0]).lower()
    args = [_word_text(w) for w in words[1:]]
    is_ro, why = _program_is_readonly(program, args)
    return is_ro, why


def _redirect_is_readonly(redirect) -> bool:
    """A redirect node. `.type` is the executor string ('>', '>>', '<', ...).
    `.output` is the word node for the target."""
    op = redirect.type
    if op in ("<", "<<", "<<<", "<<-", "0<"):
        return True
    # Output: only /dev/null-ish targets are safe.
    target = _word_text(redirect.output) if redirect.output else ""
    return target in _SAFE_REDIRECT_TARGETS


def _word_text(node) -> str:
    if node is None:
        return ""
    return getattr(node, "word", "") or ""


# ---------------------------------------------------------------------------
# Per-program readonly checks
# ---------------------------------------------------------------------------

def _program_is_readonly(program: str, args: list[str]) -> tuple[bool, str | None]:
    if not program:
        return False, "empty_program"
    if program in _ALWAYS_READONLY:
        # Special-case `tee` — without args it just writes to stdout; with args it writes files.
        if program == "tee" and any(not a.startswith("-") for a in args):
            return False, "tee_writes_file"
        return True, None
    if program == "find":
        return _find_readonly(args)
    if program == "xargs":
        return _xargs_readonly(args)
    if program == "sed":
        return _sed_readonly(args)
    if program == "awk":
        return _awk_readonly(args)
    if program == "git":
        return _git_readonly(args)
    if program == "kubectl":
        return _kubectl_readonly(args)
    if program == "docker":
        return _docker_readonly(args)
    if program == "ssh":
        return _ssh_readonly(args)
    if program == "aws":
        return _aws_readonly(args)
    if program == "gcloud":
        return _gcloud_readonly(args)
    if program in ("psql", "mysql", "sqlite3"):
        return _sql_cli_readonly(args)
    if program == "redis-cli":
        return _redis_readonly(args)
    return False, f"unknown_program:{program}"


def _find_readonly(args: list[str]) -> tuple[bool, str | None]:
    """`find` defaults to traversal-only (readonly). Becomes mutating with
    `-delete` or `-exec PROG ...` where PROG isn't readonly."""
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-delete":
            return False, "find_delete"
        if a in ("-exec", "-execdir", "-ok", "-okdir"):
            # Next token is the program; followed by args terminated by `;` or `+`.
            j = i + 1
            if j >= len(args):
                return False, "find_exec_no_program"
            prog = args[j].lower()
            inner_args: list[str] = []
            k = j + 1
            while k < len(args) and args[k] not in (";", "\\;", "+"):
                inner_args.append(args[k])
                k += 1
            inner_ok, why = _program_is_readonly(prog, inner_args)
            if not inner_ok:
                return False, f"find_exec:{why}"
            i = k + 1
            continue
        if a == "-fprint" or a == "-fprintf" or a == "-fls":
            return False, f"find_writes:{a}"
        i += 1
    return True, None


_XARGS_VALUE_FLAGS: frozenset[str] = frozenset({
    "-I", "-E", "-n", "-L", "-P", "-s", "-d", "-a",
    "--replace", "--max-args", "--max-procs", "--max-lines",
    "--max-chars", "--delimiter", "--arg-file", "--eof",
})


def _xargs_readonly(args: list[str]) -> tuple[bool, str | None]:
    """`xargs` runs another program. Recursively classify it."""
    i = 0
    while i < len(args):
        a = args[i]
        if a in _XARGS_VALUE_FLAGS:
            i += 2
            continue
        # GNU long-flag with value: --replace=X
        if a.startswith("--") and "=" in a and a.split("=", 1)[0] in _XARGS_VALUE_FLAGS:
            i += 1
            continue
        if a.startswith("-"):
            i += 1
            continue
        # found the target program
        return _program_is_readonly(a.lower(), args[i + 1:])
    # No target program → defaults to /bin/echo
    return True, None


def _sed_readonly(args: list[str]) -> tuple[bool, str | None]:
    """`sed` reads stdin / files and writes to stdout. It's mutating only when
    `-i`/`--in-place` is given. Other risky flags: `-f` script file (may itself
    be malicious — but reading a file isn't mutation; the SCRIPT may contain
    writes via `w` command, but that's an edge case we'll default-deny on)."""
    for a in args:
        if a == "-i" or a.startswith("-i.") or a == "--in-place" or a.startswith("--in-place="):
            return False, "sed_in_place"
        # macOS sed: -i '' (empty backup suffix); checked above via `-i`.
    # Watch out for `w FILE` write commands inside the script; cheap detection:
    # if the script body contains an unquoted ` w ` followed by a file, flag it.
    # Heuristic only.
    for a in args:
        if a.startswith("-") or not a:
            continue
        if " w " in f" {a} ":
            return False, "sed_w_command"
        break  # only inspect the first non-flag arg (the script)
    return True, None


def _awk_readonly(args: list[str]) -> tuple[bool, str | None]:
    """`awk` is generally read-only — reads input, writes stdout. Mutating
    only if the script calls `system()` or prints to a file with `>`/`|`.
    Best-effort: deny if script contains those patterns."""
    for a in args:
        if a.startswith("-"):
            continue
        if "system(" in a or ">" in a or "|&" in a or "| \"" in a:
            return False, "awk_side_effect"
        # only first script
        break
    return True, None


def _first_positional(args: list[str]) -> str | None:
    for a in args:
        if not a.startswith("-"):
            return a
    return None


def _git_readonly(args: list[str]) -> tuple[bool, str | None]:
    sub = _first_positional(args)
    if sub is None:
        return False, "git_no_subcommand"
    if sub not in _GIT_READONLY_SUB:
        return False, f"git_subcommand:{sub}"
    if sub == "config" and "--get" not in args and "-l" not in args and "--list" not in args:
        return False, "git_config_write"
    if sub == "branch" and any(a in ("-d", "-D", "-m", "-M", "-c", "-C") for a in args):
        return False, "git_branch_mutating"
    if sub == "tag" and any(not a.startswith("-") for a in args[args.index(sub) + 1:]):
        # `git tag` with no positional args lists tags; with a name it creates one.
        return False, "git_tag_create"
    return True, None


def _kubectl_readonly(args: list[str]) -> tuple[bool, str | None]:
    sub = _first_positional(args)
    if sub is None:
        return False, "kubectl_no_subcommand"
    if sub not in _KUBECTL_READONLY_SUB:
        return False, f"kubectl_subcommand:{sub}"
    if sub == "auth":
        # `kubectl auth can-i` is read-only; `kubectl auth reconcile` mutates.
        rest = [a for a in args if not a.startswith("-")]
        if len(rest) >= 2 and rest[1] not in ("can-i", "whoami"):
            return False, "kubectl_auth_mutating"
    return True, None


# SSH flags that take an argument (the value after the flag) per OpenSSH's
# manpage. Anything not in this set, but starts with `-`, is treated as a
# valueless flag (-A, -C, -tt, -v, …). Conservative: if a new flag arrived
# upstream and takes an arg, we'd over-skip the host — the only consequence
# is that ssh's remote command gets classified as "no_host" and we land
# mutating, which is the safe failure mode.
_SSH_FLAG_TAKES_ARG: frozenset[str] = frozenset({
    "-B", "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l",
    "-m", "-O", "-o", "-p", "-Q", "-R", "-S", "-W", "-w",
})


def _ssh_readonly(args: list[str]) -> tuple[bool, str | None]:
    """`ssh <flags> <host> <remote command…>` — the SSH transport itself
    isn't the blast surface; the *remote command* is. Strip ssh's own
    flags and host, then recursively run the remote command back through
    `_classify_bash`. The same rules that auto-allow `docker ps` locally
    will auto-allow it over SSH.

    `ssh host` with no remote command (= interactive login) is treated as
    mutating: an agent opening an interactive remote shell is not what we
    want auto-allowed, even though it doesn't "modify" anything by itself."""
    i = 0
    host: str | None = None
    cmd_parts: list[str] = []
    while i < len(args):
        a = args[i]
        if a == "--":
            i += 1
            continue
        if a in _SSH_FLAG_TAKES_ARG:
            # Flag + value pair; skip both. If `-iname.pem` (joined form)
            # showed up, it'd already not match _SSH_FLAG_TAKES_ARG and
            # fall through to the generic short-flag branch below.
            i += 2
            continue
        if a.startswith("-") and len(a) > 1:
            i += 1
            continue
        if host is None:
            host = a
            i += 1
            continue
        cmd_parts.append(a)
        i += 1
    if host is None:
        return False, "ssh_no_host"
    if not cmd_parts:
        return False, "ssh_interactive_session"
    # If the shell quoted the remote command as a single arg (the usual
    # case: `ssh host 'docker ps'`), cmd_parts is a one-element list with
    # the inner shell line. If the caller passed multiple unquoted words
    # (`ssh host docker ps`), join restores what the remote shell will
    # actually execute.
    remote_cmd = " ".join(cmd_parts)
    inner = _classify_bash(remote_cmd)
    if inner.kind == ClassifierVerdict.READONLY:
        return True, None
    if inner.kind == ClassifierVerdict.CATASTROPHIC:
        return False, f"ssh_inner_catastrophic:{inner.reason}"
    return False, f"ssh_inner:{inner.reason or 'mutating'}"


def _docker_readonly(args: list[str]) -> tuple[bool, str | None]:
    sub = _first_positional(args)
    if sub is None:
        return False, "docker_no_subcommand"
    if sub not in _DOCKER_READONLY_SUB:
        return False, f"docker_subcommand:{sub}"
    return True, None


def _aws_readonly(args: list[str]) -> tuple[bool, str | None]:
    # `aws <service> <verb> [...]` — first two positional tokens.
    positionals = [a for a in args if not a.startswith("-")]
    if len(positionals) < 2:
        return False, "aws_no_verb"
    service, verb = positionals[0], positionals[1]
    if service == "s3":
        # high-level s3 operations: ls is read-only; cp/mv/rm/sync mutate.
        return (True, None) if verb == "ls" else (False, f"aws_s3:{verb}")
    if verb.startswith(("describe-", "list-", "get-", "head-")):
        return True, None
    return False, f"aws_verb:{verb}"


def _gcloud_readonly(args: list[str]) -> tuple[bool, str | None]:
    positionals = [a for a in args if not a.startswith("-")]
    if not positionals:
        return False, "gcloud_empty"
    # last positional before flags is usually the verb in gcloud's deep nesting
    verb = positionals[-1]
    if verb in ("list", "describe", "get", "get-iam-policy"):
        return True, None
    return False, f"gcloud_verb:{verb}"


def _sql_cli_readonly(args: list[str]) -> tuple[bool, str | None]:
    for i, a in enumerate(args):
        if a == "-c" and i + 1 < len(args):
            return (True, None) if _sql_is_readonly(args[i + 1]) else (False, "sql_mutating")
        if a.startswith("--command="):
            return (True, None) if _sql_is_readonly(a.split("=", 1)[1]) else (False, "sql_mutating")
    return False, "sql_cli_no_-c"


def _redis_readonly(args: list[str]) -> tuple[bool, str | None]:
    # Skip connection flags. Find first non-flag (skipping -h/-p/-n/-a + values).
    skip_value = False
    flags_with_value = {"-h", "-p", "-n", "-a", "--user", "--pass", "-u", "--scan-pattern"}
    for a in args:
        if skip_value:
            skip_value = False
            continue
        if a in flags_with_value:
            skip_value = True
            continue
        if a.startswith("-"):
            continue
        return (True, None) if a.upper() in _REDIS_READONLY_CMDS else (False, f"redis_cmd:{a}")
    return False, "redis_no_command"


def _sql_is_readonly(sql: str) -> bool:
    """Parse SQL with sqlglot; true iff every top-level statement is a pure read."""
    try:
        statements = sqlglot.parse(sql)
    except Exception:
        return False
    if not statements:
        return False
    for stmt in statements:
        if stmt is None:
            continue
        if not _stmt_is_read_only(stmt):
            return False
    return True


_WRITE_EXP = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop,
              exp.Alter, exp.AlterTable if hasattr(exp, "AlterTable") else exp.Alter,
              exp.Merge, exp.Copy if hasattr(exp, "Copy") else exp.Insert,
              exp.TruncateTable if hasattr(exp, "TruncateTable") else exp.Insert)


def _stmt_is_read_only(stmt: exp.Expression) -> bool:
    # Top-level must be Select, With(Select), Describe, Show, Pragma, or Command{name in READ_KEYWORDS}.
    READ_KEYWORDS = {"SELECT", "WITH", "EXPLAIN", "DESCRIBE", "DESC", "SHOW", "PRAGMA"}
    top_key = (stmt.key or "").upper()
    if top_key not in READ_KEYWORDS and not isinstance(stmt, (exp.Select,)):
        # Sometimes sqlglot wraps in a Command for SHOW/DESCRIBE
        if not (isinstance(stmt, exp.Command) and (stmt.name or "").upper() in READ_KEYWORDS):
            return False
    # Walk the whole tree looking for embedded writes (defensive against CTE writes)
    for node in stmt.walk():
        if isinstance(node, _WRITE_EXP):
            return False
    return True


# ---------------------------------------------------------------------------
# MCP tool classifiers
# ---------------------------------------------------------------------------

def _classify_messenger(tool_input: dict[str, Any]) -> Verdict:
    op = str(tool_input.get("op", ""))
    if op in (
        "list", "read", "mark_read", "style", "read_image", "transcribe",
        "history", "search", "search_messages", "list_chats",
    ):
        return Verdict(
            kind=ClassifierVerdict.READONLY,
            canonical=f"messenger_inbox.{op}",
            blast_radius="Read-only messenger access.",
        )
    if op == "react":
        # Reactions are auto-allowed: single curated emoji, reversible,
        # no content leakage. Server validates the emoji allowlist.
        chat = str(tool_input.get("chat_id", "?"))
        emoji = str(tool_input.get("emoji", ""))
        return Verdict(
            kind=ClassifierVerdict.READONLY,
            canonical=f"messenger_inbox.react({chat}, {emoji!r})",
            blast_radius=(
                f"Sends a Telegram reaction ({emoji}) to one message in "
                f"chat {chat}. Reversible; no message content."
            ),
        )
    if op == "send":
        chat = str(tool_input.get("chat_id", "?"))
        text = str(tool_input.get("text", ""))
        return Verdict(
            kind=ClassifierVerdict.MUTATING,
            canonical=f"Send to chat {chat}: {text!r}",
            blast_radius=(
                f"Sends a Telegram message AS the user to chat {chat}. "
                f"Visible to recipient; cannot be unsent reliably."
            ),
        )
    return Verdict(
        kind=ClassifierVerdict.MUTATING,
        canonical=f"messenger_inbox.{op}",
        blast_radius=f"Unknown messenger op '{op}'.",
        reason="unknown_op",
    )


def _classify_memory(tool_input: dict[str, Any]) -> Verdict:
    """Operator memory store ops. `save` writes a row to local SQLite with
    no external blast radius and the operator already does it without
    approval — the executor inherits the same trust. Both ops are
    classified READONLY so the broker auto-allows."""
    op = str(tool_input.get("op", ""))
    if op == "query":
        q = str(tool_input.get("query", ""))[:60]
        return Verdict(
            kind=ClassifierVerdict.READONLY,
            canonical=f"memory.query({q!r})",
            blast_radius="Read-only memory lookup (local SQLite).",
        )
    if op == "save":
        text = str(tool_input.get("text", ""))[:80]
        return Verdict(
            kind=ClassifierVerdict.READONLY,
            canonical=f"memory.save({text!r})",
            blast_radius="Writes one local-DB row; no external effect.",
        )
    return Verdict(
        kind=ClassifierVerdict.MUTATING,
        canonical=f"memory.{op}",
        blast_radius=f"Unknown memory op '{op}'.",
        reason="unknown_op",
    )


def _readonly_verdict(tool_name: str, tool_input: dict[str, Any]) -> Verdict:
    if tool_name == "Read":
        target = tool_input.get("file_path", "")
        return Verdict(
            kind=ClassifierVerdict.READONLY,
            canonical=f"Read({target})",
            blast_radius=f"Reads file '{target}'.",
        )
    return Verdict(
        kind=ClassifierVerdict.READONLY,
        canonical=f"{tool_name}({tool_input!r})",
        blast_radius=f"Read-only {tool_name}.",
    )
