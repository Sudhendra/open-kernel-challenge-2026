#!/usr/bin/env python3
"""comment_trim.py -- trim explanatory comments that OUR patch added, until the judge's reduced diff fits a byte target.

  python3 harness/comment_trim.py lanes/<tree> --base 297e54fd --target-bytes 385000 [--dry]
  python3 harness/comment_trim.py lanes/<tree> --base 297e54fd --verify-against <rev>   # proof only, no edits

What it touches: only lines that the patch ADDED relative to --base under python/sglang (lines inside pure-insertion
hunks of `git diff -U0 <base>`, or any line of a new file), in .py/.cu/.cuh/.h/.cpp files. Stock lines are never
edited (dropping a stock comment would ADD diff bytes and change stock source).
Candidates, applied largest saving first (ties by path, line) until the reduced diff (harness/min_diff.py format) is
<= --target-bytes:
  py-comment   run of whole-line `#` comments, removed (plus one now-doubled added blank line)
  cpp-comment  run of whole-line `//` or whole-line `/* ... */` comments, removed (same blank rule)
  docstring    multi-line docstring -> its first sentence on one line (never removed: no body ever becomes empty)
  env-shorten  comment run in srt/environ.py (user-facing env docs) -> one line (first sentence or first line)
  header       first comment run / module docstring of a NEW file -> first line / first sentence (one header kept)
Kept untouched: licence / SPDX / Copyright runs; `noqa`, `type:`, `fmt:`, pragma, NOLINT, clang-format, isort,
pylint, mypy comments; any run or docstring stating a numerical or precision property (precision, fp32, bf16, fp8,
fp16, rounding, arithmetic order, exact, lossless, greedy, bitwise, deterministic, +0.0, ...); C++ runs next to a line continuation.

Proof (always run before writing; `--verify-against REV` re-runs it between REV and the current tree):
  .py   ast.dump(parse(before)) == ast.dump(parse(after)) with docstrings stripped on both sides; tokenize streams equal
        after dropping COMMENT / NL tokens and docstring STRING tokens; docstring presence per node unchanged;
        compile() of the result; every produced line <= 400 chars.
  C++   two independent lexers: clang's raw lexer (`clang -Xclang -dump-raw-tokens`, no preprocessing, so no headers
        needed) and this file's own lexer (string / char / raw-string / digit-separator / line-continuation aware):
        token streams (kind, spelling, start-of-logical-line, adjacency) without comments must be identical.
  diff  no `-` (stock) line appears in `git diff -U0 base` that was not there before, i.e. no stock line was touched
        (one may disappear: a shortened docstring can equal the stock line it replaced).

Re-run on a later tree (e.g. the final assembly = stack_k36w + a branch lane): make the new lane worktree, commit its
code, then run this tool on THAT tree with the same command (it recomputes "added" lines against the base, so comments
the new commits added are trimmed too and already-trimmed ones are simply gone). Do not cherry-pick an older trim
commit: it would conflict with, or silently miss, the new commits' comments. Then commit by file name and run
`python3 harness/min_diff.py <tree> <out.diff>` + `bash harness/judge.sh lint <out.diff>`.
"""
import argparse
import ast
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter

SUFFIXES = (".py", ".cu", ".cuh", ".h", ".cpp")
HUNK_RE = re.compile(rb"^(@@ -[0-9,]+ \+[0-9,]+ @@).*$")
KEEP_PRAGMA = re.compile(
    r"noqa|type:\s|type:ignore|fmt:|pragma|NOLINT|clang-format|isort:|pylint|mypy|pyright|ruff:|SPDX|Copyright|"
    r"Licen[cs]e|^#!|coding[:=]",
    re.I,
)
# "order" counts only where it is about the order of arithmetic (fixed / summation / index order ...): "FCFS order",
# "stream order" or a (half, hk, b) layout order state no numerical property. Bare "round" / "rounds" (a sync round,
# placement rounds) do not either; "rounding" / "rounded" do.
_ORDER = (
    r"(?:fixed|same|summation|sum|reduction|merge|addition|add|accumulation|fold|index|slot|rank|ascending|"
    r"deterministic|any)[- ]order|order of (?:the )?(?:add|sum|fold|reduc|accumul|merge|partial)|"
    r"order(?:ed)? (?:sum|add|fold|merge|reduc)|sum order"
)
KEEP_WORDS = re.compile(
    r"precision|fp32|bf16|fp16|fp8|float32|bfloat16|rounding|rounded|" + _ORDER + r"|\bexact|lossless|"
    r"greedy|bitwise|bit-identical|bit-exact|deterministic|[+-]0\.0\b",
    re.I,
)
LICENCE = re.compile(r"SPDX|Copyright|Licen[cs]e", re.I)
MAX_LINE = 400
DOC_MAX = 220  # longest one-line docstring / env line produced, chars incl. indent


# COMMENT_TRIM_KEEP: an extra regex (env) of comment text that must survive the trim (e.g. the rationale of a bare constant).
if os.environ.get("COMMENT_TRIM_KEEP"):
    KEEP_WORDS = re.compile(KEEP_WORDS.pattern + "|" + os.environ["COMMENT_TRIM_KEEP"], KEEP_WORDS.flags)

def git(tree, *args, env=None, inp=None, check=True):
    r = subprocess.run(["git", "-C", tree, *args], capture_output=True, env=env, input=inp)
    if check and r.returncode:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.decode()[:400]}")
    return r.stdout


# ----------------------------------------------------------------------------------------------- diff helpers
def reduce_raw(raw):
    """Same transform as harness/min_diff.py trimmed_diff()."""
    out = []
    for line in raw.split(b"\n"):
        m = HUNK_RE.match(line)
        if m:
            line = m.group(1)
        elif line.startswith(b"index ") and b".." in line:
            continue
        out.append(line)
    return b"\n".join(out)


def diff_with_overrides(tree, base, overrides, context):
    """`git diff -U<context> base` of HEAD with `overrides` {path: bytes} applied, via a temporary index."""
    with tempfile.TemporaryDirectory() as td:
        env = dict(os.environ, GIT_INDEX_FILE=os.path.join(td, "index"))
        git(tree, "read-tree", "HEAD", env=env)
        for p, data in sorted(overrides.items()):
            mode = git(tree, "ls-tree", "HEAD", "--", p).split()[0].decode()
            sha = git(tree, "hash-object", "-w", "--stdin", inp=data).strip().decode()
            git(tree, "update-index", "--cacheinfo", f"{mode},{sha},{p}", env=env)
        return git(tree, "diff", "--cached", "--no-renames", f"-U{context}", base, "--", "python/sglang", env=env)


def reduced_bytes(tree, base, overrides):
    return reduce_raw(diff_with_overrides(tree, base, overrides, 1))


def stock_minus_lines(raw):
    return Counter(l for l in raw.split(b"\n") if l.startswith(b"-") and not l.startswith(b"---"))


def added_lines(tree, base, path, is_new, nlines):
    """1-based new-side line numbers of `path` (HEAD) that the patch added (`+` lines of `git diff -U0 base`; all lines
    of a new file). `+` lines of mixed hunks count: the stock `-` lines stay byte-identical (proved in prove())."""
    if is_new:
        return set(range(1, nlines + 1))
    raw = git(tree, "diff", "--no-renames", "-U0", base, "HEAD", "--", path)
    out = set()
    for line in raw.split(b"\n"):
        m = re.match(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if not m:
            continue
        new_s = int(m.group(3))
        new_n = 1 if m.group(4) is None else int(m.group(4))
        out.update(range(new_s, new_s + new_n))
    return out


# ----------------------------------------------------------------------------------------------- C++ lexer
_IDENT = re.compile(r"[A-Za-z_]\w*")
_RAW_PREFIX = {"R", "LR", "uR", "UR", "u8R"}
_STR_PREFIX = {"L", "u", "U", "u8"}


def cpp_lex(text):
    """Own lexer. Returns (comments, tokens). comments: [(start, end)] offsets (end exclusive).
    tokens: [(kind, spelling, start_of_logical_line, adjacent_to_previous)]. Multi-char punctuators are emitted
    char by char (with the adjacency flag that is equivalent). Backslash-newline is treated as a splice."""
    comments, toks = [], []
    i, n = 0, len(text)
    sol, adj = True, True

    def emit(kind, s):
        nonlocal sol, adj
        toks.append((kind, s.replace("\\\n", ""), sol, adj))
        sol, adj = False, True

    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n and text[i + 1] == "\n":  # splice
            i += 2
            continue
        if c == "\n":
            sol, adj = True, False
            i += 1
            continue
        if c in " \t\r\f\v":
            adj = False
            i += 1
            continue
        if text.startswith("//", i):
            j = i
            while j < n:
                if text[j] == "\n" and not (j > 0 and text[j - 1] == "\\"):
                    break
                j += 1
            comments.append((i, j))
            sol, adj = False, False
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            if j < 0:
                raise ValueError("unterminated /* comment")
            comments.append((i, j + 2))
            sol, adj = False, False
            i = j + 2
            continue
        m = _IDENT.match(text, i)
        if m:
            word = m.group(0)
            j = m.end()
            if word in _RAW_PREFIX and j < n and text[j] == '"':
                k = text.index("(", j)
                delim = text[j + 1:k]
                end = text.index(")" + delim + '"', k)
                emit("str", text[i:end + len(delim) + 2])
                i = end + len(delim) + 2
                continue
            if word in _STR_PREFIX and j < n and text[j] in "\"'":
                e = _quoted_end(text, j)
                emit("str", text[i:e])
                i = e
                continue
            emit("id", word)
            i = j
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i + 1
            while j < n:
                d = text[j]
                if d in "eEpP" and j + 1 < n and text[j + 1] in "+-":
                    j += 2
                elif d.isalnum() or d in "._":
                    j += 1
                elif d == "'" and j + 1 < n and text[j + 1].isalnum():
                    j += 1  # C++14 digit separator
                elif d == "\\" and j + 1 < n and text[j + 1] == "\n":
                    j += 2
                else:
                    break
            emit("num", text[i:j])
            i = j
            continue
        if c in "\"'":
            e = _quoted_end(text, i)
            emit("str", text[i:e])
            i = e
            continue
        emit("punct", c)
        i += 1
    return comments, toks


def _quoted_end(text, i):
    q = text[i]
    j = i + 1
    while True:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == q:
            return j + 1
        if c == "\n":
            raise ValueError(f"newline in literal at offset {i}")
        j += 1


def cpp_token_stream(text):
    return cpp_lex(text)[1]


_CLANG = shutil.which("clang")
_CLANG_END = re.compile(r"Loc=<[^>\n]*>\n?")


def clang_token_stream(text, suffix):
    """Token stream from clang's raw lexer (`-dump-raw-tokens`: no preprocessing, no includes; C++ mode for all
    suffixes since a CUDA run lexes twice). Whitespace and comment tokens dropped; each kept token carries its
    kind, spelling, StartOfLine flag and whether whitespace or a comment preceded it."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.cpp")
        with open(p, "w") as f:
            f.write(text)
        r = subprocess.run([_CLANG, "-x", "c++", "-std=c++17", "-Xclang", "-dump-raw-tokens", "-fsyntax-only", p],
                           capture_output=True, text=True)
        out = r.stderr
    toks, gap, prev = [], False, 0
    for m in _CLANG_END.finditer(out):
        rec = out[prev:m.start()]
        prev = m.end()
        kind = rec.split(" ", 1)[0]
        sm = re.match(r"\w+ '(.*)'\t", rec, re.S)
        if not sm:
            raise RuntimeError("cannot parse clang token record: " + rec[:200])
        spell = sm.group(1)
        flags = rec[sm.end():]
        if kind == "comment" or (kind == "unknown" and spell.strip() in ("", "\\")):
            gap = True
            continue
        toks.append((kind, spell, "StartOfLine" in flags, not (gap or "LeadingSpace" in flags)))
        gap = False
    if prev != len(out) or (not toks and text.strip()):
        raise RuntimeError("clang raw-token output not fully parsed: " + out[prev:prev + 300])
    return toks


# ----------------------------------------------------------------------------------------------- candidates
class Cand:
    __slots__ = ("path", "kind", "start", "end", "repl", "saving")

    def __init__(self, path, kind, start, end, repl, saving):
        self.path, self.kind, self.start, self.end, self.repl, self.saving = path, kind, start, end, repl, saving


def _lbytes(lines):
    return sum(len(l.encode()) + 2 for l in lines)  # "+" prefix and newline in the diff


def _first_sentence(text):
    para = re.split(r"\n\s*\n", text.strip(), maxsplit=1)[0]
    para = " ".join(para.split())
    m = re.search(r"(?<=[.!?])\s", para)
    return para[: m.start()] if m else para


def _runs(line_nos):
    runs, cur = [], []
    for ln in sorted(line_nos):
        if cur and ln == cur[-1] + 1:
            cur.append(ln)
        else:
            if cur:
                runs.append(cur)
            cur = [ln]
    if cur:
        runs.append(cur)
    return runs


def _blank_extra(lines, run, added):
    """Line number of an added blank line made redundant by removing `run` (blank before and blank after)."""
    s, e = run[0], run[-1]
    if s - 2 >= 0 and e < len(lines) and lines[s - 2].strip() == "" and lines[e].strip() == "" and (e + 1) in added:
        return e + 1
    return None


def _comment_run_cands(path, lines, run, added, new_file, is_header, env_file, text_of, kind):
    body = [lines[ln - 1] for ln in run]
    joined = "\n".join(text_of(l) for l in body)
    if any(KEEP_PRAGMA.search(l) for l in body):
        return []
    if KEEP_WORDS.search(joined):
        return _paragraph_cands(path, lines, run, text_of, kind)
    indent = re.match(r"\s*", body[0]).group(0)
    lead = body[0].strip()
    lead_tok = "#" if kind == "py-comment" else ("//" if lead.startswith("//") else None)
    if env_file or is_header:
        if len(run) == 1 or lead_tok is None:
            return []
        sent = _first_sentence(joined)
        one = f"{indent}{lead_tok} {sent}"
        if len(one) > DOC_MAX or "*/" in sent:
            one = body[0]
        return [Cand(path, "env-shorten" if env_file else "header", run[0], run[-1], [one], _lbytes(body) - _lbytes([one]))]
    end = run[-1]
    extra = _blank_extra(lines, run, added)
    removed = list(body)
    if extra is not None:
        end = extra
        removed.append(lines[extra - 1])
    return [Cand(path, kind, run[0], end, [], _lbytes(removed))]


def _paragraph_cands(path, lines, run, text_of, kind):
    """A run that states a numerical property: keep the paragraphs (split at empty comment lines) that state it,
    remove the others. Only for runs made of single-line `#` / `//` comments (never splits a /* */ block)."""
    lead = "#" if kind == "py-comment" else "//"
    if any(not lines[ln - 1].strip().startswith(lead) for ln in run):
        return []
    paras, cur, seps = [], [], []
    for ln in run:
        if text_of(lines[ln - 1]) == "":
            if cur:
                paras.append(cur)
            cur = []
            seps.append(ln)
        else:
            cur.append(ln)
    if cur:
        paras.append(cur)
    if len(paras) < 2:
        return []
    keep = [bool(KEEP_WORDS.search("\n".join(text_of(lines[ln - 1]) for ln in p))) for p in paras]
    out = []
    for i, p in enumerate(paras):
        if keep[i]:
            continue
        span = list(p)
        nxt = p[-1] + 1
        prv = p[0] - 1
        if i < len(paras) - 1 and nxt in seps:
            span.append(nxt)
        elif i == len(paras) - 1 and keep[i - 1] and prv in seps:
            span.insert(0, prv)
        if span != list(range(span[0], span[-1] + 1)):
            continue
        out.append(Cand(path, kind + "-para", span[0], span[-1], [], _lbytes([lines[ln - 1] for ln in span])))
    return out


def py_candidates(path, src, added, new_file):
    lines = src.split("\n")
    env_file = path.endswith("srt/environ.py")
    cands = []
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    first_code = next((t.start[0] for t in toks if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                                                                   tokenize.ENCODING)), len(lines) + 1)
    cl = set()
    for t in toks:
        if t.type == tokenize.COMMENT and lines[t.start[0] - 1].strip().startswith("#") and t.start[0] in added:
            cl.add(t.start[0])
    tree = ast.parse(src)
    has_moddoc = ast.get_docstring(tree, clean=False) is not None
    header_done = False
    for run in _runs(cl):
        is_header = new_file and not has_moddoc and not header_done and run[0] < first_code
        if is_header:
            header_done = True
        cands += _comment_run_cands(path, lines, run, added, new_file, is_header, env_file,
                                    lambda l: l.strip().lstrip("#").strip(), "py-comment")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or not node.body:
            continue
        d = node.body[0]
        if not (isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str)):
            continue
        s, e = d.lineno, d.end_lineno
        if s == e or not all(ln in added for ln in range(s, e + 1)):
            continue
        seg = "\n".join(lines[s - 1:e])
        before, after = seg[: d.col_offset], seg[d.col_offset:]
        if before.strip() or lines[e - 1][d.end_col_offset:].strip():
            continue
        m = re.match(r"^([rRuU]?)(\"\"\"|''')", after)
        if not m or not after.endswith(m.group(2)):
            continue
        q = m.group(2)
        inner = after[m.end(): -3]
        kind = "header" if isinstance(node, ast.Module) and new_file else "docstring"
        if KEEP_WORDS.search(inner):
            # states a numerical property: keep the summary paragraph and every paragraph that states one
            groups, cur = [], []
            last = e - 1 if lines[e - 1].strip() == q else e  # closing quotes alone on the last line
            for ln in range(s, last + 1):
                if lines[ln - 1].strip() == "":
                    if cur:
                        groups.append(cur)
                    cur = []
                else:
                    cur.append(ln)
            if cur:
                groups.append(cur)
            drop = []
            for g in groups[1:]:
                if s in g or (last == e and e in g) or KEEP_WORDS.search("\n".join(lines[ln - 1] for ln in g)):
                    continue
                if lines[g[0] - 2].strip() == "":
                    drop.append(g[0] - 1)  # its blank separator
                drop += g
            if not drop:
                continue
            kept = [lines[ln - 1] for ln in range(s, e + 1) if ln not in drop]
            cands.append(Cand(path, kind + "-para", s, e, kept, _lbytes(lines[s - 1:e]) - _lbytes(kept)))
            continue
        sent = _first_sentence(inner)
        if not sent or "\\" in sent or sent.endswith(q[0]) or q in inner or "\\" in inner:
            continue
        one = f"{before}{m.group(1)}{q}{sent}{q}"
        if len(one) > DOC_MAX:
            continue
        cands.append(Cand(path, kind, s, e, [one], _lbytes(lines[s - 1:e]) - _lbytes([one])))
    return cands


def cpp_candidates(path, src, added, new_file):
    lines = src.split("\n")
    comments, _ = cpp_lex(src)
    starts = [0]
    for l in lines[:-1]:
        starts.append(starts[-1] + len(l) + 1)

    def line_of(off):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    covered = {}
    spans = []
    for a, b in comments:
        la, lb = line_of(a), line_of(max(a, b - 1))
        spans.append((la, lb))
        for ln in range(la, lb + 1):
            covered.setdefault(ln, []).append((a, b))
    only = set()
    for ln, cs in covered.items():
        s0 = starts[ln - 1]
        line = lines[ln - 1]
        mask = [ch.isspace() for ch in line]
        for a, b in cs:
            for k in range(max(a, s0), min(b, s0 + len(line))):
                mask[k - s0] = True
        if all(mask):
            only.add(ln)
    first_code = None
    for ln, line in enumerate(lines, 1):
        if line.strip() and ln not in only:
            first_code = ln
            break
    first_code = first_code or len(lines) + 1
    cands = []
    header_done = False
    for run in _runs(ln for ln in only if ln in added):
        rs, re_ = run[0], run[-1]
        if any((la <= re_ and lb >= rs) and (la < rs or lb > re_) for la, lb in spans):
            continue  # a comment crosses the run boundary
        if (rs >= 2 and lines[rs - 2].rstrip().endswith("\\")) or lines[re_ - 1].rstrip().endswith("\\"):
            continue  # inside a continued macro
        is_header = new_file and not header_done and rs < first_code
        if is_header:
            header_done = True
            if any(LICENCE.search(lines[ln - 1]) for ln in run):
                continue
        cands += _comment_run_cands(path, lines, run, added, new_file, is_header, False,
                                    lambda l: l.strip().lstrip("/").strip(), "cpp-comment")
    return cands


def apply_cands(src, cands):
    lines = src.split("\n")
    for c in sorted(cands, key=lambda c: -c.start):
        lines[c.start - 1:c.end] = c.repl
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- proof
def _strip_docs(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            d = node.body[0]
            if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str):
                node.body = node.body[1:]
    return tree


def _doc_starts(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            d = node.body[0]
            if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str):
                out.append((d.lineno, d.col_offset))
    return out


def _py_code_tokens(src):
    docs = set(_doc_starts(ast.parse(src)))
    out = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type in (tokenize.COMMENT, tokenize.NL):
            continue
        if t.type == tokenize.STRING and t.start in docs:
            out.append(("DOC", ""))
            continue
        out.append((t.type, t.string))
    return out


def prove_py(before, after, path):
    errs = []
    try:
        compile(after, path, "exec")
    except SyntaxError as e:
        return [f"{path}: compile failed: {e}"]
    tb, ta = ast.parse(before), ast.parse(after)
    if len(_doc_starts(tb)) != len(_doc_starts(ta)):
        errs.append(f"{path}: docstring count changed")
    if ast.dump(_strip_docs(tb)) != ast.dump(_strip_docs(ast.parse(after))):
        errs.append(f"{path}: AST (docstrings stripped) differs")
    if _py_code_tokens(before) != _py_code_tokens(after):
        errs.append(f"{path}: code token stream differs")
    return errs


def prove_cpp(before, after, path):
    errs = []
    if cpp_token_stream(before) != cpp_token_stream(after):
        errs.append(f"{path}: own-lexer token stream differs")
    if _CLANG:
        suf = os.path.splitext(path)[1]
        if clang_token_stream(before, suf) != clang_token_stream(after, suf):
            errs.append(f"{path}: clang raw token stream differs")
    else:
        errs.append(f"{path}: clang not found (own lexer only); refusing")
    return errs


def prove(tree, base, before_map, after_map):
    errs = []
    produced_long = 0
    for p in sorted(after_map):
        b, a = before_map[p], after_map[p]
        if b == a:
            continue
        errs += prove_py(b, a, p) if p.endswith(".py") else prove_cpp(b, a, p)
        bl = set(b.split("\n"))
        for l in a.split("\n"):
            if l not in bl and len(l) > MAX_LINE:
                produced_long += 1
    if produced_long:
        errs.append(f"{produced_long} produced lines over {MAX_LINE} chars")
    changed = {p: a.encode() for p, a in after_map.items() if a != before_map[p]}
    raw0 = diff_with_overrides(tree, base, {}, 0)
    raw1 = diff_with_overrides(tree, base, changed, 0)
    gained = stock_minus_lines(raw1) - stock_minus_lines(raw0)
    if gained:  # a `-` line may only disappear (an edit that restores the stock line), never appear
        errs.append(f"new stock (-) lines in the diff, a stock line was touched: {list(gained)[:3]}")
    red = reduce_raw(diff_with_overrides(tree, base, changed, 1))
    longest = max((len(l) for l in red.split(b"\n")), default=0)
    if longest > MAX_LINE:
        errs.append(f"reduced diff has a line of {longest} chars")
    return errs, len(red), longest


# ----------------------------------------------------------------------------------------------- self test
def self_test():
    tricky = r'''#define A 1 // c \
  still comment
int a = 1'000'000; // sep
const char* s = "x // not /* a */ comment \" still";
char c = '"'; char d = '\''; auto w = L'x'; auto u8s = u8"//";
auto r = R"delim(// raw /* not */ )" still)delim";
/* block
   // inside */ int b = 2;
#define B(x) \
  /* mid */ \
  (x)
// whole line
int z = a/b; float f = 1e+5f; double h = 0x1p-3;
'''
    comments, toks = cpp_lex(tricky)
    texts = [tricky[a:b] for a, b in comments]
    want = ["// c \\\n  still comment", "// sep", "/* block\n   // inside */", "/* mid */", "// whole line"]
    assert texts == want, texts
    assert ("str", 'R"delim(// raw /* not */ )" still)delim"', False, False) in [(k, s, False, False) for k, s, _, _ in toks]
    if _CLANG:
        cl = clang_token_stream(tricky, ".cu")
        assert len(cl) == len(toks), (len(cl), len(toks))
        assert [t[2] for t in cl] == [t[2] for t in toks], "start-of-line flags differ"
        assert [t[3] for t in cl] == [t[3] for t in toks], "adjacency flags differ"
    # removing the whole-line comment is neutral; removing the spliced line is not
    neutral = tricky.replace("// whole line\n", "")
    assert cpp_token_stream(neutral) == cpp_token_stream(tricky)
    broken = tricky.replace("#define B(x) \\\n  /* mid */ \\\n", "#define B(x) \\\n")
    assert cpp_token_stream(broken) == cpp_token_stream(tricky)  # comment-only continued line: still neutral
    broken2 = tricky.replace("int a = 1'000'000;", "int a = 1'000'001;")
    assert cpp_token_stream(broken2) != cpp_token_stream(tricky)
    if _CLANG:
        assert clang_token_stream(neutral, ".cu") == clang_token_stream(tricky, ".cu")
        assert clang_token_stream(broken2, ".cu") != clang_token_stream(tricky, ".cu")
        spliced = tricky.replace("#define A 1 // c \\\n  still comment\n", "#define A 1 \\\n")
        assert clang_token_stream(spliced, ".cu") != clang_token_stream(tricky, ".cu")
        assert cpp_token_stream(spliced) != cpp_token_stream(tricky)
    py = 'def f(x):\n    """Do f.\n\n    More text here.\n    """\n    # gone\n    return x  # keep\n'
    py2 = 'def f(x):\n    """Do f."""\n    return x  # keep\n'
    assert prove_py(py, py2, "t.py") == []
    assert prove_py(py, py2.replace("return x", "return -x"), "t.py")
    print("self-test OK (own lexer" + (" + clang" if _CLANG else "") + ")")


# ----------------------------------------------------------------------------------------------- main
def load(tree, base, rev="HEAD"):
    names = git(tree, "diff", "--no-renames", "--name-status", base, rev, "--", "python/sglang").decode().split("\n")
    files = {}
    for row in names:
        if not row.strip():
            continue
        st, p = row.split("\t", 1)
        if st.startswith("D") or not p.endswith(SUFFIXES):
            continue
        files[p] = (st.startswith("A"), git(tree, "show", f"{rev}:{p}").decode())
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tree", nargs="?")
    ap.add_argument("--base", default="297e54fd")
    ap.add_argument("--target-bytes", type=int)
    ap.add_argument("--dry", action="store_true", help="select, prove and report, but write nothing")
    ap.add_argument("--verify-against", metavar="REV", help="prove HEAD of tree is a comment-only edit of REV")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not a.tree:
        ap.error("tree required")
    tree = a.tree
    if git(tree, "status", "--porcelain", "--", "python/sglang").strip():
        sys.exit("tree has uncommitted changes under python/sglang; commit first")
    self_test()

    if a.verify_against:
        old = {p: t for p, (_, t) in load(tree, a.verify_against, "HEAD").items()}
        names = sorted(old)
        before = {p: git(tree, "show", f"{a.verify_against}:{p}").decode() for p in names}
        after = {p: old[p] for p in names}
        errs = []
        for p in names:
            errs += prove_py(before[p], after[p], p) if p.endswith(".py") else prove_cpp(before[p], after[p], p)
            for l in after[p].split("\n"):
                if l not in set(before[p].split("\n")) and len(l) > MAX_LINE:
                    errs.append(f"{p}: produced line over {MAX_LINE}")
        other = git(tree, "diff", "--name-only", a.verify_against, "HEAD", "--").decode().split()
        stray = [x for x in other if x not in names]
        if stray:
            errs.append(f"files changed outside the checked set: {stray[:8]}")
        print(f"verify {a.verify_against}..HEAD: {len(names)} files changed, "
              f"{sum(p.endswith('.py') for p in names)} .py (ast+tokenize+compile), "
              f"{sum(not p.endswith('.py') for p in names)} C++ (own lexer + {'clang raw lexer' if _CLANG else 'NO clang'})")
        for e in errs:
            print("FAIL", e)
        print("VERIFY OK" if not errs else "VERIFY FAILED")
        sys.exit(1 if errs else 0)

    if a.target_bytes is None:
        ap.error("--target-bytes required")
    files = load(tree, a.base)
    cands = []
    for p, (is_new, src) in sorted(files.items()):
        add = added_lines(tree, a.base, p, is_new, src.count("\n") + 1)
        fn = py_candidates if p.endswith(".py") else cpp_candidates
        cands += fn(p, src, add, is_new)
    cands.sort(key=lambda c: (-c.saving, c.path, c.start))
    cands = [c for c in cands if c.saving > 0]
    start = len(reduced_bytes(tree, a.base, {}))
    print(f"reduced diff now {start} B, target {a.target_bytes} B, {len(cands)} candidates "
          f"(~{sum(c.saving for c in cands)} B available)")
    if start <= a.target_bytes:
        print("already at or under target: nothing to do")
        return
    need = start - a.target_bytes
    chosen = []
    for _ in range(12):
        chosen, acc = [], 0
        for c in cands:
            if acc >= need:
                break
            chosen.append(c)
            acc += c.saving
        bypath = {}
        for c in chosen:
            bypath.setdefault(c.path, []).append(c)
        after = {p: apply_cands(files[p][1], cs) for p, cs in bypath.items()}
        got = len(reduced_bytes(tree, a.base, {p: t.encode() for p, t in after.items()}))
        if got <= a.target_bytes or len(chosen) == len(cands):
            break
        need += got - a.target_bytes
    before_map = {p: files[p][1] for p in after}
    errs, final, longest = prove(tree, a.base, before_map, after)
    print(f"\n{'file':<92} {'cands':>5} {'saved':>7}  kinds")
    for p in sorted(bypath, key=lambda p: -sum(c.saving for c in bypath[p])):
        cs = bypath[p]
        kinds = Counter(c.kind for c in cs)
        print(f"{p:<92} {len(cs):>5} {sum(c.saving for c in cs):>7}  "
              + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    kinds = Counter()
    for c in chosen:
        kinds[c.kind] += c.saving
    print(f"\nby kind (line bytes): " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print(f"applied {len(chosen)}/{len(cands)} candidates in {len(after)} files; reduced diff {start} -> {final} B "
          f"(saved {start - final}); target {a.target_bytes}, margin {a.target_bytes - final} B; "
          f"limit margin {409600 - final} B; longest line {longest}")
    for e in errs:
        print("PROOF FAIL", e)
    if errs:
        sys.exit("proof failed: nothing written")
    print("PROOF OK: ast/tokenize/compile (.py), own + clang raw token streams (C++), stock lines untouched")
    if final > a.target_bytes:
        print(f"WARNING: candidates exhausted above target ({final} > {a.target_bytes})")
    if a.dry:
        print("--dry: nothing written")
        return
    for p, t in after.items():
        with open(os.path.join(tree, p), "w") as f:
            f.write(t)
    for p in after:
        if p.endswith(".py"):
            r = subprocess.run([sys.executable, "-m", "py_compile", os.path.join(tree, p)], capture_output=True)
            if r.returncode:
                sys.exit(f"py_compile failed on {p}: {r.stderr.decode()[:300]}")
    print(f"wrote {len(after)} files (py_compile OK); commit them by name:")
    print("  git -C " + tree + " add -- " + " ".join(sorted(after)))


if __name__ == "__main__":
    main()
