#!/usr/bin/env python3
"""min_diff.py -- write a judge patch file with one line of context and trimmed headers, and prove it equals the U3 file.

  python3 harness/min_diff.py lanes/<tree> patches/<name>/<file>.diff [--base 297e54fd] [--context 1]

What it changes versus `git diff 297e54fd -- python/sglang` (3 lines of context):
  - `-U1`: one context line per hunk side instead of three;
  - the text after the second `@@` of every hunk header (the enclosing function name) is dropped;
  - `index <a>..<b> <mode>` lines are dropped (new-file / deleted-file mode lines are kept).
Both `git apply` and `patch -p1` accept the result. The script then proves it: it extracts the base tree twice,
applies the trimmed file with `git apply` in one copy and with `patch -p1` in the other, and compares every file
under python/sglang against the lane tree's HEAD. It exits non-zero on any difference.

Judge acceptance of this format is NOT proven by this script: the first use needs one check run on the judge.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile

HUNK = re.compile(rb"^(@@ -[0-9,]+ \+[0-9,]+ @@).*$")


def run(cmd, cwd=None, stdin=None):
    return subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True)


def trimmed_diff(tree, base, context):
    raw = run(["git", "-C", tree, "diff", f"-U{context}", base, "--", "python/sglang"]).stdout
    out = []
    for line in raw.split(b"\n"):
        m = HUNK.match(line)
        if m:
            line = m.group(1)
        elif line.startswith(b"index ") and b".." in line:
            continue
        out.append(line)
    return b"\n".join(out)


def tree_files(root):
    files = {}
    top = os.path.join(root, "python", "sglang")
    for d, dirs, names in os.walk(top, followlinks=False):
        dirs[:] = [x for x in dirs if not os.path.islink(os.path.join(d, x))]
        for n in names:
            p = os.path.join(d, n)
            if os.path.islink(p):
                continue
            files[os.path.relpath(p, root)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return files


def extract(tree, rev, dest):
    tar = run(["git", "-C", tree, "archive", rev, "python/sglang"]).stdout
    r = run(["tar", "-x", "-C", dest], stdin=tar)
    if r.returncode:
        sys.exit(f"extract {rev} failed: {r.stderr.decode()[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("out")
    ap.add_argument("--base", default="297e54fd")
    ap.add_argument("--context", type=int, default=1)
    a = ap.parse_args()
    if run(["git", "-C", a.tree, "status", "--porcelain", "--", "python/sglang"]).stdout.strip():
        sys.exit("tree has uncommitted changes under python/sglang; commit first")
    body = trimmed_diff(a.tree, a.base, a.context)
    full = run(["git", "-C", a.tree, "diff", a.base, "--", "python/sglang"]).stdout
    with tempfile.TemporaryDirectory() as tmp:
        patch = os.path.join(tmp, "p.diff")
        open(patch, "wb").write(body)
        want_dir = os.path.join(tmp, "want")
        os.mkdir(want_dir)
        extract(a.tree, "HEAD", want_dir)
        want = tree_files(want_dir)
        for tool, cmd in (("git apply", ["git", "apply", patch]), ("patch -p1", ["patch", "-p1", "-s", "-i", patch])):
            d = os.path.join(tmp, tool.split()[0])
            os.mkdir(d)
            extract(a.tree, a.base, d)
            r = run(cmd, cwd=d)
            if r.returncode:
                sys.exit(f"{tool} FAILED: {(r.stdout + r.stderr).decode()[:500]}")
            got = {k: v for k, v in tree_files(d).items() if not k.endswith((".orig", ".rej"))}
            if got != want:
                bad = sorted(k for k in set(got) | set(want) if got.get(k) != want.get(k))
                sys.exit(f"{tool}: tree differs from HEAD in {len(bad)} files, e.g. {bad[:5]}")
            print(f"{tool}: applies, {len(want)} files equal to HEAD")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "wb").write(body)
    print(f"wrote {a.out}: {len(body)} bytes (U3 file would be {len(full)}), sha256 {hashlib.sha256(body).hexdigest()}")


if __name__ == "__main__":
    main()
