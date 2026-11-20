#!/usr/bin/env python3
"""Rules lint: candidate vs best. Hard-fails on things never allowed; fails when the candidate ADDS a host sync.
usage: lint.py candidate.py best.py  -> exit 0 ok, 1 fail. Prints findings."""
import re, sys
cand, best = sys.argv[1], sys.argv[2]
c = open(cand, errors="ignore").read(); b = open(best, errors="ignore").read()
fails, warns = [], []
if not c.startswith("# okc: problem=muse_decode_layer"):
    fails.append("first line must be '# okc: problem=muse_decode_layer'")
for pat in [r"^\s*(import|from)\s+reference\b", r"^\s*(import|from)\s+checks\b", r"openkernelchallenge\.dev", r"\bokc\s+(submit|check)\b"]:
    if re.search(pat, c, re.M): fails.append(f"forbidden: {pat}")
if "Apache" not in c: fails.append("Apache-2.0 attribution header missing")
if "def kernel_fn" not in c or "def prepare" not in c: fails.append("prepare/kernel_fn missing")
# host syncs / value-dependent control flow: not allowed to increase vs best
for tok in [".item(", ".cpu(", ".numpy(", ".tolist(", "torch.cuda.synchronize", "cudaDeviceSynchronize", "time.sleep(", "lru_cache"]:
    nc, nb = c.count(tok), b.count(tok)
    if nc > nb: fails.append(f"adds host sync / caching token {tok!r}: {nb} -> {nc}")
# precision: no new bf16->fp8/int8 casts of activations outside what best already does
for tok in ["float8_e5m2", "torch.int8", "quantize_"]:
    nc, nb = c.count(tok), b.count(tok)
    if nc > nb: warns.append(f"new precision token {tok!r}: {nb} -> {nc} (verify it is allowed)")
# output caching: any module-level dict keyed on input tensors is suspicious
if re.search(r"_cache\[.*(data|positions|stream)\b", c) and not re.search(r"_cache\[.*(data|positions|stream)\b", b):
    warns.append("new cache keyed on inputs: make sure nothing is replayed across calls")
for w in warns: print("WARN", w)
for f in fails: print("FAIL", f)
print("LINT", "FAIL" if fails else "OK")
sys.exit(1 if fails else 0)
