#!/usr/bin/env python3
"""Check that each file here is the exact source of the scored job named below.

The judge's content_hash for a kernel job is
    sha256(problem + NUL + team + NUL + source bytes)
so the check needs nothing but the Python standard library. No candidate code
is imported and nothing is sent anywhere.
"""
import hashlib
import pathlib
import sys

TEAM = "Sudhendra"
# problem: (scored job id, speedup of that job, the job's content_hash)
JOBS = {
    "muse_attention_decode": ("49f87f365df1496e", "2.630x", "5027cb8713f6543930c1d6db6961f031ab8f2625d51687ce48a93d7d3a4068db"),
    "muse_decode_layer": ("8178ab1226944045", "2.316x", "e6239ac6d8b9be3060c9e9a32a3731d2606f81959059b17c04e80db79e0608da"),
    "muse_moe_w4a16": ("4a34800e1b664536", "2.394x", "460b92768d610025ed4c594273ca3f295877974a4822e902f388018f50b09420"),
    "muse_prefill_layer": ("ea1c10521bde4fb5", "1.898x", "0185a333d590f4685755448cf60030ffbaf8d7c32cc3bcf4f0cc5be5bb9fd70b"),
}


def content_hash(problem: str, team: str, source: bytes) -> str:
    return hashlib.sha256(problem.encode() + b"\0" + team.encode() + b"\0" + source).hexdigest()


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    bad = 0
    for problem, (job, speedup, want) in JOBS.items():
        got = content_hash(problem, TEAM, (here / f"{problem}.py").read_bytes())
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {problem:24s} job {job} ({speedup})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
