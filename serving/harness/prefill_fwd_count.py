#!/usr/bin/env python3
"""Small-prefill forwards per rung from a variant's server.log (lane KSYNC A/B read-out).

  python3 harness/prefill_fwd_count.py out/<ts>/variants/<name>/server.log

Rungs are split on > 3 s log gaps. For each rung, the later phase starts 1 s after the last >= 8k-token prefill
(the end of the cold storm); prints its forwards, prefill forwards, admitted sequences and sequences per prefill
forward. k21 reference [measured]: c16 later phase 140 prefill forwards for 192 arrivals (1.37 per forward).
"""
import datetime
import re
import sys

ev = []
for line in open(sys.argv[1], errors="replace"):
    m = re.match(r"\[(\S+ \S+) TP0[^\]]*\] (Prefill|Decode) batch \[(\d+)\], (.*)", line)
    if m:
        t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
        ev.append((t, m.group(2), int(m.group(3)), dict(re.findall(r"#([\w-]+): ([\d.]+)", m.group(4)))))
segs = [[ev[0]]]
for a, b in zip(ev, ev[1:]):
    if b[0] - a[0] > 3:
        segs.append([])
    segs[-1].append(b)
for s in segs:
    big = [e for e in s if e[1] == "Prefill" and int(e[3]["new-token"]) >= 8000]
    maxrun = max(int(e[3].get("running-req", 0)) for e in s)
    if not big:
        continue
    late = [e for e in s if e[0] > big[-1][0] + 1.0]
    if len(late) < 5:
        print(f"rung (max running {maxrun}): no later phase after the cold storm")
        continue
    pf = [e for e in late if e[1] == "Prefill"]
    seqs = sum(int(e[3]["new-seq"]) for e in pf)
    print(f"rung (max running {maxrun}): later phase {late[-1][0] - late[0][0]:.1f} s, forwards {late[-1][2] - late[0][2]},"
          f" prefill forwards {len(pf)}, sequences {seqs}, per forward {seqs / max(1, len(pf)):.2f}")
