#!/usr/bin/env python3
"""Standalone research scanner: runs the research lane on a second provider, independent of loop.py.
usage: research2.py PROVIDER [LANE_NAME]   (e.g. research2.py opencode research-ds)
Writes new queue/*.md into the shared queue, logs to state/log.tsv, status to state/<lane>.json."""
import json, os, subprocess, sys, time, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {}
for line in open(os.path.join(HERE, "config.env")):
    line = line.split("#", 1)[0].strip()
    if "=" in line: k, v = line.split("=", 1); CFG[k.strip()] = v.strip().strip('"')
PROV = sys.argv[1]; LANE = sys.argv[2] if len(sys.argv) > 2 else f"research-{PROV}"
EVERY = int(os.environ.get("RESEARCH_EVERY_MIN", CFG.get("RESEARCH_EVERY_MIN", "180"))) * 60
ST = os.path.join(HERE, "state"); LOG = os.path.join(ST, "log.tsv"); STATUS = os.path.join(ST, f"{LANE}.json")
def now(): return dt.datetime.now().strftime("%m-%d %H:%M:%S")
def log(kind, msg):
    with open(LOG, "a") as f: f.write(f"{now()}\t{kind}\t{LANE}\t{msg}\n")
    print(f"{now()} {kind:8s} {LANE:9s} {msg}", flush=True)
S = {"provider": PROV, "attempt": None, "stage": "idle", "since": now(), "attempts": 0, "added": 0, "last": ""}
def status(): S["time"] = now(); json.dump(S, open(STATUS, "w"), indent=1)
log("start", f"provider {PROV} model {CFG.get('OPENCODE_MODEL') if PROV=='opencode' else ''} every {EVERY//60} min")
while True:
    if os.path.exists(os.path.join(ST, "PAUSE")): time.sleep(60); continue
    aid = f"{LANE}-{dt.datetime.now().strftime('%m%d-%H%M%S')}"; adir = os.path.join(HERE, "attempts", aid)
    os.makedirs(adir, exist_ok=True)
    if PROV == "opencode":  # non-interactive opencode: allow its tools inside the attempt dir
        json.dump({"$schema": "https://opencode.ai/config.json",
                   "permission": {"bash": "allow", "webfetch": "allow", "websearch": "allow", "edit": "allow", "write": "allow", "read": "allow"}},
                  open(os.path.join(adir, "opencode.json"), "w"), indent=1)
    S.update(attempt=aid, stage="scanning", since=now()); S["attempts"] += 1; status(); log("attempt", f"{aid} ({PROV})")
    t0 = time.time()
    r = subprocess.run([os.path.join(HERE, "attempt.sh"), "research", PROV, aid], capture_output=True, text=True)
    qdir = os.path.join(adir, "queue"); new = [f for f in os.listdir(qdir) if f.endswith(".md")] if os.path.isdir(qdir) else []
    added = 0
    for f in new:
        dst = os.path.join(HERE, "queue", f)
        if not os.path.exists(dst): subprocess.run(["cp", os.path.join(qdir, f), dst]); added += 1
    S["added"] += added; S.update(stage="idle", last=f"+{added} queue items in {(time.time()-t0)/60:.0f} min (rc={r.returncode})"); status()
    log("research", f"{aid}: {added} new queue items, {(time.time()-t0)/60:.0f} min, rc={r.returncode}")
    time.sleep(EVERY)
