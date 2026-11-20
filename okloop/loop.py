#!/usr/bin/env python3
"""okloop supervisor: per lane, attempt -> gate -> keep/dead -> submit. State in state/, log in state/log.tsv."""
import json, os, re, shlex, subprocess, sys, threading, time, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {}
for line in open(os.path.join(HERE, "config.env")):
    line = line.split("#", 1)[0].strip()
    if "=" in line: k, v = line.split("=", 1); CFG[k.strip()] = v.strip().strip('"')
ST = os.path.join(HERE, "state"); os.makedirs(ST, exist_ok=True)
LOG = os.path.join(ST, "log.tsv"); STATUS = os.path.join(ST, "status.json"); SUBS = os.path.join(ST, "submits.tsv")
lock = threading.Lock()

def now(): return dt.datetime.now().strftime("%m-%d %H:%M:%S")
def log(kind, lane, msg):
    with lock, open(LOG, "a") as f: f.write(f"{now()}\t{kind}\t{lane}\t{msg}\n")
    print(f"{now()} {kind:8s} {lane:9s} {msg}", flush=True)
def sh(cmd, **kw): return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
def learn(line):
    with lock, open(os.path.join(HERE, "LEARNINGS.md"), "a") as f: f.write(line.rstrip() + "\n")
def note_of(adir):
    p = os.path.join(adir, "NOTE.md")
    return open(p).read().strip().splitlines()[0][:300] if os.path.exists(p) and open(p).read().strip() else "(no NOTE.md)"

state = {"lanes": {}, "keeps": 0, "attempts": 0, "started": now(), "paused": False, "box_down": False}
def write_status():
    s = dict(state); s["best_score"] = read(os.path.join(ST, "best_score.txt")); s["aa_band"] = read(os.path.join(ST, "aa_band.txt"))
    s["submits_today"] = submits_today(); s["time"] = now()
    json.dump(s, open(STATUS, "w"), indent=1)
def read(p):
    try: return open(p).read().strip()
    except Exception: return None

# ---------- submissions ----------
def utc_day(): return dt.datetime.utcnow().strftime("%Y-%m-%d")
def submits_today():
    if not os.path.exists(SUBS): return 0
    return sum(1 for l in open(SUBS) if l.startswith(utc_day()))
def last_submitted_score():
    v = read(os.path.join(ST, "last_submitted_score.txt")); return float(v) if v else 0.0
def maybe_submit(lane, path, local_score, note):
    policy = CFG.get("SUBMIT", "auto"); auto_slots = int(CFG.get("AUTO_SLOTS", "8")); mingain = float(CFG.get("SUBMIT_MIN_GAIN", "0.005"))
    if local_score and last_submitted_score() and local_score / last_submitted_score() - 1 < mingain:
        log("hold", lane, f"keep not yet {mingain:.1%} over last submitted file ({local_score:.4f} vs {last_submitted_score():.4f}); accumulating"); return
    if policy == "manual" or submits_today() >= auto_slots:
        pend = os.path.join(ST, "pending_submit.txt"); open(pend, "w").write(path + "\n")
        log("pending", lane, f"submission waits for `okloop approve`: {path}"); return
    submit(lane, path, local_score, note)
def submit(lane, path, local_score, note):
    log("submit", lane, f"okc submit {os.path.basename(path)} ({note})")
    r = sh(f"cd {shlex.quote(HERE)} && okc submit {CFG['PROBLEM']} {shlex.quote(path)} --json", timeout=3600)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r'"(?:job|id|job_id)"\s*:\s*"([0-9a-f]{8,})"', out); sp = re.search(r'"speedup"\s*:\s*([0-9.]+)', out)
    job = m.group(1) if m else "?"; spd = sp.group(1) if sp else "?"
    with open(SUBS, "a") as f: f.write(f"{utc_day()}\t{now()}\t{job}\t{spd}\t{local_score}\t{path}\t{note}\n")
    open(os.path.join(ST, "last_submitted_score.txt"), "w").write(str(local_score))
    log("scored", lane, f"job {job} judge speedup {spd} (local {local_score})"); open(os.path.join(ST, f"submit_{job}.json"), "w").write(out)

# ---------- keep ----------
def keep(lane, adir, res):
    cand = os.path.join(adir, "candidate.py"); best = os.path.join(HERE, "best", "submission.py")
    s2 = res.get("scores2") or {}; s1 = res.get("scores1") or {}
    local = s2.get("candidate.py") or s1.get("candidate.py") or 0.0
    note = note_of(adir); tag = os.path.basename(adir)
    with lock:
        os.replace(best, best + ".prev"); subprocess.run(["cp", cand, best]); subprocess.run(["cp", cand, os.path.join(HERE, "best", "aa_copy.py")])
        open(os.path.join(ST, "best_score.txt"), "w").write(f"{local:.4f}")
        subprocess.run(["cp", cand, os.path.join(HERE, "best", f"keep_{tag}.py")])
        sh(f"cd {shlex.quote(HERE)} && git add best/submission.py LEARNINGS.md state/log.tsv >/dev/null 2>&1; git commit -qm 'keep {tag}: {note[:60]} ({res['r1']:+.4f}/{res['r2']:+.4f})' >/dev/null 2>&1")
        sh(f"{shlex.quote(os.path.join(HERE,'box.sh'))} A sync; {shlex.quote(os.path.join(HERE,'box.sh'))} B sync 2>/dev/null")
    state["keeps"] += 1
    learn(f"- KEEP {tag}: {note} | gate {res['r1']:+.4f} / {res['r2']:+.4f} | local {local:.4f}")
    log("KEEP", lane, f"{tag} {res['r1']:+.4f}/{res['r2']:+.4f} -> best {local:.4f}: {note}")
    maybe_submit(lane, best, local, f"{tag}: {note[:80]}")

# ---------- lane worker ----------
def lane_worker(lane, prov):
    L = state["lanes"][lane] = {"provider": prov, "attempt": None, "stage": "idle", "since": now(), "attempts": 0, "keeps": 0, "last": ""}
    while True:
        if state["paused"] or state["box_down"]: time.sleep(30); continue
        aid = f"{lane}-{dt.datetime.now().strftime('%m%d-%H%M%S')}"; adir = os.path.join(HERE, "attempts", aid)
        L.update(attempt=aid, stage="writing", since=now()); L["attempts"] += 1; state["attempts"] += 1; write_status()
        log("attempt", lane, f"{aid} ({prov})")
        t0 = time.time()
        r = subprocess.run([os.path.join(HERE, "attempt.sh"), lane, prov, aid], capture_output=True, text=True)
        if time.time() - t0 < 60 and r.returncode != 0:
            L.update(stage="idle", last=f"attempt died in {time.time()-t0:.0f}s rc={r.returncode}; backing off 5 min"); log("error", lane, f"{aid} died in {time.time()-t0:.0f}s rc={r.returncode}: {open(os.path.join(adir,'agent.log')).read()[-200:] if os.path.exists(os.path.join(adir,'agent.log')) else r.stderr[-200:]}"); write_status(); time.sleep(300); continue
        if r.returncode == 3:
            learn(f"- DEAD {aid} ({lane}): {note_of(adir)}"); L.update(stage="idle", last="no candidate"); log("dead", lane, f"{aid}: {note_of(adir)}"); write_status(); continue
        if r.returncode != 0:
            L.update(stage="idle", last=f"attempt error rc={r.returncode}"); log("error", lane, f"{aid} rc={r.returncode} {r.stderr[-200:]}"); write_status(); time.sleep(60); continue
        L.update(stage="gate", since=now()); write_status()
        stage_watch = threading.Thread(target=watch_stage, args=(L, adir), daemon=True); stage_watch.start()
        g = subprocess.run([os.path.join(HERE, "gate.sh"), "--stage", "all", os.path.join(adir, "candidate.py")], capture_output=True, text=True)
        resp = os.path.join(adir, "result.json"); res = json.load(open(resp)) if os.path.exists(resp) else {}
        if g.returncode == 0 and res.get("verdict") == "keep":
            L["keeps"] += 1; L["last"] = "KEEP"; keep(lane, adir, res)
        elif g.returncode == 2:
            L["last"] = "gate infra error"; log("error", lane, f"{aid} gate infra error: {g.stderr[-300:]}"); check_box(); time.sleep(60)
        else:
            why = read(os.path.join(adir, ".stage")) or "?"; ratios = f"{res.get('r1')}/{res.get('r2')}" if res else ""
            learn(f"- DEAD {aid} ({lane}): {note_of(adir)} | failed at {why} {ratios}")
            L["last"] = f"no ({why} {ratios})"; log("dead", lane, f"{aid} failed at {why} {ratios}: {note_of(adir)}")
        L.update(stage="idle", since=now()); write_status()

def watch_stage(L, adir):
    p = os.path.join(adir, ".stage")
    while L.get("stage") not in ("idle",):
        s = read(p)
        if s and L.get("stage") != s and not s.startswith("done"): L["stage"] = s; write_status()
        if s and s.startswith("done"): return
        time.sleep(10)

def research_worker(prov):
    every = int(CFG.get("RESEARCH_EVERY_MIN", "180")) * 60
    L = state["lanes"]["research"] = {"provider": prov, "attempt": None, "stage": "idle", "since": now(), "attempts": 0, "keeps": 0, "last": ""}
    while True:
        if state["paused"]: time.sleep(60); continue
        aid = f"research-{dt.datetime.now().strftime('%m%d-%H%M%S')}"; L.update(attempt=aid, stage="scanning", since=now()); L["attempts"] += 1; write_status()
        log("attempt", "research", aid)
        subprocess.run([os.path.join(HERE, "attempt.sh"), "research", prov, aid], capture_output=True, text=True)
        adir = os.path.join(HERE, "attempts", aid); new = [f for f in os.listdir(os.path.join(adir, "queue")) if f.endswith(".md")] if os.path.isdir(os.path.join(adir, "queue")) else []
        added = 0
        for f in new:
            dst = os.path.join(HERE, "queue", f)
            if not os.path.exists(dst): subprocess.run(["cp", os.path.join(adir, "queue", f), dst]); added += 1
        L.update(stage="idle", last=f"+{added} queue items"); log("research", "research", f"{aid}: {added} new queue items"); write_status()
        time.sleep(every)

def check_box():
    r = sh(f"{shlex.quote(os.path.join(HERE,'box.sh'))} A check", timeout=60)
    down = r.returncode != 0 or "DOWN" in r.stdout
    if down != state["box_down"]:
        state["box_down"] = down; log("box", "-", "box A DOWN: start it on Nebius, then `okloop box setup` if the container is gone" if down else "box A back up"); write_status()
    return not down

def pending_submit_worker():
    while True:
        pend = os.path.join(ST, "pending_submit.txt"); ok = os.path.join(ST, "approved")
        if os.path.exists(pend) and os.path.exists(ok):
            path = open(pend).read().strip(); os.remove(ok); os.remove(pend)
            submit("manual", path, float(read(os.path.join(ST, "best_score.txt")) or 0), "approved by user")
        time.sleep(20)

def main():
    log("start", "-", f"lanes {CFG['LANES']} research {CFG.get('RESEARCH_PROVIDER')}")
    if not check_box(): log("box", "-", "box A not reachable at start; waiting")
    threads = []
    for spec in CFG["LANES"].split():
        lane, prov = spec.split(":"); t = threading.Thread(target=lane_worker, args=(lane, prov), daemon=True); t.start(); threads.append(t); time.sleep(5)
    if CFG.get("RESEARCH_PROVIDER"): threading.Thread(target=research_worker, args=(CFG["RESEARCH_PROVIDER"],), daemon=True).start()
    threading.Thread(target=pending_submit_worker, daemon=True).start()
    while True:
        state["paused"] = os.path.exists(os.path.join(ST, "PAUSE")); check_box(); write_status(); time.sleep(120)

if __name__ == "__main__": main()
