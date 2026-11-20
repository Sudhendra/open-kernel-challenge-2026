"""warm_trigger.py -- arm the torch profiler on later-turn ("warm") prefill batches, after a cold first-turn storm.

profile_session.sh PROF=warm runs `watch` in the background while a fresh judge-shape load runs (research/
warm_prefill_2026-11-17.md: a 55-100 ms host-side stall precedes every warm prefill batch; all older traces are cold).

  watch: follow <server.log> from byte --offset (its size when the load started). Arm on the first "Prefill batch" line
         with #cached-token >= --min-cached and #queue-req <= --max-queue that comes after at least one "Decode batch"
         line that follows the load's first prefill line (prefill is scheduled before decode, so that decode line means the
         cold first-turn storm is over;
         --skip N arms on the N+1-th such line instead). Arming = POST /start_profile (no profile_by_stage, no
         start_step: the trace starts at once and covers every forward, prefill and decode) with CPU+GPU activities,
         with_stack, num_steps=--max-steps as a backstop; optional py-spy sidecar on the rank-0 scheduler. After
         --extend + 1 more "Prefill batch" lines (so --extend whole prefill forwards are inside the window) POST
         /stop_profile. The load ending (--done-file appears) before that also stops it. Summary JSON in --summary.
  scan:  the same detector over a saved server.log, per cold-storm rung start (offline check, no HTTP).

Why a log trigger and not the server's start_step: start_step is an absolute forward count, and the forward at which the
storm ends depends on the chunk count and on the turn-1 decode steps (acceptance-length dependent), so a fixed offset
lands tens of forwards off. The log carries the forward ids ([N] with SGLANG_LOG_FORWARD_ITERS=1); the summary records
them so the trace window maps back onto log lines. The arming batch itself is not in the trace (its line is logged
before its run_batch; the start request is read on the next scheduler iteration): the window starts at the next forward.
"""
import argparse, json, os, re, shutil, signal, subprocess, sys, time, urllib.error, urllib.request

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)")
IT = re.compile(r"(?:Prefill|Decode) batch \[(\d+)\]")
FIELDS = {k: re.compile(r"#%s: (\d+)" % k) for k in ("new-seq", "new-token", "cached-token", "running-req", "queue-req")}


def parse(line):
    """-> dict(kind=P|D, ts, it, new-seq, new-token, cached-token, running-req, queue-req) or None."""
    if "Prefill batch" in line: kind = "P"
    elif "Decode batch" in line: kind = "D"
    else: return None
    r = {"kind": kind, "ts": None, "it": None}
    m = TS.match(line); r["ts"] = m.group(1) if m else None
    m = IT.search(line); r["it"] = int(m.group(1)) if m else None
    for k, rx in FIELDS.items():
        m = rx.search(line); r[k] = int(m.group(1)) if m else None
    return r


def ts_s(ts):
    """'2026-11-17 23:55:15.171' -> seconds of day (float); None passes through."""
    if not ts: return None
    h, m, s = ts.split(" ")[1].split(":"); return int(h) * 3600 + int(m) * 60 + float(s)


class Detector:
    """Feed parsed lines in order. state: wait -> armed (feed returned 'arm') -> stop (feed returned 'stop')."""
    def __init__(self, min_cached=1024, max_queue=0, skip=0, extend=12):
        self.min_cached, self.max_queue, self.skip, self.extend = min_cached, max_queue, skip, extend
        self.seen_prefill = self.seen_decode = False; self.qualified = 0; self.state = "wait"; self.after = 0; self.decodes_after = 0
        self.arm = self.stop = None

    def feed(self, r):
        if r is None: return None
        if self.state == "wait":
            if r["kind"] == "D": self.seen_decode = self.seen_prefill; return None   # decode lines before the load's first prefill are stale
            self.seen_prefill = True
            if self.seen_decode and (r["cached-token"] or 0) >= self.min_cached and (r["queue-req"] or 0) <= self.max_queue:
                self.qualified += 1
                if self.qualified > self.skip: self.state = "armed"; self.arm = r; return "arm"
            return None
        if self.state == "armed":
            if r["kind"] == "D": self.decodes_after += 1; return None
            self.after += 1
            if self.after > self.extend: self.state = "stop"; self.stop = r; return "stop"
        return None


def post(url, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(url + path, data=data, method="POST", headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp: txt = resp.read().decode(errors="replace"); code = resp.status
    except urllib.error.HTTPError as e: txt = e.read().decode(errors="replace"); code = e.code
    except Exception as e: txt = f"{type(e).__name__}: {e}"; code = -1
    return {"path": path, "code": code, "reply": txt[:400], "s": round(time.time() - t0, 2)}


def rank0_pid(pattern):
    """pid of the rank-0 scheduler (setproctitle 'sglang::scheduler_TP0_EP0'), via pgrep -f, else ps."""
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10).stdout.split()
        if out: return int(out[0])
    except Exception: pass
    try:
        for ln in subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=10).stdout.splitlines():
            pid, _, args = ln.strip().partition(" ")
            if re.search(pattern, args) and pid.isdigit() and int(pid) != os.getpid(): return int(pid)
    except Exception: pass
    return None


def watch(a):
    t0 = time.time(); det = Detector(a.min_cached, a.max_queue, a.skip, a.extend)
    S = {"log": a.log, "offset": a.offset, "prefix": a.prefix, "params": {k: getattr(a, k) for k in ("min_cached", "max_queue", "skip", "extend", "max_steps", "with_stack")},
         "t_watch_start": t0, "first_line": None, "arm": None, "stop": None, "start_reply": None, "stop_reply": None, "pyspy": None, "outcome": None}
    def save():
        if a.summary:
            with open(a.summary, "w") as f: json.dump(S, f, indent=1)
    def log(msg): print(f"[warm] +{time.time() - t0:.1f}s {msg}", flush=True)
    spy = None; buf = b""; pos = a.offset; stopped = False
    def do_stop(why):
        nonlocal stopped
        if stopped: return
        stopped = True; log(f"stop ({why}): POST /stop_profile")
        S["stop_reply"] = post(a.url, "/stop_profile", None, a.http_timeout); S["stop_why"] = why
        log(f"stop reply {S['stop_reply']}  (an error here after the num_steps backstop already stopped it is harmless)")
    def _term(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    log(f"watching {a.log} from byte {a.offset}: arm on Prefill #cached-token >= {a.min_cached}, #queue-req <= {a.max_queue}, after a Decode line, skip {a.skip}; "
        f"stop after {a.extend} more prefill forwards; backstop num_steps {a.max_steps}; with_stack {a.with_stack}")
    try:
        while True:
            try:
                with open(a.log, "rb") as f: f.seek(pos); chunk = f.read(); pos = f.tell()   # bytes: --offset is `wc -c`
            except FileNotFoundError: chunk = b""
            buf += chunk; lines = [x.decode(errors="replace") for x in buf.split(b"\n")]; buf = buf[buf.rfind(b"\n") + 1:]; lines.pop()
            for ln in lines:
                r = parse(ln)
                if r is None: continue
                if S["first_line"] is None: S["first_line"] = r
                ev = det.feed(r)
                if ev == "arm":
                    S["arm"] = dict(r, t=round(time.time() - t0, 2)); log(f"ARM on: {ln.strip()[:220]}")
                    if a.pyspy_out:
                        pid = rank0_pid(a.pyspy_match); exe = shutil.which("py-spy")
                        if not exe or not pid: S["pyspy"] = {"skipped": f"py-spy {'missing' if not exe else 'ok'}, rank-0 pid {pid}"}; log(f"py-spy skipped: {S['pyspy']}")
                        else:
                            cmd = [exe, "record", "--pid", str(pid), "-d", str(a.pyspy_s), "--format", "raw", "-o", a.pyspy_out] + a.pyspy_args.split()
                            spy = subprocess.Popen(cmd, stdout=open(a.pyspy_out + ".log", "w"), stderr=subprocess.STDOUT)
                            S["pyspy"] = {"cmd": cmd, "pid": pid}; log(f"py-spy: {' '.join(cmd)}")
                    body = {"num_steps": a.max_steps, "activities": ["CPU", "GPU"], "with_stack": bool(a.with_stack), "record_shapes": False, "profile_prefix": a.prefix}
                    S["start_body"] = body; S["start_reply"] = post(a.url, "/start_profile", body, 120); log(f"start reply {S['start_reply']}")
                    save()
                elif ev == "stop":
                    S["stop"] = dict(r, t=round(time.time() - t0, 2)); log(f"STOP on: {ln.strip()[:220]}")
                    do_stop(f"{a.extend} prefill forwards after the arm line"); save()
            if det.state == "stop": break
            if a.done_file and os.path.exists(a.done_file):
                if det.state == "armed": do_stop("load ended before the prefill count")
                else: log("load ended before any warm prefill: nothing profiled")
                break
            if det.state == "wait" and time.time() - t0 > a.arm_timeout:
                log(f"no warm prefill within {a.arm_timeout}s: nothing profiled"); break
            time.sleep(a.poll_s)
    except KeyboardInterrupt:
        log("terminated")
        if det.state == "armed": do_stop("terminated")
    S["outcome"] = "ok" if det.state == "stop" else ("stopped_early" if stopped else "armed_no_stop") if det.state == "armed" else "not_armed"
    S["window"] = {"prefill_lines_after_arm": det.after, "decode_lines_after_arm": det.decodes_after,
                   "it_first": (S["arm"] or {}).get("it"), "it_last": (S["stop"] or {}).get("it")}
    save()
    if spy is not None:
        try: spy.wait(timeout=a.pyspy_s + 60)
        except subprocess.TimeoutExpired: spy.kill()
        S["pyspy"]["rc"] = spy.returncode; S["pyspy"]["out_bytes"] = os.path.getsize(a.pyspy_out) if os.path.exists(a.pyspy_out) else 0; save()
        log(f"py-spy rc {spy.returncode}, {S['pyspy']['out_bytes']} bytes")
    log(f"outcome {S['outcome']} window {S['window']}"); return 0


def rung_starts(recs, min_cached):
    """Cold-storm starts: a big low-cache prefill on an idle server (running 0) followed within 2 prefill lines by a queue >= 4."""
    P = [i for i, r in enumerate(recs) if r["kind"] == "P"]; out = []
    for j, i in enumerate(P):
        r = recs[i]
        if r["running-req"] == 0 and (r["new-token"] or 0) >= 8192 and (r["cached-token"] or 0) < min_cached:
            if any((recs[k]["queue-req"] or 0) >= 4 for k in P[j + 1:j + 3]) and (not out or i > out[-1] + 50): out.append(i)
    return out


def scan(a):
    lines = open(a.log, errors="replace", newline="").read().split("\n")   # raw \n only: server.log has bare \r (progress bars) that would shift line numbers
    recs = []; lno = []
    for n, ln in enumerate(lines, 1):
        r = parse(ln)
        if r: recs.append(r); lno.append(n)
    starts = rung_starts(recs, a.min_cached)
    print(f"{a.log}: {len(recs)} batch lines; cold-storm starts at log lines {[lno[i] for i in starts]}")
    for k, i0 in enumerate(starts, 1):
        if a.rung and k != a.rung: continue
        end = starts[k] if k < len(starts) else len(recs)
        det = Detector(a.min_cached, a.max_queue, a.skip, a.extend); arm = stop = None
        for i in range(i0, end):
            ev = det.feed(recs[i])
            if ev == "arm": arm = i
            if ev == "stop": stop = i; break
        r0 = recs[i0]; print(f"rung {k}: start line {lno[i0]} [{r0['it']}] {r0['ts']}")
        if arm is None: print("  no arm line"); continue
        ra = recs[arm]; dt = ts_s(ra["ts"]) - ts_s(r0["ts"]) if ra["ts"] and r0["ts"] else None
        print(f"  ARM  line {lno[arm]} [{ra['it']}] {ra['ts']}  +{dt:.1f}s after rung start: {lines[lno[arm] - 1].strip()[:200]}")
        if stop is not None:
            rs = recs[stop]; span = (rs["it"] - ra["it"]) if rs["it"] and ra["it"] else None
            print(f"  STOP line {lno[stop]} [{rs['it']}] {rs['ts']}  +{ts_s(rs['ts']) - ts_s(ra['ts']):.1f}s after arm; window = forwards {ra['it']}+1..{rs['it']} "
                  f"({span} forwards, {det.after - 1} whole prefills + decodes; {det.decodes_after} decode log lines at interval 20)")
        else: print(f"  no stop line before the rung ended ({det.after} prefill lines after arm)")
        # steady-state ratio over the rest of the rung: forwards per prefill batch after the arm line
        Pw = [recs[i] for i in range(arm, end) if recs[i]["kind"] == "P"]
        its = [r["it"] for r in recs[arm:end] if r["it"]]
        if len(Pw) > 1 and its: print(f"  rest of rung after arm: {len(Pw)} prefill batches over {max(its) - ra['it']} forwards = {(max(its) - ra['it']) / len(Pw):.1f} forwards per prefill")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("watch", "scan"):
        p = sub.add_parser(name); p.add_argument("--log", required=True)
        p.add_argument("--min-cached", type=int, default=1024); p.add_argument("--max-queue", type=int, default=0)
        p.add_argument("--skip", type=int, default=0); p.add_argument("--extend", type=int, default=12)
        if name == "scan": p.add_argument("--rung", type=int, default=0, help="only this cold-storm rung (1-based); 0 = all")
    w = sub.choices["watch"]
    w.add_argument("--offset", type=int, default=0); w.add_argument("--url", default="http://127.0.0.1:30000"); w.add_argument("--prefix", default="warm")
    w.add_argument("--max-steps", type=int, default=240, help="num_steps backstop (forwards, prefill + decode)")
    w.add_argument("--with-stack", type=int, default=1); w.add_argument("--summary", default=""); w.add_argument("--done-file", default="")
    w.add_argument("--arm-timeout", type=float, default=600); w.add_argument("--poll-s", type=float, default=0.1); w.add_argument("--http-timeout", type=float, default=900)
    w.add_argument("--pyspy-out", default="", help="py-spy raw output path; empty = no sidecar")
    w.add_argument("--pyspy-s", type=int, default=20); w.add_argument("--pyspy-args", default="-r 100 --idle --threads")
    w.add_argument("--pyspy-match", default=r"sglang::scheduler.*_TP0(_|$)")
    a = ap.parse_args()
    return watch(a) if a.cmd == "watch" else scan(a)


if __name__ == "__main__": sys.exit(main())
