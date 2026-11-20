"""CPU unit test of harness/warm_trigger.py (PROF=warm): the detector on a real c16 server.log excerpt, and the live `watch`
loop against a fake server (normal stop, load ending before the stop, load ending before any warm prefill). No GPU.
  python3 -I harness/test_warm_trigger.py [real server.log to scan as well]
"""
import http.server, importlib.util, json, os, shutil, socketserver, subprocess, sys, tempfile, threading, time
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("wt", os.path.join(HERE, "warm_trigger.py")); wt = importlib.util.module_from_spec(spec); spec.loader.exec_module(wt)
FIX = os.path.join(HERE, "mock", "warm_server_log.txt")   # out/20261117T233405Z/variants/k14/server.log lines 1395-1545 (c16 rung)
lines = open(FIX, newline="").read().split("\n")

# 1. detector: stale decode lines before the storm are ignored; arms on the first warm prefill [1457], stops 12 prefills later
det = wt.Detector(); ev = [(n, det.feed(wt.parse(l))) for n, l in enumerate(lines, 1)]
arm = [n for n, e in ev if e == "arm"]; stop = [n for n, e in ev if e == "stop"]
assert len(arm) == 1 and len(stop) == 1, (arm, stop)
assert det.arm["it"] == 1457 and det.arm["cached-token"] == 73850 and det.stop["it"] == 1546 and det.after == 13, (det.arm, det.stop, det.after)
assert lines[0].count("Decode batch") == 0 and any("Decode batch" in l for l in lines[:5])   # the fixture starts with c32-tail decode lines
d2 = wt.Detector(skip=3); [d2.feed(wt.parse(l)) for l in lines]; assert d2.arm["it"] > 1457, d2.arm
d3 = wt.Detector(min_cached=10 ** 9); [d3.feed(wt.parse(l)) for l in lines]; assert d3.state == "wait"
print(f"detector ok: arm fixture line {arm[0]} [it {det.arm['it']}], stop line {stop[0]} [it {det.stop['it']}], {det.decodes_after} decode lines between")

# 2. watch against a fake server
posts = []
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); posts.append((self.path, self.rfile.read(n).decode()))
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok": true}')
srv = socketserver.TCPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_address[1]}"
tmp = tempfile.mkdtemp()

def run(feed, name, extra=()):
    log = os.path.join(tmp, name + ".log"); open(log, "w").write("[mock server] boot line\n"); off = os.path.getsize(log)
    done = os.path.join(tmp, name + ".done"); summ = os.path.join(tmp, name + ".json"); posts.clear()
    p = subprocess.Popen([sys.executable, "-I", os.path.join(HERE, "warm_trigger.py"), "watch", "--log", log, "--offset", str(off), "--url", url,
                          "--prefix", "warm_c16", "--done-file", done, "--summary", summ, "--poll-s", "0.02", *extra], stdout=subprocess.PIPE, text=True)
    with open(log, "a", newline="") as f:
        for l in feed: f.write(l + "\n"); f.flush(); time.sleep(0.002)
    open(done, "w").close(); out = p.communicate(timeout=60)[0]
    return json.load(open(summ)), [x[0] for x in posts], out

S, P, _ = run(lines, "full")
assert S["outcome"] == "ok" and P == ["/start_profile", "/stop_profile"] and S["arm"]["it"] == 1457 and S["stop"]["it"] == 1546, (S["outcome"], P)
assert S["start_body"] == {"num_steps": 240, "activities": ["CPU", "GPU"], "with_stack": True, "record_shapes": False, "profile_prefix": "warm_c16"}, S["start_body"]
cut = lines[: next(i for i, l in enumerate(lines) if "Prefill batch [1496]" in l)]   # the load ends 6 prefill batches after the arm line
S, P, _ = run(cut, "early")
assert S["outcome"] == "stopped_early" and P == ["/start_profile", "/stop_profile"] and S["stop_why"].startswith("load ended"), (S["outcome"], P)
S, P, out = run(lines[:60], "cold_only")   # storm only: never arms, never posts
assert S["outcome"] == "not_armed" and P == [] and "nothing profiled" in out, (S["outcome"], P)
S, P, out = run(lines, "nospy", ("--pyspy-out", os.path.join(tmp, "spy.txt"), "--pyspy-match", "no-such-process-xyz"))
assert S["outcome"] == "ok" and "skipped" in (S["pyspy"] or {}), S["pyspy"]   # sidecar is best effort: no pid -> skipped, window still taken
print("watch ok: full window, early load end -> stop posted, cold-only -> nothing posted, py-spy without a pid -> skipped")
srv.shutdown(); shutil.rmtree(tmp)

# 3. optional: the whole real log (all rungs)
if len(sys.argv) > 1: sys.exit(wt.scan(type("A", (), dict(log=sys.argv[1], min_cached=1024, max_queue=0, skip=0, extend=12, rung=0))()))
