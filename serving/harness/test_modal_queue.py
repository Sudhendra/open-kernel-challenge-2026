"""CPU unit test of modal_serve.py's queue loop with a fake runner and a fake volume (no Modal, no GPU).
  python3 -I harness/test_modal_queue.py
"""
import os, re, sys, tempfile, types, shutil, json
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "modal_serve.py")).read()
# pull the pure helpers out of the module without importing modal
ns = {"os": os, "subprocess": __import__("subprocess"), "sys": sys, "signal": __import__("signal")}
for fn in ("_newest_out", "_parse_item", "_vol_sync", "_gpus_quiet", "_serve_queue"):
    m = re.search(r"^def %s\(.*?(?=^def |\Z)" % fn, src, re.S | re.M); assert m, fn; exec(m.group(0), ns)
root = tempfile.mkdtemp(); vol_dir = os.path.join(root, "vol"); os.makedirs(os.path.join(vol_dir, "out", "20260101T000000Z"))
open(os.path.join(vol_dir, "out", "20260101T000000Z", "proxy.json"), "w").write("{}")
# redirect /vol to our temp dir
for fn in ("_newest_out", "_serve_queue"):
    code = re.search(r"^def %s\(.*?(?=^def |\Z)" % fn, src, re.S | re.M).group(0).replace('"/vol', '"%s' % vol_dir)
    exec(code, ns)
class FakeVol:
    reloads = commits = 0
    def reload(self): FakeVol.reloads += 1
    def commit(self): FakeVol.commits += 1
ns["vol"] = FakeVol()
ns["_gpus_quiet"] = lambda *a, **k: None
runs = []
def fake_run_once(cmd, env):
    runs.append(dict(env)); d = os.path.join(vol_dir, "out", "2026010%dT000000Z" % (len(runs) + 1)); os.makedirs(d); open(os.path.join(d, "proxy.json"), "w").write("{}")
    return 0, d
ns["_run_once"] = fake_run_once
q = os.path.join(vol_dir, "queue", "t"); os.makedirs(q)
open(os.path.join(q, "01_a.env"), "w").write("VARIANTS=x:SRC=patched,PROF=1\nPROXY_RUNGS=1x1x2\nMAX_MIN=50\nWARM_EXTEND=8\nWARM_PYSPY_ARGS=-r 50 --idle\nFOO=1\n")
open(os.path.join(q, "02_b.env"), "w").write("VARIANTS=y:SRC=w\nSTOCK_REUSE=20260101T000000Z\nPROXY_SHAPE=short\nMEM_GATE=\nPROXY_CORPUS=/opt/x/torch\nPROXY_CLIENT=ids\n")
open(os.path.join(q, "03_c.env"), "w").write("VARIANTS=z:SRC=w\nSTOCK_REUSE=nonexistent\n")
open(os.path.join(q, "STOP"), "w").write("")   # present from the start: the loop must still drain nothing? No: STOP is checked first.
os.remove(os.path.join(q, "STOP"))
import threading, time
def stopper(): time.sleep(1); open(os.path.join(q, "STOP"), "w").write("")
threading.Thread(target=stopper, daemon=True).start()
ns["_serve_queue"]("queue/t", {"BASE": "1", "PRE_TESTS": "1"}, os.path.join(vol_dir, "out", "20260101T000000Z"), idle_min=1, hold_min=1)
assert len(runs) == 2, runs
assert runs[0]["STOCK_MODE"] == "reuse" and runs[0]["STOCK_REUSE_DIR"].endswith("20260101T000000Z") and runs[0]["PRE_TESTS"] == "" and runs[0]["VARIANTS"] == "x:SRC=patched,PROF=1" and runs[0]["MAX_MIN"] == "50"
assert runs[1]["STOCK_REUSE_DIR"].endswith("20260101T000000Z")
assert runs[1]["PROXY_SHAPE"] == "short" and runs[1]["MEM_GATE"] == "" and "PROXY_SHAPE" not in runs[0]
assert runs[1]["PROXY_CORPUS"] == "/opt/x/torch" and "PROXY_CORPUS" not in runs[0]   # item keys pass through; base env otherwise
assert runs[1]["PROXY_CLIENT"] == "ids" and "PROXY_CLIENT" not in runs[0]   # ladder client mode passes through
assert runs[0]["WARM_EXTEND"] == "8" and runs[0]["WARM_PYSPY_ARGS"] == "-r 50 --idle" and "FOO" not in runs[0] and "WARM_EXTEND" not in runs[1]   # PROF=warm knobs
# modal_queue.py push --prof-env writes exactly the keys _serve_queue passes through
import importlib.util
_spec = importlib.util.spec_from_file_location("mq", os.path.join(HERE, "modal_queue.py")); mq = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mq)
_serve_keys = re.findall(r'"(WARM_[A-Z_]+)"', re.search(r"^def _serve_queue\(.*?(?=^def |\Z)", src, re.S | re.M).group(0))
assert tuple(_serve_keys) == mq.WARM_KEYS, (_serve_keys, mq.WARM_KEYS)
assert mq.prof_env_lines("WARM_EXTEND=12, WARM_SKIP=4") == ["WARM_EXTEND=12", "WARM_SKIP=4"] and mq.prof_env_lines("") == []
try: mq.prof_env_lines("FOO=1"); raise AssertionError("FOO accepted")
except SystemExit: pass
done = sorted(os.listdir(os.path.join(q, "done"))); assert done == ["01_a.env", "01_a.result", "02_b.env", "02_b.result", "03_c.SKIPPED.env"], done
print("queue loop ok:", done, "reloads", FakeVol.reloads, "commits", FakeVol.commits); shutil.rmtree(root)
